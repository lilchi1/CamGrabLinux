// CameraThread.cpp — Поток камеры: RTSP → декодирование (nvv4l2decoder) → отображение.
//
// Отображение и JSON-логирование выполняются в отдельных потоках, чтобы в потоке
// GStreamer (внутри onNewSample) не было никакого I/O: рендер CUDA+X11 и запись
// на диск каждый кадр блокировали пайплайн и раздували decode_ms до ~100 мс
// при чистом декоде NVDEC ~5-17 мс.
#include "headers.h"
#include "Display.h"
#include "InferPipeline.h"
#include "PipelineJsonLogger.h"
#include "DetectionLog.h"
#include <X11/keysym.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <deque>
#include <condition_variable>
#include <map>

// Главный поток камеры: RTSP → декодирование → отображение
void cameraThread(std::string url, int camIdx) {
    logWrite("INFO", url, "Поток камеры запущен");

    // Открытие RTSP потока через FFmpeg
    RtspReader reader;
    if (!reader.open(url)) {
        logWrite("ERROR", url, "Не удалось открыть RTSP поток");
        setCamConnected(camIdx, false, url);
        stopCamRunning(camIdx);
        return;
    }

    int codecId = reader.codecId();
    int w = reader.width();
    int h = reader.height();

    logWrite("INFO", url, "Кодек: " + std::to_string(codecId) +
                          " Разрешение: " + std::to_string(w) + "x" +
                          std::to_string(h));

    // Обновление метаданных камеры
    {
        std::lock_guard<std::mutex> lock(g_camMtx);
        if (camIdx >= 0 && (size_t)camIdx < g_cams.size()) {
            g_cams[(size_t)camIdx].codec = codecId;
            g_cams[(size_t)camIdx].width = w;
            g_cams[(size_t)camIdx].height = h;
        }
    }

    // Режим отображения: overlay (xvimagesink в наше окно) или CUDA+X11
    bool useOverlay = (g_displayMode == "xvimagesink");

    // Создание окна отображения (размер из флагов --width/--height, чёрное до
    // первого кадра). В режиме бенчмарка окно не создаём — замер чистого декодирования.
    // В overlay-режиме окно — контейнер (без XImage/CUDA), клавиши наши.
    DisplayWindow* display = nullptr;
    if (!g_benchmarkMode) {
        display = new DisplayWindow();
        if (!display->open("Камера " + std::to_string(camIdx) + " - " + url,
                           g_winWidth, g_winHeight, useOverlay)) {
            delete display;
            display = nullptr;
        }
    }
    bool overlaySink = useOverlay && (display != nullptr);
    guintptr winHandle = overlaySink ? (guintptr)display->window() : 0;

    // Выбор декодера: аппаратный NVDEC через GStreamer nvv4l2decoder (JetPack 6)
    std::unique_ptr<GstDecoder> gstDec;

    if (codecId == AV_CODEC_ID_H264 || codecId == AV_CODEC_ID_H265) {
        gstDec = std::make_unique<GstDecoder>();
        if (gstDec->open(codecId, overlaySink, winHandle)) {
            logWrite("INFO", url, "Используется GStreamer nvv4l2decoder (аппаратный NVDEC)");
        } else {
            logWrite("ERROR", url, "GStreamer nvv4l2decoder недоступен");
            gstDec.reset();
        }
    }

    // Проверка декодера
    bool haveDecoder = gstDec && gstDec->isOpen();
    if (!haveDecoder) {
        logWrite("ERROR", url, "Нет доступного декодера");
        setCamConnected(camIdx, false, url);
        stopCamRunning(camIdx);
        reader.close();
        delete display;
        return;
    }

    // JSON-файл лога этапов обработки кадра (в папке logs/). Запись на диск
    // выполняет отдельный поток внутри PipelineJsonLogger — в потоках GStreamer
    // и отображения нет блокирующего I/O.
    PipelineJsonLogger jsonLogger;
    DetectionLogger detLogger;
    bool jsonEnabled = false;
    if (camIdx >= 0 && g_logDecodeSpeed) {
        std::string jsonPath = "logs/pipeline_log_" + std::to_string(camIdx) + ".json";
        if (jsonLogger.begin(jsonPath)) {
            jsonEnabled = true;
            logWrite("INFO", url, "JSON-логирование этапов включено: " + jsonPath);
        } else {
            logWrite("WARN", url, "Не удалось открыть JSON-файл лога: " + jsonPath);
        }
        // Боксы пишутся отдельным файлом (logs/detections_<cam>.json) — основной
        // лог этапов не засоряется массивами координат.
        std::string detPath = "logs/detections_" + std::to_string(camIdx) + ".json";
        if (detLogger.begin(detPath)) {
            logWrite("INFO", url, "Лог детекций (боксы) включён: " + detPath);
        } else {
            logWrite("WARN", url, "Не удалось открыть файл лога детекций: " + detPath);
        }
    }

    // Счётчики для FPS и статистики (обновляются из gst-потока appsink)
    uint64_t frameCount = 0;
    uint64_t lastFpsPrint = 0;
    auto lastFpsTime = std::chrono::steady_clock::now();
    double fps = 0.0;
    std::mutex statsMtx;

    // ─── Корреляция статистики декодера с кадром отображения ───────────────
    // DecodeStats из потока GStreamer кладётся в карту по PTS; поток отображения
    // забирает запись, когда кадр с этим PTS доходит до рендера. Полная запись
    // лога (декод + ИИ + рендер) собирается в потоке отображения; без него
    // (benchmark/overlay) — decode-only из колбэка.
    struct PendingStats {
        GstDecoder::DecodeStats st;
        int64_t frameNo = -1;
    };
    std::map<int64_t, PendingStats> pendingStats;  // PTS → статистика декодера
    bool fullFrameLogging = display && !overlaySink;
    std::string codecStr;
    if (codecId == AV_CODEC_ID_H264) codecStr = "H.264";
    else if (codecId == AV_CODEC_ID_H265) codecStr = "H.265";
    else if (codecId == AV_CODEC_ID_MJPEG) codecStr = "MJPEG";
    else codecStr = "Unknown";

    // ─── Детекция YOLO (TensorRT) ────────────────────────────────────────────
    // Инференс выполняется в потоке отображения (после показа кадра). Если
    // .engine не задан или инициализация не удалась — работаем без детекции.
    std::unique_ptr<InferPipeline> infer;
    int infProcessed = 0;   // кадров прогнано через детекцию
    int infDetFrames = 0;   // кадров с хотя бы одним объектом
    int infTotalDets = 0;   // всего боксов
    if (!g_modelPath.empty()) {
        infer = std::make_unique<InferPipeline>();
        const int numClasses = g_classNames.empty() ? 80 : (int)g_classNames.size();
        bool ok = false;
        if (g_yolov2Mode) {
            YoloV2Config cfg;
            cfg.grid = g_yolov2Grid;
            cfg.stride = 32;
            cfg.numAnchors = (int)g_yolov2Anchors.size() / 2;
            cfg.anchors = g_yolov2Anchors;
            const int inSize = g_modelInSize > 0 ? g_modelInSize : cfg.grid * cfg.stride;
            ok = infer->initV2(g_modelPath, inSize, inSize, numClasses, cfg,
                               g_confThresh, g_nmsThresh);
        } else {
            const int inSize = g_modelInSize > 0 ? g_modelInSize : 640;
            ok = infer->init(g_modelPath, inSize, inSize, numClasses, 8400,
                             g_confThresh, g_nmsThresh);
        }
        if (!ok) {
            logWrite("WARN", url, "Инициализация детекции не удалась — работаем без неё");
            infer.reset();
        } else {
            logWrite("INFO", url, "Детекция YOLO: " + g_modelPath +
                                  " (классов=" + std::to_string(numClasses) +
                                  ", conf=" + std::to_string(g_confThresh) +
                                  ", nms=" + std::to_string(g_nmsThresh) + ")");
        }
    }

    // ─── Асинхронное отображение ─────────────────────────────────────────────
    // Колбэк кадра (поток GStreamer) только переносит указатели NV12 в
    // ограниченную очередь (drop-oldest: при переполнении выбрасывается старый
    // кадр, показывается свежий). Пиксели не копируются: GstBuffer удерживается
    // живым через keepAlive до обработки в потоке отображения. Рендер (CUDA +
    // XPutImage) делает отдельный поток — чтобы CUDA/X11 не блокировали
    // пайплайн декодирования.
    struct DispFrame {
        uint8_t* y = nullptr;     // Y-плоскость NV12 (жива, пока keepAlive)
        uint8_t* uv = nullptr;    // UV-плоскость NV12
        int w = 0, h = 0, sy = 0, suv = 0;
        int64_t pts = -1;         // PTS кадра (для корреляции со статистикой декодера)
        std::shared_ptr<void> keepAlive;  // удержание GstBuffer (zero-copy)
    };
    std::mutex dispMtx;
    std::condition_variable dispCv;
    std::deque<DispFrame> dispQueue;
    bool dispStop = false;
    std::thread dispThread;

    if (display && !overlaySink) {
        dispThread = std::thread([&]() {
            // ─── Прогрев CUDA/TRT/CV-CUDA до цикла ─────────────────────────
            // Первый запуск ядер/движка дорогой: на 1-м кадре preprocess до
            // ~45 мс, infer до ~180 мс (прогрев контекста, компиляция ядер,
            // инициализация CV-CUDA). Прогоняем один пустой кадр — первый
            // реальный кадр пойдёт по «тёплому» пути без выбросов.
            if (infer && infer->ready()) {
                int ww = 1920, hh = 1080;
                if (gstDec && gstDec->width() > 0 && gstDec->height() > 0) {
                    ww = gstDec->width();
                    hh = gstDec->height();
                }
                std::vector<uint8_t> dummy((size_t)ww * hh * 3 / 2, 0);
                if (display->uploadNv12(dummy.data(), dummy.data() + (size_t)ww * hh,
                                        ww, ww, ww, hh)) {
                    GpuFrame wf;
                    wf.yPlane  = display->deviceY();
                    wf.uvPlane = display->deviceUV();
                    wf.width = ww; wf.height = hh;
                    wf.strideY = ww; wf.strideUV = ww;
                    wf.pts = -1;
                    infer->run(wf);
                }
            }

            while (true) {
                DispFrame f;
                {
                    std::unique_lock<std::mutex> lock(dispMtx);
                    dispCv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                        return dispStop || !dispQueue.empty();
                    });
                    if (dispStop) break;
                    if (dispQueue.empty()) continue;
                    f = std::move(dispQueue.front());
                    dispQueue.pop_front();
                }
                // Единый аплоад NV12 на GPU: один буфер используют и детекция,
                // и отображение — исключена повторная передача кадра CPU→GPU.
                // Аплоад синхронный (как исходный runHostNv12) — preprocessMs
                // остаётся чистым временем препроцессинга.
                InferTimings timings;
                auto up0 = std::chrono::steady_clock::now();
                if (!display->uploadNv12(f.y, f.uv, f.sy, f.suv, f.w, f.h))
                    continue;
                timings.uploadMs = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - up0).count();

                // Детекция ИИ на кадре (до отображения): инференс читает NV12
                // из того же device-буфера (GpuFrame без повторного аплоада) →
                // боксы, с замером этапов preprocess/infer (upload уже сделан).
                Detections dets;
                if (infer && infer->ready()) {
                    GpuFrame gf;
                    gf.yPlane  = display->deviceY();
                    gf.uvPlane = display->deviceUV();
                    gf.width = f.w; gf.height = f.h;
                    gf.strideY = f.w; gf.strideUV = f.w;
                    gf.pts = f.pts;
                    dets = infer->run(gf, &timings);
                    infProcessed++;
                    if (!dets.empty()) {
                        infDetFrames++;
                        infTotalDets += (int)dets.size();
                    }
                    if (infProcessed % 150 == 0 && infDetFrames > 0) {
                        logWrite("INFO", url, "DET: кадров с объектами=" +
                                 std::to_string(infDetFrames) + "/" +
                                 std::to_string(infProcessed) +
                                 ", боксов=" + std::to_string(infTotalDets));
                    }
                }

                // Отрисовка кадра (без/с ИИ-боксами) с замером времени рендера
                auto r0 = std::chrono::steady_clock::now();
                display->showFrameFromDevice(f.w, f.h, dets, g_classNames);
                auto r1 = std::chrono::steady_clock::now();
                double renderMs = std::chrono::duration<double, std::milli>(r1 - r0).count();

                // ─── Полная запись лога: декод + ИИ + рендер ─────────────
                if (jsonEnabled) {
                    int64_t frameNo = -1;
                    double decMs = -1.0, decFuncMs = -1.0, pushMs = -1.0, fiMs = -1.0;
                    int qDepth = 0;
                    {
                        std::lock_guard<std::mutex> lock(statsMtx);
                        auto it = pendingStats.find(f.pts);
                        if (it != pendingStats.end()) {
                            frameNo = it->second.frameNo;
                            decMs = it->second.st.decodeMs;
                            decFuncMs = it->second.st.decodeFuncMs;
                            pushMs = it->second.st.pushBlockMs;
                            fiMs = it->second.st.frameIntervalMs;
                            qDepth = it->second.st.queueDepth;
                            pendingStats.erase(it);
                        }
                    }

                    // ─── Боксы — отдельным файлом (logs/detections_<cam>.json) ─
                    if (!dets.empty()) {
                        std::string ts = getCurrentTimestamp();
                        for (const Detection& d : dets) {
                            DetectionRecord dr;
                            dr.frameNo = frameNo;
                            dr.pts = f.pts;
                            dr.timestamp = ts;
                            dr.classId = d.classId;
                            dr.className = (d.classId >= 0 && (size_t)d.classId < g_classNames.size())
                                               ? g_classNames[(size_t)d.classId]
                                               : std::to_string(d.classId);
                            dr.confidence = d.confidence;
                            dr.x1 = d.x1; dr.y1 = d.y1; dr.x2 = d.x2; dr.y2 = d.y2;
                            detLogger.append(dr);
                        }
                    }

                    // ─── Основной лог: этапы кадра (без массивов боксов) ────
                    PipelineLogRecord rec;
                    rec.frameNo = frameNo;
                    rec.pts = f.pts;
                    rec.timestamp = getCurrentTimestamp();
                    rec.source = std::to_string(f.w) + "x" + std::to_string(f.h);
                    rec.codec = codecStr;
                    rec.decodeMs = decMs;
                    rec.decodeFuncMs = decFuncMs;
                    rec.pushBlockMs = pushMs;
                    rec.frameIntervalMs = fiMs;
                    rec.queueDepth = qDepth;
                    rec.uploadMs = timings.uploadMs;
                    rec.preprocessMs = timings.preprocessMs;
                    rec.inferMs = timings.inferMs;
                    if (dets.empty()) rec.renderMs = renderMs;
                    else rec.renderAiMs = renderMs;
                    rec.detections = dets;
                    jsonLogger.append(rec);
                }
            }
        });

        FrameCallback cb = [&](const HostFrame& hf) {
            if (!g_running) return;
            DispFrame f;
            f.w = hf.width; f.h = hf.height;
            f.sy = hf.strideY; f.suv = hf.strideUV;
            f.pts = hf.pts;
            f.y = hf.yPlane; f.uv = hf.uvPlane;
            f.keepAlive = hf.keepAlive;   // удерживает GstBuffer живым (zero-copy)
            std::lock_guard<std::mutex> lock(dispMtx);
            if (dispQueue.size() >= 2) dispQueue.pop_front();  // drop-oldest
            dispQueue.push_back(std::move(f));
            dispCv.notify_one();
        };
        if (gstDec && gstDec->isOpen()) gstDec->setFrameCallback(cb);
    }

    // Колбэк статистики декодирования: накопление статистики и корреляция
    // с кадром отображения (по PTS). Вызывается из потока GStreamer — поэтому
    // без I/O и без форматирования строк (только лёгкие операции).
    double sumTotal = 0, sumPush = 0, sumFI = 0, sumDisp = 0;
    uint64_t cntStats = 0;

    GstDecoder::LatencyCb latCb = [&](int64_t pts, const GstDecoder::DecodeStats& st) {
        if (!g_running) return;
        std::lock_guard<std::mutex> lock(statsMtx);
        frameCount++;

        auto fmt = [](double v) {
            char buf[32];
            snprintf(buf, sizeof(buf), "%.1f", v);
            return std::string(buf);
        };
        auto ms = [&](double v) { return v >= 0.0 ? fmt(v) + " ms" : std::string("N/A"); };

        // ─── Корреляция статистики декодера с кадром отображения ─────────
        if (jsonEnabled) {
            if (fullFrameLogging) {
                // Полную запись соберёт поток отображения (декод+ИИ+рендер):
                // здесь только сохраняем статистику по PTS.
                if (pendingStats.size() >= 4096)
                    pendingStats.erase(pendingStats.begin());
                pendingStats[pts] = PendingStats{st, (int64_t)frameCount};
            } else {
                // Потока отображения нет (benchmark/overlay) — пишем запись
                // только с этапом декодирования.
                PipelineLogRecord rec;
                rec.frameNo = (int64_t)frameCount;
                rec.pts = pts;
                rec.timestamp = getCurrentTimestamp();
                rec.source = std::to_string(w) + "x" + std::to_string(h);
                rec.codec = codecStr;
                rec.decodeMs = st.decodeMs;
                rec.decodeFuncMs = st.decodeFuncMs;
                rec.pushBlockMs = st.pushBlockMs;
                rec.frameIntervalMs = st.frameIntervalMs;
                rec.queueDepth = st.queueDepth;
                jsonLogger.append(rec);
            }
        }

        // ─── Накопление статистики для сводки ──────────────────────────────
        sumTotal += st.decodeMs;
        sumPush += st.pushBlockMs;
        sumFI += st.frameIntervalMs;
        sumDisp += st.displayMs;
        cntStats++;

        // ─── Сводка каждые 100 кадров (только в терминал) ──────────────────
        if (frameCount - lastFpsPrint >= 100) {
            auto fpsNow = std::chrono::steady_clock::now();
            double fpsElapsed = std::chrono::duration<double>(fpsNow - lastFpsTime).count();
            if (fpsElapsed > 0) fps = 100.0 / fpsElapsed;
            lastFpsPrint = frameCount;
            lastFpsTime = fpsNow;

            {
                std::lock_guard<std::mutex> lock(g_camMtx);
                if (camIdx >= 0 && (size_t)camIdx < g_cams.size())
                    g_cams[(size_t)camIdx].fps = fps;
            }
            auto avg = [&](double s) { return cntStats > 0 ? s / cntStats : -1.0; };
            logWrite("INFO", url,
                "SUMMARY FPS=" + fmt(fps) +
                " | avg_total=" + ms(avg(sumTotal)) +
                " | avg_push_block=" + ms(avg(sumPush)) +
                " | avg_frame_interval=" + ms(avg(sumFI)) +
                " | avg_display=" + ms(avg(sumDisp)) +
                " | dropped_B=" + std::to_string(gstDec ? gstDec->droppedBFrames() : 0));
            sumTotal = sumPush = sumFI = sumDisp = 0;
            cntStats = 0;
        }
    };
    if (gstDec && gstDec->isOpen()) gstDec->setLatencyCallback(latCb);

    setCamConnected(camIdx, true, url);
    logWrite("INFO", url, "Камера подключена");

    // ─── Главный цикл: чтение RTSP → отправка пакетов в декодер ──────────
    while (g_running && isCamRunning(camIdx)) {
        // Обработка клавиатуры: ESC — выход из программы
        if (display) {
            display->pollEvents();
            int ks;
            bool pr;
            while (display->popKeyEvent(ks, pr)) {
                if (ks == XK_Escape && pr) {
                    logWrite("INFO", url, "ESC: выход из программы");
                    g_running = 0;
                }
            }
        }

        // Перезапуск конвейера при ошибке
        if (gstDec && gstDec->failed()) {
            logWrite("WARN", url, "Ошибка GStreamer-конвейера, перезапуск декодера...");
            gstDec->close();
            if (!gstDec->open(codecId, overlaySink, winHandle)) {
                logWrite("ERROR", url, "Переинициализация декодера не удалась");
                break;
            }
        }

        uint8_t* pktData;
        int pktSize;
        int64_t pts;

        // Чтение пакета из RTSP потока
        if (!reader.readPacket(pktData, pktSize, pts)) {
            // Ошибка чтения — попытка переподключения
            logWrite("WARN", url, "Ошибка чтения, попытка переподключения...");
            reader.close();
            std::this_thread::sleep_for(std::chrono::seconds(1));

            if (!reader.open(url, 10)) {
                logWrite("ERROR", url, "Переподключение не удалось");
                break;
            }

            // Переинициализация декодера после переподключения
            int newW = reader.width();
            int newH = reader.height();
            if (gstDec && gstDec->isOpen()) {
                gstDec->close();
                if (!gstDec->open(codecId, overlaySink, winHandle)) {
                    logWrite("ERROR", url, "Переинициализация декодера не удалась");
                    break;
                }
            }

            // Обновление метаданных после переподключения
            {
                std::lock_guard<std::mutex> lock(g_camMtx);
                if (camIdx >= 0 && (size_t)camIdx < g_cams.size()) {
                    g_cams[(size_t)camIdx].width = newW;
                    g_cams[(size_t)camIdx].height = newH;
                }
            }
            continue;
        }

        // Отправка пакета в аппаратный декодер NVDEC
        if (gstDec && gstDec->isOpen())
            gstDec->pushPacket(pktData, pktSize, pts);
    }  // while (g_running && ...)

    // Освобождение ресурсов: сначала остановить gst-поток, затем файлы/окно
    logWrite("INFO", url, "Поток камеры завершается");
    setCamConnected(camIdx, false, url);
    if (gstDec) gstDec->close();   // пайплайн остановлен — новых колбэков нет

    // Остановка потока отображения: сигнал → join → закрытие окна
    {
        std::lock_guard<std::mutex> lock(dispMtx);
        dispStop = true;
    }
    dispCv.notify_all();
    if (dispThread.joinable()) dispThread.join();

    // Завершение JSON-логов: закрытие массивов, flush, остановка писателей
    detLogger.end();
    jsonLogger.end();
    if (jsonEnabled) {
        logWrite("INFO", url, "JSON-лог сохранён: logs/pipeline_log_" +
                 std::to_string(camIdx) + ".json");
        logWrite("INFO", url, "Лог детекций сохранён: logs/detections_" +
                 std::to_string(camIdx) + ".json");
    }

    if (display) { delete display; display = nullptr; }
    reader.close();
    stopCamRunning(camIdx);
}
