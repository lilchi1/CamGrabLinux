// CameraThread.cpp — Поток камеры: RTSP → декодирование (nvv4l2decoder) → отображение.
//
// Отображение и CSV-логирование выполняются в отдельных потоках, чтобы в потоке
// GStreamer (внутри onNewSample) не было никакого I/O: рендер CUDA+X11 и запись
// на диск каждый кадр блокировали пайплайн и раздували decode_ms до ~100 мс
// при чистом декоде NVDEC ~5-17 мс.
#include "headers.h"
#include "Display.h"
#include "InferPipeline.h"
#include <X11/keysym.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <deque>
#include <condition_variable>

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

    // CSV-файл с детальной статистикой декодирования (в папке logs/).
    // Запись в файл выполняет отдельный поток-логгер, gst-поток только
    // кладёт готовые записи в очередь.
    std::ofstream csvFile;
    if (camIdx >= 0 && g_logDecodeSpeed) {
        mkdir("logs", 0755);
        std::string csvPath = "logs/decode_times_" + std::to_string(camIdx) + ".csv";
        csvFile.open(csvPath);
        if (csvFile.is_open()) {
            csvFile << "resolution,frame_no,pts,codec,source_width,source_height,"
                    << "decode_ms,decode_func_ms,push_block_ms,display_ms,frame_interval_ms,"
                    << "queue_depth,decoded_at\n";
            logWrite("INFO", url, "CSV-логирование включено: " + csvPath);
        } else {
            logWrite("WARN", url, "Не удалось открыть CSV-файл со статистикой декодирования");
        }
    }

    // Счётчики для FPS и статистики (обновляются из gst-потока appsink)
    uint64_t frameCount = 0;
    uint64_t lastFpsPrint = 0;
    auto lastFpsTime = std::chrono::steady_clock::now();
    double fps = 0.0;
    std::mutex statsMtx;

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
    // Колбэк кадра (поток GStreamer) только копирует NV12 в ограниченную очередь
    // (drop-oldest: при переполнении выбрасывается старый кадр, показывается
    // свежий). Рендер (CUDA + XPutImage) делает отдельный поток — чтобы CUDA/X11
    // не блокировали пайплайн декодирования.
    struct DispFrame {
        std::vector<uint8_t> y;    // Y-плоскость NV12
        std::vector<uint8_t> uv;   // UV-плоскость NV12
        int w = 0, h = 0, sy = 0, suv = 0;
    };
    std::mutex dispMtx;
    std::condition_variable dispCv;
    std::deque<DispFrame> dispQueue;
    bool dispStop = false;
    std::thread dispThread;

    if (display && !overlaySink) {
        dispThread = std::thread([&]() {
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
                display->showFrame(f.y.data(), f.uv.data(), f.w, f.h, f.sy, f.suv);

                // Детекция на показанном кадре: аплоад NV12 на GPU → инференс →
                // оверлей боксов. Координаты детекций — в пикселях исходного кадра.
                if (infer && infer->ready()) {
                    Detections dets = infer->runHostNv12(f.y.data(), f.uv.data(),
                                                         f.w, f.h, f.sy, f.suv, -1);
                    infProcessed++;
                    if (!dets.empty()) {
                        display->showDetections(dets, g_classNames, f.w, f.h);
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
            }
        });

        FrameCallback cb = [&](uint8_t* y, uint8_t* uv, int cw, int ch,
                               int sy, int suv, int64_t pts) {
            (void)pts;
            if (!g_running) return;
            DispFrame f;
            f.w = cw; f.h = ch; f.sy = sy; f.suv = suv;
            f.y.assign(y, y + (size_t)sy * ch);
            f.uv.assign(uv, uv + (size_t)suv * (ch / 2));
            std::lock_guard<std::mutex> lock(dispMtx);
            if (dispQueue.size() >= 2) dispQueue.pop_front();  // drop-oldest
            dispQueue.push_back(std::move(f));
            dispCv.notify_one();
        };
        if (gstDec && gstDec->isOpen()) gstDec->setFrameCallback(cb);
    }

    // ─── Асинхронный CSV-логгер ──────────────────────────────────────────────
    struct LogRec {
        uint64_t frameNo = 0;
        int64_t pts = -1;
        double decodeMs = -1.0;
        double decodeFuncMs = -1.0;
        double pushBlockMs = -1.0;
        double displayMs = -1.0;
        double frameIntervalMs = -1.0;
        int queueDepth = 0;
    };
    std::mutex logMtx;
    std::condition_variable logCv;
    std::deque<LogRec> logQueue;
    bool logStop = false;
    std::thread csvThread;

    if (csvFile.is_open()) {
        std::string resolution = std::to_string(w) + "x" + std::to_string(h);
        std::string codecStr;
        if (codecId == AV_CODEC_ID_H264) codecStr = "H.264";
        else if (codecId == AV_CODEC_ID_H265) codecStr = "H.265";
        else if (codecId == AV_CODEC_ID_MJPEG) codecStr = "MJPEG";
        else codecStr = "Unknown";

        csvThread = std::thread([&, resolution, codecStr]() {
            for (;;) {
                LogRec r;
                {
                    std::unique_lock<std::mutex> lock(logMtx);
                    logCv.wait_for(lock, std::chrono::milliseconds(200), [&] {
                        return logStop || !logQueue.empty();
                    });
                    if (logStop && logQueue.empty()) break;
                    if (logQueue.empty()) continue;
                    r = std::move(logQueue.front());
                    logQueue.pop_front();
                }
            csvFile << resolution << ","
                    << r.frameNo << ","
                    << r.pts << ","
                    << codecStr << ","
                    << w << ","
                    << h << ","
                    << r.decodeMs << ","
                    << r.decodeFuncMs << ","
                    << r.pushBlockMs << ","
                    << r.displayMs << ","
                    << r.frameIntervalMs << ","
                    << r.queueDepth << ","
                    << getCurrentTimestamp() << "\n";
            }
            csvFile.flush();
        });
    }

    // Колбэк статистики декодирования: накопление статистики и очередь в
    // CSV-логгер. Вызывается из потока GStreamer — поэтому без I/O и без
    // форматирования строк (только лёгкие операции).
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

        // ─── Постановка записи в очередь CSV-логгера ───────────────────────
        if (csvThread.joinable()) {
            LogRec r;
            r.frameNo = frameCount;
            r.pts = pts;
            r.decodeMs = st.decodeMs;
            r.decodeFuncMs = st.decodeFuncMs;
            r.pushBlockMs = st.pushBlockMs;
            r.displayMs = st.displayMs;
            r.frameIntervalMs = st.frameIntervalMs;
            r.queueDepth = st.queueDepth;
            {
                std::lock_guard<std::mutex> lock(logMtx);
                if (logQueue.size() >= 8192) logQueue.pop_front();
                logQueue.push_back(std::move(r));
                logCv.notify_one();
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

    // Остановка потока-логгера: сигнал → join (дочищает очередь), затем закрытие файла
    {
        std::lock_guard<std::mutex> lock(logMtx);
        logStop = true;
    }
    logCv.notify_all();
    if (csvThread.joinable()) csvThread.join();
    if (csvFile.is_open()) {
        csvFile.close();
        logWrite("INFO", url, "CSV-файл сохранён: logs/decode_times_" +
                 std::to_string(camIdx) + ".csv");
    }

    // Остановка потока отображения: сигнал → join → закрытие окна
    {
        std::lock_guard<std::mutex> lock(dispMtx);
        dispStop = true;
    }
    dispCv.notify_all();
    if (dispThread.joinable()) dispThread.join();
    if (display) { delete display; display = nullptr; }
    reader.close();
    stopCamRunning(camIdx);
}
