// CameraThread.cpp — Поток камеры: RTSP → NVDEC → ИИ → отображение.
//
// Zero-latency синхронный пайплайн (один поток на камеру):
//   readPacket → push в NVDEC → pullFrame (синхронный забор кадра) →
//   upload NV12→GPU → preprocess → infer → postprocess → рендер → CSV-лог.
//
// Очередей нет: appsink в sync-режиме (drop=FALSE, max-buffers=1) — при полном
// буфере appsrc блокируется (backpressure) вместо накопления кадров. B-кадры
// отбрасываются ДО декодера (GstDecoder::pushPacket), стабилизатора темпа нет.
// Всё I/O (X11, CSV) выполняется здесь же — отдельные потоки рендера/писателей
// не нужны, лишней задержки и копирования пикселей нет.
//
// CSV-лог (logs/pipeline_log_<cam>.csv) содержит только полезные тайминги:
//   frame_no, decode_ms (чистое NVDEC), preprocess_ms, infer_ms,
//   total_ms (общее время обработки пакета: приход пакета → кадр готов).
#include "headers.h"
#include "Display.h"
#include "InferPipeline.h"
#include <X11/keysym.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdio>

namespace {

using Clock = std::chrono::steady_clock;

inline double msSince(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

}  // namespace

void cameraThread(std::string url, int camIdx) {
    logWrite("INFO", url, "Поток камеры запущен (zero-latency sync pipeline)");

    // ─── RTSP-захват ───────────────────────────────────────────────────────
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

    {
        std::lock_guard<std::mutex> lock(g_camMtx);
        if (camIdx >= 0 && (size_t)camIdx < g_cams.size()) {
            g_cams[(size_t)camIdx].codec = codecId;
            g_cams[(size_t)camIdx].width = w;
            g_cams[(size_t)camIdx].height = h;
        }
    }

    // ─── Окно отображения (в режиме бенчмарка не создаём) ────────────────
    bool useOverlay = (g_displayMode == "xvimagesink");
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

    // ─── NVDEC в sync-режиме ──────────────────────────────────────────────
    std::unique_ptr<GstDecoder> gstDec;
    if (codecId == AV_CODEC_ID_H264 || codecId == AV_CODEC_ID_H265) {
        gstDec = std::make_unique<GstDecoder>();
        if (!gstDec->open(codecId, overlaySink, winHandle, /*syncMode=*/true)) {
            logWrite("ERROR", url, "GStreamer nvv4l2decoder (sync) недоступен");
            gstDec.reset();
        } else {
            logWrite("INFO", url, "Используется nvv4l2decoder, sync-режим (без очередей)");
        }
    }
    if (!gstDec || !gstDec->isOpen()) {
        logWrite("ERROR", url, "Нет доступного декодера");
        setCamConnected(camIdx, false, url);
        stopCamRunning(camIdx);
        reader.close();
        delete display;
        return;
    }

    // ─── Детекция YOLO (TensorRT) ──────────────────────────────────────────
    std::unique_ptr<InferPipeline> infer;
    if (!g_modelPath.empty() && !overlaySink) {
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
            logWrite("INFO", url, "Детекция YOLO: " + g_modelPath);
        }
    }

    // ─── Прогрев CUDA/TRT/CV-CUDA до цикла ────────────────────────────────
    // Первый запуск ядер/движка дорогой (preprocess ~45 мс, infer ~180 мс).
    // Прогоняем один пустой кадр — реальные кадры пойдут «тёплым» путём.
    if (infer && infer->ready()) {
        int ww = gstDec->width() > 0 ? gstDec->width() : w;
        int hh = gstDec->height() > 0 ? gstDec->height() : h;
        std::vector<uint8_t> dummy((size_t)ww * hh * 3 / 2, 0);
        if (display && !overlaySink) {
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
        } else {
            infer->runHostNv12(dummy.data(), dummy.data() + (size_t)ww * hh,
                               ww, hh, ww, ww, -1);
        }
    }

    // ─── CSV-лог (синхронно, в потоке камеры) ─────────────────────────────
    // Только полезные тайминги: чистое декодирование, препроцессинг, инференс
    // и общее время обработки пакета. Очередей/потоков-писателей нет.
    FILE* csv = nullptr;
    char csvPath[256];
    if (camIdx >= 0 && g_logDecodeSpeed) {
        snprintf(csvPath, sizeof(csvPath), "logs/pipeline_log_%d.csv", camIdx);
        mkdir("logs", 0755);
        csv = fopen(csvPath, "w");
        if (csv) {
            fprintf(csv, "frame_no,decode_ms,preprocess_ms,infer_ms,total_ms\n");
            logWrite("INFO", url, "CSV-лог таймингов: " + std::string(csvPath));
        } else {
            logWrite("WARN", url, "Не удалось открыть CSV-лог: " + std::string(csvPath));
        }
    }

    // ─── Статистика для консольной сводки и вердикта по очередям ──────────
    uint64_t frameCount = 0;
    int vclPushes = 0;      // VCL-пакетов отправлено в декодер
    int pulls = 0;          // кадров получено
    int noFrame = 0;        // pull без выхода (SPS/PPS/буфер декодера)
    int skipPreKey = 0;     // VCL-пакетов пропущено до первого ключевого кадра
    int qMax = 0;           // макс. глубина очереди за окно (VCL в полёте)
    int qMaxRun = 0;        // макс. глубина очереди за весь прогон
    int detFrames = 0, totalDets = 0;
    // Итоговые суммы (за весь прогон) и счётчики валидных замеров.
    double sumDec = 0, sumPP = 0, sumInf = 0, sumTotal = 0;
    int cntDec = 0, cntPP = 0, cntInf = 0, cntTotal = 0;
    // Суммы за текущее окно (сбрасываются каждые 100 кадров).
    double wSumDec = 0, wSumPP = 0, wSumInf = 0, wSumTotal = 0;
    int wCnt = 0;
    auto winStart = Clock::now();
    uint64_t lastSumFrame = 0;

    setCamConnected(camIdx, true, url);
    logWrite("INFO", url, "Камера подключена");

    // ─── Прогрев NVDEC/GStreamer-пути до первого кадра ─────────────────────
    // «Задержка при открытии» в total_ms складывается из двух одноразовых
    // эффектов, и оба поглощаются здесь, до начала замера:
    //  1) первый gst_app_src_push_buffer синхронно инициализирует пайплайн
    //     (сессия NVDEC, согласование caps, контекст nvvidconv) — сотни мс;
    //  2) до первого ключевого кадра NVDEC не выдаёт кадры, и каждый VCL-
    //     пакет «зависал» бы в pullFrame на таймауте 250 мс (мёртвое время).
    // Прогоняем путь до первых двух декодированных кадров без записи в CSV
    // и без статистики — реальные кадры в главном цикле идут «горячим» путём.
    {
        int warmPulls = 0;
        for (int guard = 0; g_running && warmPulls < 2 && guard < 1000; guard++) {
            uint8_t* wData = nullptr; int wSize = 0; int64_t wPts = 0;
            if (!reader.readPacket(wData, wSize, wPts)) {
                logWrite("WARN", url, "Прогрев: ошибка чтения — переподключение");
                reader.close();
                std::this_thread::sleep_for(std::chrono::seconds(1));
                if (!reader.open(url, 10)) break;
                gstDec->close();
                if (!gstDec->open(codecId, overlaySink, winHandle, true)) break;
                continue;
            }
            bool wVcl = false, wKey = false;
            GstDecoder::packetInfo(codecId, wData, wSize, wVcl, wKey);
            if (wVcl && !wKey && !gstDec->hasSeenKeyframe()) continue;
            if (!gstDec->pushPacket(wData, wSize, wPts)) break;
            if (wVcl) {
                HostFrame whf;
                GstDecoder::DecodeStats wst;
                if (gstDec->pullFrame(whf, wst, 250)) warmPulls++;
            }
        }
        if (warmPulls > 0)
            logWrite("INFO", url, "Декодер прогрет (" +
                     std::to_string(warmPulls) + " кадр) — старт замера");
    }

    // ─── Главный цикл: RTSP → NVDEC → ИИ → рендер → CSV ──────────────────
    while (g_running && isCamRunning(camIdx)) {
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

        if (gstDec->failed()) {
            logWrite("WARN", url, "Ошибка GStreamer-конвейера, перезапуск декодера...");
            gstDec->close();
            if (!gstDec->open(codecId, overlaySink, winHandle, true)) {
                logWrite("ERROR", url, "Переинициализация декодера не удалась");
                break;
            }
        }

        uint8_t* pktData;
        int pktSize;
        int64_t pts;

        if (!reader.readPacket(pktData, pktSize, pts)) {
            logWrite("WARN", url, "Ошибка чтения, попытка переподключения...");
            reader.close();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (!reader.open(url, 10)) {
                logWrite("ERROR", url, "Переподключение не удалось");
                break;
            }
            gstDec->close();
            if (!gstDec->open(codecId, overlaySink, winHandle, true)) {
                logWrite("ERROR", url, "Переинициализация NVDEC не удалась");
                break;
            }
            {
                std::lock_guard<std::mutex> lock(g_camMtx);
                if (camIdx >= 0 && (size_t)camIdx < g_cams.size()) {
                    g_cams[(size_t)camIdx].width = reader.width();
                    g_cams[(size_t)camIdx].height = reader.height();
                }
            }
            continue;
        }

        // Старт общего счётчика обработки пакета (пакет уже получен).
        auto tPkt = Clock::now();

        bool hasVcl = false, isKey = false;
        GstDecoder::packetInfo(codecId, pktData, pktSize, hasVcl, isKey);

        // Пропуск VCL-пакетов до первого ключевого кадра: без IDR NVDEC не
        // выдаёт кадры, а pullFrame ждал бы таймаут на каждый такой пакет —
        // это «мёртвое время» задержки при открытии. SPS/PPS и сам ключевой
        // кадр продолжают идти в декодер.
        if (hasVcl && !isKey && !gstDec->hasSeenKeyframe()) {
            skipPreKey++;
            continue;
        }
        if (hasVcl) vclPushes++;

        // Push в декодер (B-кадры режутся внутри pushPacket — до декодера).
        if (!gstDec->pushPacket(pktData, pktSize, pts)) {
            logWrite("ERROR", url, "pushPacket: сбой конвейера");
            break;
        }

        // Синхронный забор кадра. VCL-пакет обязан дать кадр (до 250 мс);
        // SPS/PPS/пре-ключевой выхода не дают — короткий таймаут, не держим поток.
        HostFrame hf;
        GstDecoder::DecodeStats st;
        if (!gstDec->pullFrame(hf, st, hasVcl ? 250 : 5)) {
            noFrame++;
            if (noFrame == 1 || noFrame % 500 == 0)
                logWrite("WARN", url, "SYNC: нет выхода кадра (" +
                         std::to_string(noFrame) + "), inFlight=" +
                         std::to_string(gstDec->inFlight()));
            continue;
        }
        pulls++;

        // Чистое время NVDEC (sink→src pad-пробы).
        const double decodeMs = st.decodeMs;

        // ─── ИИ: upload → preprocess → infer → postprocess ─────────────────
        double preprocessMs = -1.0, inferMs = -1.0;
        Detections dets;
        if (display && !overlaySink && hf.valid()) {
            // Единый аплоад NV12→GPU: тот же device-буфер используют детекция
            // и рендер (без повторной передачи CPU→GPU).
            display->uploadNv12(hf.yPlane, hf.uvPlane, hf.strideY, hf.strideUV,
                                hf.width, hf.height);
        }
        if (infer && infer->ready() && hf.valid()) {
            InferTimings timings;
            if (display && !overlaySink) {
                GpuFrame gf;
                gf.yPlane  = display->deviceY();
                gf.uvPlane = display->deviceUV();
                gf.width = hf.width; gf.height = hf.height;
                gf.strideY = hf.width; gf.strideUV = hf.width;
                gf.pts = hf.pts;
                dets = infer->run(gf, &timings);
            } else {
                dets = infer->runHostNv12(hf.yPlane, hf.uvPlane,
                                          hf.width, hf.height,
                                          hf.strideY, hf.strideUV,
                                          hf.pts, &timings);
            }
            preprocessMs = timings.preprocessMs;
            inferMs = timings.inferMs;
            if (!dets.empty()) {
                detFrames++;
                totalDets += (int)dets.size();
            }
        }

        // ─── Рендер (CUDA-режим; из того же device-буфера, что инференс) ───
        if (display && !overlaySink && hf.valid())
            display->showFrameFromDevice(hf.width, hf.height, dets, g_classNames);

        // ─── Общее время обработки пакета: приход пакета → кадр готов ─────
        const double totalMs = msSince(tPkt);

        // ─── CSV: только полезные тайминги ─────────────────────────────────
        frameCount++;
        if (csv) {
            fprintf(csv, "%llu,%.4f,%.4f,%.4f,%.4f\n",
                    (unsigned long long)frameCount,
                    decodeMs, preprocessMs, inferMs, totalMs);
            fflush(csv);
        }

        // ─── Накопление для сводки ─────────────────────────────────────────
        int q = st.queueDepth;
        if (q > qMax) qMax = q;
        if (q > qMaxRun) qMaxRun = q;
        if (decodeMs >= 0.0)     { sumDec += decodeMs;     wSumDec += decodeMs;     cntDec++; }
        if (preprocessMs >= 0.0) { sumPP += preprocessMs;  wSumPP += preprocessMs;  cntPP++; }
        if (inferMs >= 0.0)      { sumInf += inferMs;      wSumInf += inferMs;      cntInf++; }
        if (totalMs >= 0.0)      { sumTotal += totalMs;    wSumTotal += totalMs;    cntTotal++; }
        wCnt++;

        // ─── Сводка каждые 100 кадров (только в терминал) ─────────────────
        if (frameCount - lastSumFrame >= 100) {
            double el = std::chrono::duration<double>(Clock::now() - winStart).count();
            double fps = el > 0 ? (double)wCnt / el : 0.0;
            {
                std::lock_guard<std::mutex> lock(g_camMtx);
                if (camIdx >= 0 && (size_t)camIdx < g_cams.size())
                    g_cams[(size_t)camIdx].fps = fps;
            }
            auto avg = [](double s, int n) { return n > 0 ? s / (double)n : -1.0; };
            logWrite("INFO", url,
                "SYNC FPS=" + std::to_string(fps) +
                " | avg_total=" + std::to_string(avg(wSumTotal, wCnt)) +
                " | avg_decode=" + std::to_string(avg(wSumDec, cntDec)) +
                " | avg_preprocess=" + std::to_string(avg(wSumPP, cntPP)) +
                " | avg_infer=" + std::to_string(avg(wSumInf, cntInf)) +
                " | q_max=" + std::to_string(qMax) +
                " | dropped_B=" + std::to_string(gstDec->droppedBFrames()));
            wSumDec = wSumPP = wSumInf = wSumTotal = 0;
            wCnt = 0;
            qMax = 0;
            winStart = Clock::now();
            lastSumFrame = frameCount;
        }
    }  // while

    // ─── Итог и вердикт по очередям ───────────────────────────────────────
    const uint64_t droppedB = gstDec->droppedBFrames();
    gstDec->close();
    if (csv) { fclose(csv); csv = nullptr; }

    if (frameCount > 0) {
        auto avg = [](double s, int n) { return n > 0 ? s / (double)n : -1.0; };
        logWrite("INFO", url,
            "ИТОГ: кадров=" + std::to_string(frameCount) +
            ", push=" + std::to_string(vclPushes) + ", pull=" + std::to_string(pulls) +
            ", до_ключ.кадра=" + std::to_string(skipPreKey) +
            ", B-отброшено=" + std::to_string(droppedB) +
            ", avg_decode=" + std::to_string(avg(sumDec, cntDec)) +
            ", avg_preprocess=" + std::to_string(avg(sumPP, cntPP)) +
            ", avg_infer=" + std::to_string(avg(sumInf, cntInf)) +
            ", avg_total=" + std::to_string(avg(sumTotal, cntTotal)));
        if (detFrames > 0)
            logWrite("INFO", url, "DET: кадров с объектами=" +
                     std::to_string(detFrames) + "/" + std::to_string(frameCount) +
                     ", боксов=" + std::to_string(totalDets));
    }

    // Вердикт: есть ли скрытая очередь в пайплайне.
    if (qMaxRun > 1) {
        logWrite("WARN", url,
            "ВЕРДИКТ: ОЧЕРЕДЬ В ПАЙПЛАЙНЕ — queue_depth до " + std::to_string(qMaxRun) +
            " (VCL-пакетов в полёте). Декодер/парсер буферизует кадры.");
    } else {
        logWrite("INFO", url,
            "ВЕРДИКТ: очередей нет (queue_depth<=1), задержка = темп камеры + обработка.");
    }
    if (noFrame > 0)
        logWrite("INFO", url, "pull без выхода кадра: " + std::to_string(noFrame) +
                 " (SPS/PPS/пре-ключевые пакеты)");

    setCamConnected(camIdx, false, url);
    if (display) { delete display; display = nullptr; }
    reader.close();
    stopCamRunning(camIdx);
    logWrite("INFO", url, "Поток камеры завершён");
}
