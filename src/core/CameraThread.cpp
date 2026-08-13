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
//   frame_no, is_key (1=ключевой кадр, пики total_ms совпадают с ним),
//   decode_ms (NVDEC sink→src, включает ожидание в очереди декодера),
//   decode_func_ms (pushPacket → выход кадра из appsink),
//   queue_depth (VCL-пакетов в полёте, 1 = без очереди),
//   frame_interval_ms (темп выхода кадров из appsink), push_block_ms (backpressure),
//   preprocess_ms, infer_ms, total_ms (приход пакета → кадр готов).
// Прогрева нет: цикл стартует сразу после открытия декодера.
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
            // Прогрев TRT/ядер до старта потока камеры: первый реальный кадр
            // должен идти без разовых спайков (TRT warmup ~88 мс, первый
            // preprocess ~18 мс — все на пуске, не на рабочем кадре).
            infer->warmup();
            logWrite("INFO", url, "Прогрев TensorRT выполнен");
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
            fprintf(csv, "#build=drain_surf\n");
            fprintf(csv, "frame_no,is_key,decode_ms,decode_func_ms,queue_depth,"
                         "frame_interval_ms,push_block_ms,post_decode_ms,"
                         "appsrc_hold_ms,parse_hold_ms,"
                         "preprocess_ms,infer_ms,total_ms\n");
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
    int drainDropped = 0;  // выброшено запаздывающих кадров при дренаже конвейера
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

    // Состояние «дренаж-до-пуша»: последний VCL-кадр уже отправлен в декодер
    // и ждёт выхода. Кадр забираем ДО пуша следующего — пайплайн всегда пуст,
    // queue_depth=1, задержка «пуш → выход» ≈ время декода, а не ~2 интервала
    // камеры. При обратном порядке (push→pull) appsrc с block=FALSE принимает
    // пуши без backpressure, кадры копятся (~4 в полёте) и decode_func_ms
    // растёт до ~80 мс.
    bool pushedVcl = false;
    bool prevIsKey = false;
    std::chrono::steady_clock::time_point prevTPkt;

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
            pushedVcl = false;
        }

        // ─── Дренаж-до-пуша: кадр предыдущего VCL-пуша забираем СРАЗУ, до
        // чтения следующего пакета. Обработка этого кадра совмещается с ожиданием
        // следующего пакета камеры, а пайплайн остаётся пустым (queue_depth=1).
        // При обратном порядке (push→pull) appsrc с block=FALSE принимает пуши
        // без backpressure, кадры копятся (~4 в полёте) и задержка «пуш → выход»
        // растёт до ~2 интервалов камеры (~80 мс).
        HostFrame hf;
        GstDecoder::DecodeStats st;
        if (pushedVcl && gstDec->pullFrame(hf, st, 100)) {
            pulls++;

            // Чистое время NVDEC (sink→src pad-пробы).
            const double decodeMs = st.decodeMs;

            // ─── ИИ: upload → preprocess → infer → postprocess ─────────────
            double preprocessMs = -1.0, inferMs = -1.0;
            Detections dets;
            if (display && !overlaySink && hf.valid()) {
                // Единый аплоад NV12→GPU: тот же device-буфер используют
                // детекция и рендер (без повторной передачи CPU→GPU).
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

            // ─── Рендер (CUDA-режим; из того же device-буфера, что инференс)
            if (display && !overlaySink && hf.valid())
                display->showFrameFromDevice(hf.width, hf.height, dets, g_classNames);

            // ─── Общее время: приход пакета → кадр готов ───────────────────
            // prevTPkt — время прихода пакета того кадра, который сейчас
            // обработан (т.е. кадра, отправленного в декодер на предыдущей
            // итерации). Так total_ms и decode_func_ms описывают ОДИН кадр.
            const double totalMs = msSince(prevTPkt);

            // ─── CSV: только полезные тайминги ─────────────────────────────
            // decode_ms — время пребывания в NVDEC (sink→src, включая ожидание
            // в очереди); decode_func_ms — pushPacket → выход кадра из appsink;
            // queue_depth — VCL-пакетов в полёте (1 = без очереди);
            // frame_interval_ms — интервал между кадрами из appsink;
            // push_block_ms — блокировка push (backpressure).
            frameCount++;
            if (csv) {
                fprintf(csv, "%llu,%d,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                        (unsigned long long)frameCount,
                        prevIsKey ? 1 : 0,
                        decodeMs, st.decodeFuncMs, st.queueDepth,
                        st.frameIntervalMs, st.pushBlockMs, st.postDecodeMs,
                        st.appSrcHoldMs, st.parseHoldMs,
                        preprocessMs, inferMs, totalMs);
                fflush(csv);
            }

            // ─── Накопление для сводки ─────────────────────────────────────
            int q = st.queueDepth;
            if (q > qMax) qMax = q;
            if (q > qMaxRun) qMaxRun = q;
            if (decodeMs >= 0.0)     { sumDec += decodeMs;     wSumDec += decodeMs;     cntDec++; }
            if (preprocessMs >= 0.0) { sumPP += preprocessMs;  wSumPP += preprocessMs;  cntPP++; }
            if (inferMs >= 0.0)      { sumInf += inferMs;      wSumInf += inferMs;      cntInf++; }
            if (totalMs >= 0.0)      { sumTotal += totalMs;    wSumTotal += totalMs;    cntTotal++; }
            wCnt++;

            // ─── Сводка каждые 100 кадров (только в терминал) ─────────────
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

            // ─── Дренаж до пуша: опустошить конвейер ────────────────────────
            // appsrc с block=FALSE принимает пуши без backpressure, поэтому в
            // ступенях пайплайна (appsrc → h264parse → NVDEC → nvvidconv →
            // appsink) скапливается ~3 кадра, идущих с темпом камеры: кадр,
            // запушенный сейчас, покидает пайплайн только через ~2-3 интервала
            // (decode_func_ms ≈ 120 мс). Здесь забираем ВСЕ готовые кадры из
            // appsink (это те самые запаздывающие копии из «очереди» ступеней)
            // и отбрасываем их — конвейер опустошается, и следующий пуш идёт
            // через декодер сразу (задержка ≈ время декода ~8-10 мс вместо
            // 120 мс). В стабильном состоянии (конвейер пуст до пуша) в
            // appsink сидит ровно один кадр — тот, что забран выше, поэтому
            // тут обычно пусто и лишних кадров не теряется.
            for (int i = 0; i < 8; ++i) {
                HostFrame hd;
                GstDecoder::DecodeStats sd;
                if (!gstDec->pullFrame(hd, sd, 5)) break;
                drainDropped++;
            }
        } else if (pushedVcl) {
            noFrame++;
            if (noFrame == 1 || noFrame % 500 == 0)
                logWrite("WARN", url, "SYNC: нет выхода кадра (" +
                         std::to_string(noFrame) + "), inFlight=" +
                         std::to_string(gstDec->inFlight()));
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
            pushedVcl = false;
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

        // B-кадр: pushPacket отбросит его ДО декодера (hasBSlice) — выхода из
        // NVDEC не будет, ждать его в pullFrame бессмысленно (мёртвое ожидание
        // таймаута на каждый B-кадр потока).
        const bool isB = hasVcl && GstDecoder::packetHasB(codecId, pktData, pktSize);

        // Push в декодер (B-кадры режутся внутри pushPacket — до декодера).
        // Анализ пакета уже сделан выше (packetInfo/packetHasB) — передаём его
        // готовым, чтобы не сканировать NAL-байт-стрим второй раз.
        if (!gstDec->pushPacketParsed(pktData, pktSize, pts, hasVcl, isKey, isB)) {
            logWrite("ERROR", url, "pushPacket: сбой конвейера");
            break;
        }
        if (isB || !hasVcl) continue;

        // Пометить текущий VCL-пуш как «в полёте»: его кадр будет забран
        // в начале следующей итерации (дренаж-до-пуша).
        pushedVcl = true;
        prevIsKey = isKey;
        prevTPkt = tPkt;
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
            ", дренаж-выброшено=" + std::to_string(drainDropped) +
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
