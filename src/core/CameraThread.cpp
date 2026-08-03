// CameraThread.cpp — Поток камеры: RTSP → декодирование (nvv4l2decoder) → отображение.
#include "headers.h"
#include "Display.h"
#include <X11/keysym.h>

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

    // Выбор декодера: аппаратный NVDEC через GStreamer nvv4l2decoder (JetPack 6)
    std::unique_ptr<GstDecoder> gstDec;

    if (codecId == AV_CODEC_ID_H264 || codecId == AV_CODEC_ID_H265) {
        gstDec = std::make_unique<GstDecoder>();
        if (gstDec->open(codecId)) {
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
        return;
    }

    // Создание окна отображения (1600x900, чёрное до первого кадра)
    DisplayWindow* display = new DisplayWindow();
    if (!display->open("Камера " + std::to_string(camIdx) + " - " + url, 1600, 900)) {
        delete display;
        display = nullptr;
    }

    // Управление поворотной камерой (PTZ): стрелки, если камера поддерживает ISAPI
    std::unique_ptr<PtzControl> ptz;
    if (display) {
        ptz = std::make_unique<PtzControl>();
        if (ptz->open(url))
            logWrite("INFO", url, "PTZ доступен: стрелки управляют камерой");
        else
            logWrite("WARN", url, "PTZ недоступен (стрелки неактивны)");
    }

    // CSV-файл с детальной статистикой декодирования
    std::ofstream csvFile;
    if (camIdx >= 0 && g_logDecodeSpeed) {
        csvFile.open("decode_times_" + std::to_string(camIdx) + ".csv");
        if (csvFile.is_open()) {
            // Заголовок CSV
            csvFile << "resolution,frame_no,pts,codec,source_width,source_height,"
                    << "decode_ms,push_block_ms,app_to_dec_ms,dec_to_conv_ms,"
                    << "conv_to_sink_ms,display_ms,frame_interval_ms,queue_depth,decoded_at\n";
            logWrite("INFO", url, "CSV-логирование включено: decode_times_" + 
                     std::to_string(camIdx) + ".csv");
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

    // Колбэк кадра: декодер отдаёт NV12 → CUDA конвертирует → X11 показывает
    FrameCallback cb = [&](uint8_t* y, uint8_t* uv, int cw, int ch,
                           int sy, int suv, int64_t pts) {
        (void)pts;
        if (!g_running) return;
        if (display) display->showFrame(y, uv, cw, ch, sy, suv);
    };
    if (gstDec && gstDec->isOpen()) gstDec->setFrameCallback(cb);

    // Колбэк статистики декодирования: запись в CSV и сводка в терминал
    double sumTotal = 0, sumPush = 0, sumA2D = 0, sumD2C = 0, sumC2S = 0, sumFI = 0, sumDisp = 0;
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

        // ─── Запись в CSV ──────────────────────────────────────────────────────
        if (csvFile.is_open()) {
            std::string resolution = std::to_string(w) + "x" + std::to_string(h);
            std::string codecStr;
            if (codecId == AV_CODEC_ID_H264) codecStr = "H.264";
            else if (codecId == AV_CODEC_ID_H265) codecStr = "H.265";
            else if (codecId == AV_CODEC_ID_MJPEG) codecStr = "MJPEG";
            else codecStr = "Unknown";
            
            csvFile << resolution << ","
                    << frameCount << ","
                    << pts << ","
                    << codecStr << ","
                    << w << ","
                    << h << ","
                    << st.decodeMs << ","
                    << st.pushBlockMs << ","
                    << st.appToDecMs << ","
                    << st.decToConvMs << ","
                    << st.convToSinkMs << ","
                    << st.displayMs << ","
                    << st.frameIntervalMs << ","
                    << st.queueDepth << ","
                    << getCurrentTimestamp() << "\n";
            csvFile.flush(); // Принудительная запись на диск
        }

        // ─── Накопление статистики для сводки ──────────────────────────────
        sumTotal += st.decodeMs;
        sumPush += st.pushBlockMs;
        sumA2D += st.appToDecMs;
        sumD2C += st.decToConvMs;
        sumC2S += st.convToSinkMs;
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
                " | avg_appsrc->dec=" + ms(avg(sumA2D)) +
                " | avg_dec->conv=" + ms(avg(sumD2C)) +
                " | avg_conv->sink=" + ms(avg(sumC2S)) +
                " | avg_frame_interval=" + ms(avg(sumFI)) +
                " | avg_display=" + ms(avg(sumDisp)));
            sumTotal = sumPush = sumA2D = sumD2C = sumC2S = sumFI = sumDisp = 0;
            cntStats = 0;
        }
    };
    if (gstDec && gstDec->isOpen()) gstDec->setLatencyCallback(latCb);

    setCamConnected(camIdx, true, url);
    logWrite("INFO", url, "Камера подключена");

    // ─── Главный цикл: чтение RTSP → отправка пакетов в декодер ──────────
    while (g_running && isCamRunning(camIdx)) {
        // Обработка клавиатуры: ESC — выход, стрелки — PTZ (поворот камеры)
        if (display) {
            display->pollEvents();
            int ks;
            bool pr;
            while (display->popKeyEvent(ks, pr)) {
                if (ks == XK_Escape && pr) {
                    logWrite("INFO", url, "ESC: выход из программы");
                    g_running = 0;
                } else if (ptz && ptz->isOpen()) {
                    PtzControl::Dir dir = (PtzControl::Dir)-1;
                    switch (ks) {
                        case XK_Left:  dir = PtzControl::Dir::Left;  break;
                        case XK_Right: dir = PtzControl::Dir::Right; break;
                        case XK_Up:    dir = PtzControl::Dir::Up;    break;
                        case XK_Down:  dir = PtzControl::Dir::Down;  break;
                        default: break;
                    }
                    if ((int)dir >= 0) ptz->onKey(dir, pr);
                }
            }
        }

        // Перезапуск конвейера при ошибке
        if (gstDec && gstDec->failed()) {
            logWrite("WARN", url, "Ошибка GStreamer-конвейера, перезапуск декодера...");
            gstDec->close();
            if (!gstDec->open(codecId)) {
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
                if (!gstDec->open(codecId)) {
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
    if (gstDec) gstDec->close();
    if (ptz) ptz->close();
    if (csvFile.is_open()) {
        csvFile.close();
        logWrite("INFO", url, "CSV-файл сохранён: decode_times_" + 
                 std::to_string(camIdx) + ".csv");
    }
    if (display) { delete display; display = nullptr; }
    reader.close();
    stopCamRunning(camIdx);
}
