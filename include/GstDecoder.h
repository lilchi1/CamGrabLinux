// GstDecoder.h — Аппаратный декодер NVDEC через GStreamer (nvv4l2decoder).
// JetPack 6 (Orin, R36) не содержит V4L2 M2M NVDEC-драйвера (CONFIG_VIDEO_TEGRA
// отключён), поэтому декодирование выполняется через NVMM-элемент nvv4l2decoder.
// Пайплайн: appsrc → h264/h265parse → nvv4l2decoder → nvvidconv → video/x-raw,NV12 → appsink
// Время декодирования одного пакета замеряется как разница между отправкой
// пакета в appsrc и получением декодированного кадра из appsink (FIFO-сопоставление).
#pragma once

#include <atomic>
#include <cstdint>
#include <chrono>
#include <deque>
#include <functional>
#include <map>
#include <mutex>

#include <gst/gst.h>

#include "FrameCallback.h"

class GstDecoder {
public:
    // Статистика декодирования кадра — задержки по элементам пайплайна (мс)
    struct DecodeStats {
        double decodeMs = -1.0;         // общая: пуш в appsrc → кадр из appsink
        double pushBlockMs = -1.0;      // блокировка gst_app_src_push_buffer (backpressure)
        double appToDecMs = -1.0;       // appsrc → выход nvv4l2decoder (parse+декодер)
        double decToConvMs = -1.0;      // nvv4l2decoder → выход nvvidconv
        double convToSinkMs = -1.0;     // nvvidconv → вход appsink
        double frameIntervalMs = -1.0;  // интервал между кадрами из appsink
        double displayMs = -1.0;        // время обработки кадра в колбэке (CUDA+X11)
        int queueDepth = 0;             // число невостребованных пушей (глубина буферизации)
    };

    // Колбэк со статистикой декодирования кадра
    using LatencyCb = std::function<void(int64_t pts, const DecodeStats& stats)>;

    GstDecoder();
    ~GstDecoder();

    // Открытие декодера для указанного кодека (AV_CODEC_ID_H264 / H265)
    bool open(int codecId);

    // Закрытие декодера и остановка GStreamer-пайплайна
    void close();

    // Отправка одного закодированного пакета (Annex-B byte-stream) в декодер
    bool pushPacket(uint8_t* data, int size, int64_t pts);

    // Проверка, открыт ли декодер
    bool isOpen() const { return m_pipeline != nullptr; }

    // Сбой конвейера (ошибка/EOS) — требуется перезапуск
    bool failed() const { return m_failed.load(); }

    // Установка колбэка для получения декодированных NV12 кадров
    void setFrameCallback(FrameCallback cb) { m_frameCb = cb; }

    // Установка колбэка с временем декодирования каждого кадра
    void setLatencyCallback(LatencyCb cb) { m_latencyCb = cb; }

    // Текущее разрешение декодированного потока
    int width() const { return m_width.load(); }
    int height() const { return m_height.load(); }

private:
    GstElement* m_pipeline;   // Полный пайплайн
    GstElement* m_appsrc;     // Элемент appsrc (вход закодированных пакетов)
    GstElement* m_appsink;    // Элемент appsink (выход NV12 кадров)
    int m_codecId;            // ID кодека (H.264/H.265)

    FrameCallback m_frameCb;      // Доставка NV12 кадров
    LatencyCb m_latencyCb;        // Доставка времени декодирования кадра

    std::atomic<int> m_width;     // Разрешение декодированного потока
    std::atomic<int> m_height;
    std::atomic<bool> m_failed;   // Флаг сбоя конвейера

    std::mutex m_mtx;             // Защита m_pushTimes / m_lastPushTime / m_seenKeyframe
    std::deque<std::chrono::steady_clock::time_point> m_pushTimes; // времена отправки VCL-пакетов (FIFO)
    std::chrono::steady_clock::time_point m_lastPushTime;          // время последней отправки
    bool m_seenKeyframe;          // получен ли первый ключевой кадр

    // ─── Замер задержек по элементам пайплайна ────────────────────────────────
    GstPad* m_appsrcPad;          // src-pad appsrc (вход в пайплайн)
    GstPad* m_decPad;             // src-pad nvv4l2decoder (выход декодера)
    GstPad* m_convPad;            // src-pad nvvidconv
    GstPad* m_sinkPad;            // sink-pad appsink (приход кадра)
    gulong m_probeAppsrc, m_probeDec, m_probeConv, m_probeSink;  // ID проб

    struct SampleTs {             // времена прохождения кадра через элементы (по PTS)
        std::chrono::steady_clock::time_point appsrcOut;
        std::chrono::steady_clock::time_point decOut;
        std::chrono::steady_clock::time_point convOut;
        std::chrono::steady_clock::time_point sinkIn;
    };
    std::mutex m_tsMtx;                                     // Защита карт временных меток
    std::map<int64_t, SampleTs> m_sampleTs;                 // времена проб по PTS кадра
    std::map<int64_t, std::chrono::steady_clock::time_point> m_pushTs; // время push по PTS
    std::chrono::steady_clock::time_point m_lastSampleAt;   // время последнего кадра из appsink
    std::atomic<double> m_lastPushBlockMs;                  // последняя задержка push (мс)

    static GstFlowReturn onNewSample(GstElement* sink, gpointer userData);
    static GstBusSyncReply onBusMessage(GstBus* bus, GstMessage* msg, gpointer userData);
    static GstPadProbeReturn onTsProbe(GstPad* pad, GstPadProbeInfo* info, gpointer userData);
    static void scanPacket(int codecId, const uint8_t* data, int size,
                           bool& hasVcl, bool& isKey);
};
