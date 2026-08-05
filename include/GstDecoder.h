// GstDecoder.h — Аппаратный декодер NVDEC через GStreamer (nvv4l2decoder).
// JetPack 6 (Orin, R36) не содержит V4L2 M2M NVDEC-драйвера (CONFIG_VIDEO_TEGRA
// отключён), поэтому декодирование выполняется через NVMM-элемент nvv4l2decoder.
// Пайплайн: appsrc → h264/h265parse → nvv4l2decoder → nvvidconv → video/x-raw,NV12 → appsink
// Проект полностью изолирован от B-кадров: пакеты, содержащие B-срез, отбрасываются
// ДО декодера (только I/P-кадры) — NVDEC не выполняет reorder, задержка минимальна.
// Низкая задержка: disable-dpb, enable-max-performance, num-extra-surfaces=0,
// appsrc is-live=FALSE, стабилизатор темпа 30 мс.
// Время декодирования одного пакета замеряется как разница между отправкой
// пакета в appsrc и получением декодированного кадра из appsink (FIFO-сопоставление).
#pragma once

#include <atomic>
#include <cstdint>
#include <chrono>
#include <deque>
#include <functional>
#include <mutex>

#include <gst/gst.h>

#include "FrameCallback.h"

class GstDecoder {
public:
    // Статистика декодирования кадра (мс)
    struct DecodeStats {
        double decodeMs = -1.0;         // общая: пуш в appsrc → кадр из appsink
        double pushBlockMs = -1.0;      // блокировка gst_app_src_push_buffer (backpressure)
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

    // Сколько B-кадров отброшено (изоляция от B-кадров)
    uint64_t droppedBFrames() const { return m_droppedB.load(); }

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
    std::atomic<uint64_t> m_droppedB;  // отброшено B-кадров

    std::mutex m_mtx;             // Защита m_pushTimes / m_lastPushTime / m_seenKeyframe / m_lastEmitAt
    std::deque<std::chrono::steady_clock::time_point> m_pushTimes; // времена отправки VCL-пакетов (FIFO)
    std::chrono::steady_clock::time_point m_lastPushTime;          // время последней отправки
    bool m_seenKeyframe;          // получен ли первый ключевой кадр
    std::chrono::steady_clock::time_point m_lastEmitAt;  // время последнего выданного кадра (стабилизатор темпа)

    // ─── Замер времени декодирования (FIFO push→кадр) ─────────────────────────
    std::chrono::steady_clock::time_point m_lastSampleAt;   // время последнего кадра из appsink
    std::atomic<double> m_lastPushBlockMs;                  // последняя задержка push (мс)

    static GstFlowReturn onNewSample(GstElement* sink, gpointer userData);
    static GstBusSyncReply onBusMessage(GstBus* bus, GstMessage* msg, gpointer userData);
    static void scanPacket(int codecId, const uint8_t* data, int size,
                           bool& hasVcl, bool& isKey);
    static bool hasBSlice(int codecId, const uint8_t* data, int size);
};
