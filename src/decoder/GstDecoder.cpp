// GstDecoder.cpp — Аппаратный декодер NVDEC через GStreamer (nvv4l2decoder).
// Пайплайн: appsrc → h264/h265parse → nvv4l2decoder → nvvidconv → video/x-raw,NV12 → appsink
// Время декодирования кадра замеряется как разница между отправкой пакета
// в appsrc и получением декодированного кадра из appsink.
#include "headers.h"
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

GstDecoder::GstDecoder()
    : m_pipeline(nullptr), m_appsrc(nullptr), m_appsink(nullptr),
      m_codecId(0), m_width(0), m_height(0), m_failed(false),
      m_seenKeyframe(false),
      m_appsrcPad(nullptr), m_decPad(nullptr), m_convPad(nullptr), m_sinkPad(nullptr),
      m_probeAppsrc(0), m_probeDec(0), m_probeConv(0), m_probeSink(0),
      m_lastPushBlockMs(0.0) {}

GstDecoder::~GstDecoder() { close(); }

// Задержка между двумя моментами времени (мс). -1.0, если один из них не задан.
static double msBetween(const std::chrono::steady_clock::time_point& a,
                        const std::chrono::steady_clock::time_point& b) {
    if (a == std::chrono::steady_clock::time_point{} ||
        b == std::chrono::steady_clock::time_point{}) return -1.0;
    return std::chrono::duration<double, std::milli>(a - b).count();
}

// Поиск NAL-единиц в Annex-B пакете: наличие VCL-среза (тип 1/5 для H.264,
// 0..31 для H.265) и наличие ключевого кадра (IDR: тип 5 / 19,20).
void GstDecoder::scanPacket(int codecId, const uint8_t* data, int size,
                            bool& hasVcl, bool& isKey) {
    hasVcl = false;
    isKey = false;
    int i = 0;
    while (i + 3 < size) {
        if (data[i] == 0 && data[i + 1] == 0 &&
            (data[i + 2] == 1 || (data[i + 2] == 0 && data[i + 3] == 1))) {
            int sc = (data[i + 2] == 1) ? 3 : 4;
            int nalIdx = i + sc;
            if (nalIdx < size) {
                if (codecId == AV_CODEC_ID_H265) {
                    int t = (data[nalIdx] >> 1) & 0x3f;
                    if (t < 32) hasVcl = true;
                    if (t == 19 || t == 20) isKey = true;
                } else {
                    int t = data[nalIdx] & 0x1f;
                    if (t == 1 || t == 5) hasVcl = true;
                    if (t == 5) isKey = true;
                }
            }
            i = nalIdx;
        } else {
            i++;
        }
    }
}

// Пробинг буферов на ключевых точках пайплайна: фиксируем время прохождения
// каждого кадра (по PTS) через appsrc → декодер → conv → appsink.
GstPadProbeReturn GstDecoder::onTsProbe(GstPad* pad, GstPadProbeInfo* info,
                                        gpointer userData) {
    GstDecoder* self = static_cast<GstDecoder*>(userData);
    if (!GST_PAD_PROBE_INFO_BUFFER(info)) return GST_PAD_PROBE_OK;
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf || !GST_BUFFER_PTS_IS_VALID(buf)) return GST_PAD_PROBE_OK;
    int64_t pts = (int64_t)GST_BUFFER_PTS(buf);

    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(self->m_tsMtx);
    SampleTs& ts = self->m_sampleTs[pts];
    if (pad == self->m_appsrcPad) ts.appsrcOut = now;
    else if (pad == self->m_decPad) ts.decOut = now;
    else if (pad == self->m_convPad) ts.convOut = now;
    else if (pad == self->m_sinkPad) ts.sinkIn = now;
    return GST_PAD_PROBE_OK;
}

// Обработчик новых кадров из appsink: маппинг NV12 и вызов колбэков
GstFlowReturn GstDecoder::onNewSample(GstElement* sink, gpointer userData) {
    GstDecoder* self = static_cast<GstDecoder*>(userData);
    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) return GST_FLOW_OK;

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);

    GstVideoInfo info;
    if (buffer && caps && gst_video_info_from_caps(&info, caps)) {
        GstVideoFrame frame;
        if (gst_video_frame_map(&frame, &info, buffer, GST_MAP_READ)) {
            int w = GST_VIDEO_INFO_WIDTH(&info);
            int h = GST_VIDEO_INFO_HEIGHT(&info);
            int64_t pts = GST_BUFFER_PTS_IS_VALID(buffer)
                              ? (int64_t)GST_BUFFER_PTS(buffer) : -1;

            // Поиск времени отправки пакета, давшего этот кадр.
            // Пакет ≈ кадр, поэтому берём самый старый ещё не использованный пуш (FIFO).
            std::chrono::steady_clock::time_point submit = std::chrono::steady_clock::now();
            bool haveSubmit = false;
            {
                std::lock_guard<std::mutex> lock(self->m_mtx);
                if (!self->m_pushTimes.empty()) {
                    submit = self->m_pushTimes.front();
                    self->m_pushTimes.pop_front();
                    haveSubmit = true;
                } else {
                    // Fallback: время последней отправки пакета
                    submit = self->m_lastPushTime;
                    haveSubmit = true;
                }
            }

            double decodeMs = -1.0;
            if (haveSubmit)
                decodeMs = measureDecodeSpeed(submit, std::chrono::steady_clock::now());

            self->m_width.store(w);
            self->m_height.store(h);

            // ── Сбор статистики кадра для анализа задержек ─────────────────
            DecodeStats st;
            st.decodeMs = decodeMs;
            st.pushBlockMs = self->m_lastPushBlockMs.load();
            auto now = std::chrono::steady_clock::now();

            // Интервал между кадрами на выходе appsink (реальный темп декодера)
            if (self->m_lastSampleAt != std::chrono::steady_clock::time_point{})
                st.frameIntervalMs = std::chrono::duration<double, std::milli>(now - self->m_lastSampleAt).count();
            self->m_lastSampleAt = now;

            // Глубина FIFO пушей — сколько кадров накоплено в пайплайне
            {
                std::lock_guard<std::mutex> lock(self->m_mtx);
                st.queueDepth = (int)self->m_pushTimes.size();
            }

            // Разбивка по элементам пайплайна (по PTS кадра)
            if (pts > 0) {
                std::lock_guard<std::mutex> lock(self->m_tsMtx);
                auto it = self->m_sampleTs.find(pts);
                if (it != self->m_sampleTs.end()) {
                    const SampleTs& ts = it->second;
                    auto pit = self->m_pushTs.find(pts);
                    if (pit != self->m_pushTs.end()) {
                        st.appToDecMs = msBetween(ts.decOut, pit->second);
                        st.decToConvMs = msBetween(ts.convOut, ts.decOut);
                        st.convToSinkMs = msBetween(ts.sinkIn, ts.convOut);
                    }
                    self->m_sampleTs.erase(it);
                }
                self->m_pushTs.erase(pts);
            }
            // ────────────────────────────────────────────────────────────────

            uint8_t* y  = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&frame, 0);
            uint8_t* uv = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&frame, 1);
            int sy  = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
            int suv = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 1);

            if (self->m_frameCb && y && uv) {
                auto d0 = std::chrono::steady_clock::now();
                self->m_frameCb(y, uv, w, h, sy, suv, pts);
                auto d1 = std::chrono::steady_clock::now();
                st.displayMs = std::chrono::duration<double, std::milli>(d1 - d0).count();
            }
            if (self->m_latencyCb)
                self->m_latencyCb(pts, st);

            gst_video_frame_unmap(&frame);
        }
    }
    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

// Синхронный обработчик сообщений шины (вызывается в потоке вызывающего)
GstBusSyncReply GstDecoder::onBusMessage(GstBus* bus, GstMessage* msg, gpointer userData) {
    (void)bus;
    GstDecoder* self = static_cast<GstDecoder*>(userData);
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_ERROR: {
            GError* err = nullptr;
            gchar* dbg = nullptr;
            gst_message_parse_error(msg, &err, &dbg);
            fprintf(stderr, "[GstDecoder] ERROR: %s (%s)\n",
                    err ? err->message : "?", dbg ? dbg : "");
            if (err) g_error_free(err);
            if (dbg) g_free(dbg);
            self->m_failed.store(true);
            break;
        }
        case GST_MESSAGE_WARNING: {
            GError* err = nullptr;
            gchar* dbg = nullptr;
            gst_message_parse_warning(msg, &err, &dbg);
            fprintf(stderr, "[GstDecoder] WARNING: %s\n", err ? err->message : "?");
            if (err) g_error_free(err);
            if (dbg) g_free(dbg);
            break;
        }
        case GST_MESSAGE_EOS:
            self->m_failed.store(true);
            break;
        default:
            break;
    }
    return GST_BUS_PASS;
}

// Открытие декодера: создание и запуск GStreamer-пайплайна
bool GstDecoder::open(int codecId) {
    close();

    m_codecId = codecId;
    m_failed.store(false);

    gst_init(nullptr, nullptr);

    // Выбор элемента-парсера и caps входного потока (Annex-B byte-stream)
    const char* parseName = (codecId == AV_CODEC_ID_H265) ? "h265parse" : "h264parse";
    const char* srcCapsStr = (codecId == AV_CODEC_ID_H265)
        ? "video/x-h265, stream-format=(string)byte-stream"
        : "video/x-h264, stream-format=(string)byte-stream";

    m_appsrc  = gst_element_factory_make("appsrc", "src");
    GstElement* parse = gst_element_factory_make(parseName, "parse");
    GstElement* dec   = gst_element_factory_make("nvv4l2decoder", "dec");
    GstElement* conv  = gst_element_factory_make("nvvidconv", "conv");
    GstElement* capsf = gst_element_factory_make("capsfilter", "caps");
    m_appsink = gst_element_factory_make("appsink", "sink");
    m_pipeline = gst_pipeline_new("decpipeline");

    if (!m_appsrc || !parse || !dec || !conv || !capsf || !m_appsink || !m_pipeline) {
        fprintf(stderr, "[GstDecoder] не удалось создать элементы GStreamer\n");
        close();
        return false;
    }

    // ─── Минимальная буферизация для appsrc ──────────────────────────────
    g_object_set(m_appsrc,
                 "format", GST_FORMAT_TIME,
                 "stream-type", GST_APP_STREAM_TYPE_STREAM,
                 "is-live", TRUE,
                 "block", FALSE,
                 "do-timestamp", FALSE,
                 "max-bytes", 1024 * 1024,  // 1 МБ
                 NULL);
    GstCaps* srcCaps = gst_caps_from_string(srcCapsStr);
    gst_app_src_set_caps(GST_APP_SRC(m_appsrc), srcCaps);
    gst_caps_unref(srcCaps);

    // ─── Минимальная буферизация для парсера ──────────────────────────────
    g_object_set(parse,
                 "disable-passthrough", FALSE,
                 "fast", TRUE,
                 NULL);

    // ─── Минимальная буферизация для декодера NVDEC ──────────────────────
    g_object_set(dec,
                 "low-latency", TRUE,
                 "drop-frame", TRUE,
                 "num-extra-surfaces", 0,
                 "output-io-mode", 4,  // DMABUF
                 NULL);

    // ─── Минимальная буферизация для nvvidconv ──────────────────────────────
    g_object_set(conv,
                 "nvbuf-memory-type", 0,  // Системная память на выходе
                 NULL);

    // ─── Настройка капсфильтра для NV12 ────────────────────────────────────
    GstCaps* sinkCaps = gst_caps_from_string("video/x-raw, format=(string)NV12");
    g_object_set(capsf, "caps", sinkCaps, NULL);
    gst_caps_unref(sinkCaps);

    // ─── Минимальная буферизация для appsink ──────────────────────────────
    g_object_set(m_appsink, 
                 "emit-signals", TRUE, 
                 "sync", FALSE,
                 "drop", TRUE,
                 "max-buffers", 1,
                 NULL);
    g_signal_connect(m_appsink, "new-sample", G_CALLBACK(onNewSample), this);

    // Сборка пайплайна
    gst_bin_add_many(GST_BIN(m_pipeline), m_appsrc, parse, dec, conv, capsf, m_appsink, NULL);
    if (!gst_element_link_many(m_appsrc, parse, dec, conv, capsf, m_appsink, NULL)) {
        fprintf(stderr, "[GstDecoder] не удалось связать элементы пайплайна\n");
        close();
        return false;
    }

    // Обработка сообщений шины (ошибки/EOS) синхронно
    GstBus* bus = gst_element_get_bus(m_pipeline);
    if (bus) {
        gst_bus_set_sync_handler(bus, onBusMessage, this, nullptr);
        gst_object_unref(bus);
    }

    // Пробы на ключевых точках пайплайна для замера задержек по элементам
    m_appsrcPad = gst_element_get_static_pad(m_appsrc, "src");
    m_decPad    = gst_element_get_static_pad(dec, "src");
    m_convPad   = gst_element_get_static_pad(conv, "src");
    m_sinkPad   = gst_element_get_static_pad(m_appsink, "sink");
    if (m_appsrcPad)
        m_probeAppsrc = gst_pad_add_probe(m_appsrcPad, GST_PAD_PROBE_TYPE_BUFFER,
                                          onTsProbe, this, nullptr);
    if (m_decPad)
        m_probeDec = gst_pad_add_probe(m_decPad, GST_PAD_PROBE_TYPE_BUFFER,
                                       onTsProbe, this, nullptr);
    if (m_convPad)
        m_probeConv = gst_pad_add_probe(m_convPad, GST_PAD_PROBE_TYPE_BUFFER,
                                        onTsProbe, this, nullptr);
    if (m_sinkPad)
        m_probeSink = gst_pad_add_probe(m_sinkPad, GST_PAD_PROBE_TYPE_BUFFER,
                                        onTsProbe, this, nullptr);

    // Запуск пайплайна
    GstStateChangeReturn ret = gst_element_set_state(m_pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        fprintf(stderr, "[GstDecoder] не удалось запустить пайплайн\n");
        close();
        return false;
    }

    fprintf(stderr, "[GstDecoder] пайплайн запущен (codecId=%d) в режиме low-latency\n", codecId);
    return true;
}

// Отправка одного закодированного пакета в декодер
bool GstDecoder::pushPacket(uint8_t* data, int size, int64_t pts) {
    if (!m_pipeline || !data || size <= 0) return false;

    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        bool hasVcl = false, isKey = false;
        scanPacket(m_codecId, data, size, hasVcl, isKey);
        if (isKey) m_seenKeyframe = true;
        // Время пишем только для пакетов с VCL-срезом после первого ключевого
        // кадра — тогда FIFO строго 1:1 "пакет → кадр" и decode_ms честный.
        if (m_seenKeyframe && hasVcl) {
            m_pushTimes.push_back(now);
            m_lastPushTime = now;
            // Ограничение размера очереди времен отправки
            while (m_pushTimes.size() > 4096)
                m_pushTimes.pop_front();
        }
    }

    // Запись времени push по PTS (для разбивки задержек по элементам)
    if (pts > 0) {
        std::lock_guard<std::mutex> lock(m_tsMtx);
        m_pushTs[pts] = now;
        while (m_pushTs.size() > 4096) {
            auto it = m_pushTs.begin();
            if (it->first >= pts) break;   // не удалять будущие PTS
            m_pushTs.erase(it);
        }
    }

    GstBuffer* buf = gst_buffer_new_allocate(nullptr, (guint)size, nullptr);
    GstMapInfo map;
    if (gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
        memcpy(map.data, data, (size_t)size);
        gst_buffer_unmap(buf, &map);
    }
    GST_BUFFER_PTS(buf) = pts > 0 ? (GstClockTime)pts : GST_CLOCK_TIME_NONE;
    GST_BUFFER_DURATION(buf) = GST_CLOCK_TIME_NONE;

    // Время блокировки push: если оно велико, appsrc ждёт, пока пайплайн
    // освободится (backpressure от медленного потребления).
    auto t0 = std::chrono::steady_clock::now();
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(m_appsrc), buf);
    auto t1 = std::chrono::steady_clock::now();
    m_lastPushBlockMs.store(std::chrono::duration<double, std::milli>(t1 - t0).count());

    if (ret != GST_FLOW_OK) {
        m_failed.store(true);
        return false;
    }
    return true;
}

// Закрытие декодера: остановка пайплайна и освобождение ресурсов
void GstDecoder::close() {
    // Снятие проб
    auto removeProbe = [](GstPad*& pad, gulong& id) {
        if (pad) {
            if (id) gst_pad_remove_probe(pad, id);
            gst_object_unref(pad);
            pad = nullptr;
            id = 0;
        }
    };
    removeProbe(m_appsrcPad, m_probeAppsrc);
    removeProbe(m_decPad, m_probeDec);
    removeProbe(m_convPad, m_probeConv);
    removeProbe(m_sinkPad, m_probeSink);

    if (m_pipeline) {
        gst_element_set_state(m_pipeline, GST_STATE_NULL);
        gst_object_unref(m_pipeline);
        m_pipeline = nullptr;
    }
    m_appsrc = nullptr;
    m_appsink = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_pushTimes.clear();
        m_seenKeyframe = false;
    }
    {
        std::lock_guard<std::mutex> lock(m_tsMtx);
        m_sampleTs.clear();
        m_pushTs.clear();
        m_lastSampleAt = std::chrono::steady_clock::time_point{};
    }
    m_lastPushBlockMs.store(0.0);
    m_failed.store(false);
}