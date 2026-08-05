// GstDecoder.cpp — Аппаратный декодер NVDEC через GStreamer (nvv4l2decoder).
// Пайплайн: appsrc → h264/h265parse → nvv4l2decoder → nvvidconv → video/x-raw,NV12 → appsink
// Проект полностью изолирован от B-кадров: B-срезы отбрасываются до декодера
// (только I/P), поэтому NVDEC не выполняет reorder и задержка минимальна.
// Низкая задержка обеспечивается реальными свойствами nvv4l2decoder:
// disable-dpb=TRUE + enable-max-performance=TRUE + num-extra-surfaces=0,
// appsrc is-live=FALSE, стабилизатор темпа 30 мс в onNewSample.
// Время декодирования кадра замеряется как разница между отправкой пакета
// в appsrc и получением декодированного кадра из appsink.
#include "headers.h"
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>

namespace {

// Битовый читатель для exp-golomb кодов в slice-заголовке (после EBSP→RBSP).
class BitReader {
public:
    BitReader(const uint8_t* d, int byteCount) : p(d), bitPos(0), nbits(byteCount * 8) {}
    bool readBit() {
        if (bitPos >= nbits) return false;
        int byte = bitPos >> 3;
        int bit = 7 - (bitPos & 7);
        bitPos++;
        return (p[byte] >> bit) & 1;
    }
    uint32_t readUE() {
        int zeros = 0;
        while (!readBit()) {
            zeros++;
            if (zeros > 31) return 0;
        }
        uint32_t val = 1;
        for (int i = 0; i < zeros; i++) {
            val <<= 1;
            if (readBit()) val |= 1;
        }
        return val - 1;
    }
    bool ok() const { return bitPos <= nbits; }
private:
    const uint8_t* p;
    int bitPos;
    int nbits;
};

// Конвертация EBSP → RBSP: удаление байтов эмуляции (00 00 03 → 00 00).
int ebspToRbsp(const uint8_t* src, int size, uint8_t* dst, int dstCap) {
    int n = 0;
    for (int i = 0; i < size && n < dstCap; i++) {
        if (i + 2 < size && src[i] == 0 && src[i + 1] == 0 && src[i + 2] == 3) {
            if (n + 2 > dstCap) break;
            dst[n++] = 0;
            dst[n++] = 0;
            i += 2;
        } else {
            dst[n++] = src[i];
        }
    }
    return n;
}

}  // namespace

GstDecoder::GstDecoder()
    : m_pipeline(nullptr), m_appsrc(nullptr), m_appsink(nullptr),
      m_codecId(0), m_width(0), m_height(0), m_failed(false),
      m_droppedB(0), m_seenKeyframe(false),
      m_lastPushBlockMs(0.0) {}
GstDecoder::~GstDecoder() { close(); }

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

// Проверка, содержит ли пакет B-срез. Разбирается slice-заголовок VCL NAL:
//  - H.264: slice_type (ue(v)), B если slice_type % 5 == 1 (типы 1,6);
//  - H.265: slice_type (ue(v)), B если slice_type == 0.
// Если разобрать заголовок не удалось — кадр НЕ отбрасываем (conservative).
bool GstDecoder::hasBSlice(int codecId, const uint8_t* data, int size) {
    uint8_t rbsp[512];
    int i = 0;
    while (i + 3 < size) {
        // Поиск стартового кода 00 00 01 / 00 00 00 01
        if (!(data[i] == 0 && data[i + 1] == 0 &&
              (data[i + 2] == 1 || (data[i + 2] == 0 && data[i + 3] == 1)))) {
            i++;
            continue;
        }
        int sc = (data[i + 2] == 1) ? 3 : 4;
        int nalStart = i + sc;
        int j = nalStart;
        // Конец NAL — следующий стартовый код или конец пакета
        while (j + 3 < size) {
            if (data[j] == 0 && data[j + 1] == 0 &&
                (data[j + 2] == 1 || (data[j + 2] == 0 && data[j + 3] == 1)))
                break;
            j++;
        }
        int nalSize = j - nalStart;
        if (nalSize > 0) {
            if (codecId == AV_CODEC_ID_H265) {
                int nalType = (data[nalStart] >> 1) & 0x3f;
                // VCL (0..9) с достаточным payload: первый флаг + pps_id + slice_type
                if (nalType <= 9 && nalSize >= 3) {
                    int n = ebspToRbsp(data + nalStart + 2, nalSize - 2,
                                       rbsp, (int)sizeof(rbsp));
                    BitReader br(rbsp, n);
                    if (!br.readBit()) { i = nalStart; continue; }  // first_slice_segment=0 — не разбираем
                    if (nalType >= 16) br.readBit();                // no_output_of_prior_pics (IRAP)
                    br.readUE();                                    // slice_pic_parameter_set_id
                    uint32_t sliceType = br.readUE();               // 0=B, 1=P, 2=I
                    if (br.ok() && sliceType == 0) return true;
                }
            } else {
                int nalType = data[nalStart] & 0x1f;
                // Слайсовые NAL (1 — non-IDR, 5 — IDR)
                if ((nalType == 1 || nalType == 5) && nalSize >= 2) {
                    int n = ebspToRbsp(data + nalStart + 1, nalSize - 1,
                                       rbsp, (int)sizeof(rbsp));
                    BitReader br(rbsp, n);
                    br.readUE();                                    // first_mb_in_slice
                    uint32_t sliceType = br.readUE();
                    if (br.ok() && (sliceType % 5) == 1) return true;
                }
            }
        }
        i = nalStart;
    }
    return false;
}

// Обработчик новых кадров из appsink: маппинг NV12 и вызов колбэков
GstFlowReturn GstDecoder::onNewSample(GstElement* sink, gpointer userData) {
    GstDecoder* self = static_cast<GstDecoder*>(userData);
    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) return GST_FLOW_OK;

    // Стабилизатор темпа: кадр, пришедший раньше чем через 30 мс после
    // предыдущего, отбрасывается — выход ровный, пайплайн не забивается.
    // Отброшенный кадр «съедает» одну запись FIFO времён пушей, маппинг
    // push↔кадр остаётся 1:1. Порог 30 мс пропускает до ~33 кадров/с —
    // для камеры 25 fps (40 мс) и медленнее (10.5 fps, 95 мс) дропов нет.
    {
        auto emitNow = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(self->m_mtx);
        if (self->m_lastEmitAt != std::chrono::steady_clock::time_point{} &&
            std::chrono::duration<double, std::milli>(emitNow - self->m_lastEmitAt).count() < 30.0) {
            if (!self->m_pushTimes.empty()) self->m_pushTimes.pop_front();
            gst_sample_unref(sample);
            return GST_FLOW_OK;
        }
        self->m_lastEmitAt = emitNow;
    }

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
    // is-live=FALSE — как в рабочем e6fd622: live-режим добавлял ~130 мс
    // (NVDEC ждёт тактовых меток и буферизует ~2.5 кадра до выдачи). Пуши
    // всё равно идут с темпом сети (RtspReader).
    g_object_set(m_appsrc,
                 "format", GST_FORMAT_TIME,
                 "stream-type", GST_APP_STREAM_TYPE_STREAM,
                 "is-live", FALSE,
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
    // Свойств "low-latency"/"drop-frame" у nvv4l2decoder нет (проверено
    // gst-inspect-1.0) — они игнорировались. Реальные рычаги задержки:
    //  - disable-dpb=TRUE: отключить DPB-reorder (B-кадры уже режутся
    //    фильтром hasBSlice, переупорядочивание не нужно);
    //  - enable-max-performance=TRUE: убрать троттлинг GPU-клока NVDEC
    //    (на Jetson декодер может работать на пониженной частоте);
    //  - num-extra-surfaces=0: декодер держит только текущий кадр.
    g_object_set(dec,
                 "disable-dpb", TRUE,
                 "enable-max-performance", TRUE,
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

    // ─── Изоляция от B-кадров ──────────────────────────────────────────────
    // Пакет с B-срезом не попадает в декодер вообще: NVDEC не делает reorder,
    // декодируются только I/P-кадры (frame-by-frame, таргет 1-10 мс).
    if (hasBSlice(m_codecId, data, size)) {
        m_droppedB.fetch_add(1);
        return true;  // отброшен (успешно)
    }

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

    // Запись времени push для кадра — в кадре нет времени отправки (пакет ≈
    // кадр, поэтому используется FIFO m_pushTimes в onNewSample).

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
        m_lastEmitAt = std::chrono::steady_clock::time_point{};
    }
    m_droppedB.store(0);
    m_lastSampleAt = std::chrono::steady_clock::time_point{};
    m_lastPushBlockMs.store(0.0);
    m_failed.store(false);
}