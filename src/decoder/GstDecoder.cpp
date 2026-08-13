// GstDecoder.cpp — Аппаратный декодер NVDEC через GStreamer (nvv4l2decoder).
// Пайплайн: appsrc → h264/h265parse → nvv4l2decoder → nvvidconv → video/x-raw,NV12 → appsink
// (в overlay-режиме: appsrc → parse → nvv4l2decoder → tee → { queue → nvvidconv(NV12) →
// xvimagesink (рендер в наше окно), capsf(video/x-raw(memory:NVMM) NV12) → appsink } —
// замер без маппинга пикселей).
// Проект полностью изолирован от B-кадров: B-срезы отбрасываются до декодера
// (только I/P), поэтому NVDEC не выполняет reorder и задержка минимальна.
// Низкая задержка обеспечивается реальными свойствами nvv4l2decoder:
// disable-dpb=TRUE + enable-max-performance=TRUE + num-extra-surfaces=0,
// appsrc is-live=FALSE. Стабилизатора темпа нет — без искусственной задержки.
// Время декодирования кадра замеряется pad-пробами на nvv4l2decoder:
// вход буфера в sink-пад → выход кадра из src-пада (чистое NVDEC, без
// parse/очередей/рендера). Сопоставление по PTS — SPS/PPS-пакеты входят
// в декодер без выхода, а async-режим (drop=TRUE) может дропать кадры,
// поэтому FIFO по порядку здесь не годится.
#include "headers.h"
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <gst/video/videooverlay.h>

// Удерживает маппинг NV12 живым до обработки кадра в потоке-потребителе.
// gst_video_frame_map берёт реф буфера, gst_video_frame_unmap снимает маппинг
// и отдаёт реф — поэтому keeper'у достаточно хранить GstVideoFrame (проверено
// на GStreamer 1.20.3). Деструктор вызывается в потоке отображения.
struct HostFrameKeeper {
    GstVideoFrame frame;
    explicit HostFrameKeeper(const GstVideoFrame& f) : frame(f) {}
    ~HostFrameKeeper() { gst_video_frame_unmap(&frame); }
};

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
      m_codecId(0), m_overlayMode(false), m_syncMode(false), m_overlayWindow(0),
      m_width(0), m_height(0), m_failed(false),
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
                // Оба признака известны — дальше сканировать нечего.
                if (hasVcl && isKey) return;
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
// Все слайсы одного пакета принадлежат ОДНОМУ кадру → тип одинаков. Достаточно
// разобрать первый VCL NAL, остальное сканировать не нужно. Для заголовка
// хватает первых ~64 байт payload (first_mb + slice_type = пара ue(v)).
// Если разобрать заголовок не удалось — кадр НЕ отбрасываем (conservative).
bool GstDecoder::hasBSlice(int codecId, const uint8_t* data, int size) {
    uint8_t rbsp[64];
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
                    const int cap = std::min(nalSize - 2, (int)sizeof(rbsp));
                    int n = ebspToRbsp(data + nalStart + 2, cap,
                                       rbsp, (int)sizeof(rbsp));
                    BitReader br(rbsp, n);
                    if (!br.readBit()) { i = nalStart; continue; }  // first_slice_segment=0 — не разбираем
                    if (nalType >= 16) br.readBit();                // no_output_of_prior_pics (IRAP)
                    br.readUE();                                    // slice_pic_parameter_set_id
                    uint32_t sliceType = br.readUE();               // 0=B, 1=P, 2=I
                    if (br.ok() && sliceType == 0) return true;
                    if (br.ok()) return false;  // первый VCL разобран — тип кадра известен
                }
            } else {
                int nalType = data[nalStart] & 0x1f;
                // Слайсовые NAL (1 — non-IDR, 5 — IDR)
                if ((nalType == 1 || nalType == 5) && nalSize >= 2) {
                    const int cap = std::min(nalSize - 1, (int)sizeof(rbsp));
                    int n = ebspToRbsp(data + nalStart + 1, cap,
                                       rbsp, (int)sizeof(rbsp));
                    BitReader br(rbsp, n);
                    br.readUE();                                    // first_mb_in_slice
                    uint32_t sliceType = br.readUE();
                    if (br.ok() && (sliceType % 5) == 1) return true;
                    if (br.ok()) return false;  // первый VCL разобран — тип кадра известен
                }
            }
        }
        i = nalStart;
    }
    return false;
}

// Pad-проба на sink-паде nvv4l2decoder: буфер входит в декодер.
// Запоминаем время входа по PTS (буферы без PTS — с PTS=-1, матчатся по порядку).
GstPadProbeReturn GstDecoder::onDecInProbe(GstPad*, GstPadProbeInfo* info, gpointer userData) {
    GstDecoder* self = static_cast<GstDecoder*>(userData);
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;
    int64_t pts = GST_BUFFER_PTS_IS_VALID(buf) ? (int64_t)GST_BUFFER_PTS(buf) : -1;
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(self->m_mtx);
    self->m_decInPts.emplace_back(pts, now);
    if (self->m_decInPts.size() > 512) self->m_decInPts.pop_front();
    return GST_PAD_PROBE_OK;
}

// Pad-проба на src-паде nvv4l2decoder: кадр вышел из декодера.
// Сопоставляем с временем входа по PTS (берём последний вход с таким PTS —
// ближайший предшествующий выходу), decodeMs = выход − вход. Это чистое
// время NVDEC без parse/очередей/рендера.
GstPadProbeReturn GstDecoder::onDecOutProbe(GstPad*, GstPadProbeInfo* info, gpointer userData) {
    GstDecoder* self = static_cast<GstDecoder*>(userData);
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;
    int64_t pts = GST_BUFFER_PTS_IS_VALID(buf) ? (int64_t)GST_BUFFER_PTS(buf) : -1;
    auto now = std::chrono::steady_clock::now();

    std::chrono::steady_clock::time_point inTime{};
    bool found = false;
    {
        std::lock_guard<std::mutex> lock(self->m_mtx);
        if (pts >= 0) {
            for (auto it = self->m_decInPts.rbegin(); it != self->m_decInPts.rend(); ++it) {
                if (it->first == pts) {
                    inTime = it->second;
                    self->m_decInPts.erase(std::prev(it.base()));
                    found = true;
                    break;
                }
            }
        }
        if (!found && !self->m_decInPts.empty()) {
            inTime = self->m_decInPts.front().second;
            self->m_decInPts.pop_front();
            found = true;
        }
        if (found) {
            double ms = std::chrono::duration<double, std::milli>(now - inTime).count();
            self->m_lastDecodeMs = ms;
            self->m_decTimes.emplace_back(pts, ms);
            if (self->m_decTimes.size() > 256) self->m_decTimes.pop_front();
        }
    }
    return GST_PAD_PROBE_OK;
}

// Pad-проба на sink-паде appsink: фиксируем время появления кадра в appsink.
// Совместно с временем забора в pullFrame даёт postDecodeMs — ожидание кадра
// ПОСЛЕ выхода из декодера (nvvidconv + внутренняя буферизация appsink).
GstPadProbeReturn GstDecoder::onAppSinkArriveProbe(GstPad*, GstPadProbeInfo* info,
                                                   gpointer userData) {
    GstDecoder* self = static_cast<GstDecoder*>(userData);
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;
    int64_t pts = GST_BUFFER_PTS_IS_VALID(buf) ? (int64_t)GST_BUFFER_PTS(buf) : -1;
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(self->m_mtx);
    self->m_appSinkArr.emplace_back(pts, now);
    if (self->m_appSinkArr.size() > 256) self->m_appSinkArr.pop_front();
    return GST_PAD_PROBE_OK;
}

// Pad-проба на src-паде appsrc: буфер покидает appsrc (передан в h264parse).
// Разница с временем pushPacket = время ожидания кадра в очереди appsrc —
// это главный кандидат на «скрытую очередь» (appsrc с block=FALSE копит
// буферы, если downstream не успевает).
GstPadProbeReturn GstDecoder::onAppSrcOutProbe(GstPad*, GstPadProbeInfo* info,
                                               gpointer userData) {
    GstDecoder* self = static_cast<GstDecoder*>(userData);
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;
    int64_t pts = GST_BUFFER_PTS_IS_VALID(buf) ? (int64_t)GST_BUFFER_PTS(buf) : -1;
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(self->m_mtx);
    self->m_appSrcOut.emplace_back(pts, now);
    if (self->m_appSrcOut.size() > 256) self->m_appSrcOut.pop_front();
    return GST_PAD_PROBE_OK;
}

// Pad-проба на src-паде h264parse: буфер покидает парсер (передан в декодер).
// Разница с временем выхода из appsrc = время в парсере. Оставшаяся часть
// pre-decode задержки — ожидание входа в nvv4l2decoder (внутренняя очередь).
GstPadProbeReturn GstDecoder::onParseOutProbe(GstPad*, GstPadProbeInfo* info,
                                              gpointer userData) {
    GstDecoder* self = static_cast<GstDecoder*>(userData);
    GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;
    int64_t pts = GST_BUFFER_PTS_IS_VALID(buf) ? (int64_t)GST_BUFFER_PTS(buf) : -1;
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(self->m_mtx);
    self->m_parseOut.emplace_back(pts, now);
    if (self->m_parseOut.size() > 256) self->m_parseOut.pop_front();
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
        int w = GST_VIDEO_INFO_WIDTH(&info);
        int h = GST_VIDEO_INFO_HEIGHT(&info);
        int64_t pts = GST_BUFFER_PTS_IS_VALID(buffer)
                          ? (int64_t)GST_BUFFER_PTS(buffer) : -1;

        // Чистое время декодирования NVDEC (pad-пробы на декодере, sink→src).
        // Сопоставление по PTS кадра; fallback — последнее измеренное значение
        // (PTS=-1 или запись уже съедена/прунирована дропнутым кадром).
        double decodeMs = -1.0;
        if (pts >= 0) {
            std::lock_guard<std::mutex> lock(self->m_mtx);
            for (auto it = self->m_decTimes.begin(); it != self->m_decTimes.end(); ++it) {
                if (it->first == pts) {
                    decodeMs = it->second;
                    self->m_decTimes.erase(it);
                    break;
                }
            }
        }
        if (decodeMs < 0.0)
            decodeMs = self->m_lastDecodeMs;

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

        // Время от входа пакета в декодер (pushPacket) до выхода кадра из
        // appsink. Сопоставление FIFO 1:1: по одному VCL-пакету на кадр
        // (B-кадры отбрасываются до декодера, дропнутый кадр съедает запись).
        {
            std::lock_guard<std::mutex> lock(self->m_mtx);
            if (!self->m_pushTimes.empty()) {
                auto pushAt = self->m_pushTimes.front();
                self->m_pushTimes.pop_front();
                st.decodeFuncMs = std::chrono::duration<double, std::milli>(now - pushAt).count();
            }
        }

        // Доставка пикселей только если колбэк установлен. В nv3dsink-режиме
        // (NVMM-буфер) пиксели не маппятся и не копируются вовсе.
        if (self->m_frameCb) {
            GstVideoFrame frame;
            if (gst_video_frame_map(&frame, &info, buffer, GST_MAP_READ)) {
                uint8_t* y  = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&frame, 0);
                uint8_t* uv = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&frame, 1);
                int sy  = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
                int suv = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 1);

                if (y && uv) {
                    // Zero-copy: кадр доставляется без копирования пикселей.
                    // Маппинг и буфер удерживаются keepAlive до обработки кадра
                    // в потоке отображения (деструктор HostFrameKeeper).
                    HostFrame hf;
                    hf.yPlane = y; hf.uvPlane = uv;
                    hf.width = w; hf.height = h;
                    hf.strideY = sy; hf.strideUV = suv;
                    hf.pts = pts;
                    hf.keepAlive = std::make_shared<HostFrameKeeper>(frame);

                    auto d0 = std::chrono::steady_clock::now();
                    self->m_frameCb(hf);
                    auto d1 = std::chrono::steady_clock::now();
                    st.displayMs = std::chrono::duration<double, std::milli>(d1 - d0).count();
                } else {
                    gst_video_frame_unmap(&frame);
                }
            }
        }
        if (self->m_latencyCb)
            self->m_latencyCb(pts, st);
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

// Открытие декодера: создание и запуск GStreamer-пайплайна.
// Две топологии:
//  - обычная: appsrc → parse → nvv4l2decoder → nvvidconv(NV12 system) → appsink;
//  - overlay (xvimagesink): appsrc → parse → nvv4l2decoder → tee →
//      { queue → nvvidconv(NV12 system) → xvimagesink (рендер в наше окно),
//        capsf(NVMM) → appsink (только замер, пиксели не маппятся) }.
bool GstDecoder::open(int codecId, bool overlaySink, guintptr overlayWindow,
                      bool syncMode) {
    close();

    m_codecId = codecId;
    m_overlayMode = overlaySink;
    m_syncMode = syncMode;
    m_overlayWindow = overlayWindow;
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
    GstElement* tee   = nullptr;
    GstElement* disp  = nullptr;
    GstElement* dconv = nullptr;
    GstElement* dqueue = nullptr;
    GstElement* conv  = nullptr;
    GstElement* capsf = nullptr;
    if (m_overlayMode) {
        tee    = gst_element_factory_make("tee", "tee");
        disp   = gst_element_factory_make("xvimagesink", "disp");
        dconv  = gst_element_factory_make("nvvidconv", "dconv");
        dqueue = gst_element_factory_make("queue", "dqueue");
        capsf  = gst_element_factory_make("capsfilter", "caps");
    } else {
        conv   = gst_element_factory_make("nvvidconv", "conv");
        capsf  = gst_element_factory_make("capsfilter", "caps");
    }
    m_appsink = gst_element_factory_make("appsink", "sink");
    m_pipeline = gst_pipeline_new("decpipeline");

    if (!m_appsrc || !parse || !dec || !m_appsink || !m_pipeline ||
        (m_overlayMode && (!tee || !disp || !dconv || !dqueue || !capsf)) ||
        (!m_overlayMode && (!conv || !capsf))) {
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
                 NULL);
    // "fast" есть только у h264parse (у h265parse его нет — без проверки
    // свойство вызывает GLib-WARNING при каждом открытии).
    if (g_object_class_find_property(G_OBJECT_GET_CLASS(parse), "fast"))
        g_object_set(parse, "fast", TRUE, NULL);

    // ─── Минимальная буферизация для декодера NVDEC ──────────────────────
    // Свойств "low-latency"/"drop-frame" у nvv4l2decoder нет (проверено
    // gst-inspect-1.0) — они игнорировались. Реальные рычаги задержки:
    //  - disable-dpb=TRUE: отключить DPB-reorder (B-кадры уже режутся
    //    фильтром hasBSlice, переупорядочивание не нужно);
    //  - enable-max-performance=TRUE: убрать троттлинг GPU-клока NVDEC
    //    (на Jetson декодер может работать на пониженной частоте);
    //  - num-extra-surfaces=4: НЕ 0! С единственным surface декодер не может
    //    принять следующий вход, пока выход не забрали — вход блокируется
    //    нашим pull'ом, кадр висит в парсере ~2 интервала (decode_func≈120 мс).
    //    Дополнительные поверхности дают декодеру декодировать кадр СРАЗУ по
    //    пушу (задержка пуш→выход ≈ время декода ~8 мс), а наш loop по-прежнему
    //    пушит/забирает 1 кадр за итерацию (queue_depth=1);
    //  - output-io-mode=0 (auto): значения DMABUF(4) в этой версии JetPack нет
    //    (диапазон 0..2), DMABUF-ветка недоступна — auto выдаёт NVMM, что
    //    нужно и nvvidconv, и замеру.
    g_object_set(dec,
                 "disable-dpb", TRUE,
                 "enable-max-performance", TRUE,
                 "num-extra-surfaces", 4,
                 "output-io-mode", 0,  // auto (в JetPack 6 поддерживается 0/2)
                 NULL);

    // ─── Ветка вывода (nvvidconv+NV12 или xvimagesink-overlay) ─────────────
    if (m_overlayMode) {
        // Ветка рендера: nvvidconv (NVMM→system NV12) → xvimagesink.
        // Рендер идёт через Xv-блайт X-сервера (GPU/композитор, без CUDA-ядер
        // и без копий в приложении). sync=FALSE — non-live appsrc с сырыми PTS;
        // qos=FALSE — не генерировать QoS-события вверх по конвейеру
        // (исключаем влияние дропов на измерение decode_ms).
        g_object_set(dconv,
                     "nvbuf-memory-type", 0,  // Системная память на выходе
                     NULL);
        g_object_set(disp,
                     "sync", FALSE,
                     "qos", FALSE,
                     "force-aspect-ratio", TRUE,
                     "draw-borders", TRUE,
                     "double-buffer", TRUE,
                     NULL);

        // Очередь на ветке рендера (макс. 1 кадр): tee пушит синхронно,
        // рендер идёт в потоке очереди и не блокирует ветку appsink —
        // decode_ms остаётся чистым (без времени отображения).
        g_object_set(dqueue,
                     "max-size-buffers", 1,
                     "max-size-bytes", 0,
                     "max-size-time", 0,
                     "leaky", 2,  // LEAKY_QUEUE: сбрасывать старые кадры
                     NULL);

        // Встраивание в наше X11-окно (окно уже отображено): ESC/PTZ остаются
        // нашими, xvimagesink рендерит только пиксели. Вызывается до PLAYING.
        if (m_overlayWindow) {
            gst_video_overlay_set_window_handle(GST_VIDEO_OVERLAY(disp),
                                                m_overlayWindow);
        }

        // Ветка замера: appsink принимает NVMM-буфер, пиксели не маппятся —
        // frameCb в этом режиме не ставится, конверсий нет вообще.
        GstCaps* nvmmCaps = gst_caps_from_string(
            "video/x-raw(memory:NVMM), format=(string)NV12");
        g_object_set(capsf, "caps", nvmmCaps, NULL);
        gst_caps_unref(nvmmCaps);
    } else {
        g_object_set(conv,
                     "nvbuf-memory-type", 0,  // Системная память на выходе
                     NULL);

        // ─── Настройка капсфильтра для NV12 ────────────────────────────────
        GstCaps* sinkCaps = gst_caps_from_string("video/x-raw, format=(string)NV12");
        g_object_set(capsf, "caps", sinkCaps, NULL);
        gst_caps_unref(sinkCaps);
    }

    // ─── Минимальная буферизация для appsink ──────────────────────────────
    if (m_syncMode) {
        // Zero-latency sync-режим: без сигналов, без дропов, ёмкость 1 кадр.
        // Если декодер выдал кадр, а он ещё не забран pullFrame — appsrc
        // заблокируется (backpressure), а не накопит очередь.
        g_object_set(m_appsink,
                     "emit-signals", FALSE,
                     "sync", FALSE,
                     "drop", FALSE,
                     "max-buffers", 1,
                     NULL);
    } else {
        g_object_set(m_appsink, 
                     "emit-signals", TRUE, 
                     "sync", FALSE,
                     "drop", TRUE,
                     "max-buffers", 1,
                     NULL);
        g_signal_connect(m_appsink, "new-sample", G_CALLBACK(onNewSample), this);
    }

    // Сборка пайплайна
    if (m_overlayMode) {
        gst_bin_add_many(GST_BIN(m_pipeline), m_appsrc, parse, dec, tee,
                         dqueue, dconv, disp, capsf, m_appsink, NULL);
        if (!gst_element_link_many(m_appsrc, parse, dec, tee, NULL) ||
            !gst_element_link_many(tee, dqueue, dconv, disp, NULL) ||
            !gst_element_link_many(tee, capsf, m_appsink, NULL)) {
            fprintf(stderr, "[GstDecoder] не удалось связать элементы пайплайна (overlay)\n");
            close();
            return false;
        }
    } else {
        gst_bin_add_many(GST_BIN(m_pipeline), m_appsrc, parse, dec, conv, capsf, m_appsink, NULL);
        if (!gst_element_link_many(m_appsrc, parse, dec, conv, capsf, m_appsink, NULL)) {
            fprintf(stderr, "[GstDecoder] не удалось связать элементы пайплайна\n");
            close();
            return false;
        }
    }

    // Чистое время декодирования: pad-пробы на nvv4l2decoder (sink → src).
    // Ставятся после линковки, до PLAYING — пробы ловят только поток данных.
    m_decSinkPad = gst_element_get_static_pad(dec, "sink");
    m_decSrcPad  = gst_element_get_static_pad(dec, "src");
    if (m_decSinkPad) m_decInProbeId = gst_pad_add_probe(
        m_decSinkPad, GST_PAD_PROBE_TYPE_BUFFER, onDecInProbe, this, nullptr);
    if (m_decSrcPad) m_decOutProbeId = gst_pad_add_probe(
        m_decSrcPad, GST_PAD_PROBE_TYPE_BUFFER, onDecOutProbe, this, nullptr);

    // Время появления кадра в appsink (для postDecodeMs).
    m_appSinkPad = gst_element_get_static_pad(m_appsink, "sink");
    if (m_appSinkPad) m_appSinkProbeId = gst_pad_add_probe(
        m_appSinkPad, GST_PAD_PROBE_TYPE_BUFFER, onAppSinkArriveProbe, this, nullptr);

    // Время выхода буфера из appsrc и из h264parse (для appSrcHoldMs/parseHoldMs).
    // Разбивает pre-decode задержку: очередь appsrc / парсер / вход декодера.
    m_appSrcPad = gst_element_get_static_pad(m_appsrc, "src");
    if (m_appSrcPad) m_appSrcOutProbeId = gst_pad_add_probe(
        m_appSrcPad, GST_PAD_PROBE_TYPE_BUFFER, onAppSrcOutProbe, this, nullptr);
    m_parseSrcPad = gst_element_get_static_pad(parse, "src");
    if (m_parseSrcPad) m_parseOutProbeId = gst_pad_add_probe(
        m_parseSrcPad, GST_PAD_PROBE_TYPE_BUFFER, onParseOutProbe, this, nullptr);

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

    fprintf(stderr, "[GstDecoder] пайплайн запущен (codecId=%d) %s\n",
            codecId, m_overlayMode ? "overlay-режим (xvimagesink)" : "режим low-latency");
    return true;
}

// Отправка одного закодированного пакета в декодер (сам анализирует пакет)
bool GstDecoder::pushPacket(uint8_t* data, int size, int64_t pts) {
    if (!m_pipeline || !data || size <= 0) return false;
    bool hasVcl = false, isKey = false;
    packetInfo(m_codecId, data, size, hasVcl, isKey);
    const bool isB = packetHasB(m_codecId, data, size);
    return pushPacketParsed(data, size, pts, hasVcl, isKey, isB);
}

// pushPacket с готовым анализом пакета — без повторного сканирования NAL.
bool GstDecoder::pushPacketParsed(uint8_t* data, int size, int64_t pts,
                                  bool hasVcl, bool isKey, bool isB) {
    if (!m_pipeline || !data || size <= 0) return false;

    // ─── Изоляция от B-кадров ──────────────────────────────────────────────
    // Пакет с B-срезом не попадает в декодер вообще: NVDEC не делает reorder,
    // декодируются только I/P-кадры (frame-by-frame, таргет 1-10 мс).
    if (isB) {
        m_droppedB.fetch_add(1);
        return true;  // отброшен (успешно)
    }

    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(m_mtx);
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

// Синхронный забор декодированного кадра из appsink (только syncMode).
// Ждёт до timeoutMs мс. При успехе:
//  - st.decodeMs      — чистое время NVDEC (pad-пробы sink→src, по PTS);
//  - st.decodeFuncMs  — от pushPacket (вход VCL-пакета) до выхода кадра;
//  - st.pushBlockMs   — последняя блокировка gst_app_src_push_buffer;
//  - st.frameIntervalMs — интервал между кадрами из appsink;
//  - st.queueDepth    — VCL-пакетов в полёте (наш кадр = 1, больше — очередь).
// В overlay-режиме пиксели не маппятся (hf.valid()==false) — только тайминги;
// иначе hf заполняется NV12-указателями, буфер держится keepAlive.
bool GstDecoder::pullFrame(HostFrame& hf, DecodeStats& st, int timeoutMs) {
    if (!m_pipeline || !m_appsink || !m_syncMode)
        return false;

    GstSample* sample = gst_app_sink_try_pull_sample(
        GST_APP_SINK(m_appsink), timeoutMs * GST_MSECOND);
    if (!sample)
        return false;

    bool got = false;
    GstBuffer* buffer = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);
    if (buffer && caps) {
        GstVideoInfo info;
        if (gst_video_info_from_caps(&info, caps)) {
            int w = GST_VIDEO_INFO_WIDTH(&info);
            int h = GST_VIDEO_INFO_HEIGHT(&info);
            int64_t pts = GST_BUFFER_PTS_IS_VALID(buffer)
                              ? (int64_t)GST_BUFFER_PTS(buffer) : -1;
            auto now = std::chrono::steady_clock::now();

            // Чистое время декодирования NVDEC (pad-пробы, sink→src).
            double decodeMs = -1.0;
            if (pts >= 0) {
                std::lock_guard<std::mutex> lock(m_mtx);
                for (auto it = m_decTimes.begin(); it != m_decTimes.end(); ++it) {
                    if (it->first == pts) {
                        decodeMs = it->second;
                        m_decTimes.erase(it);
                        break;
                    }
                }
            }
            if (decodeMs < 0.0)
                decodeMs = m_lastDecodeMs;

            // Ожидание кадра ПОСЛЕ выхода из декодера: время появления в appsink
            // (pad-проба) → время забора pullFrame. Большое значение = кадр долго
            // ждал в nvvidconv/очереди после декодирования.
            double postDecodeMs = -1.0;
            if (pts >= 0) {
                std::lock_guard<std::mutex> lock(m_mtx);
                for (auto it = m_appSinkArr.begin(); it != m_appSinkArr.end(); ++it) {
                    if (it->first == pts) {
                        postDecodeMs = std::chrono::duration<double, std::milli>(
                            now - it->second).count();
                        m_appSinkArr.erase(it);
                        break;
                    }
                }
            }

            m_width.store(w);
            m_height.store(h);

            st.decodeMs = decodeMs;
            st.postDecodeMs = postDecodeMs;
            st.pushBlockMs = m_lastPushBlockMs.load();

            if (m_lastSampleAt != std::chrono::steady_clock::time_point{})
                st.frameIntervalMs = std::chrono::duration<double, std::milli>(
                    now - m_lastSampleAt).count();
            m_lastSampleAt = now;

            {
                std::lock_guard<std::mutex> lock(m_mtx);
                st.queueDepth = (int)m_pushTimes.size();
            }

            std::chrono::steady_clock::time_point pushAt{};
            {
                std::lock_guard<std::mutex> lock(m_mtx);
                if (!m_pushTimes.empty()) {
                    pushAt = m_pushTimes.front();
                    m_pushTimes.pop_front();
                    st.decodeFuncMs = std::chrono::duration<double, std::milli>(
                        now - pushAt).count();
                }
            }

            // Ожидание ДО декодера: разбиваем pre-decode задержку на очередь
            // appsrc (appSrcHoldMs) и парсер (parseHoldMs) — время каждого
            // относительно СВОЕГО push (pushAt из FIFO). Остаток
            // (decodeFuncMs − appSrcHold − parseHold − decodeMs − postDecode)
            // — ожидание входа в nvv4l2decoder (внутренняя очередь декодера).
            if (pts >= 0 && pushAt != std::chrono::steady_clock::time_point{}) {
                std::lock_guard<std::mutex> lock(m_mtx);
                for (auto it = m_appSrcOut.begin(); it != m_appSrcOut.end(); ++it) {
                    if (it->first == pts) {
                        st.appSrcHoldMs = std::chrono::duration<double, std::milli>(
                            it->second - pushAt).count();
                        m_appSrcOut.erase(it);
                        break;
                    }
                }
                for (auto it = m_parseOut.begin(); it != m_parseOut.end(); ++it) {
                    if (it->first == pts) {
                        st.parseHoldMs = std::chrono::duration<double, std::milli>(
                            it->second - pushAt).count();
                        m_parseOut.erase(it);
                        break;
                    }
                }
            }

            if (!m_overlayMode) {
                GstVideoFrame frame;
                if (gst_video_frame_map(&frame, &info, buffer, GST_MAP_READ)) {
                    hf.yPlane  = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&frame, 0);
                    hf.uvPlane = (uint8_t*)GST_VIDEO_FRAME_PLANE_DATA(&frame, 1);
                    hf.width = w;
                    hf.height = h;
                    hf.strideY = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
                    hf.strideUV = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 1);
                    hf.pts = pts;
                    hf.keepAlive = std::make_shared<HostFrameKeeper>(frame);
                }
                // Даже если маппинг не удался — кадр декодирован: тайминги
                // уже записаны, считаем его (hf.valid()==false → без пикселей).
            }
            got = true;  // кадр получен из appsink (в overlay — только тайминги)
        }
    }
    gst_sample_unref(sample);
    return got;
}

// Статический хелпер: анализ Annex-B пакета (VCL-срез / ключевой кадр).
bool GstDecoder::packetInfo(int codecId, const uint8_t* data, int size,
                            bool& hasVcl, bool& isKey) {
    scanPacket(codecId, data, size, hasVcl, isKey);
    return hasVcl;
}

// Статический хелпер: есть ли в пакете B-срез (будет отброшен pushPacket).
bool GstDecoder::packetHasB(int codecId, const uint8_t* data, int size) {
    return hasBSlice(codecId, data, size);
}

// Число VCL-пакетов, отправленных в декодер, но без выхода кадра.
int GstDecoder::inFlight() const {
    std::lock_guard<std::mutex> lock(m_mtx);
    return (int)m_pushTimes.size();
}

// Закрытие декодера: остановка пайплайна и освобождение ресурсов
void GstDecoder::close() {
    if (m_pipeline) {
        gst_element_set_state(m_pipeline, GST_STATE_NULL);
        if (m_decSinkPad) {
            if (m_decInProbeId) gst_pad_remove_probe(m_decSinkPad, m_decInProbeId);
            gst_object_unref(m_decSinkPad);
            m_decSinkPad = nullptr;
        }
        if (m_decSrcPad) {
            if (m_decOutProbeId) gst_pad_remove_probe(m_decSrcPad, m_decOutProbeId);
            gst_object_unref(m_decSrcPad);
            m_decSrcPad = nullptr;
        }
        if (m_appSinkPad) {
            if (m_appSinkProbeId) gst_pad_remove_probe(m_appSinkPad, m_appSinkProbeId);
            gst_object_unref(m_appSinkPad);
            m_appSinkPad = nullptr;
        }
        if (m_appSrcPad) {
            if (m_appSrcOutProbeId) gst_pad_remove_probe(m_appSrcPad, m_appSrcOutProbeId);
            gst_object_unref(m_appSrcPad);
            m_appSrcPad = nullptr;
        }
        if (m_parseSrcPad) {
            if (m_parseOutProbeId) gst_pad_remove_probe(m_parseSrcPad, m_parseOutProbeId);
            gst_object_unref(m_parseSrcPad);
            m_parseSrcPad = nullptr;
        }
        m_decInProbeId = m_decOutProbeId = m_appSinkProbeId = 0;
        m_appSrcOutProbeId = m_parseOutProbeId = 0;
        gst_object_unref(m_pipeline);
        m_pipeline = nullptr;
    }
    m_appsrc = nullptr;
    m_appsink = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_pushTimes.clear();
        m_decInPts.clear();
        m_decTimes.clear();
        m_appSinkArr.clear();
        m_appSrcOut.clear();
        m_parseOut.clear();
        m_lastDecodeMs = -1.0;
        m_seenKeyframe = false;
    }
    m_droppedB.store(0);
    m_lastSampleAt = std::chrono::steady_clock::time_point{};
    m_lastPushBlockMs.store(0.0);
    m_failed.store(false);
}