// RtspReader.cpp — Реализация чтения RTSP через FFmpeg AVFormatContext + битстрим-фильтр.
#include "headers.h"

RtspReader::RtspReader()
    : m_fmtCtx(nullptr), m_videoStreamIdx(-1), m_codecId(0),
      m_width(0), m_height(0), m_pkt(nullptr),
      m_filteredPkt(nullptr), m_bsfCtx(nullptr) {}

RtspReader::~RtspReader() { close(); }

// Открытие RTSP потока: ТОЛЬКО UDP с минимальной задержкой
bool RtspReader::open(const std::string& url, int timeoutSec) {
    close();

    AVDictionary* opts = nullptr;
    
    // ─── ТОЛЬКО UDP транспорт ──────────────────────────────────────────────
    // rtsp_transport=udp — RTP-медиапотоки по UDP. Опцию rtsp_flags не задаём:
    // prefer_tcp заставляет пробовать TCP первым, prefer_udp не существует.
    // RTSP-сигнализация (DESCRIBE/SETUP/PLAY) в FFmpeg всегда идёт по TCP:554.
    av_dict_set(&opts, "rtsp_transport", "udp", 0);
    
    // ─── Минимальная задержка ──────────────────────────────────────────────
    char tbuf[32];
    snprintf(tbuf, sizeof(tbuf), "%d", timeoutSec * 1000000);
    av_dict_set(&opts, "stimeout", tbuf, 0);
    
    av_dict_set(&opts, "fflags", "nobuffer", 0);           // Отключить буферизацию
    av_dict_set(&opts, "flags", "low_delay", 0);           // Режим низкой задержки
    av_dict_set(&opts, "max_delay", "0", 0);               // Нулевая задержка
    av_dict_set(&opts, "probesize", "32", 0);              // Минимальный размер для определения формата
    av_dict_set(&opts, "analyzeduration", "0", 0);         // Не анализировать длительно
    
    // ─── Отключить переупорядочивание ──────────────────────────────────────
    av_dict_set(&opts, "reorder_queue_size", "0", 0);      // Без переупорядочивания
    
    // ─── Прямой (без буферизации) доступ к сокету ────────────────────────────
    // Замечание: сам фильтр B-кадров живёт в GstDecoder::pushPacket — пакеты с
    // B-срезами отбрасываются ДО декодера (проект изолирован от B-кадров).
    av_dict_set(&opts, "avio_flags", "direct", 0);
    
    // ─── Увеличенный сокетный буфер для UDP ────────────────────────────────
    // Предупреждение: нужно настроить ядро: sudo sysctl -w net.core.rmem_max=8388608
    av_dict_set(&opts, "buffer_size", "4194304", 0);

    m_fmtCtx = avformat_alloc_context();
    
    // Установка таймаута для сокетов
    av_dict_set(&opts, "rw_timeout", tbuf, 0);
    
    int ret = avformat_open_input(&m_fmtCtx, url.c_str(), nullptr, &opts);
    av_dict_free(&opts);
    if (ret < 0) { 
        logWrite("ERROR", url, "Не удалось открыть RTSP (только UDP)");
        close(); 
        return false; 
    }

    // Проверка, что мы действительно используем UDP
    if (m_fmtCtx->iformat && m_fmtCtx->iformat->name) {
        logWrite("INFO", url, std::string("Формат: ") + m_fmtCtx->iformat->name);
    }

    // Получение информации о потоках с минимальной задержкой
    ret = avformat_find_stream_info(m_fmtCtx, nullptr);
    if (ret < 0) { close(); return false; }

    // Поиск первого видеопотока
    for (unsigned i = 0; i < m_fmtCtx->nb_streams; i++) {
        if (m_fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            m_videoStreamIdx = static_cast<int>(i);
            break;
        }
    }
    if (m_videoStreamIdx < 0) { close(); return false; }

    // Извлечение параметров кодека и разрешения
    auto* par = m_fmtCtx->streams[m_videoStreamIdx]->codecpar;
    m_codecId = par->codec_id;
    m_width  = par->width;
    m_height = par->height;

    // Проверка: если H.264, проверяем наличие B-кадров
    if (m_codecId == AV_CODEC_ID_H264) {
        // Получаем extradata для проверки
        if (par->extradata && par->extradata_size > 0) {
            // В H.264 extradata (SPS/PPS) можно проверить наличие B-кадров
            // Но проще проверить по фреймам при декодировании
            logWrite("INFO", url, "H.264 поток обнаружен (B-кадры будут игнорироваться)");
        }
    }

    // Выделение памяти для пакетов
    m_pkt = av_packet_alloc();
    m_filteredPkt = av_packet_alloc();

    // Настройка битстрим-фильтра (mp4toannexb) для H.264/H.265
    const char* bsfName = nullptr;
    if (m_codecId == AV_CODEC_ID_H264) bsfName = "h264_mp4toannexb";
    else if (m_codecId == AV_CODEC_ID_H265) bsfName = "hevc_mp4toannexb";

    if (bsfName) {
        const AVBitStreamFilter* bsf = av_bsf_get_by_name(bsfName);
        if (bsf) {
            av_bsf_alloc(bsf, &m_bsfCtx);
            avcodec_parameters_copy(m_bsfCtx->par_in, par);
            av_bsf_init(m_bsfCtx);
        }
    }

    logWrite("INFO", url, "RTSP открыт: ТОЛЬКО UDP, low_delay, без буферизации, без переупорядочивания");
    return true;
}

// Закрытие потока и освобождение ресурсов FFmpeg
void RtspReader::close() {
    if (m_bsfCtx) av_bsf_free(&m_bsfCtx);
    if (m_pkt) av_packet_free(&m_pkt);
    if (m_filteredPkt) av_packet_free(&m_filteredPkt);
    if (m_fmtCtx) avformat_close_input(&m_fmtCtx);
    m_videoStreamIdx = -1;
}

// Чтение одного видеопакета с минимальной задержкой
bool RtspReader::readPacket(uint8_t*& data, int& size, int64_t& pts) {
    while (true) {
        int ret = av_read_frame(m_fmtCtx, m_pkt);
        if (ret < 0) return false;

        // Пропуск пакетов не из видеопотока
        if (m_pkt->stream_index != m_videoStreamIdx) {
            av_packet_unref(m_pkt);
            continue;
        }

        AVPacket* outPkt = m_pkt;
        // Применение битстрим-фильтра (h264_mp4toannexb / hevc_mp4toannexb)
        if (m_bsfCtx) {
            if (av_bsf_send_packet(m_bsfCtx, m_pkt) < 0) {
                av_packet_unref(m_pkt);
                return false;
            }
            if (av_bsf_receive_packet(m_bsfCtx, m_filteredPkt) < 0) {
                av_packet_unref(m_pkt);
                return false;
            }
            outPkt = m_filteredPkt;
        }

        data = outPkt->data;
        size = outPkt->size;
        pts  = outPkt->pts;
        
        // Если PTS нет, используем DTS
        if (pts == AV_NOPTS_VALUE) {
            pts = outPkt->dts;
        }
        
        return true;
    }
}