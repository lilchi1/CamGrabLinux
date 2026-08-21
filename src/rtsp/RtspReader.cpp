// RtspReader.cpp — Реализация чтения RTSP через FFmpeg AVFormatContext + битстрим-фильтр.
#include "headers.h"
#include <cstdio>

RtspReader::RtspReader()
    : m_fmtCtx(nullptr), m_videoStreamIdx(-1), m_codecId(0),
      m_width(0), m_height(0), m_pkt(nullptr),
      m_filteredPkt(nullptr), m_bsfCtx(nullptr) {}

RtspReader::~RtspReader() { close(); }

// Открытие RTSP потока с минимальной задержкой.
// Транспорт определяется --rtsp-transport: tcp / udp / auto (udp → tcp fallback).
// Важно: RTSP-сигнализация (DESCRIBE/SETUP/PLAY) всегда идёт по TCP:554, опция
// rtsp_transport влияет только на медиапоток (RTP). Поэтому «Connection refused»
// по tcp://...:554 означает, что камера не приняла управляющее соединение —
// обычно это занятый лимит RTSP-сессий, неверный порт или firewall.
bool RtspReader::open(const std::string& url, int timeoutSec) {
    close();

    // ─── Порядок попыток транспорта ────────────────────────────────────────
    std::vector<std::string> transports;
    if (g_rtspTransport == "rtp")      transports = { "rtp" };
    else if (g_rtspTransport == "udp") transports = { "udp" };
    else                               transports = { "udp", "rtp" };  // auto

    // ─── Минимальная задержка: общие опции ────────────────────────────────
    char tbuf[32];
    snprintf(tbuf, sizeof(tbuf), "%d", timeoutSec * 1000000);

    // ─── Запрос разрешения у камеры ────────────────────────────────────────
    // FFmpeg-RTSP не умеет сам запрашивать разрешение (см. `ffmpeg -h demuxer=rtsp`),
    // поэтому при явно заданных -w/-H добавляем vendor-параметры width/height
    // к RTSP-URL. Большинство IP-камер (Hikvision/Dahua/ONVIF-совместимые)
    // воспринимают их в query-строке; если камера игнорирует — поток просто
    // придёт в исходном разрешении.
    std::string openUrl = url;
    if (g_camResRequested) {
        std::string q;
        if (g_winWidth > 0)
            q += (q.empty() ? "" : "&") + std::string("width=") + std::to_string(g_winWidth);
        if (g_winHeight > 0)
            q += (q.empty() ? "" : "&") + std::string("height=") + std::to_string(g_winHeight);
        if (!q.empty()) {
            openUrl += (openUrl.find('?') == std::string::npos ? "?" : "&") + q;
            logWrite("INFO", url, "Запрос разрешения у камеры: " + q);
        }
    }

    // ─── Попытки открытия: каждая со своими опциями ────────────────────────
    std::string usedTransport;
    int lastRet = -1;
    for (const auto& tr : transports) {
        AVDictionary* opts = nullptr;
        av_dict_set(&opts, "rtsp_transport", tr.c_str(), 0);
        av_dict_set(&opts, "stimeout", tbuf, 0);
        av_dict_set(&opts, "fflags", "nobuffer", 0);          // Отключить буферизацию
        av_dict_set(&opts, "flags", "low_delay", 0);          // Режим низкой задержки
        av_dict_set(&opts, "max_delay", "0", 0);              // Нулевая задержка
        av_dict_set(&opts, "probesize", "32", 0);             // Минимальный размер
        av_dict_set(&opts, "analyzeduration", "0", 0);        // Не анализировать долго
        av_dict_set(&opts, "reorder_queue_size", "0", 0);     // Без переупорядочивания
        av_dict_set(&opts, "avio_flags", "direct", 0);        // Прямой доступ к сокету
        av_dict_set(&opts, "buffer_size", "4194304", 0);      // Сокетный буфер UDP
        av_dict_set(&opts, "rw_timeout", tbuf, 0);

        m_fmtCtx = avformat_alloc_context();
        int ret = avformat_open_input(&m_fmtCtx, openUrl.c_str(), nullptr, &opts);
        av_dict_free(&opts);
        if (ret >= 0) {
            usedTransport = tr;
            lastRet = ret;
            break;
        }
        char errbuf[AV_ERROR_MAX_STRING_SIZE] = { 0 };
        av_strerror(ret, errbuf, sizeof(errbuf));
        logWrite("WARN", url, "RTSP-транспорт " + tr + " не открылся: " + errbuf);
        close();
        m_fmtCtx = nullptr;
        lastRet = ret;
    }

    if (!m_fmtCtx) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE] = { 0 };
        av_strerror(lastRet, errbuf, sizeof(errbuf));
        logWrite("ERROR", url, std::string("Не удалось открыть RTSP поток: ") + errbuf);
        logWrite("ERROR", url,
                 "Проверьте URL/порт, число RTSP-сессий камеры (обычно лимит 1-2, "
                 "второе подключение к той же камере отклоняется) и firewall.");
        return false;
    }

    logWrite("INFO", url, "RTSP открыт (транспорт медиа: " + usedTransport + ")");

    // Проверка, что мы действительно используем нужный транспорт
    if (m_fmtCtx->iformat && m_fmtCtx->iformat->name) {
        logWrite("INFO", url, std::string("Формат: ") + m_fmtCtx->iformat->name);
    }

    // Получение информации о потоках с минимальной задержкой
    int ret = avformat_find_stream_info(m_fmtCtx, nullptr);
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

    logWrite("INFO", url, "RTSP открыт: " + usedTransport + ", low_delay, без буферизации, без переупорядочивания");
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
            // Освободить предыдущий вывод фильтра перед новым приёмом
            // (av_bsf_receive_packet отдаёт реф на внутренний буфер фильтра).
            av_packet_unref(m_filteredPkt);
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