// RtspReader.h — Чтение RTSP потока через FFmpeg, H.264/H.265 пакеты с Annex-B фильтрацией.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavutil/avutil.h>
}

class RtspReader {
public:
    RtspReader();
    ~RtspReader();

    // Открытие RTSP потока с таймаутом (секунды)
    bool open(const std::string& url, int timeoutSec = 10);

    // Закрытие потока
    void close();

    // Чтение одного видеопакета. data/size/pts валидны при успехе.
    bool readPacket(uint8_t*& data, int& size, int64_t& pts);

    // Доступ к метаданным потока
    int codecId() const { return m_codecId; }
    int width() const { return m_width; }
    int height() const { return m_height; }

private:
    AVFormatContext* m_fmtCtx;       // Контекст формата FFmpeg
    int m_videoStreamIdx;            // Индекс видеопотока
    int m_codecId;                   // ID кодека (H.264, H.265)
    int m_width;                     // Разрешение по ширине
    int m_height;                    // Разрешение по высоте
    AVPacket* m_pkt;                 // Буфер для чтения пакетов
    AVPacket* m_filteredPkt;         // Выход битстрим-фильтра
    AVBSFContext* m_bsfCtx;          // Контекст фильтра (h264_mp4toannexb / hevc_mp4toannexb)
};
