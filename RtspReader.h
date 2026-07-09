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

    bool open(const std::string& url, int timeoutSec = 10);
    void close();

    bool readPacket(uint8_t*& data, int& size, int64_t& pts, bool& isKeyFrame);
    int videoStreamIndex() const { return m_videoStreamIdx; }

    int codecId() const { return m_codecId; }
    int width() const { return m_width; }
    int height() const { return m_height; }
    bool isOpen() const { return m_fmtCtx != nullptr; }

    const uint8_t* extradata() const { return m_extradata; }
    int extradataSize() const { return m_extradataSize; }

private:
    AVFormatContext* m_fmtCtx;
    int m_videoStreamIdx;
    int m_codecId;
    int m_width;
    int m_height;
    AVPacket* m_pkt;
    AVPacket* m_filteredPkt;
    AVBSFContext* m_bsfCtx;
    uint8_t* m_extradata;
    int m_extradataSize;
};