// RtspReader — opens an RTSP stream via FFmpeg, reads H.264/H.265 packets,
// applies Annex-B bitstream filter, and exposes raw NAL data.
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

    // Reads one video packet. data/size/pts/isKeyFrame are valid on success.
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
    AVPacket* m_filteredPkt;       // output of bitstream filter
    AVBSFContext* m_bsfCtx;        // h264_mp4toannexb / hevc_mp4toannexb
    uint8_t* m_extradata;          // copy of codec extradata (SPS/PPS)
    int m_extradataSize;
};
