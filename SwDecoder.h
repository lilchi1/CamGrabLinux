// SwDecoder — software H.264/H.265 decoder using FFmpeg libavcodec.
// Produces NV12 frames delivered via FrameCallback.
#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}
#include <cstdint>
#include <string>

#include "FrameCallback.h"

class SwDecoder {
public:
    SwDecoder();
    ~SwDecoder();

    bool open(int codecId, const uint8_t* extradata, int extradataSize,
              int width, int height);
    void close();
    bool decode(uint8_t* data, int size, int64_t pts, bool isKeyFrame);

    void setFrameCallback(FrameCallback cb) { m_callback = cb; }
    bool isOpen() const { return m_codecCtx != nullptr; }
    int fps() const { return m_fps; }
    uint64_t totalFrames() const { return m_totalFrames; }

private:
    AVCodecContext* m_codecCtx;
    AVFrame* m_frame;
    AVFrame* m_nv12Frame;
    SwsContext* m_swsCtx;
    FrameCallback m_callback;

    uint8_t* m_sps;           // cached SPS for decoder re-init
    int m_spsSize;
    uint8_t* m_pps;
    int m_ppsSize;
    int64_t m_prevPts;
    double m_fps;
    int m_width;
    int m_height;
    uint64_t m_totalFrames;
};
