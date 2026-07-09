#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

using FrameCallback = std::function<void(uint8_t* yPlane, uint8_t* uvPlane,
                                         int width, int height,
                                         int strideY, int strideUV,
                                         int64_t pts)>;

class SwDecoder {
public:
    SwDecoder();
    ~SwDecoder();

    bool init(int avCodecId, FrameCallback cb = nullptr,
              int knownWidth = 0, int knownHeight = 0);
    void close();

    bool decode(const uint8_t* data, size_t size, int64_t pts, bool isKeyFrame);
    bool flush();

    bool setExtradata(const uint8_t* data, int size);

    int width() const { return m_width; }
    int height() const { return m_height; }
    bool isOpen() const { return m_codecCtx != nullptr; }

    void setFrameCallback(FrameCallback cb) { m_callback = cb; }

private:
    AVCodecContext* m_codecCtx;
    AVFrame* m_frame;
    bool m_codecValid;

    int m_width;
    int m_height;

    FrameCallback m_callback;

    uint8_t* m_nv12Buf;
    AVFrame* m_nv12Frame;
    SwsContext* m_swsCtx;
    AVPixelFormat m_srcFmt;

    uint8_t* m_extradata;
    int m_extradataSize;

    bool ensureNv12Converter(int width, int height, AVPixelFormat srcFmt);
};
