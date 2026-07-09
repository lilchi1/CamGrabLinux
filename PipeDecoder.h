#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <functional>

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

using FrameCallback = std::function<void(uint8_t* yPlane, uint8_t* uvPlane,
                                         int width, int height,
                                         int strideY, int strideUV,
                                         int64_t pts)>;

class PipeDecoder {
public:
    PipeDecoder();
    ~PipeDecoder();

    bool init(const std::string& url, FrameCallback cb = nullptr);
    void close();

    int readFrame(int64_t pts);

    int width() const { return m_width; }
    int height() const { return m_height; }
    bool isOpen() const { return m_pipe != nullptr; }

    void setFrameCallback(FrameCallback cb) { m_callback = cb; }

private:
    FILE* m_pipe;
    std::vector<uint8_t> m_buf;
    int m_width;
    int m_height;
    FrameCallback m_callback;

    SwsContext* m_swsCtx;
    uint8_t* m_nv12Buf;
    AVFrame* m_nv12Frame;
    int m_nv12Width;
    int m_nv12Height;

    bool ensureNv12Converter(int width, int height);
    bool findJpegFrame(std::vector<uint8_t>& jpeg);
};
