// PipeDecoder — decodes MJPEG streams by piping to an external ffmpeg process.
// Useful when no hardware decoder is available for MJPEG.
#pragma once

#include <cstdint>
#include <string>

#include "FrameCallback.h"

class PipeDecoder {
public:
    PipeDecoder();
    ~PipeDecoder();

    bool open(const std::string& url, int width, int height);
    void close();
    bool decode(uint8_t* data, int size, int64_t pts);

    void setFrameCallback(FrameCallback cb) { m_callback = cb; }
    bool isOpen() const { return m_fp != nullptr; }

private:
    FILE* m_fp;          // pipe to ffmpeg process
    int m_width;
    int m_height;
    FrameCallback m_callback;
};
