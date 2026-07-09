// NvV4l2Decoder — hardware-accelerated video decoder on Jetson using V4L2 NVDEC.
// Produces NV12 frames delivered via FrameCallback.
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <linux/videodev2.h>

#include "FrameCallback.h"

class NvV4l2Decoder {
public:
    NvV4l2Decoder();
    ~NvV4l2Decoder();

    bool open(int codecId, int width, int height);
    void close();
    bool decode(uint8_t* data, int size, int64_t pts);

    void setFrameCallback(FrameCallback cb) { m_callback = cb; }
    bool isOpen() const { return m_fd >= 0; }

private:
    int m_fd;                    // V4L2 device handle (/dev/nvhost-nvdec or similar)
    int m_width;
    int m_height;
    int m_codecId;
    FrameCallback m_callback;

    // V4L2 buffer tracking
    std::vector<void*> m_outputBufs;
    std::vector<void*> m_captureBufs;
    int m_numOutputBufs;
    int m_numCaptureBufs;

    bool requestBuffers(int type, int count);
    bool queueBuffer(int type, int index, uint8_t* data = nullptr, int size = 0);
    bool dequeueBuffer(int type, int* index);
    bool streamOn(int type);
    bool streamOff(int type);
};
