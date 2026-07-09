#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <functional>

#include <linux/videodev2.h>
#include "NvVideoDecoder.h"
#include "NvBuffer.h"
#include "NvBufSurface.h"

using FrameCallback = std::function<void(uint8_t* yPlane, uint8_t* uvPlane,
                                         int width, int height,
                                         int strideY, int strideUV,
                                         int64_t pts)>;

class NvV4l2Decoder {
public:
    NvV4l2Decoder();
    ~NvV4l2Decoder();

    bool init(int codecId, FrameCallback cb = nullptr,
              int knownWidth = 0, int knownHeight = 0);
    void close();

    bool decode(const uint8_t* data, size_t size, int64_t pts, bool isKeyFrame);
    bool flush();

    bool setExtradata(const uint8_t* data, int size);

    int width() const { return m_width; }
    int height() const { return m_height; }
    bool isOpen() const { return m_decoder != nullptr; }

    void setFrameCallback(FrameCallback cb) { m_callback = cb; }

private:
    NvVideoDecoder* m_decoder;
    bool m_decoderValid;
    int m_codecId;
    int m_width;
    int m_height;
    bool m_firstPacket;

    int m_numOutputBuffers;
    int m_numCaptureBuffers;
    int m_nextOutputIndex;

    FrameCallback m_callback;

    int m_captureFds[32];
    bool m_captureSetupDone;

    uint8_t* m_extradata;
    int m_extradataSize;
    bool m_spsPpsSent;
    int m_outputBuffersQueued;
    int m_maxOutputBuffersQueued;

    bool createCaptureBuffers();
    void destroyCaptureBuffers();
    bool startCapture(int width, int height);
    bool setupCapture();
    bool recycleOutputBuffers();
    bool sendExtradata();
    int  numQueuedOutputBuffers();
};
