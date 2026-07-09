// NvV4l2Decoder implementation — Jetson V4L2 NVDEC (H.264/H.265).
#include "headers.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

#ifndef V4L2_PIX_FMT_H264
#define V4L2_PIX_FMT_H264 v4l2_fourcc('H','2','6','4')
#endif
#ifndef V4L2_PIX_FMT_H265
#define V4L2_PIX_FMT_H265 v4l2_fourcc('H','2','6','5')
#endif
#ifndef V4L2_EVENT_RESOLUTION_CHANGE
#define V4L2_EVENT_RESOLUTION_CHANGE 5
#endif

int avCodecToV4l2(int avCodecId) {
    if (avCodecId == AV_CODEC_ID_H264) return V4L2_PIX_FMT_H264;
    if (avCodecId == AV_CODEC_ID_H265) return V4L2_PIX_FMT_H265;
    return 0;
}

NvV4l2Decoder::NvV4l2Decoder()
    : m_fd(-1), m_width(0), m_height(0), m_codecId(0),
      m_numOutputBufs(0), m_numCaptureBufs(0) {}

NvV4l2Decoder::~NvV4l2Decoder() { close(); }

bool NvV4l2Decoder::open(int codecId, int width, int height) {
    close();

    m_codecId = codecId;
    m_width = width;
    m_height = height;

    int v4l2Codec = avCodecToV4l2(codecId);
    if (!v4l2Codec) return false;

    // Open V4L2 decoder device (::open = POSIX, not the class method)
    m_fd = ::open("/dev/nvhost-nvdec", O_RDWR | O_NONBLOCK, 0);
    if (m_fd < 0) {
        m_fd = ::open("/dev/nvhost-msenc", O_RDWR | O_NONBLOCK, 0);
        if (m_fd < 0) { m_fd = ::open("/dev/video0", O_RDWR | O_NONBLOCK, 0); }
    }
    if (m_fd < 0) return false;

    // Subscribe to resolution change events
    struct v4l2_event_subscription sub;
    memset(&sub, 0, sizeof(sub));
    sub.type = V4L2_EVENT_RESOLUTION_CHANGE;
    ioctl(m_fd, VIDIOC_SUBSCRIBE_EVENT, &sub);

    // Set up OUTPUT (encoded data) queue
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    fmt.fmt.pix_mp.pixelformat = v4l2Codec;
    fmt.fmt.pix_mp.width = width;
    fmt.fmt.pix_mp.height = height;
    fmt.fmt.pix_mp.num_planes = 1;

    if (ioctl(m_fd, VIDIOC_S_FMT, &fmt) < 0) { close(); return false; }

    // Request and mmap OUTPUT buffers
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 8;
    req.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(m_fd, VIDIOC_REQBUFS, &req) < 0) { close(); return false; }
    m_numOutputBufs = req.count;
    m_outputBufs.resize(m_numOutputBufs, nullptr);

    for (int i = 0; i < m_numOutputBufs; i++) {
        struct v4l2_plane planes[1];
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        memset(planes, 0, sizeof(planes));
        buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        buf.m.planes = planes;
        buf.length = 1;
        if (ioctl(m_fd, VIDIOC_QUERYBUF, &buf) < 0) { close(); return false; }
        m_outputBufs[i] = mmap(nullptr, buf.m.planes[0].length,
                               PROT_READ | PROT_WRITE, MAP_SHARED,
                               m_fd, buf.m.planes[0].m.mem_offset);
    }

    streamOn(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
    return true;
}

void NvV4l2Decoder::close() {
    if (m_fd >= 0) {
        streamOff(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE);
        streamOff(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE);
        for (auto* p : m_outputBufs) if (p) munmap(p, 0);
        for (auto* p : m_captureBufs) if (p) munmap(p, 0);
        ::close(m_fd);
        m_fd = -1;
    }
    m_outputBufs.clear();
    m_captureBufs.clear();
    m_numOutputBufs = 0;
    m_numCaptureBufs = 0;
}

bool NvV4l2Decoder::decode(uint8_t* data, int size, int64_t pts) {
    if (m_fd < 0 || !data || size <= 0) return false;

    // Enqueue encoded data to OUTPUT queue
    struct v4l2_plane planes[1];
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    memset(planes, 0, sizeof(planes));
    buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.m.planes = planes;
    buf.length = 1;
    buf.m.planes[0].bytesused = 0;

    // Find an available OUTPUT buffer by DQBUF
    int idx;
    if (!dequeueBuffer(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, &idx)) return false;
    if (idx >= 0 && idx < m_numOutputBufs) {
        memcpy(m_outputBufs[idx], data, size);
        buf.index = idx;
        buf.m.planes[0].bytesused = size;
        if (ioctl(m_fd, VIDIOC_QBUF, &buf) < 0) return false;
    }

    // Dequeue CAPTURE frame
    int capIdx;
    if (dequeueBuffer(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, &capIdx) && capIdx >= 0) {
        // Deliver NV12 data via callback
        if (m_callback && capIdx < m_numCaptureBufs && m_captureBufs[capIdx]) {
            struct v4l2_plane cplanes[2];
            struct v4l2_buffer cbuf;
            memset(&cbuf, 0, sizeof(cbuf));
            memset(cplanes, 0, sizeof(cplanes));
            cbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
            cbuf.memory = V4L2_MEMORY_MMAP;
            cbuf.index = capIdx;
            cbuf.m.planes = cplanes;
            cbuf.length = 2;
            if (ioctl(m_fd, VIDIOC_QUERYBUF, &cbuf) == 0) {
                uint8_t* yPlane  = (uint8_t*)m_captureBufs[capIdx];
                uint8_t* uvPlane = yPlane + cbuf.m.planes[0].bytesused;
                m_callback(yPlane, uvPlane, m_width, m_height,
                           cbuf.m.planes[0].bytesused / m_height,
                           cbuf.m.planes[1].bytesused / (m_height / 2), pts);
            }
        }
        // Re-queue the capture buffer
        struct v4l2_plane cplanes[2];
        struct v4l2_buffer cbuf;
        memset(&cbuf, 0, sizeof(cbuf));
        memset(cplanes, 0, sizeof(cplanes));
        cbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        cbuf.memory = V4L2_MEMORY_MMAP;
        cbuf.index = capIdx;
        cbuf.m.planes = cplanes;
        cbuf.length = 2;
        ioctl(m_fd, VIDIOC_QBUF, &cbuf);
    }

    return true;
}

bool NvV4l2Decoder::requestBuffers(int type, int count) {
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = count;
    req.type = type;
    req.memory = V4L2_MEMORY_MMAP;
    return ioctl(m_fd, VIDIOC_REQBUFS, &req) >= 0;
}

bool NvV4l2Decoder::queueBuffer(int type, int index,
                                uint8_t* data, int size) {
    struct v4l2_buffer buf;
    struct v4l2_plane planes[1];
    memset(&buf, 0, sizeof(buf));
    memset(planes, 0, sizeof(planes));
    buf.type = type;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = index;
    buf.m.planes = planes;
    buf.length = 1;
    if (data) {
        buf.m.planes[0].bytesused = size;
        if (index >= 0 && index < m_numOutputBufs && m_outputBufs[index])
            memcpy(m_outputBufs[index], data, size);
    }
    return ioctl(m_fd, VIDIOC_QBUF, &buf) >= 0;
}

bool NvV4l2Decoder::dequeueBuffer(int type, int* index) {
    struct v4l2_buffer buf;
    struct v4l2_plane planes[1];
    memset(&buf, 0, sizeof(buf));
    memset(planes, 0, sizeof(planes));
    buf.type = type;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.m.planes = planes;
    buf.length = 1;
    if (ioctl(m_fd, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN) return true; // No buffer available — not an error
        return false;
    }
    *index = buf.index;
    return true;
}

bool NvV4l2Decoder::streamOn(int type) {
    return ioctl(m_fd, VIDIOC_STREAMON, &type) >= 0;
}

bool NvV4l2Decoder::streamOff(int type) {
    return ioctl(m_fd, VIDIOC_STREAMOFF, &type) >= 0;
}
