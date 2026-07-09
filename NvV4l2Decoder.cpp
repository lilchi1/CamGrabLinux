#include "NvV4l2Decoder.h"
#include "NvBufSurface.h"
#include <cstring>
#include <cstdio>
#include <cerrno>
#include <unistd.h>
#include <poll.h>

NvV4l2Decoder::NvV4l2Decoder()
    : m_decoder(nullptr)
    , m_decoderValid(false)
    , m_codecId(0)
    , m_width(0)
    , m_height(0)
    , m_firstPacket(true)
    , m_numOutputBuffers(0)
    , m_numCaptureBuffers(0)
    , m_nextOutputIndex(0)
    , m_captureSetupDone(false)
    , m_extradata(nullptr)
    , m_extradataSize(0)
    , m_spsPpsSent(false)
    , m_outputBuffersQueued(0)
    , m_maxOutputBuffersQueued(0)
{
    for (int i = 0; i < 32; i++) m_captureFds[i] = -1;
}

NvV4l2Decoder::~NvV4l2Decoder()
{
    close();
}

bool NvV4l2Decoder::createCaptureBuffers()
{
    NvBufSurf::NvCommonAllocateParams capParams = {};
    capParams.width = (unsigned)m_width;
    capParams.height = (unsigned)m_height;
    capParams.layout = NVBUF_LAYOUT_PITCH;
    capParams.colorFormat = NVBUF_COLOR_FORMAT_NV12;
    capParams.memType = NVBUF_MEM_SURFACE_ARRAY;
    capParams.memtag = NvBufSurfaceTag_VIDEO_DEC;

    int ret = NvBufSurf::NvAllocate(&capParams, m_numCaptureBuffers, m_captureFds);
    if (ret != 0) {
        fprintf(stderr, "NvV4l2Decoder: NvBufSurf::NvAllocate failed (%d)\n", ret);
        return false;
    }

    return true;
}

void NvV4l2Decoder::destroyCaptureBuffers()
{
    for (int i = 0; i < 32; i++) {
        if (m_captureFds[i] >= 0) {
            NvBufSurf::NvDestroy(m_captureFds[i]);
            m_captureFds[i] = -1;
        }
    }
}

bool NvV4l2Decoder::setupCapture()
{
    struct v4l2_format format = {};
    int ret = m_decoder->capture_plane.getFormat(format);
    if (ret < 0) {
        fprintf(stderr, "NvV4l2Decoder: capture getFormat failed\n");
        return false;
    }

    struct v4l2_crop crop = {};
    crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ret = m_decoder->capture_plane.getCrop(crop);
    if (ret < 0) {
        fprintf(stderr, "NvV4l2Decoder: capture getCrop failed\n");
        return false;
    }

    m_width = crop.c.width ? crop.c.width : format.fmt.pix_mp.width;
    m_height = crop.c.height ? crop.c.height : format.fmt.pix_mp.height;

    if (m_decoder->setCapturePlaneFormat(format.fmt.pix_mp.pixelformat,
                                          format.fmt.pix_mp.width,
                                          format.fmt.pix_mp.height) < 0) {
        fprintf(stderr, "NvV4l2Decoder: setCapturePlaneFormat after resolution change failed\n");
        return false;
    }

    int minCaptureBuffers = 0;
    ret = m_decoder->getMinimumCapturePlaneBuffers(minCaptureBuffers);
    if (ret < 0) {
        fprintf(stderr, "NvV4l2Decoder: getMinimumCapturePlaneBuffers failed\n");
        return false;
    }

    m_numCaptureBuffers = minCaptureBuffers + 2;

    ret = m_decoder->capture_plane.reqbufs(V4L2_MEMORY_DMABUF,
                                           (uint32_t)m_numCaptureBuffers);
    if (ret < 0) {
        fprintf(stderr, "NvV4l2Decoder: capture reqbufs failed\n");
        return false;
    }

    m_numCaptureBuffers = m_decoder->capture_plane.getNumBuffers();

    if (!createCaptureBuffers()) {
        return false;
    }

    ret = m_decoder->capture_plane.setStreamStatus(true);
    if (ret < 0) {
        fprintf(stderr, "NvV4l2Decoder: capture streamon failed\n");
        return false;
    }

    for (int i = 0; i < m_numCaptureBuffers; i++) {
        struct v4l2_buffer v4l2_buf = {};
        struct v4l2_plane planes[3] = {};
        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        v4l2_buf.memory = V4L2_MEMORY_DMABUF;
        v4l2_buf.length = 3;
        planes[0].m.fd = m_captureFds[i];

        ret = m_decoder->capture_plane.qBuffer(v4l2_buf, nullptr);
        if (ret < 0) {
            fprintf(stderr, "NvV4l2Decoder: qBuffer capture %d failed\n", i);
            return false;
        }
    }

    m_captureSetupDone = true;
    return true;
}

static int alignUp(int val, int align) {
    return (val + align - 1) & ~(align - 1);
}

bool NvV4l2Decoder::startCapture(int width, int height)
{
    m_width = width;
    m_height = height;

    if (m_decoder->setCapturePlaneFormat(V4L2_PIX_FMT_NV12M, width, height) < 0) {
        fprintf(stderr, "NvV4l2Decoder: setCapturePlaneFormat failed\n");
        return false;
    }

    int minCaptureBuffers = 4;
    int ret = m_decoder->getMinimumCapturePlaneBuffers(minCaptureBuffers);
    if (ret < 0) {
        fprintf(stderr, "NvV4l2Decoder: getMinimumCapturePlaneBuffers failed, using default\n");
        minCaptureBuffers = 4;
    }

    m_numCaptureBuffers = minCaptureBuffers + 2;

    ret = m_decoder->capture_plane.reqbufs(V4L2_MEMORY_DMABUF,
                                           (uint32_t)m_numCaptureBuffers);
    if (ret < 0) {
        fprintf(stderr, "NvV4l2Decoder: capture reqbufs failed\n");
        return false;
    }

    m_numCaptureBuffers = m_decoder->capture_plane.getNumBuffers();

    if (!createCaptureBuffers()) {
        return false;
    }

    ret = m_decoder->capture_plane.setStreamStatus(true);
    if (ret < 0) {
        fprintf(stderr, "NvV4l2Decoder: capture streamon failed\n");
        return false;
    }

    for (int i = 0; i < m_numCaptureBuffers; i++) {
        struct v4l2_buffer v4l2_buf = {};
        struct v4l2_plane planes[3] = {};
        v4l2_buf.index = i;
        v4l2_buf.m.planes = planes;
        v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        v4l2_buf.memory = V4L2_MEMORY_DMABUF;
        v4l2_buf.length = 3;
        planes[0].m.fd = m_captureFds[i];

        ret = m_decoder->capture_plane.qBuffer(v4l2_buf, nullptr);
        if (ret < 0) {
            fprintf(stderr, "NvV4l2Decoder: qBuffer capture %d failed\n", i);
            return false;
        }
    }

    m_captureSetupDone = true;
    return true;
}

bool NvV4l2Decoder::init(int codecId, FrameCallback cb,
                          int knownWidth, int knownHeight)
{
    m_codecId = codecId;
    m_callback = cb;

    m_decoder = NvVideoDecoder::createVideoDecoder("rtsp_decoder");
    if (!m_decoder) {
        fprintf(stderr, "NvV4l2Decoder: createVideoDecoder failed\n");
        return false;
    }

    // Early NVDEC health check: if the NVDEC HW channel failed to
    // allocate (e.g. all 16 channels exhausted), any NVDEC-dependent
    // V4L2 operation (STREAMOFF, REQBUFS, G_CTRL) can corrupt heap.
    // Detect it now before setting up any plane state.
    {
        int32_t dummy = 0;
        if (m_decoder->getControl(V4L2_CID_MIN_BUFFERS_FOR_CAPTURE, dummy) < 0) {
            fprintf(stderr, "NvV4l2Decoder: NVDEC hardware not available\n");
            close();
            return false;
        }
    }

    if (knownWidth <= 0 || knownHeight <= 0) {
        if (m_decoder->subscribeEvent(V4L2_EVENT_RESOLUTION_CHANGE, 0, 0) < 0) {
            fprintf(stderr, "NvV4l2Decoder: subscribeEvent failed\n");
            close();
            return false;
        }
    }

    if (m_decoder->setOutputPlaneFormat(m_codecId, 2097152) < 0) {
        fprintf(stderr, "NvV4l2Decoder: setOutputPlaneFormat failed\n");
        close();
        return false;
    }

    m_decoder->setFrameInputMode(0);

    if (m_decoder->output_plane.setupPlane(V4L2_MEMORY_MMAP, 10, true, false) < 0) {
        fprintf(stderr, "NvV4l2Decoder: output_plane.setupPlane failed\n");
        close();
        return false;
    }

    m_numOutputBuffers = m_decoder->output_plane.getNumBuffers();

    if (m_decoder->output_plane.setStreamStatus(true) < 0) {
        fprintf(stderr, "NvV4l2Decoder: output streamon failed\n");
        close();
        return false;
    }

    m_firstPacket = true;
    m_captureSetupDone = false;

    // NVDEC HW is verified working and output plane is set up.
    // Set flag now so that close() runs full cleanup if startCapture
    // or resolution-change setup fail later.
    m_decoderValid = true;

    if (knownWidth > 0 && knownHeight > 0) {
        if (!startCapture(knownWidth, knownHeight)) {
            close();
            return false;
        }
    }

    return true;
}

void NvV4l2Decoder::close()
{
    if (m_decoder) {
        if (m_decoderValid) {
            if (m_numOutputBuffers > 0) {
                m_decoder->output_plane.setStreamStatus(false);
                m_decoder->output_plane.deinitPlane();
            }
            if (m_captureSetupDone) {
                m_decoder->capture_plane.setStreamStatus(false);
                m_decoder->capture_plane.deinitPlane();
            }
        }

        destroyCaptureBuffers();

        delete m_decoder;
        m_decoder = nullptr;
    }

    if (m_extradata) {
        free(m_extradata);
        m_extradata = nullptr;
    }
    m_extradataSize = 0;

    m_numOutputBuffers = 0;
    m_numCaptureBuffers = 0;
    m_nextOutputIndex = 0;
    m_captureSetupDone = false;
    m_decoderValid = false;
    m_spsPpsSent = false;
    m_outputBuffersQueued = 0;
}

bool NvV4l2Decoder::recycleOutputBuffers()
{
    struct v4l2_plane dqPlanes[1] = {};
    struct v4l2_buffer dqBuf = {};
    dqBuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    dqBuf.memory = V4L2_MEMORY_MMAP;
    dqBuf.m.planes = dqPlanes;
    dqBuf.length = 1;

    int count = 0;
    while (m_decoder->output_plane.dqBuffer(dqBuf, nullptr, nullptr, 0) == 0) {
        count++;
    }
    m_outputBuffersQueued -= count;
    if (m_outputBuffersQueued < 0) m_outputBuffersQueued = 0;
    return true;
}

int NvV4l2Decoder::numQueuedOutputBuffers()
{
    struct v4l2_plane dqPlanes[1] = {};
    struct v4l2_buffer dqBuf = {};
    dqBuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    dqBuf.memory = V4L2_MEMORY_MMAP;
    dqBuf.m.planes = dqPlanes;
    dqBuf.length = 1;

    int count = 0;
    while (m_decoder->output_plane.dqBuffer(dqBuf, nullptr, nullptr, 0) == 0) {
        count++;
    }
    m_outputBuffersQueued -= count;
    if (m_outputBuffersQueued < 0) m_outputBuffersQueued = 0;
    return m_outputBuffersQueued;
}

bool NvV4l2Decoder::setExtradata(const uint8_t* data, int size)
{
    if (m_extradata) {
        free(m_extradata);
        m_extradata = nullptr;
    }
    m_extradataSize = 0;
    if (!data || size <= 0) return true;
    m_extradata = (uint8_t*)malloc(size);
    if (!m_extradata) return false;
    memcpy(m_extradata, data, size);
    m_extradataSize = size;
    return true;
}

bool NvV4l2Decoder::sendExtradata()
{
    if (!m_extradata || m_extradataSize <= 0) return true;

    size_t offset = 0;
    while (offset < (size_t)m_extradataSize) {
        if (offset + 4 > (size_t)m_extradataSize) break;
        int startCode = 0;
        if (m_extradata[offset] == 0 && m_extradata[offset+1] == 0 &&
            m_extradata[offset+2] == 0 && m_extradata[offset+3] == 1) {
            startCode = 4;
        } else if (m_extradata[offset] == 0 && m_extradata[offset+1] == 0 &&
                   m_extradata[offset+2] == 1) {
            startCode = 3;
        } else {
            offset++;
            continue;
        }
        if (offset + startCode >= (size_t)m_extradataSize) break;
        int nalType = m_extradata[offset + startCode] & 0x1F;
        if (nalType != 7 && nalType != 8) { offset += startCode + 1; continue; }

        size_t nalEnd = offset + startCode;
        while (nalEnd < (size_t)m_extradataSize) {
            if (nalEnd + 4 <= (size_t)m_extradataSize &&
                m_extradata[nalEnd] == 0 && m_extradata[nalEnd+1] == 0 &&
                m_extradata[nalEnd+2] == 0 && m_extradata[nalEnd+3] == 1) break;
            if (nalEnd + 3 <= (size_t)m_extradataSize &&
                m_extradata[nalEnd] == 0 && m_extradata[nalEnd+1] == 0 &&
                m_extradata[nalEnd+2] == 1) break;
            if (nalEnd + 3 <= (size_t)m_extradataSize &&
                nalEnd > offset + startCode &&
                m_extradata[nalEnd-1] == 0 && m_extradata[nalEnd] != 0 &&
                m_extradata[nalEnd-2] == 0 && m_extradata[nalEnd-3] == 0) break;
            nalEnd++;
        }
        int nalSize = (int)(nalEnd - offset);

        NvBuffer* buf = m_decoder->output_plane.getNthBuffer(m_nextOutputIndex);
        if (!buf) return false;

        if (nalSize > (int)buf->planes[0].length) {
            offset = nalEnd;
            continue;
        }

        memcpy(buf->planes[0].data, m_extradata + offset, nalSize);
        buf->planes[0].bytesused = nalSize;

        struct v4l2_buffer v4l2_buf = {};
        struct v4l2_plane planes[1] = {};
        v4l2_buf.index = m_nextOutputIndex;
        v4l2_buf.m.planes = planes;
        v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
        v4l2_buf.memory = V4L2_MEMORY_MMAP;
        v4l2_buf.length = 1;
        v4l2_buf.flags = 0;

        if (m_decoder->output_plane.qBuffer(v4l2_buf, nullptr) < 0) break;

        m_nextOutputIndex = (m_nextOutputIndex + 1) % m_numOutputBuffers;
        m_outputBuffersQueued++;

        offset = nalEnd;
    }

    m_spsPpsSent = true;
    return true;
}

bool NvV4l2Decoder::decode(const uint8_t* data, size_t size, int64_t pts, bool isKeyFrame)
{
    if (!m_decoder || !data || size == 0) return false;

    if (!m_captureSetupDone) {
        recycleOutputBuffers();
        struct v4l2_event ev;
        int ret = m_decoder->dqEvent(ev, 0);
        if (ret == 0 && ev.type == V4L2_EVENT_RESOLUTION_CHANGE) {
            if (!setupCapture()) return false;
        }
    }

    if (!m_spsPpsSent && m_extradata && m_extradataSize > 0) {
        sendExtradata();
    }

    if (m_outputBuffersQueued >= m_numOutputBuffers - 1) {
        recycleOutputBuffers();
        if (m_outputBuffersQueued >= m_numOutputBuffers - 1) {
            if (!isKeyFrame) return true;
            struct v4l2_plane dqPlanes[1] = {};
            struct v4l2_buffer dqBuf = {};
            dqBuf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
            dqBuf.memory = V4L2_MEMORY_MMAP;
            dqBuf.m.planes = dqPlanes;
            dqBuf.length = 1;
            if (m_decoder->output_plane.dqBuffer(dqBuf, nullptr, nullptr, 1000000) != 0) {
                return false;
            }
            m_outputBuffersQueued--;
            if (m_outputBuffersQueued < 0) m_outputBuffersQueued = 0;
        }
    }

    NvBuffer* buffer = m_decoder->output_plane.getNthBuffer(m_nextOutputIndex);
    if (!buffer) return false;

    if (size > (int)buffer->planes[0].length) {
        fprintf(stderr, "NvV4l2Decoder: packet %zu > buffer %u\n", size, buffer->planes[0].length);
        return false;
    }
    memcpy(buffer->planes[0].data, data, size);
    buffer->planes[0].bytesused = size;

    struct v4l2_buffer v4l2_buf = {};
    struct v4l2_plane planes[1] = {};
    v4l2_buf.index = m_nextOutputIndex;
    v4l2_buf.m.planes = planes;
    v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    v4l2_buf.memory = V4L2_MEMORY_MMAP;
    v4l2_buf.length = 1;
    v4l2_buf.flags = isKeyFrame ? V4L2_BUF_FLAG_KEYFRAME : 0;
    v4l2_buf.timestamp.tv_sec = pts / 1000000;
    v4l2_buf.timestamp.tv_usec = pts % 1000000;

    if (m_decoder->output_plane.qBuffer(v4l2_buf, nullptr) < 0) {
        fprintf(stderr, "NvV4l2Decoder: qBuffer output failed\n");
        return false;
    }
    m_outputBuffersQueued++;

    m_nextOutputIndex = (m_nextOutputIndex + 1) % m_numOutputBuffers;
    if (isKeyFrame) m_spsPpsSent = true;

    if (!m_captureSetupDone) return true;

    recycleOutputBuffers();

    struct v4l2_plane dqPlanes[3] = {};
    struct v4l2_buffer dqBuf = {};
    dqBuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    dqBuf.memory = V4L2_MEMORY_DMABUF;
    dqBuf.m.planes = dqPlanes;
    dqBuf.length = 3;

    while (m_decoder->capture_plane.dqBuffer(dqBuf, nullptr, nullptr, 0) == 0) {
        if (m_callback) {
            int index = dqBuf.index;
            int fd = m_captureFds[index];

            NvBufSurface* nvbuf_surf = nullptr;
            if (NvBufSurfaceFromFd(fd, (void**)(&nvbuf_surf)) == 0) {
                int pitchY = nvbuf_surf->surfaceList->planeParams.pitch[0];
                int pitchUV = nvbuf_surf->surfaceList->planeParams.pitch[1];

                static bool diag = false;
                if (!diag) {
                    diag = true;
                    fprintf(stderr, "NvV4l2Decoder: surf %dx%d pitch[0]=%d psize[0]=%lu "
                            "pitch[1]=%d psize[1]=%lu layout=%d\n",
                            nvbuf_surf->surfaceList->width,
                            nvbuf_surf->surfaceList->height,
                            pitchY,
                            (unsigned long)nvbuf_surf->surfaceList->planeParams.psize[0],
                            pitchUV,
                            (unsigned long)nvbuf_surf->surfaceList->planeParams.psize[1],
                            nvbuf_surf->surfaceList->layout);
                }

                int strideY = pitchY;
                int strideUV = pitchUV;
                if (strideY < m_width || strideY > m_width * 2)
                    strideY = alignUp(m_width, 64);
                if (strideUV < m_width || strideUV > m_width * 2)
                    strideUV = alignUp(m_width, 64);

                NvBufSurfaceMap(nvbuf_surf, 0, 0, NVBUF_MAP_READ);
                NvBufSurfaceMap(nvbuf_surf, 0, 1, NVBUF_MAP_READ);

                NvBufSurfaceSyncForCpu(nvbuf_surf, 0, 0);
                NvBufSurfaceSyncForCpu(nvbuf_surf, 0, 1);

                uint8_t* yPtr = (uint8_t*)nvbuf_surf->surfaceList->mappedAddr.addr[0];
                uint8_t* uvPtr = (uint8_t*)nvbuf_surf->surfaceList->mappedAddr.addr[1];

                if (!uvPtr || uvPtr <= yPtr)
                    uvPtr = yPtr + nvbuf_surf->surfaceList->planeParams.psize[0];

                int64_t framePts = pts;
                if (dqBuf.timestamp.tv_sec != 0 || dqBuf.timestamp.tv_usec != 0)
                    framePts = dqBuf.timestamp.tv_sec * 1000000LL + dqBuf.timestamp.tv_usec;

                if (yPtr && uvPtr) {
                    m_callback(yPtr, uvPtr, m_width, m_height,
                               strideY, strideUV, framePts);
                }

                NvBufSurfaceUnMap(nvbuf_surf, 0, 0);
                NvBufSurfaceUnMap(nvbuf_surf, 0, 1);
            }
        }

        dqBuf.m.planes[0].m.fd = m_captureFds[dqBuf.index];
        if (m_decoder->capture_plane.qBuffer(dqBuf, nullptr) < 0) {
            fprintf(stderr, "NvV4l2Decoder: qBuffer capture requeue failed\n");
            break;
        }
    }

    return true;
}

bool NvV4l2Decoder::flush()
{
    if (!m_decoder) return false;

    struct v4l2_plane planes[1] = {};
    struct v4l2_buffer buf = {};
    buf.index = m_nextOutputIndex;
    buf.m.planes = planes;
    buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.length = 1;
    planes[0].bytesused = 0;

    if (m_decoder->output_plane.qBuffer(buf, nullptr) < 0) {
        return false;
    }

    if (!m_captureSetupDone) return true;

    struct v4l2_plane dqPlanes[3] = {};
    struct v4l2_buffer dqBuf = {};
    dqBuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    dqBuf.memory = V4L2_MEMORY_DMABUF;
    dqBuf.m.planes = dqPlanes;
    dqBuf.length = 3;

    while (m_decoder->capture_plane.dqBuffer(dqBuf, nullptr, nullptr, 50) == 0) {
        if (dqPlanes[0].bytesused == 0) break;

        dqBuf.m.planes[0].m.fd = m_captureFds[dqBuf.index];
        if (m_decoder->capture_plane.qBuffer(dqBuf, nullptr) < 0) break;
    }

    return true;
}
