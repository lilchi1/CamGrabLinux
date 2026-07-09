// SwDecoder implementation — FFmpeg software decode → NV12.
#include "headers.h"

SwDecoder::SwDecoder()
    : m_codecCtx(nullptr), m_frame(nullptr), m_nv12Frame(nullptr),
      m_swsCtx(nullptr), m_sps(nullptr), m_spsSize(0),
      m_pps(nullptr), m_ppsSize(0),
      m_prevPts(-1), m_fps(0.0), m_width(0), m_height(0),
      m_totalFrames(0) {}

SwDecoder::~SwDecoder() { close(); }

bool SwDecoder::open(int codecId, const uint8_t* extradata, int extradataSize,
                     int width, int height) {
    close();

    const AVCodec* codec = avcodec_find_decoder((AVCodecID)codecId);
    if (!codec) return false;

    m_codecCtx = avcodec_alloc_context3(codec);
    if (!codec) { close(); return false; }

    // Inject extradata (SPS/PPS) for out-of-band codec data
    if (extradata && extradataSize > 0) {
        m_codecCtx->extradata = (uint8_t*)av_malloc(extradataSize + AV_INPUT_BUFFER_PADDING_SIZE);
        memcpy(m_codecCtx->extradata, extradata, extradataSize);
        m_codecCtx->extradata_size = extradataSize;
    }

    m_codecCtx->width = width;
    m_codecCtx->height = height;
    m_codecCtx->pix_fmt = AV_PIX_FMT_YUV420P;

    if (avcodec_open2(m_codecCtx, codec, nullptr) < 0) { close(); return false; }

    m_frame = av_frame_alloc();
    m_nv12Frame = av_frame_alloc();
    // Pre-allocate NV12 frame buffer
    m_nv12Frame->format = AV_PIX_FMT_NV12;
    m_nv12Frame->width = width;
    m_nv12Frame->height = height;
    av_frame_get_buffer(m_nv12Frame, 0);

    m_width = width;
    m_height = height;
    return true;
}

void SwDecoder::close() {
    if (m_swsCtx) { sws_freeContext(m_swsCtx); m_swsCtx = nullptr; }
    if (m_nv12Frame) { av_frame_free(&m_nv12Frame); }
    if (m_frame) { av_frame_free(&m_frame); }
    if (m_codecCtx) { avcodec_free_context(&m_codecCtx); }
    if (m_sps) { delete[] m_sps; m_sps = nullptr; }
    if (m_pps) { delete[] m_pps; m_pps = nullptr; }
    m_spsSize = m_ppsSize = 0;
}

// Store SPS/PPS for later decoder re-init
static void cacheSpsPps(SwDecoder* dec, const uint8_t* data, int size) {}

bool SwDecoder::decode(uint8_t* data, int size, int64_t pts, bool isKeyFrame) {
    if (!m_codecCtx) return false;

    AVPacket* pkt = av_packet_alloc();
    pkt->data = data;
    pkt->size = size;
    pkt->pts = pts;
    pkt->dts = pts;

    int ret = avcodec_send_packet(m_codecCtx, pkt);
    av_packet_free(&pkt);
    if (ret < 0) return false;

    ret = avcodec_receive_frame(m_codecCtx, m_frame);
    if (ret < 0) return false;

    // YUV420P → NV12 conversion
    m_swsCtx = sws_getCachedContext(m_swsCtx,
        m_frame->width, m_frame->height, AV_PIX_FMT_YUV420P,
        m_nv12Frame->width, m_nv12Frame->height, AV_PIX_FMT_NV12,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!m_swsCtx) return false;

    uint8_t* dstData[4] = { m_nv12Frame->data[0], m_nv12Frame->data[1], nullptr, nullptr };
    int dstStride[4] = { m_nv12Frame->linesize[0], m_nv12Frame->linesize[1], 0, 0 };
    sws_scale(m_swsCtx, m_frame->data, m_frame->linesize, 0, m_frame->height,
              dstData, dstStride);

    m_totalFrames++;

    // Update FPS estimation every 30 frames
    if (m_totalFrames % 30 == 0 && pts >= 0 && m_prevPts >= 0) {
        m_fps = 30.0 / ((double)(pts - m_prevPts) / 90000.0);
        if (m_fps < 0) m_fps = 0;
    }
    if (pts >= 0) m_prevPts = pts;

    // Invoke user callback with NV12 data
    if (m_callback) {
        m_callback(m_nv12Frame->data[0], m_nv12Frame->data[1],
                   m_nv12Frame->width, m_nv12Frame->height,
                   m_nv12Frame->linesize[0], m_nv12Frame->linesize[1],
                   pts);
    }

    return true;
}
