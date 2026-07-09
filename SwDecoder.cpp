#include "SwDecoder.h"
#include <cstdio>
#include <cstring>

SwDecoder::SwDecoder()
    : m_codecCtx(nullptr)
    , m_frame(nullptr)
    , m_codecValid(false)
    , m_width(0)
    , m_height(0)
    , m_nv12Buf(nullptr)
    , m_nv12Frame(nullptr)
    , m_swsCtx(nullptr)
    , m_srcFmt(AV_PIX_FMT_NONE)
    , m_extradata(nullptr)
    , m_extradataSize(0)
{
}

SwDecoder::~SwDecoder()
{
    close();
}

bool SwDecoder::ensureNv12Converter(int width, int height, AVPixelFormat srcFmt)
{
    if (m_swsCtx && m_width == width && m_height == height && m_srcFmt == srcFmt)
        return true;

    if (m_swsCtx) sws_freeContext(m_swsCtx);
    if (m_nv12Buf) av_freep(&m_nv12Buf);
    if (m_nv12Frame) av_frame_free(&m_nv12Frame);

    m_swsCtx = nullptr;
    m_nv12Buf = nullptr;
    m_nv12Frame = nullptr;

    m_nv12Frame = av_frame_alloc();
    if (!m_nv12Frame) return false;

    int ret = av_image_alloc(m_nv12Frame->data, m_nv12Frame->linesize,
                             width, height, AV_PIX_FMT_NV12, 32);
    if (ret < 0) {
        av_frame_free(&m_nv12Frame);
        return false;
    }
    m_nv12Buf = m_nv12Frame->data[0];
    m_nv12Frame->width = width;
    m_nv12Frame->height = height;
    m_nv12Frame->format = AV_PIX_FMT_NV12;

    m_swsCtx = sws_getContext(width, height, srcFmt,
                              width, height, AV_PIX_FMT_NV12,
                              SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!m_swsCtx) {
        av_freep(&m_nv12Buf);
        av_frame_free(&m_nv12Frame);
        return false;
    }

    m_width = width;
    m_height = height;
    m_srcFmt = srcFmt;
    return true;
}

bool SwDecoder::init(int avCodecId, FrameCallback cb, int knownWidth, int knownHeight)
{
    m_callback = cb;

    const AVCodec* codec = avcodec_find_decoder((AVCodecID)avCodecId);
    if (!codec) {
        fprintf(stderr, "SwDecoder: avcodec_find_decoder failed for id %d\n", avCodecId);
        return false;
    }

    m_codecCtx = avcodec_alloc_context3(codec);
    if (!m_codecCtx) {
        fprintf(stderr, "SwDecoder: avcodec_alloc_context3 failed\n");
        return false;
    }

    if (knownWidth > 0 && knownHeight > 0) {
        m_codecCtx->width = knownWidth;
        m_codecCtx->height = knownHeight;
    }

    if (m_extradata && m_extradataSize > 0) {
        m_codecCtx->extradata = (uint8_t*)av_malloc(m_extradataSize + AV_INPUT_BUFFER_PADDING_SIZE);
        if (m_codecCtx->extradata) {
            memcpy(m_codecCtx->extradata, m_extradata, m_extradataSize);
            memset(m_codecCtx->extradata + m_extradataSize, 0, AV_INPUT_BUFFER_PADDING_SIZE);
            m_codecCtx->extradata_size = m_extradataSize;
        }
    }

    m_codecCtx->thread_count = 2;

    int ret = avcodec_open2(m_codecCtx, codec, nullptr);
    if (ret < 0) {
        fprintf(stderr, "SwDecoder: avcodec_open2 failed (%d)\n", ret);
        close();
        return false;
    }

    m_frame = av_frame_alloc();
    if (!m_frame) {
        fprintf(stderr, "SwDecoder: av_frame_alloc failed\n");
        close();
        return false;
    }

    m_codecValid = true;
    return true;
}

void SwDecoder::close()
{
    if (m_codecCtx) {
        if (m_codecValid) {
            avcodec_flush_buffers(m_codecCtx);
        }
        avcodec_free_context(&m_codecCtx);
        m_codecCtx = nullptr;
    }

    if (m_frame) {
        av_frame_free(&m_frame);
        m_frame = nullptr;
    }

    if (m_swsCtx) {
        sws_freeContext(m_swsCtx);
        m_swsCtx = nullptr;
    }

    if (m_nv12Buf) {
        av_freep(&m_nv12Buf);
        m_nv12Buf = nullptr;
    }

    if (m_nv12Frame) {
        av_frame_free(&m_nv12Frame);
        m_nv12Frame = nullptr;
    }

    if (m_extradata) {
        av_free(m_extradata);
        m_extradata = nullptr;
    }
    m_extradataSize = 0;

    m_width = 0;
    m_height = 0;
    m_codecValid = false;
}

bool SwDecoder::setExtradata(const uint8_t* data, int size)
{
    if (m_extradata) {
        av_free(m_extradata);
        m_extradata = nullptr;
    }
    m_extradataSize = 0;
    if (!data || size <= 0) return true;
    m_extradata = (uint8_t*)av_malloc(size + AV_INPUT_BUFFER_PADDING_SIZE);
    if (!m_extradata) return false;
    memcpy(m_extradata, data, size);
    memset(m_extradata + size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
    m_extradataSize = size;
    return true;
}

static bool isValidNal(const uint8_t* data, size_t size)
{
    if (!data || size < 4) return false;
    if (size > 4 * 1024 * 1024) return false;
    int startCode = 0;
    if (data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) {
        startCode = 4;
    } else if (data[0] == 0 && data[1] == 0 && data[2] == 1) {
        startCode = 3;
    } else {
        return false;
    }
    if ((size_t)startCode >= size) return false;
    int nalType = data[startCode] & 0x1F;
    if (nalType > 23) return false;
    return true;
}

bool SwDecoder::decode(const uint8_t* data, size_t size, int64_t pts, bool isKeyFrame)
{
    if (!m_codecCtx || !m_frame || !m_codecValid) return false;
    if (!isValidNal(data, size)) return false;

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) return false;

    pkt->data = (uint8_t*)data;
    pkt->size = size;
    pkt->pts = pts;
    pkt->dts = AV_NOPTS_VALUE;

    int ret = avcodec_send_packet(m_codecCtx, pkt);
    av_packet_free(&pkt);

    if (ret < 0) {
        if (ret == AVERROR(EAGAIN)) return true;
        return false;
    }

    while (ret >= 0) {
        ret = avcodec_receive_frame(m_codecCtx, m_frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            break;
        }
        if (ret < 0) {
            return false;
        }

        if (m_callback) {
            int strideY, strideUV;
            uint8_t *yPlane, *uvPlane;
            int fw = m_frame->width, fh = m_frame->height;

            if ((AVPixelFormat)m_frame->format != AV_PIX_FMT_NV12) {
                if (!ensureNv12Converter(fw, fh,
                                         (AVPixelFormat)m_frame->format))
                    continue;

                sws_scale(m_swsCtx,
                          (const uint8_t* const*)m_frame->data,
                          m_frame->linesize, 0, fh,
                          m_nv12Frame->data, m_nv12Frame->linesize);

                yPlane = m_nv12Frame->data[0];
                uvPlane = m_nv12Frame->data[1];
                strideY = m_nv12Frame->linesize[0];
                strideUV = m_nv12Frame->linesize[1];
            } else {
                yPlane = m_frame->data[0];
                uvPlane = m_frame->data[1];
                strideY = m_frame->linesize[0];
                strideUV = m_frame->linesize[1];
            }

            if (yPlane && uvPlane) {
                int computedStrideY = (int)(uvPlane - yPlane) / (fh > 0 ? fh : 1);
                if (computedStrideY > strideY && computedStrideY < strideY * 4)
                    strideY = computedStrideY;
                if (strideY < fw) strideY = fw;
                if (strideUV < fw) strideUV = fw;

                m_callback(yPlane, uvPlane,
                           fw, fh,
                           strideY, strideUV,
                           m_frame->pts);
            }
        }
    }

    return true;
}

bool SwDecoder::flush()
{
    if (!m_codecCtx || !m_codecValid) return false;

    int ret = avcodec_send_packet(m_codecCtx, nullptr);
    if (ret < 0) return false;

    while (ret >= 0) {
        ret = avcodec_receive_frame(m_codecCtx, m_frame);
        if (ret == AVERROR_EOF) break;
        if (ret < 0) break;

        if (m_callback) {
            if ((AVPixelFormat)m_frame->format != AV_PIX_FMT_NV12) {
                if (ensureNv12Converter(m_frame->width, m_frame->height,
                                        (AVPixelFormat)m_frame->format)) {
                    sws_scale(m_swsCtx,
                              (const uint8_t* const*)m_frame->data,
                              m_frame->linesize, 0, m_frame->height,
                              m_nv12Frame->data, m_nv12Frame->linesize);
                    m_callback(m_nv12Frame->data[0], m_nv12Frame->data[1],
                               m_frame->width, m_frame->height,
                               m_nv12Frame->linesize[0], m_nv12Frame->linesize[1],
                               m_frame->pts);
                }
            } else {
                m_callback(m_frame->data[0], m_frame->data[1],
                           m_frame->width, m_frame->height,
                           m_frame->linesize[0], m_frame->linesize[1],
                           m_frame->pts);
            }
        }
    }

    return true;
}
