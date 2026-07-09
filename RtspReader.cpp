#include "RtspReader.h"
#include <cstdio>
#include <cstring>

RtspReader::RtspReader()
    : m_fmtCtx(nullptr)
    , m_videoStreamIdx(-1)
    , m_codecId(0)
    , m_width(0)
    , m_height(0)
    , m_pkt(nullptr)
    , m_filteredPkt(nullptr)
    , m_bsfCtx(nullptr)
    , m_extradata(nullptr)
    , m_extradataSize(0)
{
}

RtspReader::~RtspReader()
{
    close();
}

bool RtspReader::open(const std::string& url, int timeoutSec)
{
    avformat_network_init();

    AVDictionary* opts = nullptr;
    char timeout[32];
    snprintf(timeout, sizeof(timeout), "%d", timeoutSec * 1000000);
    av_dict_set(&opts, "fflags", "nobuffer", 0);
    av_dict_set(&opts, "flags", "low_delay", 0);
    av_dict_set(&opts, "reorder_queue_size", "0", 0);
    av_dict_set(&opts, "max_delay", "0", 0);
    av_dict_set(&opts, "rtsp_flags", "prefer_tcp", 0);

    m_fmtCtx = avformat_alloc_context();
    if (!m_fmtCtx) {
        fprintf(stderr, "RtspReader: avformat_alloc_context failed\n");
        av_dict_free(&opts);
        return false;
    }

    av_dict_set(&opts, "buffer_size", "2097152", 0);

    int ret = -1;

    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "stimeout", timeout, 0);
    ret = avformat_open_input(&m_fmtCtx, url.c_str(), nullptr, &opts);

    if (ret < 0) {
        fprintf(stderr, "RtspReader: TCP failed, trying UDP...\n");
        close();
        m_fmtCtx = avformat_alloc_context();
        if (!m_fmtCtx) {
            fprintf(stderr, "RtspReader: avformat_alloc_context failed\n");
            av_dict_free(&opts);
            return false;
        }
        av_dict_set(&opts, "rtsp_transport", "udp", 0);
        av_dict_set(&opts, "stimeout", timeout, 0);
        av_dict_set(&opts, "buffer_size", "4194304", 0);
        av_dict_set(&opts, "fpsprobesize", "0", 0);
        ret = avformat_open_input(&m_fmtCtx, url.c_str(), nullptr, &opts);
    }

    av_dict_free(&opts);
    if (ret < 0) {
        char errBuf[256] = {};
        av_strerror(ret, errBuf, sizeof(errBuf));
        fprintf(stderr, "RtspReader: avformat_open_input failed: %s\n", errBuf);
        close();
        return false;
    }

    ret = avformat_find_stream_info(m_fmtCtx, nullptr);
    if (ret < 0) {
        fprintf(stderr, "RtspReader: avformat_find_stream_info failed\n");
        close();
        return false;
    }

    for (unsigned int i = 0; i < m_fmtCtx->nb_streams; i++) {
        if (m_fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            m_videoStreamIdx = i;
            AVCodecParameters* par = m_fmtCtx->streams[i]->codecpar;
            m_codecId = par->codec_id;
            m_width = par->width;
            m_height = par->height;
            if (par->extradata && par->extradata_size > 0) {
                m_extradata = (uint8_t*)av_malloc(par->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
                if (m_extradata) {
                    memcpy(m_extradata, par->extradata, par->extradata_size);
                    memset(m_extradata + par->extradata_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
                    m_extradataSize = par->extradata_size;
                }
            }
            break;
        }
    }

    if (m_videoStreamIdx < 0) {
        fprintf(stderr, "RtspReader: no video stream found\n");
        close();
        return false;
    }

    m_pkt = av_packet_alloc();
    if (!m_pkt) {
        fprintf(stderr, "RtspReader: av_packet_alloc failed\n");
        close();
        return false;
    }

    const char* bsfName = nullptr;
    if (m_codecId == AV_CODEC_ID_H264) bsfName = "h264_mp4toannexb";
    else if (m_codecId == AV_CODEC_ID_HEVC) bsfName = "hevc_mp4toannexb";

    if (bsfName) {
        const AVBitStreamFilter* bsf = av_bsf_get_by_name(bsfName);
        if (bsf) {
            if (av_bsf_alloc(bsf, &m_bsfCtx) == 0) {
                AVCodecParameters* par = m_fmtCtx->streams[m_videoStreamIdx]->codecpar;
                if (avcodec_parameters_copy(m_bsfCtx->par_in, par) >= 0) {
                    if (av_bsf_init(m_bsfCtx) == 0) {
                        m_filteredPkt = av_packet_alloc();
                        if (!m_filteredPkt) {
                            av_bsf_free(&m_bsfCtx);
                            m_bsfCtx = nullptr;
                        }
                    } else {
                        av_bsf_free(&m_bsfCtx);
                        m_bsfCtx = nullptr;
                    }
                } else {
                    av_bsf_free(&m_bsfCtx);
                    m_bsfCtx = nullptr;
                }
            }
        }
        if (!m_bsfCtx)
            fprintf(stderr, "RtspReader: %s bsf init skipped\n", bsfName);
    }

    return true;
}

void RtspReader::close()
{
    if (m_pkt) {
        av_packet_free(&m_pkt);
        m_pkt = nullptr;
    }
    if (m_filteredPkt) {
        av_packet_free(&m_filteredPkt);
        m_filteredPkt = nullptr;
    }
    if (m_bsfCtx) {
        av_bsf_free(&m_bsfCtx);
        m_bsfCtx = nullptr;
    }
    if (m_fmtCtx) {
        avformat_close_input(&m_fmtCtx);
        m_fmtCtx = nullptr;
    }
    if (m_extradata) {
        av_free(m_extradata);
        m_extradata = nullptr;
    }
    m_extradataSize = 0;
    m_videoStreamIdx = -1;
    m_codecId = 0;
    m_width = 0;
    m_height = 0;
}

bool RtspReader::readPacket(uint8_t*& data, int& size, int64_t& pts, bool& isKeyFrame)
{
    if (!m_fmtCtx || !m_pkt) return false;

    while (true) {
        av_packet_unref(m_pkt);

        int ret = av_read_frame(m_fmtCtx, m_pkt);
        if (ret < 0) {
            return false;
        }

        if (m_pkt->stream_index != m_videoStreamIdx)
            continue;

        if (m_bsfCtx && m_filteredPkt) {
            av_packet_unref(m_filteredPkt);
            ret = av_bsf_send_packet(m_bsfCtx, m_pkt);
            if (ret < 0) return false;

            ret = av_bsf_receive_packet(m_bsfCtx, m_filteredPkt);
            if (ret < 0) return false;

            data = m_filteredPkt->data;
            size = m_filteredPkt->size;
            pts = m_filteredPkt->pts;
            isKeyFrame = (m_filteredPkt->flags & AV_PKT_FLAG_KEY) != 0;
        } else {
            data = m_pkt->data;
            size = m_pkt->size;
            pts = m_pkt->pts;
            isKeyFrame = (m_pkt->flags & AV_PKT_FLAG_KEY) != 0;
        }
        return true;
    }
}
