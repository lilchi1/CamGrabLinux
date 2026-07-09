// RtspReader implementation — FFmpeg AVFormatContext + bitstream filter.
#include "headers.h"

RtspReader::RtspReader()
    : m_fmtCtx(nullptr), m_videoStreamIdx(-1), m_codecId(0),
      m_width(0), m_height(0), m_pkt(nullptr),
      m_filteredPkt(nullptr), m_bsfCtx(nullptr),
      m_extradata(nullptr), m_extradataSize(0) {}

RtspReader::~RtspReader() { close(); }

bool RtspReader::open(const std::string& url, int timeoutSec) {
    close();

    // Set network / RTSP transport to TCP (more reliable than UDP).
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    char tbuf[32];
    snprintf(tbuf, sizeof(tbuf), "%d", timeoutSec * 1000000);
    av_dict_set(&opts, "stimeout", tbuf, 0);
    av_dict_set(&opts, "fflags", "nobuffer", 0);
    av_dict_set(&opts, "flags", "low_delay", 0);
    av_dict_set(&opts, "max_delay", "0", 0);
    av_dict_set(&opts, "probesize", "5000000", 0);
    av_dict_set(&opts, "analyzeduration", "5000000", 0);

    m_fmtCtx = avformat_alloc_context();
    int ret = avformat_open_input(&m_fmtCtx, url.c_str(), nullptr, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        // Retry with UDP transport if TCP fails
        av_dict_set(&opts, "rtsp_transport", "udp", 0);
        ret = avformat_open_input(&m_fmtCtx, url.c_str(), nullptr, &opts);
        av_dict_free(&opts);
        if (ret < 0) { close(); return false; }
    }

    ret = avformat_find_stream_info(m_fmtCtx, nullptr);
    if (ret < 0) { close(); return false; }

    // Locate first video stream
    for (unsigned i = 0; i < m_fmtCtx->nb_streams; i++) {
        if (m_fmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            m_videoStreamIdx = i;
            break;
        }
    }
    if (m_videoStreamIdx < 0) { close(); return false; }

    auto* par = m_fmtCtx->streams[m_videoStreamIdx]->codecpar;
    m_codecId = par->codec_id;
    m_width  = par->width;
    m_height = par->height;

    // Save extradata (contains SPS/PPS for h.264/h.265)
    if (par->extradata && par->extradata_size > 0) {
        m_extradata = (uint8_t*)av_malloc(par->extradata_size);
        memcpy(m_extradata, par->extradata, par->extradata_size);
        m_extradataSize = par->extradata_size;
    }

    m_pkt = av_packet_alloc();
    m_filteredPkt = av_packet_alloc();

    // Set up bitstream filter (mp4toannexb)
    const char* bsfName = nullptr;
    if (m_codecId == AV_CODEC_ID_H264) bsfName = "h264_mp4toannexb";
    else if (m_codecId == AV_CODEC_ID_H265) bsfName = "hevc_mp4toannexb";

    if (bsfName) {
        const AVBitStreamFilter* bsf = av_bsf_get_by_name(bsfName);
        if (bsf) {
            av_bsf_alloc(bsf, &m_bsfCtx);
            avcodec_parameters_copy(m_bsfCtx->par_in, par);
            av_bsf_init(m_bsfCtx);
        }
    }

    return true;
}

void RtspReader::close() {
    if (m_bsfCtx) av_bsf_free(&m_bsfCtx);
    if (m_pkt) av_packet_free(&m_pkt);
    if (m_filteredPkt) av_packet_free(&m_filteredPkt);
    if (m_fmtCtx) avformat_close_input(&m_fmtCtx);
    if (m_extradata) { av_free(m_extradata); m_extradata = nullptr; }
    m_extradataSize = 0;
    m_videoStreamIdx = -1;
}

bool RtspReader::readPacket(uint8_t*& data, int& size,
                            int64_t& pts, bool& isKeyFrame) {
    while (true) {
        int ret = av_read_frame(m_fmtCtx, m_pkt);
        if (ret < 0) return false;
        if (m_pkt->stream_index != m_videoStreamIdx) {
            av_packet_unref(m_pkt);
            continue;
        }

        AVPacket* outPkt = m_pkt;
        // Apply bitstream filter if available
        if (m_bsfCtx) {
            if (av_bsf_send_packet(m_bsfCtx, m_pkt) < 0) {
                av_packet_unref(m_pkt);
                return false;
            }
            if (av_bsf_receive_packet(m_bsfCtx, m_filteredPkt) < 0) {
                av_packet_unref(m_pkt);
                return false;
            }
            outPkt = m_filteredPkt;
        }

        data = outPkt->data;
        size = outPkt->size;
        pts  = outPkt->pts;
        isKeyFrame = (outPkt->flags & AV_PKT_FLAG_KEY);
        return true;
    }
}
