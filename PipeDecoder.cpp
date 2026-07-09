#include "PipeDecoder.h"
#include <cstdio>
#include <cstring>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

PipeDecoder::PipeDecoder()
    : m_pipe(nullptr)
    , m_width(0)
    , m_height(0)
    , m_swsCtx(nullptr)
    , m_nv12Buf(nullptr)
    , m_nv12Frame(nullptr)
    , m_nv12Width(0)
    , m_nv12Height(0)
{
}

PipeDecoder::~PipeDecoder()
{
    close();
}

bool PipeDecoder::init(const std::string& url, FrameCallback cb)
{
    m_callback = cb;

    std::string cmd = "ffmpeg -rtsp_transport tcp -timeout 10000000 "
                      "-i \"" + url + "\" "
                      "-f image2pipe -vcodec mjpeg -q 2 -an - 2>/dev/null";

    m_pipe = popen(cmd.c_str(), "r");
    if (!m_pipe) {
        fprintf(stderr, "PipeDecoder: popen failed for: %s\n", cmd.c_str());
        return false;
    }

    setvbuf(m_pipe, nullptr, _IONBF, 0);

    return true;
}

void PipeDecoder::close()
{
    if (m_pipe) {
        pclose(m_pipe);
        m_pipe = nullptr;
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

    m_buf.clear();
    m_width = 0;
    m_height = 0;
    m_nv12Width = 0;
    m_nv12Height = 0;
}

bool PipeDecoder::ensureNv12Converter(int width, int height)
{
    if (m_swsCtx && m_nv12Width == width && m_nv12Height == height)
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

    m_swsCtx = sws_getContext(width, height, AV_PIX_FMT_BGR24,
                              width, height, AV_PIX_FMT_NV12,
                              SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!m_swsCtx) {
        av_freep(&m_nv12Buf);
        av_frame_free(&m_nv12Frame);
        return false;
    }

    m_nv12Width = width;
    m_nv12Height = height;
    return true;
}

bool PipeDecoder::findJpegFrame(std::vector<uint8_t>& jpeg)
{
    size_t soi = SIZE_MAX;
    for (size_t i = 0; i + 1 < m_buf.size(); i++) {
        if (m_buf[i] == 0xFF && m_buf[i + 1] == 0xD8) {
            soi = i;
            break;
        }
    }
    if (soi == SIZE_MAX) return false;

    size_t eoi = SIZE_MAX;
    for (size_t i = soi + 2; i + 1 < m_buf.size(); i++) {
        if (m_buf[i] == 0xFF && m_buf[i + 1] == 0xD9) {
            eoi = i + 2;
            break;
        }
    }
    if (eoi == SIZE_MAX) return false;

    jpeg.assign(m_buf.begin() + soi, m_buf.begin() + eoi);
    m_buf.erase(m_buf.begin(), m_buf.begin() + eoi);
    return true;
}

int PipeDecoder::readFrame(int64_t pts)
{
    if (!m_pipe) return -1;

    uint8_t tmp[65536];
    size_t n = fread(tmp, 1, sizeof(tmp), m_pipe);
    if (n <= 0) {
        if (feof(m_pipe) || ferror(m_pipe)) return -1;
        return -1;
    }

    m_buf.insert(m_buf.end(), tmp, tmp + n);

    if (m_buf.size() > 4 * 1024 * 1024)
        m_buf.erase(m_buf.begin(), m_buf.end() - 2 * 1024 * 1024);

    std::vector<uint8_t> jpeg;
    if (!findJpegFrame(jpeg)) return 0;

    cv::Mat raw(1, (int)jpeg.size(), CV_8UC1, jpeg.data());
    cv::Mat frame = cv::imdecode(raw, cv::IMREAD_COLOR);
    if (frame.empty()) {
        fprintf(stderr, "PipeDecoder: imdecode failed (%zu bytes)\n", jpeg.size());
        return 0;
    }

    int w = frame.cols;
    int h = frame.rows;
    m_width = w;
    m_height = h;

    if (!m_callback) return 1;

    if (!ensureNv12Converter(w, h)) return 0;

    uint8_t* bgrData[1] = { frame.data };
    int bgrStride[1] = { (int)frame.step };

    sws_scale(m_swsCtx, (const uint8_t* const*)bgrData, bgrStride,
              0, h, m_nv12Frame->data, m_nv12Frame->linesize);

    m_callback(m_nv12Frame->data[0], m_nv12Frame->data[1],
               w, h,
               m_nv12Frame->linesize[0], m_nv12Frame->linesize[1],
               pts);

    return 1;
}
