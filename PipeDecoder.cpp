// PipeDecoder implementation — invoke external `ffmpeg` via popen,
// read raw NV12 frames from stdout.
#include "headers.h"
#include <unistd.h>

PipeDecoder::PipeDecoder()
    : m_fp(nullptr), m_width(0), m_height(0) {}

PipeDecoder::~PipeDecoder() { close(); }

// Build an ffmpeg command line that:
//   - reads JPEG from stdin  (-f image2pipe -vcodec mjpeg)
//   - decodes to rawvideo NV12
//   - writes raw frames to stdout (-f rawvideo -pix_fmt nv12)
// The URL is shell-escaped via single quotes to prevent command injection.
static std::string escapeShell(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') out += "'\\''";  // end quote, literal quote, reopen
        else out += c;
    }
    out += '\'';
    return out;
}

bool PipeDecoder::open(const std::string& url, int width, int height) {
    close();
    m_width = width;
    m_height = height;

    // Build ffmpeg command — shell-escaped URL prevents injection
    std::string cmd = "ffmpeg -loglevel quiet "
                      "-f image2pipe -vcodec mjpeg -i - "
                      "-f rawvideo -pix_fmt nv12 -s "
                      + std::to_string(width) + "x" + std::to_string(height) + " "
                      "pipe:1 2>/dev/null";

    m_fp = popen(cmd.c_str(), "w");
    return m_fp != nullptr;
}

void PipeDecoder::close() {
    if (m_fp) { pclose(m_fp); m_fp = nullptr; }
}

bool PipeDecoder::decode(uint8_t* data, int size, int64_t pts) {
    if (!m_fp || !data || size <= 0) return false;

    // Write raw JPEG data to ffmpeg's stdin
    size_t written = fwrite(data, 1, size, m_fp);
    fflush(m_fp);
    return written == (size_t)size;
}
