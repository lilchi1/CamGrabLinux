// func.cpp — globals, utilities, camera thread, NAL parsing.
#include "headers.h"
#include "Display.h"
#include <unistd.h>

std::vector<CamInfo> g_cams;
std::vector<std::atomic<bool>*> g_camRunning;
std::mutex g_camMtx;
std::mutex g_printMtx;

// Atomic-wide running flag (declared in main.cpp)
extern volatile std::sig_atomic_t g_running;

// Thread-safe logging with timestamp
void logWrite(const std::string& level, const std::string& url,
              const std::string& msg) {
    std::lock_guard<std::mutex> lock(g_printMtx);
    time_t now = time(nullptr);
    struct tm* tm = localtime(&now);
    char tbuf[32];
    strftime(tbuf, sizeof(tbuf), "%H:%M:%S", tm);
    std::cout << "[" << tbuf << "][" << level << "][" << url << "] "
              << msg << std::endl;
}

// Print status of all cameras
void printAllStatus() {
    std::lock_guard<std::mutex> lock(g_printMtx);
    std::lock_guard<std::mutex> lock2(g_camMtx);
    std::cout << "\n=== Camera Status ===" << std::endl;
    for (int i = 0; i < (int)g_cams.size(); i++) {
        auto& c = g_cams[i];
        std::string codecStr;
        if (c.codec == AV_CODEC_ID_H264) codecStr = "H.264";
        else if (c.codec == AV_CODEC_ID_H265) codecStr = "H.265";
        else if (c.codec == AV_CODEC_ID_MJPEG) codecStr = "MJPEG";
        else codecStr = "Unknown";
        std::cout << "[" << i << "] " << c.url
                  << " | " << codecStr
                  << " | " << c.width << "x" << c.height
                  << " | " << (c.fps > 0 ? std::to_string(c.fps) : "N/A") << " fps"
                  << " | " << (c.connected ? "Connected" : "Disconnected")
                  << std::endl;
    }
    std::cout << "=====================\n" << std::endl;
}

void setCamConnected(int camIdx, bool connected, const std::string& url) {
    std::lock_guard<std::mutex> lock(g_camMtx);
    if (camIdx >= 0 && camIdx < (int)g_cams.size()) {
        g_cams[camIdx].connected = connected;
    }
}

// Main per-camera thread: RTSP → decode → display
void cameraThread(std::string url, int camIdx) {
    logWrite("INFO", url, "Camera thread started");

    RtspReader reader;
    if (!reader.open(url)) {
        logWrite("ERROR", url, "Failed to open RTSP stream");
        setCamConnected(camIdx, false, url);
        if (camIdx >= 0 && camIdx < (int)g_camRunning.size())
            *g_camRunning[camIdx] = false;
        return;
    }

    int codecId = reader.codecId();
    int w = reader.width();
    int h = reader.height();
    double fps = 0.0;

    logWrite("INFO", url, "Codec: " + std::to_string(codecId) +
                          " Resolution: " + std::to_string(w) + "x" +
                          std::to_string(h));

    {
        std::lock_guard<std::mutex> lock(g_camMtx);
        if (camIdx >= 0 && camIdx < (int)g_cams.size()) {
            g_cams[camIdx].codec = codecId;
            g_cams[camIdx].width = w;
            g_cams[camIdx].height = h;
        }
    }

    // Frame callback: create display on first frame, show subsequent frames
    DisplayWindow* display = nullptr;
    FrameCallback cb = [&](uint8_t* y, uint8_t* uv, int cw, int ch,
                           int sy, int suv, int64_t pts) {
        if (!g_running) return;
        if (!display) {
            display = new DisplayWindow();
            if (!display->open("Camera " + std::to_string(camIdx) + " - " + url,
                               cw, ch)) {
                delete display;
                display = nullptr;
                return;
            }
        }
        display->showFrame(y, uv, cw, ch, sy, suv);
    };

    // Select decoder based on codec type
    // H.264/H.265 → try NvV4l2Decoder (hardware), fallback to SwDecoder (software)
    // MJPEG → PipeDecoder
    std::unique_ptr<SwDecoder> swDec;
    std::unique_ptr<NvV4l2Decoder> nvDec;
    std::unique_ptr<PipeDecoder> pipeDec;

    if (codecId == AV_CODEC_ID_MJPEG) {
        pipeDec = std::make_unique<PipeDecoder>();
        if (pipeDec->open(url, w, h)) {
            pipeDec->setFrameCallback(cb);
            logWrite("INFO", url, "Using PipeDecoder (MJPEG)");
        } else {
            logWrite("ERROR", url, "Failed to open PipeDecoder");
            pipeDec.reset();
        }
    } else {
        // Try hardware V4L2 decoder first
        nvDec = std::make_unique<NvV4l2Decoder>();
        if (nvDec->open(codecId, w, h)) {
            nvDec->setFrameCallback(cb);
            logWrite("INFO", url, "Using NvV4l2Decoder (hardware)");
        } else {
            nvDec.reset();
            // Fallback to software FFmpeg decoder
            swDec = std::make_unique<SwDecoder>();
            if (swDec->open(codecId, reader.extradata(),
                            reader.extradataSize(), w, h)) {
                swDec->setFrameCallback(cb);
                logWrite("INFO", url, "Using SwDecoder (software fallback)");
            } else {
                logWrite("ERROR", url, "Failed to open any decoder");
                swDec.reset();
            }
        }
    }

    // Verify at least one decoder is active
    bool haveDecoder = (swDec && swDec->isOpen()) ||
                       (nvDec && nvDec->isOpen()) ||
                       (pipeDec && pipeDec->isOpen());
    if (!haveDecoder) {
        logWrite("ERROR", url, "No decoder available");
        setCamConnected(camIdx, false, url);
        if (camIdx >= 0 && camIdx < (int)g_camRunning.size())
            *g_camRunning[camIdx] = false;
        if (display) { delete display; display = nullptr; }
        reader.close();
        return;
    }

    setCamConnected(camIdx, true, url);
    logWrite("INFO", url, "Camera connected");

    uint64_t frameCount = 0;
    uint64_t lastFpsPrint = 0;
    auto lastFpsTime = std::chrono::steady_clock::now();
    uint64_t lastDecodeTime = 0;

    // Decode loop: read packets, feed decoder, track FPS
    while (g_running &&
           (camIdx < 0 || camIdx >= (int)g_camRunning.size() ||
            *g_camRunning[camIdx])) {
        uint8_t* pktData;
        int pktSize;
        int64_t pts;
        bool isKeyFrame;

        if (!reader.readPacket(pktData, pktSize, pts, isKeyFrame)) {
            // Read failure — may need reconnection
            logWrite("WARN", url, "Read error, attempting reconnect...");
            reader.close();
            std::this_thread::sleep_for(std::chrono::seconds(1));

            if (!reader.open(url, 10)) {
                logWrite("ERROR", url, "Reconnect failed");
                break; // will set disconnected, clean up, and exit thread
            }

            // Re-init decoder after reconnection
            if (codecId != AV_CODEC_ID_MJPEG) {
                if (nvDec && nvDec->isOpen()) {
                    nvDec->close();
                    if (!nvDec->open(codecId, reader.width(), reader.height())) {
                        nvDec.reset();
                    }
                }
                if (!nvDec && swDec) {
                    swDec->close();
                    if (!swDec->open(codecId, reader.extradata(),
                                     reader.extradataSize(),
                                     reader.width(), reader.height())) {
                        swDec.reset();
                    }
                }
            }
            continue;
        }

        // Feed packets to the active decoder
        bool decoded = false;
        if (nvDec && nvDec->isOpen()) {
            decoded = nvDec->decode(pktData, pktSize, pts);
        } else if (swDec && swDec->isOpen()) {
            decoded = swDec->decode(pktData, pktSize, pts, isKeyFrame);
        } else if (pipeDec && pipeDec->isOpen()) {
            decoded = pipeDec->decode(pktData, pktSize, pts);
        }

        if (decoded) {
            frameCount++;
            lastDecodeTime = frameCount;

            // FPS counter: log every 100 frames
            if (frameCount - lastFpsPrint >= 100) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - lastFpsTime).count();
                if (elapsed > 0) fps = 100.0 / elapsed;
                lastFpsPrint = frameCount;
                lastFpsTime = now;

                {
                    std::lock_guard<std::mutex> lock(g_camMtx);
                    if (camIdx >= 0 && camIdx < (int)g_cams.size())
                        g_cams[camIdx].fps = fps;
                }
                logWrite("INFO", url, "FPS: " + std::to_string(fps));
            }
        }

        // Check for stuck decoder — if too many frames fail, break
        if (frameCount - lastDecodeTime > 300) {
            logWrite("WARN", url, "Too many failed decode attempts, reconnecting...");
            reader.close();
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (reader.open(url, 10)) {
                lastDecodeTime = frameCount;
            } else {
                logWrite("ERROR", url, "Reconnect failed");
                break;
            }
        }
    }

    logWrite("INFO", url, "Camera thread shutting down");
    setCamConnected(camIdx, false, url);
    if (display) { delete display; display = nullptr; }
    reader.close();
    if (camIdx >= 0 && camIdx < (int)g_camRunning.size())
        *g_camRunning[camIdx] = false;
}

// Check if data starts with a valid NAL start code (0x00000001 or 0x000001).
bool isValidNalUnit(const uint8_t* data, size_t size) {
    if (!data || size < 4) return false;
    if (data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1) return true;
    if (size >= 3 && data[0] == 0 && data[1] == 0 && data[2] == 1) return true;
    return false;
}

// Scan buffer for NAL unit boundaries. Returns pairs of (pointer, size).
int findNalUnits(const uint8_t* data, size_t size,
                 std::vector<std::pair<const uint8_t*, int>>& nals) {
    nals.clear();
    if (!data || size < 4) return 0;

    size_t start = 0;
    bool found = false;

    for (size_t i = 0; i + 3 < size; i++) {
        if ((data[i] == 0 && data[i+1] == 0 && data[i+2] == 0 && data[i+3] == 1) ||
            (data[i] == 0 && data[i+1] == 0 && data[i+2] == 1)) {
            int nalSize = (data[i+2] == 1) ? 3 : 4;
            if (found) {
                nals.push_back({data + start, (int)(i - start)});
            }
            start = i;
            found = true;
            // Skip past start code to avoid re-matching
            i += (nalSize - 1);
        }
    }

    if (found) {
        nals.push_back({data + start, (int)(size - start)});
    }

    return (int)nals.size();
}
