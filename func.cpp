#include "headers.h"
#include "Display.h"

#include <cstdlib>
#include <linux/videodev2.h>

std::vector<CamInfo> g_cams;
std::vector<std::atomic<bool>*> g_camRunning;
std::mutex g_camMtx;

int avCodecToV4l2(int avCodecId) {
    if (avCodecId == AV_CODEC_ID_H264) return V4L2_PIX_FMT_H264;
    if (avCodecId == AV_CODEC_ID_HEVC) return V4L2_PIX_FMT_H265;
    return 0;
}

void logWrite(const std::string& level, const std::string& url, const std::string& msg) {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    std::cout << "[" << level << "][" << buf << "][" << url << "] " << msg << std::endl;
}

void printAllStatus() {
    std::lock_guard<std::mutex> lock(g_camMtx);
    std::cout << "\033[2J\033[1;1H";
    std::cout << "===========================================" << std::endl;
    std::cout << "         CAMERA STATUSES" << std::endl;
    std::cout << "===========================================" << std::endl;
    for (size_t i = 0; i < g_cams.size(); i++) {
        auto& c = g_cams[i];
        bool alive = i < g_camRunning.size() && g_camRunning[i] && g_camRunning[i]->load();
        std::cout << "  Camera:  " << c.url
             << "  [" << (alive ? "ACTIVE" : "CLOSED") << "]"
             << "  [" << (c.connected ? "CONNECTED" : "DISCONNECTED") << "]" << std::endl;
        std::cout << "  Codec:   " << (c.codec == AV_CODEC_ID_H264 ? "H.264" :
                                  c.codec == AV_CODEC_ID_HEVC ? "H.265" : "Unknown") << std::endl;
        std::cout << "  Res:     " << c.width << "x" << c.height << std::endl;
        std::cout << "  FPS:     " << (int)c.fps << std::endl;
        std::cout << "-------------------------------------------" << std::endl;
    }
    std::cout << "Enter new URL to add camera | 'exit' to quit" << std::endl;
    std::cout << "===========================================" << std::endl;
}

void setCamConnected(int camIdx, bool connected, const std::string& url) {
    std::lock_guard<std::mutex> lock(g_camMtx);
    if (g_cams[camIdx].connected == connected) return;
    g_cams[camIdx].connected = connected;
    logWrite(connected ? "INFO" : "WARN", url,
             connected ? "Connected" : "Disconnected");
}

static void scanNalUnits(const uint8_t* data, int size,
                          std::vector<std::pair<const uint8_t*, int>>& sps,
                          std::vector<std::pair<const uint8_t*, int>>& pps)
{
    size_t i = 0;
    while (i < (size_t)size) {
        int startCode = 0;
        if (i + 4 <= (size_t)size && data[i] == 0 && data[i+1] == 0 && data[i+2] == 0 && data[i+3] == 1)
            startCode = 4;
        else if (i + 3 <= (size_t)size && data[i] == 0 && data[i+1] == 0 && data[i+2] == 1)
            startCode = 3;
        else {
            i++;
            continue;
        }
        if (i + startCode >= (size_t)size) break;
        int nalType = data[i + startCode] & 0x1F;
        size_t nalStart = i;
        i += startCode + 1;
        while (i < (size_t)size) {
            if (i + 3 < (size_t)size && data[i] == 0 && data[i+1] == 0 && data[i+2] == 1) break;
            if (i + 4 < (size_t)size && data[i] == 0 && data[i+1] == 0 && data[i+2] == 0 && data[i+3] == 1) break;
            i++;
        }
        int nalSize = (int)(i - nalStart);
        if (nalType == 7) {
            sps.push_back({data + nalStart, nalSize});
        } else if (nalType == 8) {
            pps.push_back({data + nalStart, nalSize});
        }
    }
}

static std::vector<uint8_t> buildSpsPpsBuffer(
    const std::vector<std::pair<const uint8_t*, int>>& sps,
    const std::vector<std::pair<const uint8_t*, int>>& pps)
{
    std::vector<uint8_t> buf;
    for (auto& n : sps) {
        buf.insert(buf.end(), n.first, n.first + n.second);
    }
    for (auto& n : pps) {
        buf.insert(buf.end(), n.first, n.first + n.second);
    }
    return buf;
}

bool isValidNalUnit(const uint8_t* data, size_t size)
{
    if (!data || size < 4) return false;
    if (size > 4 * 1024 * 1024) return false;
    int startCode = 0;
    if (data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 1)
        startCode = 4;
    else if (data[0] == 0 && data[1] == 0 && data[2] == 1)
        startCode = 3;
    else
        return false;
    if ((size_t)startCode >= size) return false;
    int nalType = data[startCode] & 0x1F;
    if (nalType > 23) return false;
    return true;
}

int findNalUnits(const uint8_t* data, size_t size,
                 std::vector<std::pair<const uint8_t*, int>>& nals)
{
    int count = 0;
    size_t i = 0;
    while (i < size) {
        int startCode = 0;
        if (i + 4 <= size && data[i] == 0 && data[i+1] == 0 && data[i+2] == 0 && data[i+3] == 1)
            startCode = 4;
        else if (i + 3 <= size && data[i] == 0 && data[i+1] == 0 && data[i+2] == 1)
            startCode = 3;
        else { i++; continue; }
        size_t nalStart = i;
        i += startCode;
        while (i < size) {
            if (i + 3 < size && data[i] == 0 && data[i+1] == 0 && data[i+2] == 1) break;
            if (i + 4 < size && data[i] == 0 && data[i+1] == 0 && data[i+2] == 0 && data[i+3] == 1) break;
            i++;
        }
        nals.push_back({data + nalStart, (int)(i - nalStart)});
        count++;
    }
    return count;
}

void cameraThread(std::string url, int camIdx) {
    auto& running = *g_camRunning[camIdx];
    running = true;

    bool useNvdec = true;
    bool usePipeDecoder = false;
    const char* nvdecEnv = getenv("USE_NVDEC");
    if (nvdecEnv && nvdecEnv[0] == '0' && nvdecEnv[1] == '\0')
        useNvdec = false;
    const char* pipeEnv = getenv("PIPE_DECODER");
    if (pipeEnv && pipeEnv[0] == '1' && pipeEnv[1] == '\0')
        usePipeDecoder = true;
    int consecutiveFail = 0;
    std::unique_ptr<VideoDisplay> display;

    while (running && g_running) {
        if (usePipeDecoder) {
            PipeDecoder pipeDecoder;
            if (!pipeDecoder.init(url, nullptr)) {
                logWrite("ERROR", url, "PipeDecoder: popen failed. Retry in 5s...");
                for (int i = 0; i < 5 && running && g_running; i++)
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }

            {
                std::lock_guard<std::mutex> lock(g_camMtx);
                g_cams[camIdx].codec = AV_CODEC_ID_MJPEG;
                g_cams[camIdx].width = 0;
                g_cams[camIdx].height = 0;
            }

            auto pipeFrameCb = [&display, camIdx, url](uint8_t* y, uint8_t* uv,
                                                       int w, int h,
                                                       int sy, int suv,
                                                       int64_t pts) {
                if (!display) {
                    display.reset(new VideoDisplay("Camera " + std::to_string(camIdx) + ": " + url, w, h));
                    if (display->init()) {
                        std::lock_guard<std::mutex> lock(g_camMtx);
                        g_cams[camIdx].width = w;
                        g_cams[camIdx].height = h;
                    } else {
                        logWrite("ERROR", url, "Failed to create display window");
                        display.reset();
                    }
                }
                if (display && !display->shouldQuit())
                    display->showFrame(y, uv, w, h, sy, suv);
            };
            pipeDecoder.setFrameCallback(pipeFrameCb);

            logWrite("INFO", url, "PipeDecoder started (ffmpeg MJPEG)");
            setCamConnected(camIdx, true, url);

            int64_t pts = 0;
            uint64_t frameCount = 0;
            auto lastFpsTime = std::chrono::steady_clock::now();

            while (running && g_running) {
                if (display && display->shouldQuit()) {
                    logWrite("INFO", url, "Window closed");
                    running = false;
                    break;
                }

                int ret = pipeDecoder.readFrame(++pts);
                if (ret < 0) {
                    logWrite("WARN", url, "PipeDecoder read failed, reconnecting...");
                    break;
                }
                if (ret == 0) continue;

                frameCount++;
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration<double>(now - lastFpsTime).count();
                if (elapsed >= 1.0) {
                    {
                        std::lock_guard<std::mutex> lock(g_camMtx);
                        g_cams[camIdx].fps = frameCount / elapsed;
                        g_cams[camIdx].width = pipeDecoder.width();
                        g_cams[camIdx].height = pipeDecoder.height();
                    }
                    frameCount = 0;
                    lastFpsTime = now;
                    printAllStatus();
                }
            }

            logWrite("INFO", url, "PipeDecoder stream ended");
            pipeDecoder.close();
            setCamConnected(camIdx, false, url);

            if (!g_running || (display && display->shouldQuit())) break;

            logWrite("INFO", url, "Reconnecting in 3s...");
            for (int i = 0; i < 3 && running && g_running; i++)
                std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        RtspReader reader;
        if (!reader.open(url)) {
            logWrite("ERROR", url, "Failed to open RTSP. Retry in 5s...");
            for (int i = 0; i < 5 && running && g_running; i++)
                std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        SwDecoder swDecoder;
        NvV4l2Decoder nvDecoder;
        bool decoderReady = false;

        if (useNvdec) {
            int v4l2Codec = avCodecToV4l2(reader.codecId());
            if (v4l2Codec) {
                if (reader.extradata() && reader.extradataSize() > 0)
                    nvDecoder.setExtradata(reader.extradata(), reader.extradataSize());
                decoderReady = nvDecoder.init(v4l2Codec, nullptr, 0, 0);
            }
            if (!decoderReady) {
                logWrite("WARN", url, "NVDEC init failed, falling back to software decoder");
                useNvdec = false;
            }
        }
        if (!useNvdec) {
            if (reader.extradata() && reader.extradataSize() > 0)
                swDecoder.setExtradata(reader.extradata(), reader.extradataSize());
            decoderReady = swDecoder.init(reader.codecId(), nullptr, 0, 0);
        }
        if (!decoderReady) {
            logWrite("ERROR", url, "Failed to init decoder. Retry in 5s...");
            reader.close();
            for (int i = 0; i < 5 && running && g_running; i++)
                std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(g_camMtx);
            g_cams[camIdx].codec = reader.codecId();
            g_cams[camIdx].width = 0;
            g_cams[camIdx].height = 0;
        }

        auto frameCb = [&display, camIdx, url](uint8_t* y, uint8_t* uv,
                                                int w, int h,
                                                int sy, int suv,
                                                int64_t pts) {
            if (!display) {
                display.reset(new VideoDisplay("Camera " + std::to_string(camIdx) + ": " + url, w, h));
                if (display->init()) {
                    {
                        std::lock_guard<std::mutex> lock(g_camMtx);
                        g_cams[camIdx].width = w;
                        g_cams[camIdx].height = h;
                    }
                } else {
                    logWrite("ERROR", url, "Failed to create display window");
                    display.reset();
                }
            }
            if (display && !display->shouldQuit())
                display->showFrame(y, uv, w, h, sy, suv);
        };

        if (useNvdec)
            nvDecoder.setFrameCallback(frameCb);
        else
            swDecoder.setFrameCallback(frameCb);

        logWrite("INFO", url, useNvdec ? "NVDEC decoder started" : "Software decoder started");
        setCamConnected(camIdx, true, url);

        uint8_t* pktData = nullptr;
        int pktSize = 0;
        int64_t pts = 0;
        bool isKeyFrame = false;
        uint64_t frameCount = 0;
        auto lastFpsTime = std::chrono::steady_clock::now();

        std::vector<std::pair<const uint8_t*, int>> cachedSps, cachedPps;
        bool spsPpsReceived = (reader.extradata() && reader.extradataSize() > 0);
        std::vector<uint8_t> combinedBuf;
        int droppedFrames = 0;
        int maxDroppedFrames = 30;
        auto lastDecodeTime = std::chrono::steady_clock::now();

        while (running && g_running) {
            if (display && display->shouldQuit()) {
                logWrite("INFO", url, "Window closed");
                running = false;
                break;
            }

            bool ok = reader.readPacket(pktData, pktSize, pts, isKeyFrame);
            if (!ok) {
                logWrite("WARN", url, "RTSP read failed, reconnecting...");
                break;
            }

            if (!isValidNalUnit(pktData, pktSize)) {
                if (++consecutiveFail > 50) {
                    logWrite("WARN", url, "Too many invalid packets, reconnecting...");
                    break;
                }
                continue;
            }

            scanNalUnits(pktData, pktSize, cachedSps, cachedPps);
            if (!cachedSps.empty() && !cachedPps.empty())
                spsPpsReceived = true;

            if (!spsPpsReceived) {
                consecutiveFail = 0;
                continue;
            }

            std::vector<uint8_t> spsPpsBuf = buildSpsPpsBuffer(cachedSps, cachedPps);
            combinedBuf.clear();
            combinedBuf.insert(combinedBuf.end(), spsPpsBuf.begin(), spsPpsBuf.end());
            combinedBuf.insert(combinedBuf.end(), pktData, pktData + pktSize);
            const uint8_t* decodeData = combinedBuf.data();
            int decodeSize = (int)combinedBuf.size();

            auto now = std::chrono::steady_clock::now();
            double decodeLatency = std::chrono::duration<double>(now - lastDecodeTime).count();
            if (decodeLatency > 0.5 && !isKeyFrame && droppedFrames < maxDroppedFrames) {
                droppedFrames++;
                continue;
            }
            if (isKeyFrame) droppedFrames = 0;

            lastDecodeTime = std::chrono::steady_clock::now();

            ok = useNvdec ? nvDecoder.decode(decodeData, decodeSize, pts, isKeyFrame)
                          : swDecoder.decode(decodeData, decodeSize, pts, isKeyFrame);
            if (ok) {
                consecutiveFail = 0;
                frameCount++;

                now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration<double>(now - lastFpsTime).count();
                if (elapsed >= 1.0) {
                    {
                        std::lock_guard<std::mutex> lock(g_camMtx);
                        g_cams[camIdx].fps = frameCount / elapsed;
                        int w = useNvdec ? nvDecoder.width() : swDecoder.width();
                        int h = useNvdec ? nvDecoder.height() : swDecoder.height();
                        g_cams[camIdx].width = w;
                        g_cams[camIdx].height = h;
                    }
                    frameCount = 0;
                    lastFpsTime = now;
                    printAllStatus();
                }
            } else {
                if (++consecutiveFail > 50) {
                    logWrite("WARN", url, "Too many decode failures, reconnecting...");
                    break;
                }
            }
        }

        logWrite("INFO", url, "Stream ended, flushing decoder...");
        if (useNvdec) {
            nvDecoder.flush();
            nvDecoder.close();
        } else {
            swDecoder.flush();
            swDecoder.close();
        }
        reader.close();
        setCamConnected(camIdx, false, url);

        if (!g_running || (display && display->shouldQuit())) break;

        logWrite("INFO", url, "Reconnecting in 3s...");
        for (int i = 0; i < 3 && running && g_running; i++)
            std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    display.reset();
    running = false;
    logWrite("INFO", url, "Thread finished");
}
