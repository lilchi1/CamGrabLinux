// Master header — includes, types, and global declarations.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cctype>
#include <csignal>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <memory>

#include "FrameCallback.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

#include "SwDecoder.h"
#include "NvV4l2Decoder.h"
#include "PipeDecoder.h"
#include "RtspReader.h"

// Per-camera runtime info.
struct CamInfo {
    std::string url;
    int codec;
    int width;
    int height;
    double fps;
    bool connected;
};

extern std::vector<CamInfo> g_cams;
extern std::vector<std::atomic<bool>*> g_camRunning;
extern std::mutex g_camMtx;    // protects g_cams / g_camRunning
extern std::mutex g_printMtx;  // serialises stdout output
extern volatile std::sig_atomic_t g_running;

int avCodecToV4l2(int avCodecId);
void logWrite(const std::string& level, const std::string& url, const std::string& msg);
void printAllStatus();
void setCamConnected(int camIdx, bool connected, const std::string& url);
void cameraThread(std::string url, int camIdx);
bool isValidNalUnit(const uint8_t* data, size_t size);
int findNalUnits(const uint8_t* data, size_t size,
                 std::vector<std::pair<const uint8_t*, int>>& nals);
