// headers.h — Главный заголовочный файл. Стандартные включения и модули проекта.
#pragma once

// Стандартные C++ заголовки
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
#include <fstream>

#include "FrameCallback.h"

// FFmpeg (внешняя linkage для C-API)
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

// Модули проекта
#include "Globals.h"
#include "Log.h"
#include "CameraThread.h"
#include "GstDecoder.h"
#include "RtspReader.h"

// Замер скорости декодирования кадра: время между отправкой пакета в декодер
// и получением декодированного кадра (мс). Определена в main.cpp.
double measureDecodeSpeed(const std::chrono::steady_clock::time_point& pushedAt,
                          const std::chrono::steady_clock::time_point& arrivedAt);
