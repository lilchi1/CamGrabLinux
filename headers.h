// headers.h — Главный заголовочный файл. Все включения, типы и глобальные объявления.
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

// FFmpeg (внешняя linkage для C--API)
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

// Проектные модули
#include "GstDecoder.h"
#include "RtspReader.h"
#include "PtzControl.h"

// Коды клавиш X11 (ESC, стрелки)
#include <X11/keysym.h>

// ─── Информация о камере ─────────────────────────────────────────────────────

struct CamInfo {
    std::string url;     // RTSP URL камеры
    int codec;           // ID кодека (H.264, H.265)
    int width;           // Разрешение по ширине
    int height;          // Разрешение по высоте
    double fps;          // Текущий FPS
    bool connected;      // Флаг подключения
};

// ─── Глобальные переменные (определены в func.cpp) ───────────────────────────

extern std::vector<CamInfo> g_cams;                  // Все камеры
extern std::vector<std::atomic<bool>*> g_camRunning; // Флаги работы потоков
extern std::mutex g_camMtx;    // Защита g_cams / g_camRunning
extern std::mutex g_printMtx;  // Сериализация вывода в stdout
extern volatile std::sig_atomic_t g_running;         // Глобальный флаг работы
extern std::atomic<bool> g_logDecodeSpeed;           // Логировать скорость декодирования

// ─── Прототипы функций ───────────────────────────────────────────────────────

// Замер скорости декодирования кадра: время между отправкой пакета в декодер
// и получением декодированного кадра (мс). Определена в main.cpp.
double measureDecodeSpeed(const std::chrono::steady_clock::time_point& pushedAt,
                          const std::chrono::steady_clock::time_point& arrivedAt);

// Потокобезопасный вывод лога с меткой времени
void logWrite(const std::string& level, const std::string& url, const std::string& msg);

// Печать статуса всех камер
void printAllStatus();

// Установка статуса подключения камеры
void setCamConnected(int camIdx, bool connected, const std::string& url);

// Главный поток камеры: RTSP → декодирование → отображение
void cameraThread(std::string url, int camIdx);
