// Globals.h — Глобальное состояние камер: данные и потокобезопасные аксессоры.
#pragma once

#include <cstdint>
#include <csignal>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>

// Информация о камере
struct CamInfo {
    std::string url;     // RTSP URL камеры
    int codec;           // ID кодека (H.264, H.265)
    int width;           // Разрешение по ширине
    int height;          // Разрешение по высоте
    double fps;          // Текущий FPS
    bool connected;      // Флаг подключения
};

// Глобальные переменные (определены в Globals.cpp / main.cpp)
extern std::vector<CamInfo> g_cams;                  // Все камеры
extern std::vector<std::atomic<bool>*> g_camRunning; // Флаги работы потоков
extern std::mutex g_camMtx;    // Защита g_cams / g_camRunning
extern std::mutex g_printMtx;  // Сериализация вывода в stdout
extern volatile std::sig_atomic_t g_running;         // Глобальный флаг работы (main.cpp)
extern std::atomic<bool> g_logDecodeSpeed;           // Логировать скорость декодирования
extern bool g_benchmarkMode;                         // Режим бенчмарка (main.cpp)
extern int g_winWidth;                               // Размер окна отображения (main.cpp)
extern int g_winHeight;

// Потокобезопасные операции над состоянием камер
bool isCamRunning(int camIdx);
void stopCamRunning(int camIdx);
void setCamConnected(int camIdx, bool connected, const std::string& url);
