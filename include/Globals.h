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
extern bool g_camResRequested;                       // true, если -w/-H заданы явно → запросить у камеры
extern std::string g_displayMode;                    // Режим отображения: cuda | xvimagesink (main.cpp)
extern std::string g_rtspTransport;                  // RTSP транспорт: tcp | udp | auto (main.cpp)

// Детекция YOLO (main.cpp)
extern std::string g_modelPath;                      // путь к TensorRT .engine (пусто = без детекции)
extern std::string g_labelsPath;                     // файл имён классов (по одному в строке)
extern float g_confThresh;                           // порог уверенности
extern float g_nmsThresh;                            // порог NMS
extern std::vector<std::string> g_classNames;        // загруженные имена классов

// Детекция YOLOv2 (anchor-based, main.cpp)
extern bool g_yolov2Mode;                            // включить YOLOv2-декод (вместо YOLOv8/11/12)
extern int g_yolov2Grid;                             // сетка выхода (19 для 608x608)
extern std::vector<float> g_yolov2Anchors;           // якоря: пары (w,h) в пикселях входа модели
extern int g_modelInSize;                            // размер входа модели (0 = авто)

// Потокобезопасные операции над состоянием камер
bool isCamRunning(int camIdx);
void stopCamRunning(int camIdx);
void setCamConnected(int camIdx, bool connected, const std::string& url);
