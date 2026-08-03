// Globals.cpp — Определение глобальных данных и потокобезопасных аксессоров.
#include "headers.h"

std::vector<CamInfo> g_cams;                  // Все камеры
std::vector<std::atomic<bool>*> g_camRunning; // Флаги работы потоков
std::mutex g_camMtx;    // Защита g_cams / g_camRunning
std::mutex g_printMtx;  // Сериализация вывода в stdout
std::atomic<bool> g_logDecodeSpeed{false};    // Логировать скорость декодирования

// Потокобезопасная проверка: работает ли поток камеры
bool isCamRunning(int camIdx) {
    std::lock_guard<std::mutex> lock(g_camMtx);
    if (camIdx < 0 || (size_t)camIdx >= g_camRunning.size()) return false;
    return *g_camRunning[(size_t)camIdx];
}

// Потокобезопасная остановка потока камеры
void stopCamRunning(int camIdx) {
    std::lock_guard<std::mutex> lock(g_camMtx);
    if (camIdx >= 0 && (size_t)camIdx < g_camRunning.size())
        *g_camRunning[(size_t)camIdx] = false;
}

// Установка статуса подключения камеры
void setCamConnected(int camIdx, bool connected, const std::string& url) {
    (void)url;
    std::lock_guard<std::mutex> lock(g_camMtx);
    if (camIdx >= 0 && static_cast<size_t>(camIdx) < g_cams.size()) {
        g_cams[(size_t)camIdx].connected = connected;
    }
}
