// Log.cpp — Потокобезопасный вывод логов и статуса камер.
#include "headers.h"

// Получение текущего времени в формате "YYYY-MM-DD HH:MM:SS"
std::string getCurrentTimestamp() {
    time_t now = time(nullptr);
    struct tm* tm = localtime(&now);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
    return std::string(buf);
}

// Потокобезопасный вывод лога с временем [HH:MM:SS][УРОВЕНЬ][URL] сообщение
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

// Вывод статуса всех камер в консоль
void printAllStatus() {
    std::lock_guard<std::mutex> lock(g_camMtx);
    std::lock_guard<std::mutex> lock2(g_printMtx);
    std::cout << "\n=== Статус камер ===" << std::endl;
    for (size_t i = 0; i < g_cams.size(); i++) {
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
                  << " | " << (c.connected ? "Подключена" : "Отключена")
                  << std::endl;
    }
    std::cout << "=====================\n" << std::endl;
}
