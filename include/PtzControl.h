// PtzControl.h — Управление поворотной камерой (PTZ) через Hikvision ISAPI.
// Непрерывное движение: PUT /ISAPI/PTZCtrl/channels/1/continuous.
// HTTP выполняется curl'ом (fork+execvp) с Digest-аутентификацией.
#pragma once

#include <string>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <map>

class PtzControl {
public:
    enum class Dir { Left, Right, Up, Down };

    PtzControl();
    ~PtzControl();

    // Парсит RTSP URL (учётка/хост), проверяет доступность ISAPI PTZ и
    // запускает поток команд. isapiPort — HTTP-порт камеры (обычно 80).
    bool open(const std::string& rtspUrl, int isapiPort = 80);

    bool isOpen() const { return m_open.load(); }

    // Нажатие/отпускание стрелки: направление и состояние клавиши
    void onKey(Dir dir, bool pressed);

    // Остановка потока и отправка команды «стоп»
    void close();

private:
    struct Event { Dir dir; bool pressed; };

    void workerLoop();
    bool sendVelocity(float pan, float tilt);
    bool httpRequest(const std::string& method, const std::string& path,
                     const std::string& body, std::string& resp);
    static std::string urlDecode(const std::string& s);

    std::string m_host;          // Хост камеры
    int m_isapiPort = 80;        // HTTP-порт ISAPI
    std::string m_userPass;      // "user:pass" для Digest-аутентификации
    std::atomic<bool> m_open{false};
    std::atomic<bool> m_closing{false};
    std::atomic<bool> m_hasMove{false};  // Камера в движении

    std::thread m_thread;
    std::mutex m_mtx;
    std::condition_variable m_cv;
    std::deque<Event> m_queue;   // Очередь событий клавиш
    std::map<int, bool> m_held;  // Направление → удерживается
};
