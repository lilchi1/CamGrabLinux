// PtzControl.cpp — Управление поворотной камерой через Hikvision ISAPI.
// Непрерывное движение: PUT /ISAPI/PTZCtrl/channels/1/continuous.
// HTTP-запросы выполняются curl'ом через fork+execvp (без оболочки — нет
// shell-инъекций), с Digest-аутентификацией.
#include "PtzControl.h"
#include "headers.h"
#include <unistd.h>
#include <sys/wait.h>
#include <cctype>
#include <cstdio>
#include <cstdlib>

// Декодирование %XX в URL (для пароля из RTSP URL, например %21 → !)
PtzControl::PtzControl() {}

PtzControl::~PtzControl() { close(); }

std::string PtzControl::urlDecode(const std::string& s) {
    std::string out;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '%' && i + 2 < s.size() &&
            std::isxdigit((unsigned char)s[i + 1]) &&
            std::isxdigit((unsigned char)s[i + 2])) {
            int v = 0;
            sscanf(s.c_str() + i + 1, "%2x", &v);
            out += (char)v;
            i += 2;
        } else if (s[i] == '+') {
            out += ' ';
        } else {
            out += s[i];
        }
    }
    return out;
}

bool PtzControl::open(const std::string& rtspUrl, int isapiPort) {
    close();
    m_isapiPort = isapiPort;

    // Разбор RTSP URL: rtsp://user:pass@host:port/path
    std::string rest = rtspUrl;
    size_t scheme = rest.find("://");
    if (scheme == std::string::npos) return false;
    rest = rest.substr(scheme + 3);

    size_t at = rest.find('@');
    if (at != std::string::npos) {
        std::string cred = rest.substr(0, at);
        rest = rest.substr(at + 1);
        size_t colon = cred.find(':');
        if (colon != std::string::npos)
            m_userPass = urlDecode(cred.substr(0, colon)) + ":" +
                         urlDecode(cred.substr(colon + 1));
        else
            m_userPass = urlDecode(cred) + ":";
    }

    size_t end = rest.find_first_of(":/");
    m_host = (end == std::string::npos) ? rest : rest.substr(0, end);
    if (m_host.empty()) return false;

    // Проверка доступности ISAPI PTZ (канал 1)
    std::string resp;
    if (!httpRequest("GET", "/ISAPI/PTZCtrl/channels/1", "", resp))
        return false;

    m_open.store(true);
    m_closing.store(false);
    m_thread = std::thread(&PtzControl::workerLoop, this);
    return true;
}

void PtzControl::close() {
    if (m_thread.joinable()) {
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            m_closing.store(true);
        }
        m_cv.notify_all();
        m_thread.join();
    }
    if (m_open.load()) sendVelocity(0.0f, 0.0f);  // финальный стоп
    m_open.store(false);
}

void PtzControl::onKey(Dir dir, bool pressed) {
    if (!m_open.load()) return;
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_queue.push_back({dir, pressed});
    }
    m_cv.notify_one();
}

// Рабочий поток: обработка событий клавиш → вычисление скорости → HTTP.
// Если камера двигалась, а событий нет более 2 с — аварийный стоп.
void PtzControl::workerLoop() {
    using namespace std::chrono;
    auto lastActivity = steady_clock::now();
    while (true) {
        Event ev;
        {
            std::unique_lock<std::mutex> lock(m_mtx);
            m_cv.wait_for(lock, std::chrono::milliseconds(500), [&] {
                return m_closing.load() || !m_queue.empty();
            });
            if (m_closing.load()) break;
            if (m_queue.empty()) {
                if (m_hasMove.load() &&
                    duration_cast<milliseconds>(steady_clock::now() -
                                                lastActivity).count() > 2000) {
                    sendVelocity(0.0f, 0.0f);
                    lastActivity = steady_clock::now();
                }
                continue;
            }
            ev = m_queue.front();
            m_queue.pop_front();
            lastActivity = steady_clock::now();
        }

        m_held[(int)ev.dir] = ev.pressed;
        float pan = 0.0f, tilt = 0.0f;
        if (m_held[(int)Dir::Right]) pan = 0.5f;
        if (m_held[(int)Dir::Left])  pan = -0.5f;
        if (m_held[(int)Dir::Up])    tilt = 0.5f;
        if (m_held[(int)Dir::Down])  tilt = -0.5f;
        sendVelocity(pan, tilt);
    }
}

// Отправка команды непрерывного движения (Hikvision: pan>0 — вправо, tilt>0 — вверх)
bool PtzControl::sendVelocity(float pan, float tilt) {
    char p[16], t[16];
    snprintf(p, sizeof(p), "%.2f", pan);
    snprintf(t, sizeof(t), "%.2f", tilt);
    std::string body =
        "<PTZData xmlns=\"http://www.hikvision.com/ver10/XMLSchema\">"
        "<pan>" + std::string(p) + "</pan><tilt>" + std::string(t) +
        "</tilt><zoom>0</zoom></PTZData>";

    std::string resp;
    bool ok = httpRequest("PUT", "/ISAPI/PTZCtrl/channels/1/continuous",
                          body, resp);
    m_hasMove.store(pan != 0.0f || tilt != 0.0f);
    return ok;
}

// HTTP-запрос через curl (fork+execvp, без оболочки). Возвращает true при HTTP 2xx
// (curl с --fail возвращает ненулевой код при ответе >= 400).
bool PtzControl::httpRequest(const std::string& method, const std::string& path,
                             const std::string& body, std::string& resp) {
    (void)resp;
    std::string url = "http://" + m_host + ":" + std::to_string(m_isapiPort) + path;

    pid_t pid = fork();
    if (pid < 0) return false;
    if (pid == 0) {
        // Дочерний процесс: вызов curl, ответ — в файл /dev/null
        char tmax[] = "3";
        const char* args[24];
        int i = 0;
        args[i++] = "curl";
        args[i++] = "--silent";
        args[i++] = "--show-error";
        args[i++] = "--fail";
        args[i++] = "--digest";
        args[i++] = "--max-time";
        args[i++] = tmax;
        args[i++] = "-u";
        args[i++] = m_userPass.c_str();
        args[i++] = "-X";
        args[i++] = method.c_str();
        args[i++] = "-o";
        args[i++] = "/dev/null";
        if (!body.empty()) {
            args[i++] = "--data-binary";
            args[i++] = body.c_str();
        }
        args[i++] = url.c_str();
        args[i] = nullptr;
        execvp("curl", (char* const*)args);
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return false;
    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}
