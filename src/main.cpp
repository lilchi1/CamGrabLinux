// main.cpp — Точка входа. Создаёт поток на каждую камеру, управляет динамическим добавлением.
#include "headers.h"
#include <poll.h>
#include <unistd.h>
#include <getopt.h>

// Глобальный флаг работы программы (атомарный для безопасного доступа из потоков)
volatile std::sig_atomic_t g_running = 1;

// Флаги режимов работы
bool g_benchmarkMode = false;      // Режим бенчмарка (без отображения)

// Обработчик сигналов SIGINT/SIGTERM — корректное завершение
static void signalHandler(int) {
    g_running = 0;
}

// Замер скорости декодирования кадра: время между отправкой пакета в декодер
// и получением декодированного кадра из appsink (мс).
double measureDecodeSpeed(const std::chrono::steady_clock::time_point& pushedAt,
                          const std::chrono::steady_clock::time_point& arrivedAt) {
    return std::chrono::duration<double, std::milli>(arrivedAt - pushedAt).count();
}

// Чтение строки из stdin без блокировки (poll + read). Возвращает true, если
// строка получена; уважает g_running (ESC/SIGINT прерывают ожидание).
static std::string g_stdinBuf;
static bool readLineAsync(std::string& out, int timeoutMs) {
    while (g_running) {
        size_t nl = g_stdinBuf.find('\n');
        if (nl != std::string::npos) {
            out = g_stdinBuf.substr(0, nl);
            g_stdinBuf.erase(0, nl + 1);
            return true;
        }
        struct pollfd pfd = { STDIN_FILENO, POLLIN, 0 };
        int r = poll(&pfd, 1, timeoutMs);
        if (r <= 0) return false;  // таймаут/ошибка — ввода нет
        char buf[512];
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n <= 0) return false;  // EOF
        g_stdinBuf.append(buf, (size_t)n);
    }
    return false;
}

// Печать справки
static void printUsage(const char* progname) {
    std::cout << "Использование: " << progname << " [опции]" << std::endl;
    std::cout << "Опции:" << std::endl;
    std::cout << "  -b, --benchmark    Режим бенчмарка (без отображения окна)" << std::endl;
    std::cout << "  -h, --help         Показать эту справку" << std::endl;
    std::cout << std::endl;
    std::cout << "Примеры:" << std::endl;
    std::cout << "  " << progname << "                      # Обычный режим с отображением" << std::endl;
    std::cout << "  " << progname << " -b                  # Режим бенчмарка (без окна)" << std::endl;
    std::cout << "  " << progname << " --benchmark         # Бенчмарк с логированием" << std::endl;
}

int main(int argc, char* argv[]) {
    // Регистрация обработчиков сигналов
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // ─── Разбор аргументов командной строки ──────────────────────────────────
    static struct option long_options[] = {
        {"benchmark", no_argument, 0, 'b'},
        {"help",      no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "bh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'b':
                g_benchmarkMode = true;
                break;
            case 'h':
                printUsage(argv[0]);
                return 0;
            default:
                printUsage(argv[0]);
                return 1;
        }
    }

    std::cout << "===========================================" << std::endl;
    std::cout << "     Jetson RTSP Decoder" << std::endl;
    std::cout << "     NVDEC + CUDA + X11" << std::endl;
    std::cout << "===========================================" << std::endl;

    if (g_benchmarkMode) {
        std::cout << "🔬 РЕЖИМ БЕНЧМАРКА (отображение ОТКЛЮЧЕНО)" << std::endl;
    } else {
        std::cout << "🖥️  ОБЫЧНЫЙ РЕЖИМ (с отображением)" << std::endl;
    }
    std::cout << "📊 Логирование: ВКЛ (CSV)" << std::endl;
    std::cout << std::endl;

    // Включить логирование скорости декодирования (CSV) - всегда включено
    g_logDecodeSpeed.store(true);

    std::cout << "Введите RTSP URL (через запятую):" << std::endl;
    std::cout << "  Пример: rtsp://admin:pass@192.168.1.100:554/stream" << std::endl;
    std::cout << "URL: ";

    std::string input;
    std::getline(std::cin, input);

    // Парсинг URL из ввода пользователя (разделитель — запятая)
    std::vector<std::string> urls;
    std::stringstream ss(input);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Удаление пробелов вокруг URL
        item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
        if (!item.empty()) urls.push_back(item);
    }

    std::vector<std::thread> threads;

    // Запуск потока на каждую введённую камеру
    auto startCamera = [&](const std::string& u) {
        int idx;
        {
            std::lock_guard<std::mutex> lock(g_camMtx);
            idx = (int)g_cams.size();
            g_cams.push_back(CamInfo{u, 0, 0, 0, 0, false});
            g_camRunning.push_back(new std::atomic<bool>(true));
        }
        threads.emplace_back(cameraThread, u, idx);
        printAllStatus();
    };

    for (auto& u : urls) startCamera(u);

    // Главный цикл: мониторинг камер, добавление новых, выход по ESC/сигналу.
    // Ввод читается неблокирующе (poll), чтобы ESC из окна камеры мгновенно
    // прерывал цикл, даже если пользователь ничего не вводит.
    bool menuShown = false;
    bool hintShown = false;
    while (g_running) {
        bool anyAlive = false;
        {
            std::lock_guard<std::mutex> lock(g_camMtx);
            for (auto* f : g_camRunning)
                if (f && f->load()) { anyAlive = true; break; }
        }

        // Если все камеры закрыты — предложить переподключение или выход
        if (!anyAlive) {
            if (!menuShown) {
                std::cout << "\nВсе камеры отключены." << std::endl;
                std::cout << "[1] Подключить новую камеру" << std::endl;
                std::cout << "[2] Выход" << std::endl;
                std::cout << "Выбор: " << std::flush;
                menuShown = true;
                hintShown = false;
            }
            std::string opt;
            if (readLineAsync(opt, 400)) {
                menuShown = false;
                if (opt != "1") break;
                std::cout << "Введите RTSP URL: " << std::flush;
                std::string newUrl;
                while (g_running && !readLineAsync(newUrl, 400)) {}
                if (!g_running) break;
                if (!newUrl.empty()) startCamera(newUrl);
            }
            continue;
        }

        // Камеры работают: неблокирующий приём команд добавления камер
        menuShown = false;
        if (!hintShown) {
            std::cout << "\nКамеры работают. Введите RTSP URL для добавления камеры "
                         "или 'exit' для выхода: " << std::flush;
            hintShown = true;
        }
        std::string cmd;
        if (readLineAsync(cmd, 200) && g_running) {
            if (cmd == "exit" || cmd == "q") break;
            if (!cmd.empty()) startCamera(cmd);
            hintShown = false;
        }
    }

    // Завершение: отправить сигнал всем потокам, дождаться их завершения
    {
        std::lock_guard<std::mutex> lock(g_camMtx);
        for (auto* f : g_camRunning) if (f) *f = false;
    }
    for (auto& t : threads) if (t.joinable()) t.join();
    {
        std::lock_guard<std::mutex> lock(g_camMtx);
        for (auto* f : g_camRunning) delete f;
        g_camRunning.clear();
        g_cams.clear();
    }

    std::cout << "Выход." << std::endl;
    return 0;
}