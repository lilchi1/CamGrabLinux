// main.cpp — Точка входа. Создаёт поток на каждую камеру, управляет динамическим добавлением.
#include "headers.h"
#include "Detection.h"
#include <poll.h>
#include <unistd.h>
#include <getopt.h>

// Глобальный флаг работы программы (атомарный для безопасного доступа из потоков)
volatile std::sig_atomic_t g_running = 1;

// Флаги режимов работы
bool g_benchmarkMode = false;      // Режим бенчмарка (без отображения)
int g_winWidth  = 1600;            // Размер окна отображения (по умолчанию)
int g_winHeight = 900;
std::string g_displayMode = "xvimagesink"; // Режим отображения: xvimagesink (default) | cuda

// Детекция YOLO (TensorRT engine, вход 640×640)
std::string g_modelPath;                              // путь к .engine (пусто = без детекции)
std::string g_labelsPath;                             // файл имён классов
float g_confThresh = 0.35f;                           // порог уверенности
float g_nmsThresh = 0.45f;                            // порог NMS
std::vector<std::string> g_classNames;                // имена классов (loadClassNames)

// Детекция YOLOv2 (anchor-based, Keras-модель из yolo/yolo_model_complete.h5).
// Выход NCHW [1, C, grid, grid], C = numAnchors*(5+nc). Якоря — в пикселях входа.
bool g_yolov2Mode = false;
int g_yolov2Grid = 19;                                // сетка выхода (608/32 = 19)
std::vector<float> g_yolov2Anchors = {                // YAD2K COCO (пиксели входа 608/416)
    18.32736f, 21.67632f,
    59.98272f, 66.00096f,
    106.82976f, 175.17888f,
    252.25024f, 112.88896f,
    312.65664f, 293.38496f,
};
int g_modelInSize = 0;                                // размер входа модели (0 = авто)

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
    std::cout << "  -w, --width W      Ширина окна отображения (по умолчанию 1600)" << std::endl;
    std::cout << "  -H, --height H     Высота окна отображения (по умолчанию 900)" << std::endl;
    std::cout << "  -d, --display M    Режим отображения: xvimagesink (по умолчанию) | cuda" << std::endl;
    std::cout << "  -m, --model PATH   TensorRT .engine (YOLOv8/v11/v12, вход 640x640)" << std::endl;
    std::cout << "  -l, --labels PATH  Файл имён классов (по одному в строке, COCO=80)" << std::endl;
    std::cout << "  -c, --conf F       Порог уверенности (по умолчанию 0.35)" << std::endl;
    std::cout << "  -n, --nms F        Порог NMS (по умолчанию 0.45)" << std::endl;
    std::cout << "  --yolov2           YOLOv2-декод (anchor-based, Keras h5 → .engine)" << std::endl;
    std::cout << "  --v2-grid N        Сетка выхода YOLOv2 (по умолчанию 19 = 608/32)" << std::endl;
    std::cout << "  --v2-anchors L     Якоря YOLOv2, пары w,h в пикселях (по умолчанию COCO YAD2K)" << std::endl;
    std::cout << "  --in-size N        Размер входа модели (по умолчанию: 640 для v8/11/12," << std::endl;
    std::cout << "                       608 = grid*32 для YOLOv2)" << std::endl;
    std::cout << "  -h, --help         Показать эту справку" << std::endl;
    std::cout << std::endl;
    std::cout << "Примеры:" << std::endl;
    std::cout << "  " << progname << "                      # Обычный режим с отображением" << std::endl;
    std::cout << "  " << progname << " -b                  # Режим бенчмарка (без окна)" << std::endl;
    std::cout << "  " << progname << " --width 1280 --height 720   # Окно 1280x720" << std::endl;
    std::cout << "  " << progname << " --display cuda       # Фолбэк: рендер CUDA + XPutImage" << std::endl;
    std::cout << "  " << progname << " --benchmark         # Бенчмарк с логированием" << std::endl;
    std::cout << "  " << progname << " --model yolo.engine --labels coco.names --conf 0.4 --nms 0.5  # Детекция YOLO" << std::endl;
    std::cout << "  " << progname << " --model yolo_v2.engine --labels yolo/coco.names --yolov2  # YOLOv2 (608x608, COCO)" << std::endl;
}

int main(int argc, char* argv[]) {
    // Регистрация обработчиков сигналов
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    // ─── Разбор аргументов командной строки ──────────────────────────────────
    static struct option long_options[] = {
        {"benchmark", no_argument,       0, 'b'},
        {"width",     required_argument, 0, 'w'},
        {"height",    required_argument, 0, 'H'},
        {"display",   required_argument, 0, 'd'},
        {"model",     required_argument, 0, 'm'},
        {"labels",    required_argument, 0, 'l'},
        {"conf",      required_argument, 0, 'c'},
        {"nms",       required_argument, 0, 'n'},
        {"yolov2",    no_argument,       0, 1000},
        {"v2-grid",   required_argument, 0, 1001},
        {"v2-anchors", required_argument, 0, 1002},
        {"in-size",   required_argument, 0, 1003},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "bw:H:d:m:l:c:n:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'b':
                g_benchmarkMode = true;
                break;
            case 'w': {
                int v = std::atoi(optarg);
                if (v > 0) g_winWidth = v;
                break;
            }
            case 'H': {
                int v = std::atoi(optarg);
                if (v > 0) g_winHeight = v;
                break;
            }
            case 'd': {
                std::string mode = optarg;
                if (mode == "xvimagesink" || mode == "cuda") {
                    g_displayMode = mode;
                } else {
                    std::cerr << "Неизвестный режим отображения: " << mode
                              << " (допустимо: cuda, xvimagesink)" << std::endl;
                    return 1;
                }
                break;
            }
            case 'm':
                g_modelPath = optarg;
                break;
            case 'l':
                g_labelsPath = optarg;
                break;
            case 'c': {
                float v = (float)std::atof(optarg);
                if (v > 0.0f && v < 1.0f) g_confThresh = v;
                break;
            }
            case 'n': {
                float v = (float)std::atof(optarg);
                if (v > 0.0f && v < 1.0f) g_nmsThresh = v;
                break;
            }
            case 1000:
                g_yolov2Mode = true;
                break;
            case 1001: {
                int v = std::atoi(optarg);
                if (v > 0) g_yolov2Grid = v;
                break;
            }
            case 1002: {
                std::vector<float> anchors;
                std::stringstream ss(optarg);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    float v = (float)std::atof(item.c_str());
                    if (v > 0.0f) anchors.push_back(v);
                }
                if (anchors.size() >= 2 && (anchors.size() % 2) == 0)
                    g_yolov2Anchors = anchors;
                else
                    std::cerr << "Предупреждение: --v2-anchors должен содержать пары w,h > 0"
                              << std::endl;
                break;
            }
            case 1003: {
                int v = std::atoi(optarg);
                if (v > 0) g_modelInSize = v;
                break;
            }
            case 'h':
                printUsage(argv[0]);
                return 0;
            default:
                printUsage(argv[0]);
                return 1;
        }
    }

    // Загрузка имён классов для детекции
    if (!g_labelsPath.empty()) {
        g_classNames = loadClassNames(g_labelsPath);
        if (g_classNames.empty()) {
            std::cerr << "Ошибка: не удалось загрузить имена классов из " << g_labelsPath << std::endl;
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
        std::cout << "🖥️  ОБЫЧНЫЙ РЕЖИМ (с отображением, режим: " << g_displayMode << ")" << std::endl;
    }
    std::cout << "📊 Логирование: ВКЛ (CSV)" << std::endl;
    if (!g_modelPath.empty()) {
        int numClasses = g_classNames.empty() ? 80 : (int)g_classNames.size();
        if (g_yolov2Mode) {
            std::cout << "🧠 ДЕТЕКЦИЯ YOLOv2 (anchor-based): " << g_modelPath
                      << " (классов=" << numClasses
                      << ", сетка=" << g_yolov2Grid
                      << ", якорей=" << (int)(g_yolov2Anchors.size() / 2)
                      << ", conf=" << g_confThresh
                      << ", nms=" << g_nmsThresh << ")" << std::endl;
        } else {
            std::cout << "🧠 ДЕТЕКЦИЯ YOLO: " << g_modelPath
                      << " (классов=" << numClasses
                      << ", conf=" << g_confThresh
                      << ", nms=" << g_nmsThresh << ")" << std::endl;
        }
    } else {
        std::cout << "🧠 Детекция YOLO: ВЫКЛ (укажите --model для включения)" << std::endl;
    }
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