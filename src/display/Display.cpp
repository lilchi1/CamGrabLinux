// Display.cpp — X11 + MIT-SHM окно отображения. Конвертация NV12→BGRA и масштабирование на GPU (CUDA).
#include "Display.h"
#include "CudaDisplay.h"
#include "headers.h"
#include <X11/keysym.h>
#include <dlfcn.h>
#include <unistd.h>

// Функции MIT-SHM загружаем через dlopen (избегаем жёсткой зависимости от libXext)
typedef Bool (*XShmQueryExtFunc)(::Display*);
typedef XImage* (*XShmCreateImageFunc)(::Display*, Visual*, unsigned int, int,
                                        char*, void*, unsigned int, unsigned int);
typedef Bool (*XShmAttachFunc)(::Display*, void*);
typedef Bool (*XShmDetachFunc)(::Display*, void*);
typedef int (*XShmPutImageFunc)(::Display*, Drawable, GC, XImage*,
                                 int, int, int, int,
                                 unsigned int, unsigned int, Bool);

static void* xshmHandle = nullptr;
static XShmQueryExtFunc pXShmQueryExtension = nullptr;
static XShmCreateImageFunc pXShmCreateImage = nullptr;
static XShmAttachFunc pXShmAttach = nullptr;
static XShmDetachFunc pXShmDetach = nullptr;
static XShmPutImageFunc pXShmPutImage = nullptr;

// Загрузка MIT-SHM функций из libXext.so.6
static bool loadXShm() {
    if (xshmHandle) return pXShmQueryExtension != nullptr;
    xshmHandle = dlopen("libXext.so.6", RTLD_LAZY | RTLD_LOCAL);
    if (!xshmHandle) return false;
    pXShmQueryExtension = (XShmQueryExtFunc)dlsym(xshmHandle, "XShmQueryExtension");
    pXShmCreateImage = (XShmCreateImageFunc)dlsym(xshmHandle, "XShmCreateImage");
    pXShmAttach = (XShmAttachFunc)dlsym(xshmHandle, "XShmAttach");
    pXShmDetach = (XShmDetachFunc)dlsym(xshmHandle, "XShmDetach");
    pXShmPutImage = (XShmPutImageFunc)dlsym(xshmHandle, "XShmPutImage");
    if (!pXShmQueryExtension || !pXShmCreateImage || !pXShmAttach ||
        !pXShmDetach || !pXShmPutImage) {
        dlclose(xshmHandle); xshmHandle = nullptr;
        return false;
    }
    return true;
}

DisplayWindow::DisplayWindow()
    : m_display(nullptr), m_window(0), m_gc(nullptr),
      m_image(nullptr), m_visual(nullptr), m_depth(0),
      m_width(0), m_height(0),
      m_cuda(nullptr), m_cudaReady(false),
      m_bpp(0), m_useShm(false),
      m_frameCounter(0), m_overlayOnly(false) {
    memset(&m_shmInfo, 0, sizeof(m_shmInfo));
}

DisplayWindow::~DisplayWindow() { close(); }

// Открытие окна X11, создание XImage (SHM или обычный), инициализация CUDA
bool DisplayWindow::open(const std::string& title, int width, int height,
                         bool overlayOnly) {
    close();
    m_overlayOnly = overlayOnly;
    m_width = width;
    m_height = height;

    // Потокобезопасность Xlib: X-коннект используется и из appsink-потока
    // (showFrame), и из потока камеры (pollEvents)
    XInitThreads();

    // Подключение к X-серверу
    m_display = XOpenDisplay(nullptr);
    if (!m_display) return false;

    int screen = DefaultScreen(m_display);
    Window root = RootWindow(m_display, screen);

    // Создание окна
    m_window = XCreateSimpleWindow(m_display, root,
                                   0, 0, static_cast<unsigned>(width),
                                   static_cast<unsigned>(height), 1,
                                   BlackPixel(m_display, screen),
                                   BlackPixel(m_display, screen));
    if (!m_window) { close(); return false; }
    XStoreName(m_display, m_window, title.c_str());
    XSelectInput(m_display, m_window,
                 KeyPressMask | KeyReleaseMask | ExposureMask | StructureNotifyMask);
    XMapWindow(m_display, m_window);
    m_gc = XCreateGC(m_display, m_window, 0, nullptr);

    // Overlay-only: окно отдаётся внешнему рендеру (nv3dsink), пиксельный
    // буфер и CUDA не нужны — но окно должно быть отображено до set_window_handle.
    XSync(m_display, False);
    if (m_overlayOnly) return true;

    // Выбор TrueColor 24-бит визуала
    XVisualInfo visInfo;
    if (!XMatchVisualInfo(m_display, screen, 24, TrueColor, &visInfo)) {
        close(); return false;
    }
    m_visual = visInfo.visual;
    m_depth = visInfo.depth;

    // Выделение XImage (MIT-SHM или обычный)
    allocateImage(width, height);

    // Инициализация CUDA для конвертации NV12→BGRA и масштабирования
    m_cuda = new CudaDisplay();
    if (m_cuda->init(width, height)) {
        m_cudaReady = true;
        fprintf(stderr, "[Display] Используется GPU-ускорение CUDA\n");
    } else {
        delete m_cuda; m_cuda = nullptr;
        m_cudaReady = false;
        fprintf(stderr, "[Display] ошибка инициализации CUDA\n");
        close();
        return false;
    }

    XSync(m_display, False);
    return true;
}

// Выделение XImage: попытка MIT-SHM (zero-copy), fallback на обычный XImage
void DisplayWindow::allocateImage(int width, int height) {
    // Попытка MIT-SHM (zero-copy, корректный fallback при BadAccess)
    m_useShm = loadXShm() && pXShmQueryExtension(m_display);
    if (m_useShm) {
        m_image = pXShmCreateImage(m_display, m_visual, (unsigned)m_depth,
                                   ZPixmap, nullptr, &m_shmInfo,
                                   static_cast<unsigned>(width),
                                   static_cast<unsigned>(height));
        if (m_image) {
            // Выделение разделяемой памяти
            m_shmInfo.shmid = shmget(IPC_PRIVATE,
                                     static_cast<size_t>(m_image->bytes_per_line)
                                         * static_cast<size_t>(m_image->height),
                                     IPC_CREAT | 0600);
            if (m_shmInfo.shmid >= 0) {
                m_shmInfo.shmaddr = (char*)shmat(m_shmInfo.shmid, nullptr, 0);
                if (m_shmInfo.shmaddr != (char*)-1) {
                    m_image->data = m_shmInfo.shmaddr;
                    m_shmInfo.readOnly = False;

                    // Установка обработчика ошибок для перехвата BadShmSeg
                    static bool s_shmError;
                    s_shmError = false;
                    auto oldHandler = XSetErrorHandler(
                        [](::Display*, XErrorEvent*) -> int {
                            s_shmError = true;
                            return 0;
                        });
                    pXShmAttach(m_display, &m_shmInfo);
                    XSync(m_display, False);
                    XSetErrorHandler(oldHandler);
                    if (s_shmError) m_useShm = false;
                } else { m_useShm = false; }
            } else { m_useShm = false; }
        } else { m_useShm = false; }

        // Очистка при неудаче MIT-SHM
        if (!m_useShm) {
            if (m_shmInfo.shmaddr && m_shmInfo.shmaddr != (char*)-1) {
                shmdt(m_shmInfo.shmaddr);
                shmctl(m_shmInfo.shmid, IPC_RMID, nullptr);
            }
            if (m_image) { XDestroyImage(m_image); m_image = nullptr; }
            memset(&m_shmInfo, 0, sizeof(m_shmInfo));
        }
    }

    // Fallback: обычный XImage (без SHM)
    if (!m_image) {
        m_image = XCreateImage(m_display, m_visual, (unsigned)m_depth,
                               ZPixmap, 0, nullptr,
                               static_cast<unsigned>(width),
                               static_cast<unsigned>(height), 32, 0);
        if (m_image) {
            m_image->data = (char*)malloc(static_cast<size_t>(height)
                                          * (size_t)m_image->bytes_per_line);
            if (!m_image->data) {
                XDestroyImage(m_image);
                m_image = nullptr;
                return;
            }
        }
    }

    if (!m_image) return;

    m_bpp = m_image->bits_per_pixel / 8;
    // Заполнение чёрным (letterbox/pillarbox)
    memset(m_image->data, 0, static_cast<size_t>(height) * (size_t)m_image->bytes_per_line);

    m_width = width;
    m_height = height;
}

// Уничтожение XImage и освобождение SHM ресурсов
void DisplayWindow::destroyImage() {
    if (m_image) {
        if (m_useShm) {
            pXShmDetach(m_display, &m_shmInfo);
            XDestroyImage(m_image);
            if (m_shmInfo.shmaddr && m_shmInfo.shmaddr != (char*)-1) {
                shmdt(m_shmInfo.shmaddr);
                shmctl(m_shmInfo.shmid, IPC_RMID, nullptr);
            }
        } else {
            XDestroyImage(m_image);
        }
        m_image = nullptr;
    }
    memset(&m_shmInfo, 0, sizeof(m_shmInfo));
}

// Закрытие окна и освобождение всех ресурсов (CUDA, XImage, GC, Display)
void DisplayWindow::close() {
    std::lock_guard<std::mutex> lock(m_xMtx);
    if (m_cuda) { delete m_cuda; m_cuda = nullptr; m_cudaReady = false; }

    destroyImage();

    if (m_gc) { XFreeGC(m_display, m_gc); m_gc = nullptr; }
    if (m_window) { XDestroyWindow(m_display, m_window); m_window = 0; }
    if (m_display) { XCloseDisplay(m_display); m_display = nullptr; }
}

// Обработка событий X11: изменение размера окна (пересоздание XImage) и клавиши.
// Клавиши кладутся в очередь; авто-повтор (удержание) игнорируется.
void DisplayWindow::pollEvents() {
    std::lock_guard<std::mutex> lock(m_xMtx);
    XEvent ev;
    while (XPending(m_display)) {
        XNextEvent(m_display, &ev);
        if (ev.type == ConfigureNotify) {
            int newW = ev.xconfigure.width;
            int newH = ev.xconfigure.height;
            if (newW > 0 && newH > 0 &&
                (newW != m_width || newH != m_height)) {
                destroyImage();
                allocateImage(newW, newH);
                if (m_cuda) m_cuda->init(newW, newH);
            }
        } else if (ev.type == KeyPress) {
            KeySym key = XLookupKeysym(&ev.xkey, 0);
            if (key != NoSymbol) enqueueKey((int)key, true);
        } else if (ev.type == KeyRelease) {
            // Пропуск авто-повтора: если сразу за KeyRelease идёт KeyPress того
            // же keycode — это автоповтор удерживаемой клавиши.
            XEvent peek;
            if (XPending(m_display)) {
                XPeekEvent(m_display, &peek);
                if (peek.type == KeyPress &&
                    peek.xkey.keycode == ev.xkey.keycode) {
                    continue;
                }
            }
            KeySym key = XLookupKeysym(&ev.xkey, 0);
            if (key != NoSymbol) enqueueKey((int)key, false);
        }
    }
}

// Добавление события клавиши в очередь
void DisplayWindow::enqueueKey(int keysym, bool pressed) {
    std::lock_guard<std::mutex> lock(m_keyMtx);
    m_keyEvents.push_back(std::make_pair(keysym, pressed));
}

// Извлечение события клавиши из очереди
bool DisplayWindow::popKeyEvent(int& keysym, bool& pressed) {
    std::lock_guard<std::mutex> lock(m_keyMtx);
    if (m_keyEvents.empty()) return false;
    keysym = m_keyEvents.front().first;
    pressed = m_keyEvents.front().second;
    m_keyEvents.pop_front();
    return true;
}

// Отображение NV12 кадра: конвертация на GPU → копирование в XImage → вывод в окно
void DisplayWindow::showFrame(uint8_t* yPlane, uint8_t* uvPlane,
                              int srcW, int srcH,
                              int strideY, int strideUV) {
    if (m_overlayOnly) return;  // рендер внешний (nv3dsink)
    if (!m_image || !m_image->data || !m_cudaReady || !m_cuda) return;

    // Масштабирование с сохранением пропорций (вычисляем без блокировки)
    double scale = std::min((double)m_width / srcW, (double)m_height / srcH);
    int outW = (int)(srcW * scale + 0.5);
    int outH = (int)(srcH * scale + 0.5);
    int offX = (m_width  - outW) / 2;
    int offY = (m_height - outH) / 2;

    // CUDA: масштабирование + конвертация NV12→BGRA на GPU (без блокировки X)
    int outStride = 0;
    uint8_t* bgra = m_cuda->process(yPlane, uvPlane, strideY, strideUV,
                                    srcW, srcH, outW, outH, outStride);
    if (!bgra) return;

    // ─── Критическая секция: только копирование в XImage и вывод ──────────
    {
        std::lock_guard<std::mutex> lock(m_xMtx);
        
        // Проверка, что окно не изменилось во время CUDA-обработки
        if (!m_image || !m_image->data) return;

        // Заполнение чёрным (letterbox/pillarbox)
        memset(m_image->data, 0,
               static_cast<size_t>(m_height) * (size_t)m_image->bytes_per_line);

        // Копирование BGRA данных в XImage построчно
        for (int y = 0; y < outH; y++) {
            int dstRow = offY + y;
            if (dstRow >= m_height) break;
            const uint8_t* srcRow = bgra + y * outStride;
            char* dstRowPtr = m_image->data + dstRow * m_image->bytes_per_line
                              + offX * m_bpp;
            memcpy(dstRowPtr, srcRow, outW * 4);
        }

        // Вывод кадра в окно: MIT-SHM (zero-copy) или обычный XPutImage
        if (m_useShm)
            pXShmPutImage(m_display, m_window, m_gc, m_image,
                          0, 0, 0, 0, static_cast<unsigned>(m_width),
                          static_cast<unsigned>(m_height), False);
        else
            XPutImage(m_display, m_window, m_gc, m_image,
                      0, 0, 0, 0, static_cast<unsigned>(m_width),
                      static_cast<unsigned>(m_height));
        
        m_frameCounter++;
        
        // Периодический flush для синхронизации (каждые 10 кадров)
        if (m_frameCounter % 10 == 0) {
            XFlush(m_display);
        }
    }
    // ─── Конец критической секции ──────────────────────────────────────────
}