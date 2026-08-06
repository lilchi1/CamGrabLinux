// Display.h — X11 + MIT-SHM окно отображения. Конвертация NV12→BGRA и масштабирование на GPU (CUDA).
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <utility>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <sys/shm.h>

#include "Detection.h"

class CudaDisplay;

// Структура MIT-SHM сегмента (аналог XShmSegmentInfo из Xext)
typedef struct {
    int shmid;          // ID сегмента разделяемой памяти
    char* shmaddr;      // Адрес mmap'енного сегмента
    int readOnly;       // Флаг «только чтение» (обычно False)
} XShmSegmentInfo;

class DisplayWindow {
public:
    DisplayWindow();
    ~DisplayWindow();

    // Открытие окна с указанным заголовком и размером.
    // overlayOnly: окно без XImage/SHM/CUDA — используется как контейнер для
    // внешнего рендера (nv3dsink через GstVideoOverlay), клавиши остаются нашими.
    bool open(const std::string& title, int width, int height,
              bool overlayOnly = false);

    // Закрытие окна и освобождение ресурсов
    void close();

    // Отображение NV12 кадра (конвертация + масштабирование через CUDA)
    void showFrame(uint8_t* yPlane, uint8_t* uvPlane,
                   int srcW, int srcH,
                   int strideY, int strideUV);

    // Оверлей детекций поверх последнего showFrame(). Координаты боксов —
    // в пикселях исходного кадра (srcW x srcH); масштаб/смещение те же,
    // что в showFrame(). classNames — имена классов (по индексу classId).
    void showDetections(const Detections& dets,
                        const std::vector<std::string>& classNames,
                        int srcW, int srcH);

    // X11-хендл окна (для nv3dsink set_window_handle)
    ::Display* xDisplay() const { return m_display; }
    Window window() const { return m_window; }

    // Флаг overlay-only режима (рендер внешний, XImage/CUDA не создаются)
    bool overlayOnly() const { return m_overlayOnly; }

    // Обработка событий X11 (изменение размера окна, клавиши) — вызывать из главного цикла
    void pollEvents();

    // Получение нажатой клавиши из очереди (keysym; pressed = нажатие/отпускание).
    // Возвращает false, если очередь пуста.
    bool popKeyEvent(int& keysym, bool& pressed);

private:
    ::Display* m_display;    // Подключение к X-серверу
    Window m_window;         // X11 окно
    GC m_gc;                 // Graphics context
    XImage* m_image;         // XImage для вывода кадров (ZPixmap 32bpp)
    XShmSegmentInfo m_shmInfo;  // MIT-SHM сегмент
    Visual* m_visual;        // TrueColor 24-бит визуал
    int m_depth;             // Глубина цвета (24 бита)
    int m_width;             // Текущий размер окна
    int m_height;

    CudaDisplay* m_cuda;     // Обёртка для CUDA операций
    bool m_cudaReady;        // Флаг готовности CUDA

    int m_bpp;               // Байт на пиксель в XImage (4 для 32bpp)
    bool m_useShm;           // Флаг использования MIT-SHM

    std::mutex m_keyMtx;                         // Защита очереди клавиш
    std::deque<std::pair<int, bool>> m_keyEvents; // Очередь клавиш (keysym, нажата)

    std::mutex m_xMtx;   // Сериализация доступа к X11-коннекту и окну (только для вывода)
    
    int m_frameCounter;  // Счётчик кадров для периодического XFlush
    bool m_overlayOnly;  // Флаг: окно только как контейнер для внешнего рендера

    void allocateImage(int width, int height);  // Выделение XImage (SHM или обычный)
    void destroyImage();                         // Уничтожение XImage и SHM
    void enqueueKey(int keysym, bool pressed);   // Добавление клавиши в очередь
};