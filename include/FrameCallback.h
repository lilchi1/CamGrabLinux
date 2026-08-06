// FrameCallback.h — Типы колбэков для доставки кадров из декодера.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>

// Колбэк для доставки NV12 кадров в host-памяти (ранее использовался для
// отображения через CUDA; теперь кадры доставляются в GPU-памяти — GpuFrame).
// Параметры:
//   yPlane   — указатель на Y-плоскость NV12
//   uvPlane  — указатель на UV-плоскость NV12
//   width    — ширина кадра в пикселях
//   height   — высота кадра в пикселях
//   strideY  — шаг строки Y-плоскости (в байтах)
//   strideUV — шаг строки UV-плоскости (в байтах)
//   pts      — временная метка кадра (в единицах времени)
using FrameCallback = std::function<void(uint8_t* yPlane, uint8_t* uvPlane,
                                         int width, int height,
                                         int strideY, int strideUV,
                                         int64_t pts)>;

// Кадр, декодированный NVDEC, в CUDA-памяти GPU.
//
// Плоскости NV12 (Y и UV) адресуются указателями на device-память; буфер
// удерживается живым полем keepAlive (shared_ptr на реф GstBuffer), поэтому
// кадр безопасно хранить в очереди и обрабатывать в отдельном потоке.
struct GpuFrame {
    uint8_t* yPlane = nullptr;    // device-указатель на Y-плоскость NV12
    uint8_t* uvPlane = nullptr;   // device-указатель на UV-плоскость NV12
    int width = 0;                // ширина кадра (px)
    int height = 0;               // высота кадра (px)
    int strideY = 0;              // pitch Y-плоскости (байт)
    int strideUV = 0;             // pitch UV-плоскости (байт)
    int64_t pts = -1;             // временная метка кадра

    // Удерживает источник (GstBuffer) живым, пока жива копия GpuFrame.
    std::shared_ptr<void> keepAlive;

    bool valid() const { return yPlane && uvPlane && width > 0 && height > 0; }
};

// Колбэк доставки декодированного кадра в GPU-памяти. Вызывается из потока
// GStreamer; не должен блокироваться на I/O (только копирование указателей и
// постановка в очередь).
using GpuFrameCallback = std::function<void(const GpuFrame& frame)>;
