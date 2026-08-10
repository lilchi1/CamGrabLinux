// FrameCallback.h — Типы колбэков для доставки кадров из декодера.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>

// HostFrame — кадр NV12 в host-памяти с zero-copy доставкой из декодера.
// Буфер GStreamer удерживается живым полем keepAlive (реф буфера + состояние
// маппинга), поэтому кадр безопасно хранить в очереди и обрабатывать в
// отдельном потоке без копирования пикселей.
struct HostFrame {
    uint8_t* yPlane = nullptr;    // указатель на Y-плоскость NV12
    uint8_t* uvPlane = nullptr;   // указатель на UV-плоскость NV12
    int width = 0;                // ширина кадра (px)
    int height = 0;               // высота кадра (px)
    int strideY = 0;              // pitch Y-плоскости (байт)
    int strideUV = 0;             // pitch UV-плоскости (байт)
    int64_t pts = -1;             // временная метка кадра

    // Удерживает GstBuffer + маппинг живыми, пока жива копия HostFrame.
    std::shared_ptr<void> keepAlive;

    bool valid() const { return yPlane && uvPlane && width > 0 && height > 0; }
};

// Колбэк для доставки NV12 кадров в host-памяти (zero-copy — кадр не
// копируется, буфер держится keepAlive до обработки в потоке-потребителе).
using FrameCallback = std::function<void(const HostFrame& frame)>;

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
