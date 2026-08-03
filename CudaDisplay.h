// CudaDisplay.h — Обёртка для CUDA операций: конвертация NV12→BGRA + масштабирование.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

class CudaDisplay {
public:
    CudaDisplay();
    ~CudaDisplay();

    // Инициализация CUDA для указанного размера входного кадра
    bool init(int srcW, int srcH);

    // Освобождение всех CUDA ресурсов
    void cleanup();

    // Конвертация NV12→BGRA с масштабированием; возвращает указатель на host-буфер BGRA
    uint8_t* process(const uint8_t* yPlane, const uint8_t* uvPlane,
                     int srcStrideY, int srcStrideUV,
                     int srcW, int srcH, int dstW, int dstH,
                     int& outStride);

private:
    bool m_ready;            // Флаг готовности CUDA
    cudaStream_t m_stream;   // CUDA поток для асинхронных операций

    uint8_t* d_y;            // Device-буфер Y-плоскости NV12
    uint8_t* d_uv;           // Device-буфер UV-плоскости NV12
    uint8_t* d_bgr;          // Device-буфер BGRA (результат)
    int m_devYSize;          // Текущий размер device Y-буфера
    int m_devUVSize;         // Текущий размер device UV-буфера
    int m_devBgrSize;        // Текущий размер device BGRA-буфера
    int m_devBgrStride;      // Шаг строки device BGRA-буфера

    uint8_t* h_bgr;          // Host-буфер BGRA (для копирования обратно из GPU)
    int m_hBgrSize;          // Текущий размер host BGRA-буфера

    bool reallocDev(int ySize, int uvSize, int bgrStride, int bgrH);  // Динамическое выделение device-буферов
};
