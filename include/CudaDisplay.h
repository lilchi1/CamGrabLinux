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

    // Загрузка NV12 на GPU (один раз на кадр). Синхронная: после вызова данные
    // готовы для чтения инференсом (deviceY()/deviceUV()) и используются
    // processFromDevice() без повторной передачи CPU→GPU. Возвращает d_y или
    // nullptr при ошибке.
    uint8_t* uploadNv12(const uint8_t* yPlane, const uint8_t* uvPlane,
                        int srcStrideY, int srcStrideUV,
                        int srcW, int srcH);

    // Конвертация уже загруженного на GPU NV12 (см. uploadNv12) → BGRA
    // с масштабированием; возвращает указатель на host-буфер BGRA.
    uint8_t* processFromDevice(int srcW, int srcH, int dstW, int dstH,
                               int& outStride);

    // Конвертация NV12→BGRA с масштабированием (uploadNv12 + processFromDevice);
    // возвращает указатель на host-буфер BGRA.
    uint8_t* process(const uint8_t* yPlane, const uint8_t* uvPlane,
                     int srcStrideY, int srcStrideUV,
                     int srcW, int srcH, int dstW, int dstH,
                     int& outStride);

    // Device-указатели загруженного NV12 (после uploadNv12)
    uint8_t* deviceY() const { return d_y; }
    uint8_t* deviceUV() const { return d_uv; }

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
