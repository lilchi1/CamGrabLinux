// CudaDisplay.cpp — Обёртка для CUDA операций: конвертация NV12→BGRA + билинейное масштабирование.
#include "CudaDisplay.h"
#include <cstdio>
#include <cstring>
#include <algorithm>

// Объявления CUDA-ядр из CudaKernels.cu (extern "C" для совместимости)
extern "C" cudaError_t cudaNv12ToBgrScale(const uint8_t* yPlane, const uint8_t* uvPlane,
                                           int srcStride, int srcW, int srcH,
                                           uint8_t* d_dst, int dstStride,
                                           int dstW, int dstH, cudaStream_t stream);
extern "C" cudaError_t cudaNv12ToBgr(const uint8_t* yPlane, const uint8_t* uvPlane,
                                      int srcStride, int width, int height,
                                      uint8_t* d_dst, int dstStride, cudaStream_t stream);

CudaDisplay::CudaDisplay()
    : m_ready(false), m_stream(nullptr),
      d_y(nullptr), d_uv(nullptr), d_bgr(nullptr),
      m_devYSize(0), m_devUVSize(0), m_devBgrSize(0), m_devBgrStride(0),
      h_bgr(nullptr), m_hBgrSize(0) {}

CudaDisplay::~CudaDisplay() { cleanup(); }

// Инициализация CUDA: создание потока, предварительное выделение буферов
bool CudaDisplay::init(int srcW, int srcH) {
    cleanup();

    // Создание CUDA потока для асинхронных операций
    cudaError_t err = cudaStreamCreate(&m_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] cudaStreamCreate ошибка: %s\n", cudaGetErrorString(err));
        return false;
    }

    // Предварительное выделение device-буферов для типичных разрешений
    int ySize  = srcW * srcH;
    int uvSize = srcW * (srcH / 2);
    reallocDev(ySize, uvSize, srcW * 3, srcH);

    m_ready = true;
    fprintf(stderr, "[CUDA] инициализирован для %dx%d\n", srcW, srcH);
    return true;
}

// Освобождение всех CUDA ресурсов (буферы, поток)
void CudaDisplay::cleanup() {
    if (d_y)    { cudaFree(d_y);    d_y = nullptr; }
    if (d_uv)   { cudaFree(d_uv);   d_uv = nullptr; }
    if (d_bgr)  { cudaFree(d_bgr);  d_bgr = nullptr; }
    if (h_bgr)  { free(h_bgr);      h_bgr = nullptr; }
    if (m_stream) { cudaStreamDestroy(m_stream); m_stream = nullptr; }
    m_devYSize = m_devUVSize = m_devBgrSize = m_devBgrStride = 0;
    m_hBgrSize = 0;
    m_ready = false;
}

// Динамическое выделение/расширение device-буферов (только при необходимости)
bool CudaDisplay::reallocDev(int ySize, int uvSize, int bgrStride, int bgrH) {
    // Y-плоскость на GPU
    if (ySize > m_devYSize) {
        if (d_y) cudaFree(d_y);
        cudaError_t err = cudaMalloc(&d_y, ySize);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA] cudaMalloc Y ошибка: %s\n", cudaGetErrorString(err));
            d_y = nullptr; return false;
        }
        m_devYSize = ySize;
    }

    // UV-плоскость на GPU
    if (uvSize > m_devUVSize) {
        if (d_uv) cudaFree(d_uv);
        cudaError_t err = cudaMalloc(&d_uv, uvSize);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA] cudaMalloc UV ошибка: %s\n", cudaGetErrorString(err));
            d_uv = nullptr; return false;
        }
        m_devUVSize = uvSize;
    }

    // BGRA-буфер на GPU (результат конвертации)
    int bgrSize = bgrStride * bgrH;
    if (bgrSize > m_devBgrSize) {
        if (d_bgr) cudaFree(d_bgr);
        cudaError_t err = cudaMalloc(&d_bgr, bgrSize);
        if (err != cudaSuccess) {
            fprintf(stderr, "[CUDA] cudaMalloc BGR ошибка: %s\n", cudaGetErrorString(err));
            d_bgr = nullptr; return false;
        }
        m_devBgrSize = bgrSize;
        m_devBgrStride = bgrStride;
    }

    // BGRA-буфер на хосте (для копирования обратно из GPU)
    if (bgrSize > m_hBgrSize) {
        uint8_t* newBuf = (uint8_t*)realloc(h_bgr, bgrSize);
        if (!newBuf) {
            fprintf(stderr, "[CUDA] realloc host BGR ошибка\n");
            return false;
        }
        h_bgr = newBuf;
        m_hBgrSize = bgrSize;
    }

    return true;
}

// Загрузка NV12 на GPU (один раз на кадр). Синхронная: после возврата данные
// готовы для чтения инференсом (deviceY()/deviceUV()) и конвертацией.
uint8_t* CudaDisplay::uploadNv12(const uint8_t* yPlane, const uint8_t* uvPlane,
                                 int srcStrideY, int srcStrideUV,
                                 int srcW, int srcH) {
    if (!m_ready || !yPlane || !uvPlane) return nullptr;

    // Только NV12-буферы; BGRA (bgrStride=0) перевыделит processFromDevice.
    if (!reallocDev(srcW * srcH, srcW * (srcH / 2), 0, 0))
        return nullptr;

    // Копирование NV12 из хоста в device одним вызовом 2D (cudaMemcpy2DAsync):
    // аппаратный DMA сам учитывает stride источника и устраняет сотни мелких
    // memcpyAsync на кадр (по одному на строку). dest pitch = srcW (плотно).
    cudaError_t err = cudaMemcpy2DAsync(
        d_y,  (size_t)srcW,
        yPlane, (size_t)srcStrideY,
        (size_t)srcW, (size_t)srcH,
        cudaMemcpyHostToDevice, m_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] cudaMemcpy2D Y ошибка: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    // Копирование UV-плоскости (с учётом stride)
    int uvH = srcH / 2;
    err = cudaMemcpy2DAsync(
        d_uv, (size_t)srcW,
        uvPlane, (size_t)srcStrideUV,
        (size_t)srcW, (size_t)uvH,
        cudaMemcpyHostToDevice, m_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] cudaMemcpy2D UV ошибка: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    // Синхронизация: аплоад завершён, NV12 готов для чтения инференсом и
    // конвертацией. Полная синхронизация — как в исходном runHostNv12:
    // preprocessMs остаётся «чистым» временем препроцессинга (без хвоста
    // аплоада в замере) и CV-CUDA не пересекается с кросс-стримным событием.
    cudaStreamSynchronize(m_stream);
    return d_y;
}

// Конвертация уже загруженного на GPU NV12 → BGRA + копирование обратно на хост
uint8_t* CudaDisplay::processFromDevice(int srcW, int srcH, int dstW, int dstH,
                                        int& outStride) {
    if (!m_ready) return nullptr;

    int bgrStride = dstW * 4;  // BGRA: 4 байта на пиксель
    if (!reallocDev(0, 0, bgrStride, dstH))
        return nullptr;

    cudaError_t err;

    // Запуск CUDA-ядра: комбинированное масштабирование + конвертация NV12→BGRA
    bool needScale = (dstW != srcW || dstH != srcH);

    if (needScale) {
        err = cudaNv12ToBgrScale(d_y, d_uv, srcW, srcW, srcH,
                                  d_bgr, bgrStride, dstW, dstH, m_stream);
    } else {
        err = cudaNv12ToBgr(d_y, d_uv, srcW, srcW, srcH,
                             d_bgr, bgrStride, m_stream);
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] запуск ядра ошибка: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    // Копирование BGRA-результата обратно на хост
    int totalBgr = bgrStride * dstH;
    err = cudaMemcpyAsync(h_bgr, d_bgr, totalBgr,
                          cudaMemcpyDeviceToHost, m_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA] cudaMemcpy BGR обратно ошибка: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    // Синхронизация потока: ожидание завершения всех CUDA операций
    cudaStreamSynchronize(m_stream);
    outStride = bgrStride;
    return h_bgr;
}

// Основной процесс: uploadNv12 + processFromDevice
uint8_t* CudaDisplay::process(const uint8_t* yPlane, const uint8_t* uvPlane,
                               int srcStrideY, int srcStrideUV,
                               int srcW, int srcH, int dstW, int dstH,
                               int& outStride) {
    if (!uploadNv12(yPlane, uvPlane, srcStrideY, srcStrideUV, srcW, srcH))
        return nullptr;
    return processFromDevice(srcW, srcH, dstW, dstH, outStride);
}
