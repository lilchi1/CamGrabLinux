// CvcudaPreprocessor.h — Препроцессинг кадра NV12 (CUDA-память) для YOLO на CV-CUDA.
//
// Пайплайн (всё на GPU, один CUDA-поток):
//   NV12 (device) → [CUDA-ядро] → RGB8 → Resize (letterbox inner, CV-CUDA) →
//   [CUDA-ядро] → letterbox-паддинг + normalize 1/255 + NCHW float [1,3,H,W]
// NV12→RGB и финальный паддинг — собственные ядра (CV-CUDA CvtColor не читает
// полупланарный NV12, а var-shape pad-операторы этой сборки дают неверный результат;
// Resize от CV-CUDA проверен и работает). Результат — device-указатель для TensorRT.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "FrameCallback.h"

namespace cvcuda {
class Resize;
}

namespace nvcv {
class Image;
class ImageBatchVarShape;
}

// CUDA-ядро NV12→RGB8 (interleaved), объявлено в CudaKernels.cu.
extern "C" cudaError_t cudaNv12ToRgb(const uint8_t* yPlane, const uint8_t* uvPlane,
                                     int srcStride, int width, int height,
                                     uint8_t* d_dst, int dstStride, cudaStream_t stream);

// CUDA-ядро letterbox-паддинг + normalize: RGB8 (newW x newH) → NCHW F32 [1,3,outH,outW].
extern "C" cudaError_t cudaLetterboxNchw(const uint8_t* rgb, int rgbStride, int rgbW, int rgbH,
                                         float* d_out, int cStride, int hStride, int wStride,
                                         int outW, int outH, int padX, int padY,
                                         cudaStream_t stream);

class CvcudaPreprocessor {
public:
    CvcudaPreprocessor();
    ~CvcudaPreprocessor();

    // Инициализация препроцессинга под выходной размер модели (например 640x640)
    // и CUDA-поток, на котором будет выполняться вся работа.
    bool init(int outW, int outH, cudaStream_t stream);

    // Освобождение всех ресурсов.
    void cleanup();

    // Препроцессинг одного кадра (NV12 в CUDA-памяти). Летсплэйбокс (aspect
    // сохраняется) → RGB → нормализация 1/255 → NCHW float.
    // Возвращает false при ошибке. Выполняется асинхронно на m_stream.
    bool preprocess(const GpuFrame& frame);

    // Указатель на результат: NCHW float [1,3,outH,outW] в device-памяти.
    // Действителен после preprocess() на этом же потоке.
    const float* nchwOutput() const;

    // Параметры летсплэйбокса для обратного маппинга координат:
    //   srcX = (modelX - padX) / scaleX, srcY = (modelY - padY) / scaleY
    float scaleX() const { return m_scaleX; }
    float scaleY() const { return m_scaleY; }
    int padX() const { return m_padX; }
    int padY() const { return m_padY; }

    // Текущее разрешение исходного кадра (обновляется при смене размера).
    int inWidth() const { return m_inW; }
    int inHeight() const { return m_inH; }

private:
    // Пересоздание промежуточных образов/тензоров при смене размера кадра.
    bool ensureBuffers(int srcW, int srcH);

    int m_inW = 0, m_inH = 0;       // текущий размер исходного NV12
    int m_outW = 0, m_outH = 0;     // размер входа модели
    cudaStream_t m_stream = nullptr;

    cvcuda::Resize* m_opResize = nullptr;

    nvcv::Image* m_rgb = nullptr;                      // RGB (W x H), пишет CUDA-ядро
    nvcv::ImageBatchVarShape* m_rgbBatch = nullptr;
    nvcv::Image* m_resized = nullptr;                  // RGB letterbox-внутренний (newW x newH)
    nvcv::ImageBatchVarShape* m_resizedBatch = nullptr;
    float* m_out = nullptr;                            // NCHW F32 [1,3,outH,outW] (cudaMalloc)

    float m_scaleX = 1.0f;   // масштаб source→model по X
    float m_scaleY = 1.0f;   // масштаб source→model по Y
    int m_padX = 0, m_padY = 0; // летсплэйбокс-отступы (px модели)
};
