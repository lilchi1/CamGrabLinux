// CvcudaPreprocessor.h — Препроцессинг кадра NV12 (CUDA-память) для YOLO на CV-CUDA.
//
// Пайплайн (всё на GPU, один CUDA-поток):
//   NV12 (device) → [CUDA-ядро] → RGB8 → Resize (letterbox inner, CV-CUDA) →
//   [CUDA-ядро] → letterbox-паддинг + normalize 1/255 + NCHW float [1,3,H,W]
//
// ОПТИМИЗАЦИИ:
//  - Предвыделенный RGB-буфер (не перевыделяется на каждый кадр)
//  - cuStreamSynchronize убран (препроцессинг в том же потоке что infer)
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

    bool init(int outW, int outH, cudaStream_t stream);
    void cleanup();
    bool preprocess(const GpuFrame& frame);
    const float* nchwOutput() const;

    float scaleX() const { return m_scaleX; }
    float scaleY() const { return m_scaleY; }
    int padX() const { return m_padX; }
    int padY() const { return m_padY; }
    int inWidth() const { return m_inW; }
    int inHeight() const { return m_inH; }

private:
    bool ensureBuffers(int srcW, int srcH);

    int m_inW = 0, m_inH = 0;
    int m_outW = 0, m_outH = 0;
    cudaStream_t m_stream = nullptr;

    cvcuda::Resize* m_opResize = nullptr;

    nvcv::Image* m_rgb = nullptr;
    nvcv::ImageBatchVarShape* m_rgbBatch = nullptr;
    nvcv::Image* m_resized = nullptr;
    nvcv::ImageBatchVarShape* m_resizedBatch = nullptr;
    float* m_out = nullptr;          // NCHW F32 [1,3,outH,outW]
    uint8_t* m_rgbBuf = nullptr;     // Предвыделенный RGB буфер
    size_t m_rgbBufCap = 0;          // Ёмкость RGB буфера

    float m_scaleX = 1.0f;
    float m_scaleY = 1.0f;
    int m_padX = 0, m_padY = 0;
};
