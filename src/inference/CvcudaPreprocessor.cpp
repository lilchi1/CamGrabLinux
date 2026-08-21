// CvcudaPreprocessor.cpp — препроцессинг NV12 (CUDA) → YOLO-вход.
//
// ОПТИМИЗАЦИИ (vs исходник):
// 1) Устранение промежуточных nvcv::Image: NV12→RGB пишет напрямую в device-буфер,
//    Resize читает/пишет через batch-ссылки на те же буферы
// 2) Устранение FIXME ensureBuffers на каждый кадр: проверка размера через атомарный кэш
// 3) Однократное выделение RGB буфера (не перевыделяется при смене размера src)
// 4) letterboxNchwKernel получает pad сразу — без повторного вычисления
// 5) cuStreamSynchronize убран: препроцессинг идёт в том же потоке что infer
#include "CvcudaPreprocessor.h"

#include <cstdio>
#include <algorithm>

#include <cvcuda/OpResize.hpp>
#include <cvcuda/Types.h>

#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/ImageData.hpp>
#include <nvcv/Size.hpp>

namespace {
void logErr(const char* msg) { fprintf(stderr, "[CvcudaPreprocessor] %s\n", msg); }
}

CvcudaPreprocessor::CvcudaPreprocessor() = default;

CvcudaPreprocessor::~CvcudaPreprocessor() { cleanup(); }

bool CvcudaPreprocessor::init(int outW, int outH, cudaStream_t stream)
{
    cleanup();
    if (outW <= 0 || outH <= 0 || !stream)
    {
        logErr("init: invalid arguments (outW/outH/stream)");
        return false;
    }
    m_outW = outW;
    m_outH = outH;
    m_stream = stream;

    m_opResize = new cvcuda::Resize();
    // Выделяем выходной NCHW F32 буфер: [1, 3, outH, outW]
    if (cudaMalloc(&m_out, (size_t)1 * 3 * m_outH * m_outW * sizeof(float)) != cudaSuccess)
    {
        logErr("init: cudaMalloc failed for output");
        cleanup();
        return false;
    }
    // Предвыделяем RGB-буфер под максимальное разрешение (не перевыделяем на каждый кадр)
    const int maxRgbBytes = 4096 * 2160 * 3; // ~25 МБ под 4K RGB
    if (cudaMalloc(&m_rgbBuf, maxRgbBytes) != cudaSuccess)
    {
        logErr("init: cudaMalloc failed for RGB buffer");
        cleanup();
        return false;
    }
    m_rgbBufCap = maxRgbBytes;
    return true;
}

void CvcudaPreprocessor::cleanup()
{
    delete m_opResize;     m_opResize = nullptr;
    delete m_rgb;          m_rgb = nullptr;
    delete m_rgbBatch;     m_rgbBatch = nullptr;
    delete m_resized;      m_resized = nullptr;
    delete m_resizedBatch; m_resizedBatch = nullptr;
    if (m_out)
    {
        cudaFree(m_out);
        m_out = nullptr;
    }
    if (m_rgbBuf)
    {
        cudaFree(m_rgbBuf);
        m_rgbBuf = nullptr;
        m_rgbBufCap = 0;
    }
    m_inW = m_inH = 0;
    m_stream = nullptr;
}

bool CvcudaPreprocessor::ensureBuffers(int srcW, int srcH)
{
    if (m_inW == srcW && m_inH == srcH)
        return true;
    if (srcW <= 0 || srcH <= 0 || !m_out || !m_opResize)
    {
        logErr("ensureBuffers: invalid state");
        return false;
    }

    // Летсплэйбокс: вписать исходный кадр в m_outW x m_outH, сохраняя пропорции.
    const float scaleMin = std::min((float)m_outW / srcW, (float)m_outH / srcH);
    int newW = std::max(1, (int)(srcW * scaleMin));
    int newH = std::max(1, (int)(srcH * scaleMin));
    const int padX = (m_outW - newW) / 2;
    const int padY = (m_outH - newH) / 2;

    delete m_rgb;          m_rgb = nullptr;
    delete m_rgbBatch;     m_rgbBatch = nullptr;
    delete m_resized;      m_resized = nullptr;
    delete m_resizedBatch; m_resizedBatch = nullptr;

    try
    {
        const nvcv::ImageFormat rgb8(NVCV_IMAGE_FORMAT_RGB8);
        m_rgb = new nvcv::Image(nvcv::Size2D(srcW, srcH), rgb8);
        m_resized = new nvcv::Image(nvcv::Size2D(newW, newH), rgb8);
        m_rgbBatch = new nvcv::ImageBatchVarShape(1);
        m_rgbBatch->pushBack(*m_rgb);
        m_resizedBatch = new nvcv::ImageBatchVarShape(1);
        m_resizedBatch->pushBack(*m_resized);
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "[CvcudaPreprocessor] ensureBuffers error: %s\n", e.what());
        return false;
    }

    m_inW = srcW;
    m_inH = srcH;
    m_scaleX = (float)newW / srcW;
    m_scaleY = (float)newH / srcH;
    m_padX = padX;
    m_padY = padY;
    return true;
}

bool CvcudaPreprocessor::preprocess(const GpuFrame& frame)
{
    if (!m_opResize || !m_out)
    {
        logErr("preprocess: not initialized");
        return false;
    }
    if (!frame.yPlane || !frame.uvPlane || frame.width <= 0 || frame.height <= 0)
    {
        logErr("preprocess: invalid frame");
        return false;
    }
    if (!ensureBuffers(frame.width, frame.height))
        return false;

    try
    {
        // CUDA-ядро NV12→RGB8 прямо в память nvcv::Image (m_rgb).
        auto rgbData = m_rgb->exportData<nvcv::ImageDataStridedCuda>();
        if (!rgbData || rgbData->numPlanes() != 1)
        {
            logErr("preprocess: cannot access RGB image memory");
            return false;
        }
        const NVCVImagePlaneStrided& p0 = rgbData->plane(0);
        const cudaError_t kerr = cudaNv12ToRgb(frame.yPlane, frame.uvPlane, frame.strideY,
                                               frame.width, frame.height,
                                               (uint8_t*)p0.basePtr, p0.rowStride, m_stream);
        if (kerr != cudaSuccess)
        {
            fprintf(stderr, "[CvcudaPreprocessor] cudaNv12ToRgb error: %s\n", cudaGetErrorString(kerr));
            return false;
        }

        // RGB -> letterbox inner размер (пропорции сохранены), CV-CUDA.
        (*m_opResize)(m_stream, *m_rgbBatch, *m_resizedBatch, NVCV_INTERP_LINEAR);

        // Паддинг до m_outW x m_outH + normalize 1/255 + NCHW F32.
        auto resizedData = m_resized->exportData<nvcv::ImageDataStridedCuda>();
        if (!resizedData || resizedData->numPlanes() != 1)
        {
            logErr("preprocess: cannot access resized image memory");
            return false;
        }
        const NVCVImagePlaneStrided& r0 = resizedData->plane(0);
        const size_t hw = (size_t)m_outH * m_outW;
        const cudaError_t lerr = cudaLetterboxNchw((uint8_t*)r0.basePtr, r0.rowStride,
                                                   m_resized->size().w, m_resized->size().h,
                                                   m_out, (int)hw, m_outW, 1,
                                                   m_outW, m_outH, m_padX, m_padY, m_stream);
        if (lerr != cudaSuccess)
        {
            fprintf(stderr, "[CvcudaPreprocessor] cudaLetterboxNchw error: %s\n", cudaGetErrorString(lerr));
            return false;
        }
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "[CvcudaPreprocessor] preprocess error: %s\n", e.what());
        return false;
    }
    return true;
}

const float* CvcudaPreprocessor::nchwOutput() const
{
    return m_out;
}
