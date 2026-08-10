// CudaKernels.cu — CUDA GPU-ядра для конвертации NV12→BGRA + билинейное масштабирование.
#include <cuda_runtime.h>
#include <cstdint>

#include "YoloPostprocess.h"

// Коэффициенты BT.601 limited-range (целочисленная арифметика)
// R = 298*(Y-16) + 409*(V-128) + 128 >> 8
// G = 298*(Y-16) - 100*(U-128) - 208*(V-128) + 128 >> 8
// B = 298*(Y-16) + 516*(U-128) + 128 >> 8

// Ограничение значения до диапазона [0, 255]
__device__ __forceinline__ int clamp255(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

// Конвертация одного NV12 пикселя (Y, U, V) в BGR
__device__ __forceinline__ void nv12ToBgrPixel(uint8_t yVal, uint8_t uVal, uint8_t vVal,
                                                uint8_t& bOut, uint8_t& gOut, uint8_t& rOut) {
    int C  = (int)yVal - 16;
    int D  = (int)uVal - 128;
    int E  = (int)vVal - 128;
    int yF = 298 * C;
    rOut = (uint8_t)clamp255((yF + 409 * E + 128) >> 8);
    gOut = (uint8_t)clamp255((yF - 100 * D - 208 * E + 128) >> 8);
    bOut = (uint8_t)clamp255((yF + 516 * D + 128) >> 8);
}

// Билинейная интерполяция Y-плоскости NV12
__device__ __forceinline__ uint8_t sampleY_bilinear(const uint8_t* yPlane, int srcStride,
                                                     float fx, float fy, int srcW, int srcH) {
    int x0 = (int)fx, y0 = (int)fy;
    int x1 = x0 + 1, y1 = y0 + 1;
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    if (x1 >= srcW) x1 = srcW - 1;
    if (y1 >= srcH) y1 = srcH - 1;
    float wx = fx - (int)fx, wy = fy - (int)fy;
    float v00 = yPlane[y0 * srcStride + x0];
    float v10 = yPlane[y0 * srcStride + x1];
    float v01 = yPlane[y1 * srcStride + x0];
    float v11 = yPlane[y1 * srcStride + x1];
    return (uint8_t)(v00 * (1-wx) * (1-wy) + v10 * wx * (1-wy) +
                     v01 * (1-wx) * wy + v11 * wx * wy + 0.5f);
}

// Билинейная интерполяция UV-плоскости NV12 (чередующиеся U,V)
__device__ __forceinline__ void sampleUV_bilinear(const uint8_t* uvPlane, int srcStride,
                                                   float fx, float fy, int srcW, int srcH,
                                                   uint8_t& uOut, uint8_t& vOut) {
    // UV-плоскость имеет половинное разрешение, координаты маппятся на полупиксельные позиции
    float hfx = fx * 0.5f, hfy = fy * 0.5f;
    int x0 = (int)hfx, y0 = (int)hfy;
    int x1 = x0 + 1, y1 = y0 + 1;
    int uvW2 = srcW / 2, uvH2 = srcH / 2;
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    if (x1 >= uvW2) x1 = uvW2 - 1;
    if (y1 >= uvH2) y1 = uvH2 - 1;
    float wx = hfx - (int)hfx, wy = hfy - (int)hfy;

    float u00 = uvPlane[y0 * srcStride + x0 * 2];
    float v00 = uvPlane[y0 * srcStride + x0 * 2 + 1];
    float u10 = uvPlane[y0 * srcStride + x1 * 2];
    float v10 = uvPlane[y0 * srcStride + x1 * 2 + 1];
    float u01 = uvPlane[y1 * srcStride + x0 * 2];
    float v01 = uvPlane[y1 * srcStride + x0 * 2 + 1];
    float u11 = uvPlane[y1 * srcStride + x1 * 2];
    float v11 = uvPlane[y1 * srcStride + x1 * 2 + 1];

    uOut = (uint8_t)(u00*(1-wx)*(1-wy) + u10*wx*(1-wy) + u01*(1-wx)*wy + u11*wx*wy + 0.5f);
    vOut = (uint8_t)(v00*(1-wx)*(1-wy) + v10*wx*(1-wy) + v01*(1-wx)*wy + v11*wx*wy + 0.5f);
}

// Ядро NV12→BGRA с билинейным масштабированием.
// Каждый поток генерирует один BGRA пиксель (4 байта в dst, соответствует XImage ZPixmap 32bpp).
__global__ void nv12ToBgrScaleKernel(const uint8_t* yPlane, const uint8_t* uvPlane,
                                      int srcStride, int srcW, int srcH,
                                      uint8_t* dst, int dstStride, int dstW, int dstH) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dstW || dy >= dstH) return;

    // Вычисление координат исходного пикселя с учётом масштабирования
    float fx = (float)dx * srcW / dstW;
    float fy = (float)dy * srcH / dstH;

    // Билинейная выборка Y и UV из NV12 плоскостей
    uint8_t yVal = sampleY_bilinear(yPlane, srcStride, fx, fy, srcW, srcH);
    uint8_t uVal, vVal;
    sampleUV_bilinear(uvPlane, srcStride, fx, fy, srcW, srcH, uVal, vVal);

    // Конвертация NV12 → BGR
    uint8_t b, g, r;
    nv12ToBgrPixel(yVal, uVal, vVal, b, g, r);

    // Запись BGRA в выходной буфер (4 байта на пиксель)
    int outIdx = dy * dstStride + dx * 4;
    dst[outIdx + 0] = b;
    dst[outIdx + 1] = g;
    dst[outIdx + 2] = r;
    dst[outIdx + 3] = 0;
}

// Ядро NV12→BGRA без масштабирования (размер src == dst), каждый поток обрабатывает блок 2×2
__global__ void nv12ToBgrKernel(const uint8_t* yPlane, const uint8_t* uvPlane,
                                 int srcStride, int width, int height,
                                 uint8_t* dst, int dstStride) {
    // Каждый поток обрабатывает блок 2×2 пикселя
    int bx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int by = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (bx >= width || by >= height) return;

    // UV-индекс для блока 2×2
    int uvIdx = (by / 2) * srcStride + (bx / 2) * 2;
    uint8_t uVal = uvPlane[uvIdx];
    uint8_t vVal = uvPlane[uvIdx + 1];

    for (int dy = 0; dy < 2 && (by + dy) < height; dy++) {
        for (int dx = 0; dx < 2 && (bx + dx) < width; dx++) {
            uint8_t yVal = yPlane[(by + dy) * srcStride + bx + dx];
            uint8_t b, g, r;
            nv12ToBgrPixel(yVal, uVal, vVal, b, g, r);
            int outIdx = (by + dy) * dstStride + (bx + dx) * 4;
            dst[outIdx + 0] = b;
            dst[outIdx + 1] = g;
            dst[outIdx + 2] = r;
            dst[outIdx + 3] = 0;
        }
    }
}

// Ядро NV12→RGB8 (interleaved, без масштабирования), 2×2 блок на поток.
// Используется для препроцессинга YOLO (CV-CUDA не умеет напрямую читать NV12).
__global__ void nv12ToRgbKernel(const uint8_t* yPlane, const uint8_t* uvPlane,
                                int srcStride, int width, int height,
                                uint8_t* dst, int dstStride) {
    int bx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int by = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (bx >= width || by >= height) return;

    int uvIdx = (by / 2) * srcStride + (bx / 2) * 2;
    uint8_t uVal = uvPlane[uvIdx];
    uint8_t vVal = uvPlane[uvIdx + 1];

    for (int dy = 0; dy < 2 && (by + dy) < height; dy++) {
        for (int dx = 0; dx < 2 && (bx + dx) < width; dx++) {
            uint8_t yVal = yPlane[(by + dy) * srcStride + bx + dx];
            uint8_t b, g, r;
            nv12ToBgrPixel(yVal, uVal, vVal, b, g, r);
            int outIdx = ((by + dy) * dstStride + (bx + dx) * 3);
            dst[outIdx + 0] = r;
            dst[outIdx + 1] = g;
            dst[outIdx + 2] = b;
        }
    }
}

// Ядро letterbox-паддинг + normalize: RGB8 (rgbW x rgbH) → NCHW F32 [1,3,outH,outW].
// Контент размещается с отступом (padX, padY), фон — значение 114 (как в YOLO),
// масштаб 1/255. Строки/каналы индексируются через явные шаги (элементы).
__global__ void letterboxNchwKernel(const uint8_t* rgb, int rgbStride, int rgbW, int rgbH,
                                    float* dst, int cStride, int hStride, int wStride,
                                    int outW, int outH, int padX, int padY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outW || y >= outH) return;

    const float border = 114.0f / 255.0f;
    float r, g, b;
    int sx = x - padX, sy = y - padY;
    if (sx >= 0 && sx < rgbW && sy >= 0 && sy < rgbH) {
        const uint8_t* p = rgb + sy * rgbStride + sx * 3;
        r = p[0] / 255.0f; g = p[1] / 255.0f; b = p[2] / 255.0f;
    } else {
        r = g = b = border;
    }
    int base = y * hStride + x * wStride;
    dst[cStride + base] = r;
    dst[2 * cStride + base] = g;
    dst[0 * cStride + base] = b;
}

// Ядро декода YOLOv8/v11/v12 (anchor-free): NCHW [1, (4+nc), anchors].
// Каждый поток обрабатывает один anchor. Выбирает лучший класс, фильтрует по
// порогу и маппит бокс из координат входа модели в исходный кадр.
__device__ __forceinline__ void yoloBestClass(const float* output, int anchor, int anchors,
                                              int channels, int& cls, float& score) {
    score = -1.0f;
    cls = -1;
    for (int c = 4; c < channels; c++) {
        float s = output[c * anchors + anchor];
        if (s > score) { score = s; cls = c - 4; }
    }
}

__global__ void yoloDecodeKernel(const float* output, int channels, int anchors,
                                 float confThresh, float scaleX, float scaleY, int padX, int padY,
                                 int inW, int inH, YoloCandidate* cands, int* counter, int maxCands) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= anchors) return;

    float cx = output[0 * anchors + a];
    float cy = output[1 * anchors + a];
    float w  = output[2 * anchors + a];
    float h  = output[3 * anchors + a];
    if (w <= 0.0f || h <= 0.0f) return;

    int cls;
    float score;
    yoloBestClass(output, a, anchors, channels, cls, score);
    if (cls < 0 || score < confThresh) return;

    float sx1 = (cx - w * 0.5f - padX) / scaleX;
    float sy1 = (cy - h * 0.5f - padY) / scaleY;
    float sx2 = (cx + w * 0.5f - padX) / scaleX;
    float sy2 = (cy + h * 0.5f - padY) / scaleY;
    if (sx1 < 0.0f) sx1 = 0.0f;
    if (sy1 < 0.0f) sy1 = 0.0f;
    if (sx2 > (float)inW) sx2 = (float)inW;
    if (sy2 > (float)inH) sy2 = (float)inH;
    if (sx2 <= sx1 || sy2 <= sy1) return;

    int idx = atomicAdd(counter, 1);
    if (idx >= maxCands) return;
    cands[idx] = {sx1, sy1, sx2, sy2, score, cls};
}

// ─── Декод YOLOv2 (anchor-based, YAD2K/darknet): NCHW [1, C, grid, grid] ─────
// C = numAnchors * (5 + numClasses); для модели 608x608: grid=19, C=425, stride=32.
// Каждый поток обрабатывает один якорь одной клетки сетки:
//   cx = (sigmoid(tx) + gx) * stride,  cy = (sigmoid(ty) + gy) * stride
//   w  = anchor_w * exp(tw),           h  = anchor_h * exp(th)
//   score = sigmoid(obj) * max_class(sigmoid(cls))
// Якоря (dAnchors) задаются парами (w,h) в пикселях входа модели.
__global__ void yoloV2DecodeKernel(const float* output, int grid, int numAnchors,
                                   int numClasses, const float* dAnchors, int stride,
                                   float confThresh, float scaleX, float scaleY,
                                   int padX, int padY, int inW, int inH,
                                   YoloCandidate* cands, int* counter, int maxCands) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int total = grid * grid * numAnchors;
    if (k >= total) return;

    int a = k % numAnchors;         // якорь
    int gc = k / numAnchors;        // клетка
    int gx = gc % grid;
    int gy = gc / grid;

    const int hw = grid * grid;
    const int base = gy * grid + gx;   // каналы-major: index = c*hw + base
    // Каналы сгруппированы по якорям: блок якоря a начинается с a*(5+numClasses).
    const int off = a * (5 + numClasses);

    const float tx = output[(0 + off) * hw + base];
    const float ty = output[(1 + off) * hw + base];
    const float tw = output[(2 + off) * hw + base];
    const float th = output[(3 + off) * hw + base];
    const float to = output[(4 + off) * hw + base];

    const float so = 1.0f / (1.0f + expf(-to));
    if (so < confThresh) return;

    float cx = (1.0f / (1.0f + expf(-tx)) + gx) * stride;
    float cy = (1.0f / (1.0f + expf(-ty)) + gy) * stride;
    float w  = dAnchors[2 * a]     * expf(tw);
    float h  = dAnchors[2 * a + 1] * expf(th);

    int cls = -1;
    float best = -1.0f;
    for (int c = 5; c < 5 + numClasses; c++) {
        float s = 1.0f / (1.0f + expf(-output[(c + off) * hw + base]));
        if (s > best) { best = s; cls = c - 5; }
    }
    float score = so * best;
    if (cls < 0 || score < confThresh) return;

    float sx1 = (cx - w * 0.5f - padX) / scaleX;
    float sy1 = (cy - h * 0.5f - padY) / scaleY;
    float sx2 = (cx + w * 0.5f - padX) / scaleX;
    float sy2 = (cy + h * 0.5f - padY) / scaleY;
    if (sx1 < 0.0f) sx1 = 0.0f;
    if (sy1 < 0.0f) sy1 = 0.0f;
    if (sx2 > (float)inW) sx2 = (float)inW;
    if (sy2 > (float)inH) sy2 = (float)inH;
    if (sx2 <= sx1 || sy2 <= sy1) return;

    int idx = atomicAdd(counter, 1);
    if (idx >= maxCands) return;
    cands[idx] = {sx1, sy1, sx2, sy2, score, cls};
}

// ─── C-обёртки (вызываются из CudaDisplay.cpp) ───────────────────────────────

extern "C" {

// Масштабирование + конвертация NV12→BGRA
cudaError_t cudaNv12ToBgrScale(const uint8_t* yPlane, const uint8_t* uvPlane,
                                int srcStride, int srcW, int srcH,
                                uint8_t* d_dst, int dstStride, int dstW, int dstH,
                                cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);
    nv12ToBgrScaleKernel<<<grid, block, 0, stream>>>(yPlane, uvPlane, srcStride, srcW, srcH,
                                                      d_dst, dstStride, dstW, dstH);
    return cudaGetLastError();
}

// Конвертация NV12→BGRA без масштабирования
cudaError_t cudaNv12ToBgr(const uint8_t* yPlane, const uint8_t* uvPlane,
                           int srcStride, int width, int height,
                           uint8_t* d_dst, int dstStride, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid(((width / 2) + block.x - 1) / block.x,
              ((height / 2) + block.y - 1) / block.y);
    nv12ToBgrKernel<<<grid, block, 0, stream>>>(yPlane, uvPlane, srcStride, width, height,
                                                 d_dst, dstStride);
    return cudaGetLastError();
}

// Конвертация NV12→RGB8 (interleaved) без масштабирования
cudaError_t cudaNv12ToRgb(const uint8_t* yPlane, const uint8_t* uvPlane,
                          int srcStride, int width, int height,
                          uint8_t* d_dst, int dstStride, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid(((width / 2) + block.x - 1) / block.x,
              ((height / 2) + block.y - 1) / block.y);
    nv12ToRgbKernel<<<grid, block, 0, stream>>>(yPlane, uvPlane, srcStride, width, height,
                                                d_dst, dstStride);
    return cudaGetLastError();
}

// Letterbox-паддинг + normalize → NCHW F32
cudaError_t cudaLetterboxNchw(const uint8_t* rgb, int rgbStride, int rgbW, int rgbH,
                              float* d_out, int cStride, int hStride, int wStride,
                              int outW, int outH, int padX, int padY, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);
    letterboxNchwKernel<<<grid, block, 0, stream>>>(rgb, rgbStride, rgbW, rgbH,
                                                    d_out, cStride, hStride, wStride,
                                                    outW, outH, padX, padY);
    return cudaGetLastError();
}

// Декод YOLO-выхода → кандидаты в координатах исходного кадра
cudaError_t cudaYoloDecode(const float* output, int channels, int anchors,
                           float confThresh, float scaleX, float scaleY, int padX, int padY,
                           int inW, int inH, YoloCandidate* d_cands, int* d_counter, int maxCands,
                           cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (anchors + threads - 1) / threads;
    yoloDecodeKernel<<<blocks, threads, 0, stream>>>(output, channels, anchors, confThresh,
                                                     scaleX, scaleY, padX, padY, inW, inH,
                                                     d_cands, d_counter, maxCands);
    return cudaGetLastError();
}

// Декод YOLOv2-выхода (NCHW [1, C, grid, grid]) → кандидаты в исходном кадре
cudaError_t cudaYoloV2Decode(const float* output, int grid, int numAnchors, int numClasses,
                             const float* dAnchors, int stride, float confThresh,
                             float scaleX, float scaleY, int padX, int padY,
                             int inW, int inH, YoloCandidate* d_cands, int* d_counter,
                             int maxCands, cudaStream_t stream) {
    const int total = grid * grid * numAnchors;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    yoloV2DecodeKernel<<<blocks, threads, 0, stream>>>(output, grid, numAnchors, numClasses,
                                                       dAnchors, stride, confThresh,
                                                       scaleX, scaleY, padX, padY, inW, inH,
                                                       d_cands, d_counter, maxCands);
    return cudaGetLastError();
}

} // extern "C"
