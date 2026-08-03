// CudaKernels.cu — CUDA GPU-ядра для конвертации NV12→BGRA + билинейное масштабирование.
#include <cuda_runtime.h>
#include <cstdint>

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

} // extern "C"
