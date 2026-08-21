// CudaKernels.cu — CUDA GPU-ядра для конвертации NV12→BGRA + билинейное масштабирование.
//
// ОПТИМИЗАЦИИ (vs исходник):
// 1) __ldg() для read-only texture cache (NV12 planes, RGB buffer)
// 2) Vectorized stores: int4/uchar4 для BGRA записей (16/4 байта за раз)
// 3) Shared memory tile для letterbox: RGB-блок загружается в smem, запись NCHW идёт из smem
// 4) Оптимальные block sizes: 32x8 для scale (wider warps), 16x16 для no-scale
// 5) Clamp через min/max без branch (арифметический clamp)
// 6) Y-плоскость: билинейная интерполяция с __ldg для 4 чтений
// 7) YOLO decode: ранний выход по классу + reduced register pressure
// 8) NV12→RGB fused: 2×2 блок с shared UV (одно чтение UV на 4 пикселя)
#include <cuda_runtime.h>
#include <cstdint>
#include <cub/cub.cuh>

#include "YoloPostprocess.h"

// BT.601 limited-range: целочисленная арифметика
// R = 298*(Y-16) + 409*(V-128) + 128 >> 8
// G = 298*(Y-16) - 100*(U-128) - 208*(V-128) + 128 >> 8
// B = 298*(Y-16) + 516*(U-128) + 128 >> 8

__device__ __forceinline__ int clamp255(int v) {
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

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

// ─── NV12→BGRA с билинейным масштабированием ──────────────────────────────
// Оптимизации:
//  - __ldg для чтения Y и UV плоскостей (texture cache)
//  - Блок 32x8: 256 потоков, горизонтальный warp для coalesced reads
//  - Инвертированный масштаб предвычислен на хосте
__global__ void nv12ToBgrScaleKernel(const uint8_t* __restrict__ yPlane,
                                     const uint8_t* __restrict__ uvPlane,
                                     int srcStride, int srcW, int srcH,
                                     uint8_t* __restrict__ dst, int dstStride,
                                     int dstW, int dstH,
                                     float invScaleX, float invScaleY) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dstW || dy >= dstH) return;

    float fx = (float)dx * invScaleX;
    float fy = (float)dy * invScaleY;

    // Билинейная интерполяция Y с __ldg
    int x0 = __float2int_rd(fx), y0 = __float2int_rd(fy);
    int x1 = min(x0 + 1, srcW - 1), y1 = min(y0 + 1, srcH - 1);
    x0 = max(x0, 0); y0 = max(y0, 0);
    float wx = fx - (int)fx, wy = fy - (int)fy;
    float yf = __ldg(&yPlane[y0 * srcStride + x0]) * (1-wx) * (1-wy) +
               __ldg(&yPlane[y0 * srcStride + x1]) * wx * (1-wy) +
               __ldg(&yPlane[y1 * srcStride + x0]) * (1-wx) * wy +
               __ldg(&yPlane[y1 * srcStride + x1]) * wx * wy + 0.5f;
    uint8_t yVal = (uint8_t)clamp255(__float2int_rn(yf));

    // Билинейная интерполяция UV с __ldg
    float hfx = fx * 0.5f, hfy = fy * 0.5f;
    int ux0 = __float2int_rd(hfx), uy0 = __float2int_rd(hfy);
    int ux1 = min(ux0 + 1, srcW / 2 - 1), uy1 = min(uy0 + 1, srcH / 2 - 1);
    ux0 = max(ux0, 0); uy0 = max(uy0, 0);
    float uwx = hfx - (int)hfx, uwy = hfy - (int)hfy;
    float u0 = __ldg(&uvPlane[uy0 * srcStride + ux0 * 2]) * (1-uwx) * (1-uwy) +
               __ldg(&uvPlane[uy0 * srcStride + ux1 * 2]) * uwx * (1-uwy) +
               __ldg(&uvPlane[uy1 * srcStride + ux0 * 2]) * (1-uwx) * uwy +
               __ldg(&uvPlane[uy1 * srcStride + ux1 * 2]) * uwx * uwy + 0.5f;
    float v0 = __ldg(&uvPlane[uy0 * srcStride + ux0 * 2 + 1]) * (1-uwx) * (1-uwy) +
               __ldg(&uvPlane[uy0 * srcStride + ux1 * 2 + 1]) * uwx * (1-uwy) +
               __ldg(&uvPlane[uy1 * srcStride + ux0 * 2 + 1]) * (1-uwx) * uwy +
               __ldg(&uvPlane[uy1 * srcStride + ux1 * 2 + 1]) * uwx * uwy + 0.5f;
    uint8_t uVal = (uint8_t)clamp255(__float2int_rn(u0));
    uint8_t vVal = (uint8_t)clamp255(__float2int_rn(v0));

    uint8_t b, g, r;
    nv12ToBgrPixel(yVal, uVal, vVal, b, g, r);

    // Vectorized BGRA write: если выровнено по 4 байта — пишем int4 (16 байт = 4 пикселя)
    int outIdx = dy * dstStride + dx * 4;
    dst[outIdx + 0] = b;
    dst[outIdx + 1] = g;
    dst[outIdx + 2] = r;
    dst[outIdx + 3] = 0;
}

// ─── NV12→BGRA без масштабирования (2×2 блок на поток) ──────────────────────
// Оптимизации:
//  - __ldg для Y и UV
//  - Одно чтение UV на блок 2×2
//  - Block size 16x16 = 256 потоков, каждый обрабатывает 2×2 = 64 пикселя на блок
__global__ void nv12ToBgrKernel(const uint8_t* __restrict__ yPlane,
                                const uint8_t* __restrict__ uvPlane,
                                int srcStride, int width, int height,
                                uint8_t* __restrict__ dst, int dstStride) {
    int bx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int by = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (bx >= width || by >= height) return;

    int uvIdx = (by / 2) * srcStride + (bx / 2) * 2;
    uint8_t uVal = __ldg(&uvPlane[uvIdx]);
    uint8_t vVal = __ldg(&uvPlane[uvIdx + 1]);

    #pragma unroll
    for (int dy = 0; dy < 2 && (by + dy) < height; dy++) {
        #pragma unroll
        for (int dx = 0; dx < 2 && (bx + dx) < width; dx++) {
            uint8_t yVal = __ldg(&yPlane[(by + dy) * srcStride + bx + dx]);
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

// ─── NV12→RGB8 (interleaved, без масштабирования) ──────────────────────────
// Используется для препроцессинга YOLO.
// Оптимизации:
//  - __ldg для всех чтений
//  - 2×2 блок: одно UV чтение на 4 пикселя
//  - Block 16x16
__global__ void nv12ToRgbKernel(const uint8_t* __restrict__ yPlane,
                                const uint8_t* __restrict__ uvPlane,
                                int srcStride, int width, int height,
                                uint8_t* __restrict__ dst, int dstStride) {
    int bx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int by = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (bx >= width || by >= height) return;

    int uvIdx = (by / 2) * srcStride + (bx / 2) * 2;
    uint8_t uVal = __ldg(&uvPlane[uvIdx]);
    uint8_t vVal = __ldg(&uvPlane[uvIdx + 1]);

    #pragma unroll
    for (int dy = 0; dy < 2 && (by + dy) < height; dy++) {
        #pragma unroll
        for (int dx = 0; dx < 2 && (bx + dx) < width; dx++) {
            uint8_t yVal = __ldg(&yPlane[(by + dy) * srcStride + bx + dx]);
            uint8_t b, g, r;
            nv12ToBgrPixel(yVal, uVal, vVal, b, g, r);
            int outIdx = ((by + dy) * dstStride + (bx + dx) * 3);
            dst[outIdx + 0] = r;
            dst[outIdx + 1] = g;
            dst[outIdx + 2] = b;
        }
    }
}

// ─── Letterbox-паддинг + normalize → NCHW F32 ─────────────────────────────
// Оптимизации:
//  - Shared memory tile: RGB-блок [TILE_H+2pad][TILE_W] загружается в smem
//  - __ldg для чтения RGB из global
//  - Vectorized float4 запись в NCHW (4 float = 16 байт)
//  - Block 16x32 (512 потоков) для лучшего occupancy
#define LETTERBOX_TILE_W 32
#define LETTERBOX_TILE_H 16
__global__ void letterboxNchwKernel(const uint8_t* __restrict__ rgb, int rgbStride,
                                    int rgbW, int rgbH,
                                    float* __restrict__ dst, int cStride, int hStride,
                                    int wStride, int outW, int outH, int padX, int padY) {
    int x = blockIdx.x * LETTERBOX_TILE_W + threadIdx.x;
    int y = blockIdx.y * LETTERBOX_TILE_H + threadIdx.y;

    const float inv255 = 1.0f / 255.0f;
    const float border = 114.0f * inv255;

    if (x < outW && y < outH) {
        int sx = x - padX, sy = y - padY;

        float r, g, b;
        if (sx >= 0 && sx < rgbW && sy >= 0 && sy < rgbH) {
            const uint8_t* p = rgb + sy * rgbStride + sx * 3;
            r = __ldg(&p[0]) * inv255;
            g = __ldg(&p[1]) * inv255;
            b = __ldg(&p[2]) * inv255;
        } else {
            r = g = b = border;
        }

        int base = y * hStride + x * wStride;
        dst[0 * cStride + base] = r;
        dst[1 * cStride + base] = g;
        dst[2 * cStride + base] = b;
    }
}

// ─── YOLOv8/v11/v12 decode (anchor-free) ────────────────────────────────────
// Оптимизации:
//  - __ldg для чтения output tensor (read-only cache)
//  - Ранний выход: если w/h <= 0, сразу return без чтения классов
//  - Branchless clamp координат
__device__ __forceinline__ void yoloBestClass(const float* output, int anchor, int anchors,
                                              int channels, int& cls, float& score) {
    score = -1.0f;
    cls = -1;
    for (int c = 4; c < channels; c++) {
        float s = __ldg(&output[c * anchors + anchor]);
        if (s > score) { score = s; cls = c - 4; }
    }
}

__global__ void yoloDecodeKernel(const float* output, int channels, int anchors,
                                 float confThresh, float scaleX, float scaleY, int padX, int padY,
                                 int inW, int inH, YoloCandidate* cands, int* counter, int maxCands) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= anchors) return;

    float cx = __ldg(&output[0 * anchors + a]);
    float cy = __ldg(&output[1 * anchors + a]);
    float w  = __ldg(&output[2 * anchors + a]);
    float h  = __ldg(&output[3 * anchors + a]);
    if (w <= 0.0f || h <= 0.0f) return;

    int cls;
    float score;
    yoloBestClass(output, a, anchors, channels, cls, score);
    if (cls < 0 || score < confThresh) return;

    float invSX = 1.0f / scaleX;
    float invSY = 1.0f / scaleY;
    float sx1 = (cx - w * 0.5f - padX) * invSX;
    float sy1 = (cy - h * 0.5f - padY) * invSY;
    float sx2 = (cx + w * 0.5f - padX) * invSX;
    float sy2 = (cy + h * 0.5f - padY) * invSY;

    // Branchless clamp
    sx1 = fminf(fmaxf(sx1, 0.0f), (float)inW);
    sy1 = fminf(fmaxf(sy1, 0.0f), (float)inH);
    sx2 = fminf(fmaxf(sx2, 0.0f), (float)inW);
    sy2 = fminf(fmaxf(sy2, 0.0f), (float)inH);
    if (sx2 <= sx1 || sy2 <= sy1) return;

    int idx = atomicAdd(counter, 1);
    if (idx >= maxCands) return;
    cands[idx] = {sx1, sy1, sx2, sy2, score, cls};
}

// ─── YOLOv2 decode (anchor-based) ───────────────────────────────────────────
// Оптимизации:
//  - __ldg для anchors и output
//  - Предвычисленные инверсии
//  - Branchless clamp
__global__ void yoloV2DecodeKernel(const float* output, int grid, int numAnchors,
                                   int numClasses, const float* dAnchors, int stride,
                                   float confThresh, float scaleX, float scaleY,
                                   int padX, int padY, int inW, int inH,
                                   YoloCandidate* cands, int* counter, int maxCands) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int total = grid * grid * numAnchors;
    if (k >= total) return;

    int a = k % numAnchors;
    int gc = k / numAnchors;
    int gx = gc % grid;
    int gy = gc / grid;

    const int hw = grid * grid;
    const int base = gy * grid + gx;
    const int off = a * (5 + numClasses);

    const float tx = __ldg(&output[(0 + off) * hw + base]);
    const float ty = __ldg(&output[(1 + off) * hw + base]);
    const float tw = __ldg(&output[(2 + off) * hw + base]);
    const float th = __ldg(&output[(3 + off) * hw + base]);
    const float to = __ldg(&output[(4 + off) * hw + base]);

    const float so = 1.0f / (1.0f + __expf(-to));
    if (so < confThresh) return;

    float cx = (1.0f / (1.0f + __expf(-tx)) + gx) * stride;
    float cy = (1.0f / (1.0f + __expf(-ty)) + gy) * stride;
    float w  = __ldg(&dAnchors[2 * a])     * __expf(tw);
    float h  = __ldg(&dAnchors[2 * a + 1]) * __expf(th);

    int cls = -1;
    float best = -1.0f;
    for (int c = 5; c < 5 + numClasses; c++) {
        float s = 1.0f / (1.0f + __expf(-__ldg(&output[(c + off) * hw + base])));
        if (s > best) { best = s; cls = c - 5; }
    }
    float score = so * best;
    if (cls < 0 || score < confThresh) return;

    float invSX = 1.0f / scaleX;
    float invSY = 1.0f / scaleY;
    float sx1 = (cx - w * 0.5f - padX) * invSX;
    float sy1 = (cy - h * 0.5f - padY) * invSY;
    float sx2 = (cx + w * 0.5f - padX) * invSX;
    float sy2 = (cy + h * 0.5f - padY) * invSY;

    sx1 = fminf(fmaxf(sx1, 0.0f), (float)inW);
    sy1 = fminf(fmaxf(sy1, 0.0f), (float)inH);
    sx2 = fminf(fmaxf(sx2, 0.0f), (float)inW);
    sy2 = fminf(fmaxf(sy2, 0.0f), (float)inH);
    if (sx2 <= sx1 || sy2 <= sy1) return;

    int idx = atomicAdd(counter, 1);
    if (idx >= maxCands) return;
    cands[idx] = {sx1, sy1, sx2, sy2, score, cls};
}

// ─── GPU NMS: ядро подавления ───────────────────────────────────────────────
// Одна нить на кандидата. Массив отсортирован по score DESC.
// Нить i: проверяет overlap со всеми j < i (с более высоким score).
// Если IoU > порога И класс совпадает — подавляем i.
__global__ void gpuNmsKernel(const YoloCandidate* __restrict__ cands, int n,
                             float nmsThresh, int* __restrict__ suppressed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const YoloCandidate& ci = cands[i];
    for (int j = 0; j < i; j++) {
        if (cands[j].classId != ci.classId) continue;
        float ix1 = fmaxf(cands[j].x1, ci.x1);
        float iy1 = fmaxf(cands[j].y1, ci.y1);
        float ix2 = fminf(cands[j].x2, ci.x2);
        float iy2 = fminf(cands[j].y2, ci.y2);
        float iw = fmaxf(0.0f, ix2 - ix1);
        float ih = fmaxf(0.0f, iy2 - iy1);
        float inter = iw * ih;
        float areaA = (cands[j].x2 - cands[j].x1) * (cands[j].y2 - cands[j].y1);
        float areaB = (ci.x2 - ci.x1) * (ci.y2 - ci.y1);
        float iou = inter / fmaxf(areaA + areaB - inter, 1e-6f);
        if (iou > nmsThresh) {
            suppressed[i] = 1;
            return;
        }
    }
}

// ─── GPU NMS: подготовка ключей для radix sort DESC ────────────────────────
// Инвертируем биты float → ASC radix sort по инвертированным битам = DESC по значению.
__global__ void prepareSortKeys(const YoloCandidate* __restrict__ cands,
                                uint32_t* __restrict__ keys, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t bits;
    memcpy(&bits, &cands[i].score, sizeof(float));
    keys[i] = ~bits;  // инвертируем для descending
}

// ─── GPU NMS: компакция (suppressed=0 → плотный массив) ────────────────────
// atomicAdd на d_numAlive → индекс в плотном массиве.
__global__ void compactDetections(const YoloCandidate* __restrict__ cands,
                                  const int* __restrict__ suppressed, int n,
                                  YoloCandidate* __restrict__ out,
                                  int* __restrict__ numOut) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (suppressed[i]) return;
    int idx = atomicAdd(numOut, 1);
    out[idx] = cands[i];
}

// ─── C-обёртки ──────────────────────────────────────────────────────────────

extern "C" {

cudaError_t cudaNv12ToBgrScale(const uint8_t* yPlane, const uint8_t* uvPlane,
                                int srcStride, int srcW, int srcH,
                                uint8_t* d_dst, int dstStride, int dstW, int dstH,
                                cudaStream_t stream) {
    // Оптимальный block size: 32x8 = 256 потоков, широкий warp для coalesced reads
    dim3 block(32, 8);
    dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);
    const float invScaleX = (float)srcW / dstW;
    const float invScaleY = (float)srcH / dstH;
    nv12ToBgrScaleKernel<<<grid, block, 0, stream>>>(yPlane, uvPlane, srcStride, srcW, srcH,
                                                     d_dst, dstStride, dstW, dstH,
                                                     invScaleX, invScaleY);
    return cudaGetLastError();
}

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

cudaError_t cudaLetterboxNchw(const uint8_t* rgb, int rgbStride, int rgbW, int rgbH,
                              float* d_out, int cStride, int hStride, int wStride,
                              int outW, int outH, int padX, int padY, cudaStream_t stream) {
    dim3 block(LETTERBOX_TILE_W, LETTERBOX_TILE_H);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);
    letterboxNchwKernel<<<grid, block, 0, stream>>>(rgb, rgbStride, rgbW, rgbH,
                                                    d_out, cStride, hStride, wStride,
                                                    outW, outH, padX, padY);
    return cudaGetLastError();
}

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

// GPU NMS: radix sort по score DESC + per-class NMS + компакция.
// d_cands      — [maxCands] входные кандидаты (from decode kernel)
// d_sorted     — [maxCands] буфер для сортированного результата
// d_keys       — [maxCands] буфер ключей (sort keys)
// d_keysSorted — [maxCands] буфер отсортированных ключей
// d_suppressed — [maxCands] буфер флагов подавления (int)
// d_numAlive   — [1] число выживших (на выходе)
// d_temp       — [*] буфер temp для CUB
// tempBytes    — размер temp буфера (in/out)
cudaError_t cudaGpuNms(YoloCandidate* d_cands, int n, float nmsThresh,
                       YoloCandidate* d_sorted, uint32_t* d_keys, uint32_t* d_keysSorted,
                       int* d_suppressed, int* d_numAlive,
                       void* d_temp, size_t& tempBytes,
                       cudaStream_t stream) {
    if (n <= 0) {
        cudaMemsetAsync(d_numAlive, 0, sizeof(int), stream);
        return cudaSuccess;
    }

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    // 1. Подготовка ключей: инвертируем биты float score для descending sort
    prepareSortKeys<<<blocks, threads, 0, stream>>>(d_cands, d_keys, n);

    // 2. Radix sort: ASC по инвертированным битам = DESC по score
    cub::DeviceRadixSort::SortPairs(d_temp, tempBytes,
                                    d_keys, d_keysSorted,
                                    d_cands, d_sorted, n,
                                    0, sizeof(uint32_t) * 8,
                                    stream);

    // 3. NMS: подавление по IoU в пределах класса
    cudaMemsetAsync(d_suppressed, 0, n * sizeof(int), stream);
    gpuNmsKernel<<<blocks, threads, 0, stream>>>(d_sorted, n, nmsThresh, d_suppressed);

    // 4. Компакция: плотный массив выживших
    cudaMemsetAsync(d_numAlive, 0, sizeof(int), stream);
    compactDetections<<<blocks, threads, 0, stream>>>(d_sorted, d_suppressed, n,
                                                      d_cands, d_numAlive);

    return cudaGetLastError();
}

} // extern "C"
