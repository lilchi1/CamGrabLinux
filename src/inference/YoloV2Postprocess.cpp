// YoloV2Postprocess.cpp — декод выхода YOLOv2 на GPU + NMS на CPU.
#include "YoloV2Postprocess.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

YoloV2Postprocess::YoloV2Postprocess(int numClasses, float confThreshold, float nmsThreshold,
                                     int grid, int numAnchors, int stride,
                                     const std::vector<float>& anchors)
    : m_numClasses(numClasses)
    , m_confThreshold(confThreshold)
    , m_nmsThreshold(nmsThreshold)
    , m_grid(grid)
    , m_numAnchors(numAnchors)
    , m_stride(stride)
{
    // Дефолтные якоря COCO YAD2K (пиксели входа 608/416) — используются, если
    // переданный вектор мал или параметры некорректны.
    static const float kDefAnchors[] = {
        18.32736f, 21.67632f,
        59.98272f, 66.00096f,
        106.82976f, 175.17888f,
        252.25024f, 112.88896f,
        312.65664f, 293.38496f,
    };

    std::vector<float> useAnchors = anchors;
    bool bad = (m_grid <= 0 || m_numAnchors <= 0 || m_numClasses <= 0 || m_stride <= 0 ||
                (int)useAnchors.size() < 2 * m_numAnchors);
    if (bad) {
        fprintf(stderr, "[YoloV2Postprocess] invalid params (classes=%d grid=%d anchors=%zu/%d) "
                        "— использую дефолты\n",
                m_numClasses, m_grid, anchors.size(), 2 * m_numAnchors);
        m_grid = 19;
        m_numAnchors = 5;
        m_numClasses = 80;
        m_stride = 32;
        useAnchors.assign(kDefAnchors, kDefAnchors + 10);
    }

    if (cudaMalloc(&m_dAnchors, (size_t)(2 * m_numAnchors) * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(m_dAnchors, useAnchors.data(), (size_t)(2 * m_numAnchors) * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
    {
        fprintf(stderr, "[YoloV2Postprocess] cudaMalloc/copy anchors failed\n");
        if (m_dAnchors) { cudaFree(m_dAnchors); m_dAnchors = nullptr; }
    }

    const int maxCands = std::max(1024, m_grid * m_grid * m_numAnchors);
    if (cudaMalloc(&m_dCands, (size_t)maxCands * sizeof(YoloCandidate)) != cudaSuccess ||
        cudaMalloc(&m_dCounter, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&m_dSorted, (size_t)maxCands * sizeof(YoloCandidate)) != cudaSuccess ||
        cudaMalloc(&m_dKeys, (size_t)maxCands * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc(&m_dKeysSorted, (size_t)maxCands * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc(&m_dSuppressed, (size_t)maxCands * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&m_dNumAlive, sizeof(int)) != cudaSuccess)
    {
        fprintf(stderr, "[YoloV2Postprocess] cudaMalloc failed\n");
    }
    m_tempBytes = 1 << 20;
    cudaMalloc(&m_dTemp, m_tempBytes);
}

YoloV2Postprocess::~YoloV2Postprocess()
{
    if (m_dAnchors)      { cudaFree(m_dAnchors); m_dAnchors = nullptr; }
    if (m_dCands)        { cudaFree(m_dCands); m_dCands = nullptr; }
    if (m_dCounter)      { cudaFree(m_dCounter); m_dCounter = nullptr; }
    if (m_dSorted)       { cudaFree(m_dSorted); m_dSorted = nullptr; }
    if (m_dKeys)         { cudaFree(m_dKeys); m_dKeys = nullptr; }
    if (m_dKeysSorted)   { cudaFree(m_dKeysSorted); m_dKeysSorted = nullptr; }
    if (m_dSuppressed)   { cudaFree(m_dSuppressed); m_dSuppressed = nullptr; }
    if (m_dNumAlive)     { cudaFree(m_dNumAlive); m_dNumAlive = nullptr; }
    if (m_dTemp)         { cudaFree(m_dTemp); m_dTemp = nullptr; }
}

Detections YoloV2Postprocess::run(const float* output, float scaleX, float scaleY,
                                  int padX, int padY, int frameW, int frameH,
                                  cudaStream_t stream)
{
    Detections dets;
    if (!output || !m_dCands || !m_dCounter || !m_dAnchors || !m_dSorted || !m_dSuppressed || !m_dNumAlive)
        return dets;

    int maxCands = std::max(1024, m_grid * m_grid * m_numAnchors);

    cudaMemsetAsync(m_dCounter, 0, sizeof(int), stream);
    const cudaError_t kerr = cudaYoloV2Decode(output, m_grid, m_numAnchors, m_numClasses,
                                              m_dAnchors, m_stride, m_confThreshold,
                                              scaleX, scaleY, padX, padY, frameW, frameH,
                                              m_dCands, m_dCounter, maxCands, stream);
    if (kerr != cudaSuccess)
    {
        fprintf(stderr, "[YoloV2Postprocess] cudaYoloV2Decode error: %s\n",
                cudaGetErrorString(kerr));
        return dets;
    }

    int n = 0;
    cudaMemcpy(&n, m_dCounter, sizeof(int), cudaMemcpyDeviceToHost);
    n = std::min(n, maxCands);
    if (n <= 0)
        return dets;

    // GPU NMS: radix sort + per-class suppression + compaction
    cudaGpuNms(m_dCands, n, m_nmsThreshold,
               m_dSorted, m_dKeys, m_dKeysSorted,
               m_dSuppressed, m_dNumAlive,
               m_dTemp, m_tempBytes, stream);

    // Один sync + копирование только выживших
    cudaStreamSynchronize(stream);

    int numAlive = 0;
    cudaMemcpy(&numAlive, m_dNumAlive, sizeof(int), cudaMemcpyDeviceToHost);
    if (numAlive <= 0)
        return dets;

    dets.reserve(numAlive);
    m_cands.resize(numAlive);
    cudaMemcpy(m_cands.data(), m_dCands, (size_t)numAlive * sizeof(YoloCandidate),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < numAlive; i++)
        dets.push_back({ m_cands[i].x1, m_cands[i].y1, m_cands[i].x2, m_cands[i].y2,
                         m_cands[i].score, m_cands[i].classId });
    return dets;
}
