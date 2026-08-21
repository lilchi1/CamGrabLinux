// YoloPostprocess.cpp — декод выхода YOLO на GPU + NMS на CPU.
#include "YoloPostprocess.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

std::vector<std::string> loadClassNames(const std::string& path)
{
    std::vector<std::string> names;
    std::ifstream f(path);
    if (!f.is_open())
        return names;
    std::string line;
    while (std::getline(f, line))
    {
        if (!line.empty())
            names.push_back(line);
    }
    return names;
}

YoloPostprocess::YoloPostprocess(int numClasses, float confThreshold, float nmsThreshold, int anchors)
    : m_numClasses(numClasses)
    , m_confThreshold(confThreshold)
    , m_nmsThreshold(nmsThreshold)
    , m_anchors(anchors)
{
    if (m_anchors <= 0 || m_numClasses <= 0)
    {
        fprintf(stderr, "[YoloPostprocess] invalid params (classes=%d anchors=%d)\n", m_numClasses, m_anchors);
        m_anchors = 8400;
        m_numClasses = 80;
    }
    const int maxCands = std::max(1024, m_anchors);
    if (cudaMalloc(&m_dCands, (size_t)maxCands * sizeof(YoloCandidate)) != cudaSuccess ||
        cudaMalloc(&m_dCounter, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&m_dSorted, (size_t)maxCands * sizeof(YoloCandidate)) != cudaSuccess ||
        cudaMalloc(&m_dKeys, (size_t)maxCands * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc(&m_dKeysSorted, (size_t)maxCands * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc(&m_dSuppressed, (size_t)maxCands * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&m_dNumAlive, sizeof(int)) != cudaSuccess)
    {
        fprintf(stderr, "[YoloPostprocess] cudaMalloc failed\n");
    }
    // CUB temp storage (lazily resized by DeviceRadixSort)
    m_tempBytes = 1 << 20;  // 1 MB initial
    cudaMalloc(&m_dTemp, m_tempBytes);
}

YoloPostprocess::~YoloPostprocess()
{
    if (m_dCands)       { cudaFree(m_dCands); m_dCands = nullptr; }
    if (m_dCounter)     { cudaFree(m_dCounter); m_dCounter = nullptr; }
    if (m_dSorted)      { cudaFree(m_dSorted); m_dSorted = nullptr; }
    if (m_dKeys)        { cudaFree(m_dKeys); m_dKeys = nullptr; }
    if (m_dKeysSorted)  { cudaFree(m_dKeysSorted); m_dKeysSorted = nullptr; }
    if (m_dSuppressed)  { cudaFree(m_dSuppressed); m_dSuppressed = nullptr; }
    if (m_dNumAlive)    { cudaFree(m_dNumAlive); m_dNumAlive = nullptr; }
    if (m_dTemp)        { cudaFree(m_dTemp); m_dTemp = nullptr; }
}

Detections YoloPostprocess::run(const float* output, float scaleX, float scaleY, int padX, int padY,
                                int frameW, int frameH, cudaStream_t stream)
{
    Detections dets;
    if (!output || !m_dCands || !m_dCounter || !m_dSorted || !m_dSuppressed || !m_dNumAlive)
        return dets;

    const int channels = 4 + m_numClasses;
    int maxCands = std::max(1024, m_anchors);

    cudaMemsetAsync(m_dCounter, 0, sizeof(int), stream);
    const cudaError_t kerr = cudaYoloDecode(output, channels, m_anchors, m_confThreshold,
                                            scaleX, scaleY, padX, padY, frameW, frameH,
                                            m_dCands, m_dCounter, maxCands, stream);
    if (kerr != cudaSuccess)
    {
        fprintf(stderr, "[YoloPostprocess] cudaYoloDecode error: %s\n", cudaGetErrorString(kerr));
        return dets;
    }

    int n = 0;
    cudaMemcpy(&n, m_dCounter, sizeof(int), cudaMemcpyDeviceToHost);
    n = std::min(n, maxCands);
    if (n <= 0)
        return dets;

    // GPU NMS: radix sort + per-class suppression + compaction
    // (без sync до конца — всё на GPU, sync только после compactDetections)
    cudaGpuNms(m_dCands, n, m_nmsThreshold,
               m_dSorted, m_dKeys, m_dKeysSorted,
               m_dSuppressed, m_dNumAlive,
               m_dTemp, m_tempBytes, stream);

    // Один sync + копирование только выживших (обычно 5-20 штук vs 50-200 кандидатов)
    cudaStreamSynchronize(stream);

    int numAlive = 0;
    cudaMemcpy(&numAlive, m_dNumAlive, sizeof(int), cudaMemcpyDeviceToHost);
    if (numAlive <= 0)
        return dets;

    // m_dCands содержит плотный массив выживших после compactDetections
    dets.reserve(numAlive);
    m_cands.resize(numAlive);
    cudaMemcpy(m_cands.data(), m_dCands, (size_t)numAlive * sizeof(YoloCandidate), cudaMemcpyDeviceToHost);

    for (int i = 0; i < numAlive; i++)
        dets.push_back({ m_cands[i].x1, m_cands[i].y1, m_cands[i].x2, m_cands[i].y2,
                         m_cands[i].score, m_cands[i].classId });
    return dets;
}
