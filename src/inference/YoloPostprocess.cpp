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

namespace {

// Площадь пересечения двух боксов.
float iou(const YoloCandidate& a, const YoloCandidate& b)
{
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);
    const float iw = std::max(0.0f, ix2 - ix1);
    const float ih = std::max(0.0f, iy2 - iy1);
    const float inter = iw * ih;
    const float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    const float uni = areaA + areaB - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

} // namespace

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
    // Запас под кандидатов до NMS; при нехватке буфер перевыделяется.
    const int maxCands = std::max(1024, m_anchors);
    if (cudaMalloc(&m_dCands, (size_t)maxCands * sizeof(YoloCandidate)) != cudaSuccess ||
        cudaMalloc(&m_dCounter, sizeof(int)) != cudaSuccess)
    {
        fprintf(stderr, "[YoloPostprocess] cudaMalloc failed\n");
        m_dCands = nullptr;
        m_dCounter = nullptr;
    }
}

Detections YoloPostprocess::run(const float* output, float scaleX, float scaleY, int padX, int padY,
                                int frameW, int frameH, cudaStream_t stream)
{
    Detections dets;
    if (!output || !m_dCands || !m_dCounter)
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

    // Синхронизация нужна только здесь (после ядра), чтобы прочитать счётчик и данные.
    cudaStreamSynchronize(stream);

    int n = 0;
    cudaMemcpy(&n, m_dCounter, sizeof(int), cudaMemcpyDeviceToHost);
    n = std::min(n, maxCands);
    if (n <= 0)
        return dets;

    m_cands.resize(n);
    cudaMemcpy(m_cands.data(), m_dCands, (size_t)n * sizeof(YoloCandidate), cudaMemcpyDeviceToHost);

    // NMS: сортировка по уверенности, подавление по IoU в пределах класса.
    std::sort(m_cands.begin(), m_cands.end(),
              [](const YoloCandidate& a, const YoloCandidate& b) { return a.score > b.score; });

    std::vector<bool> removed(n, false);
    for (int i = 0; i < n; i++)
    {
        if (removed[i])
            continue;
        const YoloCandidate& keep = m_cands[i];
        dets.push_back({ keep.x1, keep.y1, keep.x2, keep.y2, keep.score, keep.classId });
        for (int j = i + 1; j < n; j++)
        {
            if (removed[j] || m_cands[j].classId != keep.classId)
                continue;
            if (iou(keep, m_cands[j]) > m_nmsThreshold)
                removed[j] = true;
        }
    }
    return dets;
}
