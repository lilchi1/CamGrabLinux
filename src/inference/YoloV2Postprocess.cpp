// YoloV2Postprocess.cpp — декод выхода YOLOv2 на GPU + NMS на CPU.
#include "YoloV2Postprocess.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

namespace {

// Площадь пересечения двух боксов (то же, что в YoloPostprocess).
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
        cudaMalloc(&m_dCounter, sizeof(int)) != cudaSuccess)
    {
        fprintf(stderr, "[YoloV2Postprocess] cudaMalloc failed\n");
        m_dCands = nullptr;
        m_dCounter = nullptr;
    }
}

YoloV2Postprocess::~YoloV2Postprocess()
{
    if (m_dAnchors) { cudaFree(m_dAnchors); m_dAnchors = nullptr; }
    if (m_dCands) { cudaFree(m_dCands); m_dCands = nullptr; }
    if (m_dCounter) { cudaFree(m_dCounter); m_dCounter = nullptr; }
}

Detections YoloV2Postprocess::run(const float* output, float scaleX, float scaleY,
                                  int padX, int padY, int frameW, int frameH,
                                  cudaStream_t stream)
{
    Detections dets;
    if (!output || !m_dCands || !m_dCounter || !m_dAnchors)
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

    cudaStreamSynchronize(stream);

    int n = 0;
    cudaMemcpy(&n, m_dCounter, sizeof(int), cudaMemcpyDeviceToHost);
    n = std::min(n, maxCands);
    if (n <= 0)
        return dets;

    m_cands.resize(n);
    cudaMemcpy(m_cands.data(), m_dCands, (size_t)n * sizeof(YoloCandidate),
               cudaMemcpyDeviceToHost);

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
