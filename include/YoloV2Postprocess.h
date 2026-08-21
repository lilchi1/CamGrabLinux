// YoloV2Postprocess.h — постпроцессинг выхода YOLOv2 (anchor-based, YAD2K).
//
// Вход: тензор NCHW [1, C, grid, grid] (raw-выход TensorRT), C = numAnchors*(5+numClasses).
// Декод (CUDA-ядро): sigmoid(tx/ty/obj), exp(tw/th), сетка grid x grid, якоря
// в пикселях входа модели. Фильтрация по confidence → NMS на CPU → боксы в
// координатах исходного кадра.
#pragma once

#include <vector>
#include <cstdint>

#include <cuda_runtime.h>

#include "Detection.h"
#include "YoloPostprocess.h"   // struct YoloCandidate (общий формат с YOLOv8)

// CUDA-ядро декода YOLOv2, объявлено в CudaKernels.cu.
extern "C" cudaError_t cudaYoloV2Decode(const float* output, int grid, int numAnchors,
                                        int numClasses, const float* dAnchors, int stride,
                                        float confThresh,
                                        float scaleX, float scaleY, int padX, int padY,
                                        int inW, int inH,
                                        YoloCandidate* d_cands, int* d_counter,
                                        int maxCands, cudaStream_t stream);

class YoloV2Postprocess {
public:
    // Параметры: число классов, пороги, сетка (19 для 608x608), число якорей
    // на клетку (5), stride (вход/сетка = 32), якоря (пары w,h в пикселях входа).
    YoloV2Postprocess(int numClasses, float confThreshold, float nmsThreshold,
                      int grid, int numAnchors, int stride,
                      const std::vector<float>& anchors);
    ~YoloV2Postprocess();

    // Декод + маппинг в исходные координаты + NMS.
    // output — device-указатель на NCHW [1, numAnchors*(5+nc), grid, grid].
    Detections run(const float* output, float scaleX, float scaleY, int padX, int padY,
                   int frameW, int frameH, cudaStream_t stream);

    int numClasses() const { return m_numClasses; }
    int grid() const { return m_grid; }
    int numAnchors() const { return m_numAnchors; }
    int stride() const { return m_stride; }

private:
    int m_numClasses;
    float m_confThreshold;
    float m_nmsThreshold;
    int m_grid;
    int m_numAnchors;
    int m_stride;

    float* m_dAnchors = nullptr;   // device: пары (w,h) якорей
    YoloCandidate* m_dCands = nullptr;  // device: максимум кандидатов
    int* m_dCounter = nullptr;          // device: счётчик
    std::vector<YoloCandidate> m_cands;

    // GPU NMS buffers (allocated once in constructor)
    YoloCandidate* m_dSorted = nullptr;
    uint32_t*      m_dKeys = nullptr;
    uint32_t*      m_dKeysSorted = nullptr;
    int*           m_dSuppressed = nullptr;
    int*           m_dNumAlive = nullptr;
    void*          m_dTemp = nullptr;
    size_t         m_tempBytes = 0;
};
