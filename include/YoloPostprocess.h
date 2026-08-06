// YoloPostprocess.h — постпроцессинг выхода YOLOv8/v11/v12 (anchor-free).
//
// Вход: тензор NCHW [1, (4+nc), anchors] с device (raw-выход TensorRT).
// Декод (CUDA-ядро): cx,cy,w,h в пикселях входа модели + классовое ядро.
// Фильтрация по confidence → NMS на CPU → боксы в координатах исходного кадра.
#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "Detection.h"

// Кандидат после декода (пишется CUDA-ядром).
struct YoloCandidate {
    float x1, y1, x2, y2;
    float score;
    int classId;
};

// CUDA-ядро декода, объявлено в CudaKernels.cu.
extern "C" cudaError_t cudaYoloDecode(const float* output, int channels, int anchors,
                                      float confThresh,
                                      float scaleX, float scaleY, int padX, int padY,
                                      int inW, int inH,
                                      YoloCandidate* d_cands, int* d_counter, int maxCands,
                                      cudaStream_t stream);

class YoloPostprocess {
public:
    // Параметры: число классов, порог уверенности, порог NMS, число анкоров
    // (для 640×640 YOLOv8/v11/v12: 8400 = 20×20 + 40×40 + 80×80).
    YoloPostprocess(int numClasses, float confThreshold, float nmsThreshold, int anchors);

    // Декод + маппинг в исходные координаты + NMS.
    // output — device-указатель на NCHW [1, (4+nc), anchors].
    // scaleX/scaleY/padX/padY — параметры летсплэйбокса препроцессора.
    // frameW/frameH — размер исходного кадра. stream — CUDA-поток ядра.
    Detections run(const float* output, float scaleX, float scaleY, int padX, int padY,
                   int frameW, int frameH, cudaStream_t stream);

    int numClasses() const { return m_numClasses; }
    int anchors() const { return m_anchors; }
    float confThreshold() const { return m_confThreshold; }
    float nmsThreshold() const { return m_nmsThreshold; }

private:
    int m_numClasses;
    float m_confThreshold;
    float m_nmsThreshold;
    int m_anchors;

    YoloCandidate* m_dCands = nullptr; // device: максимум кандидатов
    int* m_dCounter = nullptr;         // device: счётчик
    std::vector<YoloCandidate> m_cands;
};
