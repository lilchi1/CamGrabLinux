// InferPipeline.h — Связка препроцессинга, TensorRT-инференса и постпроцессинга.
//
// Полный GPU-путь для одного кадра (без копий на host до детекций):
//   GpuFrame (NV12, device) → CvcudaPreprocessor (letterbox → NCHW F32)
//   → TensorRtInfer::infer (тот же CUDA-поток) → YoloPostprocess (decode+NMS)
#pragma once

#include <memory>
#include <string>

#include <cuda_runtime.h>

#include "CvcudaPreprocessor.h"
#include "Detection.h"
#include "FrameCallback.h"
#include "TensorRtInfer.h"
#include "YoloPostprocess.h"
#include "YoloV2Postprocess.h"

// Параметры YOLOv2-постпроцессинга (anchor-based, сетка grid x grid).
struct YoloV2Config {
    int grid = 19;                        // сетка выхода (608/32 = 19)
    int numAnchors = 5;                   // якорей на клетку
    int stride = 32;                      // вход / сетка
    std::vector<float> anchors;           // пары (w,h) в пикселях входа модели
};

class InferPipeline {
public:
    InferPipeline();
    ~InferPipeline();

    // Инициализация: препроцессор, загрузка .engine, постпроцессор.
    // outW/outH — размер входа модели; numClasses/anchors — параметры выхода.
    bool init(const std::string& enginePath, int outW, int outH,
              int numClasses, int anchors, float confThresh, float nmsThresh);

    // Инициализация для YOLOv2 (anchor-based) — выход NCHW [1, C, grid, grid].
    bool initV2(const std::string& enginePath, int outW, int outH,
                int numClasses, const YoloV2Config& cfg, float confThresh, float nmsThresh);

    void cleanup();
    bool ready() const;

    // Полный прогон для одного кадра. Асинхронен до NMS; NMS синхронизирует
    // CUDA-поток и возвращает детекции в координатах исходного кадра.
    Detections run(const GpuFrame& frame);

    // Прогон для кадра NV12 в host-памяти (GstDecoder отдаёт именно такой).
    // Плоскости асинхронно загружаются в постоянный device-буфер (cudaMemcpy2DAsync
    // с учётом stride) и далее идёт полный GPU-путь run(). Возвращает детекции
    // в координатах исходного кадра.
    Detections runHostNv12(const uint8_t* yPlane, const uint8_t* uvPlane,
                           int width, int height, int strideY, int strideUV,
                           int64_t pts);

    const CvcudaPreprocessor& preprocessor() const { return m_pp; }
    const TensorRtInfer& infer() const { return m_trt; }
    const YoloPostprocess& postprocess() const { return *m_post; }
    const YoloV2Postprocess& postprocessV2() const { return *m_postV2; }

private:
    // Общая часть init/initV2: поток, препроцессор, загрузка engine, буферы.
    bool initCommon(const std::string& enginePath, int outW, int outH);

    // Перевыделение device-буфера под NV12 (Y + UV, плотно упакованы).
    bool ensureNv12Buffer(int width, int height);

    cudaStream_t m_stream = nullptr;
    CvcudaPreprocessor m_pp;
    TensorRtInfer m_trt;
    std::unique_ptr<YoloPostprocess> m_post;
    std::unique_ptr<YoloV2Postprocess> m_postV2;

    uint8_t* m_dNv12 = nullptr;  // device NV12: [Y: w*h][UV: w*(h/2)]
    size_t m_dNv12Cap = 0;       // ёмкость буфера (байт)
};
