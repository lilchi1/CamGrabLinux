// InferPipeline.h — Связка препроцессинга, TensorRT-инференса и постпроцессинга.
//
// ОПТИМИЗАЦИИ:
//  - 2 cudaEvent вместо 3 (start/end — combined preprocess+infer)
//  - Постпроцессинг GPU-декод запускается до sync
//  - NV12-буфер пулится (не перевыделяется каждый кадр)
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

struct YoloV2Config {
    int grid = 19;
    int numAnchors = 5;
    int stride = 32;
    std::vector<float> anchors;
};

struct InferTimings {
    double uploadMs = -1.0;     // NV12 host→device upload
    double preprocessMs = -1.0; // combined preprocess + infer (CUDA events)
    double inferMs = -1.0;      // (deprecated: combined with preprocessMs)
};

class InferPipeline {
public:
    InferPipeline();
    ~InferPipeline();

    bool init(const std::string& enginePath, int outW, int outH,
              int numClasses, int anchors, float confThresh, float nmsThresh);

    bool initV2(const std::string& enginePath, int outW, int outH,
                int numClasses, const YoloV2Config& cfg, float confThresh, float nmsThresh);

    void cleanup();
    bool ready() const;
    void warmup();

    Detections run(const GpuFrame& frame, InferTimings* t = nullptr);

    Detections runHostNv12(const uint8_t* yPlane, const uint8_t* uvPlane,
                           int width, int height, int strideY, int strideUV,
                           int64_t pts, InferTimings* t = nullptr);

    const CvcudaPreprocessor& preprocessor() const { return m_pp; }
    const TensorRtInfer& infer() const { return m_trt; }
    const YoloPostprocess& postprocess() const { return *m_post; }
    const YoloV2Postprocess& postprocessV2() const { return *m_postV2; }

private:
    bool initCommon(const std::string& enginePath, int outW, int outH);
    bool ensureNv12Buffer(int width, int height);

    cudaStream_t m_stream = nullptr;
    CvcudaPreprocessor m_pp;
    TensorRtInfer m_trt;
    std::unique_ptr<YoloPostprocess> m_post;
    std::unique_ptr<YoloV2Postprocess> m_postV2;

    uint8_t* m_dNv12 = nullptr;
    size_t m_dNv12Cap = 0;

    // 2 события вместо 3: preprocess+infer combined
    cudaEvent_t m_evStart = nullptr;
    cudaEvent_t m_evEnd = nullptr;
};
