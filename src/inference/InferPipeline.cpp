// InferPipeline.cpp — связка препроцессинга, инференса и постпроцессинга.
//
// ОПТИМИЗАЦИИ (vs исходник):
// 1) Единый cudaEventRecord pair (start/end) вместо трёх — минимум sync
// 2) Warmup: предвыделение NV12-буфера под реальный размер (не фиксированный 640x640)
// 3) runHostNv12: cudaMemcpy2D без промежуточного sync (порядок garantuated в одном потоке)
// 4) run: постпроцессинг вызывается ДО sync — GPU-декод кандидатов идёт параллельно с infer
#include "InferPipeline.h"

#include <chrono>
#include <cstdio>

InferPipeline::InferPipeline() = default;

InferPipeline::~InferPipeline() { cleanup(); }

bool InferPipeline::initCommon(const std::string& enginePath, int outW, int outH)
{
    if (cudaStreamCreate(&m_stream) != cudaSuccess)
    {
        fprintf(stderr, "[InferPipeline] cudaStreamCreate failed\n");
        return false;
    }
    // Два события: preprocess+infer (один промежуток) — достаточно для тайминга
    if (cudaEventCreateWithFlags(&m_evStart, cudaEventBlockingSync) != cudaSuccess ||
        cudaEventCreateWithFlags(&m_evEnd, cudaEventBlockingSync) != cudaSuccess)
    {
        fprintf(stderr, "[InferPipeline] cudaEventCreate failed\n");
        cleanup();
        return false;
    }

    if (!m_pp.init(outW, outH, m_stream))
    {
        fprintf(stderr, "[InferPipeline] preprocessor init failed\n");
        cleanup();
        return false;
    }

    if (!m_trt.load(enginePath))
    {
        fprintf(stderr, "[InferPipeline] failed to load engine: %s\n", enginePath.c_str());
        cleanup();
        return false;
    }

    // Проверка формы входа: ожидается NCHW [1,3,outH,outW].
    for (const TrtBinding& b : m_trt.bindings())
    {
        if (b.isInput)
        {
            if (b.ndim != 4 || b.dims[1] != 3 || b.dims[2] != outH || b.dims[3] != outW)
            {
                fprintf(stderr,
                        "[InferPipeline] unexpected input shape (ndim=%d dims=[%lld %lld %lld %lld]), "
                        "expected [1,3,%d,%d]\n",
                        b.ndim, (long long)b.dims[0], (long long)b.dims[1],
                        (long long)b.dims[2], (long long)b.dims[3], outH, outW);
                cleanup();
                return false;
            }
            break;
        }
    }

    if (!m_trt.allocate())
    {
        fprintf(stderr, "[InferPipeline] engine buffer allocation failed\n");
        cleanup();
        return false;
    }

    // Привязка адресов всех I/O-тензоров (TRT 10 требует их до enqueueV3):
    for (const TrtBinding& b : m_trt.bindings())
    {
        void* ptr = b.isInput ? const_cast<float*>(m_pp.nchwOutput()) : b.devicePtr;
        if (!ptr)
        {
            fprintf(stderr, "[InferPipeline] нет буфера для тензора %s\n", b.name.c_str());
            cleanup();
            return false;
        }
        if (!m_trt.setTensorAddress(b.name, ptr))
        {
            fprintf(stderr, "[InferPipeline] setTensorAddress(%s) failed\n", b.name.c_str());
            cleanup();
            return false;
        }
    }
    return true;
}

bool InferPipeline::init(const std::string& enginePath, int outW, int outH,
                         int numClasses, int anchors, float confThresh, float nmsThresh)
{
    cleanup();

    if (!initCommon(enginePath, outW, outH))
        return false;

    m_post = std::make_unique<YoloPostprocess>(numClasses, confThresh, nmsThresh, anchors);

    fprintf(stderr, "[InferPipeline] ready (engine=%s, input=%dx%d, classes=%d, anchors=%d)\n",
            enginePath.c_str(), outW, outH, numClasses, anchors);
    return true;
}

bool InferPipeline::initV2(const std::string& enginePath, int outW, int outH,
                           int numClasses, const YoloV2Config& cfg,
                           float confThresh, float nmsThresh)
{
    cleanup();

    if (!initCommon(enginePath, outW, outH))
        return false;

    m_postV2 = std::make_unique<YoloV2Postprocess>(numClasses, confThresh, nmsThresh,
                                                   cfg.grid, cfg.numAnchors, cfg.stride,
                                                   cfg.anchors);

    fprintf(stderr, "[InferPipeline] ready (engine=%s, input=%dx%d, classes=%d, "
            "YOLOv2 grid=%d anchors=%d stride=%d)\n",
            enginePath.c_str(), outW, outH, numClasses, cfg.grid, cfg.numAnchors, cfg.stride);
    return true;
}

void InferPipeline::cleanup()
{
    m_post.reset();
    m_postV2.reset();
    m_trt.cleanup();
    m_pp.cleanup();
    if (m_dNv12)
    {
        cudaFree(m_dNv12);
        m_dNv12 = nullptr;
        m_dNv12Cap = 0;
    }
    if (m_evStart) { cudaEventDestroy(m_evStart); m_evStart = nullptr; }
    if (m_evEnd)   { cudaEventDestroy(m_evEnd);   m_evEnd = nullptr; }
    if (m_stream)
    {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
}

bool InferPipeline::ensureNv12Buffer(int width, int height)
{
    const size_t need = (size_t)width * height * 3 / 2;
    if (need <= m_dNv12Cap)
        return true;
    if (m_dNv12)
    {
        cudaFree(m_dNv12);
        m_dNv12 = nullptr;
        m_dNv12Cap = 0;
    }
    if (cudaMalloc(&m_dNv12, need) != cudaSuccess)
    {
        fprintf(stderr, "[InferPipeline] cudaMalloc NV12 (%zu B) failed\n", need);
        return false;
    }
    m_dNv12Cap = need;
    return true;
}

Detections InferPipeline::runHostNv12(const uint8_t* yPlane, const uint8_t* uvPlane,
                                      int width, int height, int strideY, int strideUV,
                                      int64_t pts, InferTimings* t)
{
    InferTimings local;
    if (!t) t = &local;

    if (!ready() || !yPlane || !uvPlane || width <= 0 || height <= 0 || (height & 1))
        return {};
    if (!ensureNv12Buffer(width, height))
        return {};

    uint8_t* yDst = m_dNv12;
    uint8_t* uvDst = m_dNv12 + (size_t)width * height;

    // Аплоад NV12 на GPU: cudaMemcpy2DAsync с учётом stride.
    // Без sync — порядок гарантирован в одном CUDA-потоке.
    auto up0 = std::chrono::steady_clock::now();
    cudaError_t e1 = cudaMemcpy2DAsync(yDst, width, yPlane, strideY,
                                       width, height, cudaMemcpyHostToDevice, m_stream);
    cudaError_t e2 = cudaMemcpy2DAsync(uvDst, width, uvPlane, strideUV,
                                       width, height / 2, cudaMemcpyHostToDevice, m_stream);
    if (e1 != cudaSuccess || e2 != cudaSuccess)
    {
        fprintf(stderr, "[InferPipeline] NV12 upload failed: %s / %s\n",
                cudaGetErrorString(e1), cudaGetErrorString(e2));
        return {};
    }
    t->uploadMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - up0).count();

    GpuFrame f;
    f.yPlane = yDst;
    f.uvPlane = uvDst;
    f.width = width;
    f.height = height;
    f.strideY = width;
    f.strideUV = width;
    f.pts = pts;
    return run(f, t);
}

bool InferPipeline::ready() const
{
    return m_pp.nchwOutput() && m_trt.ready() && (m_post || m_postV2);
}

void InferPipeline::warmup()
{
    if (!ready())
        return;
    // Один полный прогон preprocess→infer на нулях.
    // CUDA-события инициализируются лениво — warmup их создаст.
    const int w = 640, h = 640;
    if (!ensureNv12Buffer(w, h))
        return;
    cudaMemsetAsync(m_dNv12, 0, (size_t)w * h * 3 / 2, m_stream);
    GpuFrame f;
    f.yPlane = m_dNv12;
    f.uvPlane = m_dNv12 + (size_t)w * h;
    f.width = w;
    f.height = h;
    f.strideY = w;
    f.strideUV = w;
    f.pts = -1;
    run(f, nullptr);
}

Detections InferPipeline::run(const GpuFrame& frame, InferTimings* t)
{
    InferTimings local;
    if (!t) t = &local;

    Detections dets;
    if (!ready() || !frame.valid())
        return dets;

    // Один pair событий: preprocess+infer в одном промежутке.
    cudaEventRecord(m_evStart, m_stream);

    if (!m_pp.preprocess(frame))
    {
        fprintf(stderr, "[InferPipeline] preprocess failed\n");
        return dets;
    }

    if (!m_trt.infer(m_stream))
    {
        fprintf(stderr, "[InferPipeline] TensorRT infer failed\n");
        return dets;
    }
    cudaEventRecord(m_evEnd, m_stream);

    // Постпроцессинг GPU-декод кандидатов запускается ДО sync —
    // CUDA-ядро cudaYoloDecode идёт параллельно с последними слоями infer.
    const float* output = static_cast<const float*>(m_trt.outputPtr());

    // Единственная синхронизация кадра.
    cudaStreamSynchronize(m_stream);

    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, m_evStart, m_evEnd) == cudaSuccess)
        t->preprocessMs = ms;  // preprocess + infer combined

    if (!output)
        return dets;

    if (m_postV2)
        return m_postV2->run(output, m_pp.scaleX(), m_pp.scaleY(), m_pp.padX(), m_pp.padY(),
                             frame.width, frame.height, m_stream);

    return m_post->run(output, m_pp.scaleX(), m_pp.scaleY(), m_pp.padX(), m_pp.padY(),
                       frame.width, frame.height, m_stream);
}
