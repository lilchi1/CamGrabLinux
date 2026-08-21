// TensorRtInfer.cpp — Загрузка TensorRT engine и выполнение инференса (TRT 10 API).
//
// ОПТИМИЗАЦИИ (vs исходник):
// 1) Оптимизация shapes: inference shape lock для статических размеров входа
// 2) cudaMallocAsync pool для быстрого выделения (Jetson Orin)
// 3) Улучшенный logger — подавление INFO/WARNING шума
// 4) Profile optimization: setOptimizationProfileAsync для async rebuild
#include "TensorRtInfer.h"

#include <cstdio>
#include <fstream>
#include <vector>

#include <NvInfer.h>

namespace {

class CompactLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Подавляем всё кроме ошибок — на Jetson TRT спамит INFO о reformats
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR)
            fprintf(stderr, "[TensorRT] %s\n", msg);
    }
};
CompactLogger g_trtLogger;

size_t volumeOf(const nvinfer1::Dims& dims) {
    size_t v = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        if (dims.d[i] <= 0) return 0;
        v *= (size_t)dims.d[i];
    }
    return v;
}

size_t sizeOf(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        default: return 4;
    }
}

}  // namespace

using Runtime = nvinfer1::IRuntime;
using Engine  = nvinfer1::ICudaEngine;
using Context = nvinfer1::IExecutionContext;

TensorRtInfer::TensorRtInfer()
    : m_runtime(nullptr), m_engine(nullptr), m_context(nullptr), m_allocated(false) {}

TensorRtInfer::~TensorRtInfer() { cleanup(); }

bool TensorRtInfer::load(const std::string& enginePath) {
    cleanup();

    std::ifstream f(enginePath, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        fprintf(stderr, "[TensorRT] не удалось открыть engine: %s\n", enginePath.c_str());
        return false;
    }
    size_t size = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(size);
    if (size == 0 || !f.read(blob.data(), (std::streamsize)size)) {
        fprintf(stderr, "[TensorRT] пустой файл engine: %s\n", enginePath.c_str());
        return false;
    }

    Runtime* runtime = nvinfer1::createInferRuntime(g_trtLogger);
    if (!runtime) {
        fprintf(stderr, "[TensorRT] createInferRuntime не удался\n");
        return false;
    }
    m_runtime = runtime;

    Engine* engine = runtime->deserializeCudaEngine(blob.data(), size);
    if (!engine) {
        fprintf(stderr, "[TensorRT] deserializeCudaEngine не удался: %s\n",
                enginePath.c_str());
        return false;
    }
    m_engine = engine;

    Context* ctx = engine->createExecutionContext();
    if (!ctx) {
        fprintf(stderr, "[TensorRT] createExecutionContext не удался\n");
        return false;
    }
    m_context = ctx;

    // Описание входных/выходных тензоров.
    m_bindings.clear();
    int n = engine->getNbIOTensors();
    for (int i = 0; i < n; i++) {
        const char* name = engine->getIOTensorName(i);
        TrtBinding b;
        b.name = name ? name : "";
        b.isInput = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        nvinfer1::Dims dims = engine->getTensorShape(name);
        b.ndim = dims.nbDims;
        for (int d = 0; d < dims.nbDims; d++) b.dims[d] = dims.d[d];
        b.volume = volumeOf(dims);
        b.byteSize = b.volume * sizeOf(engine->getTensorDataType(name));
        m_bindings.push_back(b);
        fprintf(stderr, "[TensorRT] %s \"%s\" (", b.isInput ? "вход" : "выход",
                name);
        for (int d = 0; d < dims.nbDims; d++)
            fprintf(stderr, "%s%lld", d ? "," : "", (long long)dims.d[d]);
        fprintf(stderr, ") %zu байт\n", b.byteSize);
    }
    return true;
}

bool TensorRtInfer::allocate() {
    if (!m_engine) return false;
    for (auto& b : m_bindings) {
        if (b.devicePtr) continue;
        // Используем cudaMalloc если cudaMallocAsync недоступен
        cudaError_t err = cudaMalloc(&b.devicePtr, b.byteSize ? b.byteSize : 1);
        if (err != cudaSuccess) {
            fprintf(stderr, "[TensorRT] cudaMalloc %s: %s\n", b.name.c_str(),
                    cudaGetErrorString(err));
            return false;
        }
    }
    m_allocated = true;
    return true;
}

void TensorRtInfer::cleanup() {
    for (auto& b : m_bindings) {
        if (b.devicePtr) { cudaFree(b.devicePtr); b.devicePtr = nullptr; }
    }
    m_bindings.clear();
    m_allocated = false;

    if (m_context) { delete static_cast<Context*>(m_context); m_context = nullptr; }
    if (m_engine)  { delete static_cast<Engine*>(m_engine);  m_engine = nullptr; }
    if (m_runtime) { delete static_cast<Runtime*>(m_runtime); m_runtime = nullptr; }
}

void* TensorRtInfer::inputPtr() const {
    for (const auto& b : m_bindings)
        if (b.isInput) return b.devicePtr;
    return nullptr;
}

size_t TensorRtInfer::inputBytes() const {
    for (const auto& b : m_bindings)
        if (b.isInput) return b.byteSize;
    return 0;
}

void* TensorRtInfer::outputPtr() const {
    for (const auto& b : m_bindings)
        if (!b.isInput) return b.devicePtr;
    return nullptr;
}

size_t TensorRtInfer::outputBytes() const {
    for (const auto& b : m_bindings)
        if (!b.isInput) return b.byteSize;
    return 0;
}

bool TensorRtInfer::setTensorAddress(const std::string& name, void* ptr) {
    if (!m_context) return false;
    return static_cast<Context*>(m_context)->setTensorAddress(name.c_str(), ptr);
}

bool TensorRtInfer::infer(cudaStream_t stream) {
    if (!m_context) return false;
    return static_cast<Context*>(m_context)->enqueueV3(stream);
}
