// TensorRtInfer.h — Загрузка TensorRT engine и выполнение инференса (TRT 10 API).
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// Информация об одной входной/выходной тензоре engine.
struct TrtBinding {
    std::string name;          // имя тензора
    bool isInput = false;      // вход или выход
    int ndim = 0;              // размерность
    int64_t dims[8] = {0};     // форма
    size_t volume = 0;         // число элементов
    size_t byteSize = 0;       // размер в байтах
    void* devicePtr = nullptr; // device-буфер (выделяется allocate())
};

// Обёртка над TensorRT 10: сериализованный .engine → runtime → контекст.
// Вход/выход привязываются через setTensorAddress (явная адресация, TRT 10).
class TensorRtInfer {
public:
    TensorRtInfer();
    ~TensorRtInfer();

    // Десериализация .engine файла и создание execution context.
    bool load(const std::string& enginePath);

    // Выделение device-буферов для всех входных/выходных тензоров.
    bool allocate();

    // Освобождение всех ресурсов (буферы, контекст, engine, runtime).
    void cleanup();

    bool ready() const { return m_engine && m_context; }

    // Первый входной тензор (наш препроцессинг пишет сюда) — если модель имеет
    // несколько входов, используйте setTensorAddress() напрямую.
    void* inputPtr() const;
    size_t inputBytes() const;

    // Первый выходной тензор (сырой вывод YOLO-головы).
    void* outputPtr() const;
    size_t outputBytes() const;

    // Установка адреса тензора по имени.
    bool setTensorAddress(const std::string& name, void* ptr);

    // Запуск инференса на указанном потоке (асинхронно).
    bool infer(cudaStream_t stream);

    // Доступ к привязкам.
    const std::vector<TrtBinding>& bindings() const { return m_bindings; }

private:
    // TensorRT 10 API: внутренние типы объявляем только в .cpp, чтобы не
    // тянуть NvInfer.h в каждый включающий файл.

    void* m_runtime = nullptr;         // nvinfer1::IRuntime*
    void* m_engine = nullptr;          // nvinfer1::ICudaEngine*
    void* m_context = nullptr;         // nvinfer1::IExecutionContext*
    std::vector<TrtBinding> m_bindings;
    bool m_allocated = false;
};
