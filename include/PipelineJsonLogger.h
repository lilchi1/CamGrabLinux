// PipelineJsonLogger.h — Сервис JSON-логирования этапов обработки кадра.
// Для каждой камеры создаётся свой экземпляр. Записи накапливаются в очереди
// и пишутся отдельным потоком-писателем, чтобы в потоке GStreamer/CUDA не было
// блокирующего I/O. Файл всегда остаётся валидным JSON-массивом {"frames": [...]}.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "Detection.h"

// Одна запись лога — один кадр со всеми этапами обработки.
struct PipelineLogRecord {
    // Этап 0 — идентификация кадра
    int64_t frameNo = -1;          // номер кадра декодера
    int64_t pts = -1;              // временная метка кадра
    std::string timestamp;         // "YYYY-MM-DD HH:MM:SS"
    std::string source;            // "1920x1080"
    std::string codec;             // "H.264" / "H.265"
    // Этап 1 — декодирование (NVDEC)
    double decodeMs = -1.0;        // чистое время NVDEC: sink-пад → src-пад
    double decodeFuncMs = -1.0;    // от входа пакета в декодер (pushPacket) до кадра
    double pushBlockMs = -1.0;     // блокировка appsrc push (backpressure)
    double frameIntervalMs = -1.0; // интервал между кадрами на выходе appsink
    int queueDepth = 0;            // глубина буферизации (невостребованные пуши)
    // Этап 2 — препроцессинг YOLO
    double uploadMs = -1.0;        // NV12 host→device upload
    double preprocessMs = -1.0;    // letterbox + normalize + NCHW
    // Этап 3 — инференс TensorRT
    double inferMs = -1.0;
    // Этап 4 — отрисовка
    double renderMs = -1.0;        // отрисовка видео без детекций
    double renderAiMs = -1.0;      // отрисовка видео с ИИ-боксами
    // Результат детекции
    Detections detections;         // боксы в координатах исходного кадра
};

class PipelineJsonLogger {
public:
    PipelineJsonLogger() = default;
    ~PipelineJsonLogger();

    PipelineJsonLogger(const PipelineJsonLogger&) = delete;
    PipelineJsonLogger& operator=(const PipelineJsonLogger&) = delete;

    // Открытие файла (перезапись), создание папки logs/ и запуск потока-писателя.
    bool begin(const std::string& path);

    // Постановка записи в очередь (потокобезопасно; I/O выполняет писатель).
    void append(const PipelineLogRecord& rec);

    // Завершение: закрытие JSON-массива, flush, остановка потока-писателя.
    void end();

    bool isOpen() const { return m_open; }

private:
    std::string formatRecord(const PipelineLogRecord& r) const;
    void writerLoop();

    std::ofstream m_fs;
    std::thread m_thread;
    std::deque<PipelineLogRecord> m_queue;
    std::mutex m_mtx;
    std::condition_variable m_cv;
    bool m_stop = false;
    bool m_open = false;
    bool m_wroteItem = false;        // в массиве есть хотя бы одна запись
    std::streamoff m_endOffset = 0;  // позиция конца файла (закрытие массива)
};
