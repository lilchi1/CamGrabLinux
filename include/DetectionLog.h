// DetectionLog.h — Сервис JSON-логирования детекций (боксов) отдельным файлом.
// Основной лог (PipelineJsonLogger) содержит только счётчики этапов и
// num_detections; сами боксы пишутся сюда — в logs/detections_<cam>.json.
// Дизайн повторяет PipelineJsonLogger: записи накапливаются в очереди и пишутся
// отдельным потоком-писателем (без блокирующего I/O в потоке отображения).
// Файл всегда остаётся валидным JSON-массивом {"detections": [...]}.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

// Одна запись — один обнаруженный объект (бокс) в координатах исходного кадра.
struct DetectionRecord {
    int64_t frameNo = -1;      // номер кадра декодера
    int64_t pts = -1;          // временная метка кадра
    std::string timestamp;     // "YYYY-MM-DD HH:MM:SS"
    int classId = -1;          // индекс класса
    std::string className;     // имя класса ("person", "car", ...)
    float confidence = 0;      // уверенность
    float x1 = 0, y1 = 0;      // левый-верхний угол бокса
    float x2 = 0, y2 = 0;      // правый-нижний угол бокса
};

class DetectionLogger {
public:
    DetectionLogger() = default;
    ~DetectionLogger();

    DetectionLogger(const DetectionLogger&) = delete;
    DetectionLogger& operator=(const DetectionLogger&) = delete;

    // Открытие файла (перезапись), создание папки logs/ и запуск потока-писателя.
    bool begin(const std::string& path);

    // Постановка записи в очередь (потокобезопасно; I/O выполняет писатель).
    void append(const DetectionRecord& rec);

    // Завершение: закрытие JSON-массива, flush, остановка потока-писателя.
    void end();

    bool isOpen() const { return m_open; }

private:
    std::string formatRecord(const DetectionRecord& r) const;
    void writerLoop();

    std::ofstream m_fs;
    std::thread m_thread;
    std::deque<DetectionRecord> m_queue;
    std::mutex m_mtx;
    std::condition_variable m_cv;
    bool m_stop = false;
    bool m_open = false;
    bool m_wroteItem = false;        // в массиве есть хотя бы одна запись
    std::streamoff m_endOffset = 0;  // позиция конца файла (закрытие массива)
};
