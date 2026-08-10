// PipelineJsonLogger.cpp — реализация JSON-логгера этапов обработки кадра.
#include "PipelineJsonLogger.h"

#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>

namespace {

// Форматирование времени в мс: "%.4f"; -1.0 (нет данных) → "-1".
std::string formatMs(double v) {
    char buf[32];
    if (v < 0.0)
        snprintf(buf, sizeof(buf), "-1");
    else
        snprintf(buf, sizeof(buf), "%.4f", v);
    return std::string(buf);
}

}  // namespace

PipelineJsonLogger::~PipelineJsonLogger() { end(); }

bool PipelineJsonLogger::begin(const std::string& path) {
    end();

    // Создание родительской папки (logs/) при необходимости
    size_t pos = path.find_last_of('/');
    if (pos != std::string::npos && pos > 0) {
        std::string dir = path.substr(0, pos);
        mkdir(dir.c_str(), 0755);
    }

    m_fs.open(path, std::ios::out | std::ios::trunc);
    if (!m_fs.is_open())
        return false;

    m_fs << "{\"frames\": [\n";
    m_fs.flush();
    m_endOffset = m_fs.tellp();

    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_stop = false;
        m_open = true;
        m_wroteItem = false;
        m_thread = std::thread(&PipelineJsonLogger::writerLoop, this);
    }
    return true;
}

void PipelineJsonLogger::append(const PipelineLogRecord& rec) {
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        if (!m_open)
            return;
        m_queue.push_back(rec);
    }
    m_cv.notify_one();
}

void PipelineJsonLogger::end() {
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        if (!m_open)
            return;
        m_stop = true;
    }
    m_cv.notify_all();
    if (m_thread.joinable())
        m_thread.join();

    // Закрытие JSON-массива: дописываем "]}", перезаписывая временный хвост.
    if (m_wroteItem)
        m_fs.seekp(m_endOffset - 2);
    else
        m_fs.seekp(m_endOffset);
    m_fs << "]}";
    m_fs.flush();
    m_fs.close();

    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_open = false;
    }
}

void PipelineJsonLogger::writerLoop() {
    for (;;) {
        PipelineLogRecord rec;
        {
            std::unique_lock<std::mutex> lock(m_mtx);
            m_cv.wait_for(lock, std::chrono::milliseconds(200), [&] {
                return m_stop || !m_queue.empty();
            });
            if (m_stop && m_queue.empty())
                break;
            if (m_queue.empty())
                continue;
            rec = std::move(m_queue.front());
            m_queue.pop_front();
        }
        std::string line = formatRecord(rec);
        if (m_wroteItem) {
            m_fs.seekp(m_endOffset - 2);
            m_fs << ",\n";
        } else {
            m_fs.seekp(m_endOffset);
            m_wroteItem = true;
        }
        m_fs << line << "\n]}";
        m_fs.flush();
        m_endOffset = m_fs.tellp();
    }
}

std::string PipelineJsonLogger::formatRecord(const PipelineLogRecord& r) const {
    std::string s = "{";
    s += "\"frame_no\":" + std::to_string(r.frameNo) + ",";
    s += "\"pts\":" + std::to_string(r.pts) + ",";
    s += "\"timestamp\":\"" + r.timestamp + "\",";
    s += "\"source\":\"" + r.source + "\",";
    s += "\"codec\":\"" + r.codec + "\",";
    s += "\"decode_ms\":" + formatMs(r.decodeMs) + ",";
    s += "\"decode_func_ms\":" + formatMs(r.decodeFuncMs) + ",";
    s += "\"push_block_ms\":" + formatMs(r.pushBlockMs) + ",";
    s += "\"frame_interval_ms\":" + formatMs(r.frameIntervalMs) + ",";
    s += "\"queue_depth\":" + std::to_string(r.queueDepth) + ",";
    s += "\"upload_ms\":" + formatMs(r.uploadMs) + ",";
    s += "\"preprocess_ms\":" + formatMs(r.preprocessMs) + ",";
    s += "\"infer_ms\":" + formatMs(r.inferMs) + ",";
    s += "\"render_ms\":" + formatMs(r.renderMs) + ",";
    s += "\"render_ai_ms\":" + formatMs(r.renderAiMs) + ",";
    s += "\"num_detections\":" + std::to_string(r.detections.size()) + ",";
    s += "\"detections\":[";
    for (size_t i = 0; i < r.detections.size(); ++i) {
        const Detection& d = r.detections[i];
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "{\"class\":%d,\"confidence\":%.4f,\"bbox\":[%.1f,%.1f,%.1f,%.1f]}",
                 d.classId, d.confidence, d.x1, d.y1, d.x2, d.y2);
        s += buf;
        if (i + 1 < r.detections.size())
            s += ",";
    }
    s += "]}";
    return s;
}
