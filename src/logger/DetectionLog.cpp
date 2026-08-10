// DetectionLog.cpp — реализация JSON-логгера детекций (боксов).
#include "DetectionLog.h"

#include <cstdio>
#include <sys/stat.h>
#include <sys/types.h>

DetectionLogger::~DetectionLogger() { end(); }

bool DetectionLogger::begin(const std::string& path) {
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

    m_fs << "{\"detections\": [\n";
    m_fs.flush();
    m_endOffset = m_fs.tellp();

    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_stop = false;
        m_open = true;
        m_wroteItem = false;
        m_thread = std::thread(&DetectionLogger::writerLoop, this);
    }
    return true;
}

void DetectionLogger::append(const DetectionRecord& rec) {
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        if (!m_open)
            return;
        m_queue.push_back(rec);
    }
    m_cv.notify_one();
}

void DetectionLogger::end() {
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

void DetectionLogger::writerLoop() {
    for (;;) {
        DetectionRecord rec;
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

std::string DetectionLogger::formatRecord(const DetectionRecord& r) const {
    std::string s = "{";
    s += "\"frame_no\":" + std::to_string(r.frameNo) + ",";
    s += "\"pts\":" + std::to_string(r.pts) + ",";
    s += "\"timestamp\":\"" + r.timestamp + "\",";
    s += "\"class\":" + std::to_string(r.classId) + ",";
    s += "\"class_name\":\"" + r.className + "\",";
    char buf[128];
    snprintf(buf, sizeof(buf), "\"confidence\":%.4f,", r.confidence);
    s += buf;
    snprintf(buf, sizeof(buf),
             "\"bbox\":[%.1f,%.1f,%.1f,%.1f]}",
             r.x1, r.y1, r.x2, r.y2);
    s += buf;
    return s;
}
