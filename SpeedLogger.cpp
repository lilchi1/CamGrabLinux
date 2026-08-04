// SpeedLogger.cpp — система логирования скорости аппаратного декодера (NVDEC).
// Два CSV-файла в папке logs/:
//   decode_speed.csv  — сводная таблица по разрешениям;
//   decode_packets.csv — построчный лог: сколько мс декодировался каждый пакет.
// Столбцы по-пакетного лога: resolution, frame_no, pts, codec, source_width,
// source_height, decode_ms, push_block_ms, app_to_dec_ms, dec_to_conv_ms,
// conv_to_sink_ms, display_ms, frame_interval_ms, queue_depth, decoded_at.
#include "SpeedLogger.h"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <ostream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

// Заголовок сводной CSV-таблицы
static const char* kCsvHeader =
    "resolution,codec,source_width,source_height,total_frames,avg_fps,"
    "avg_decode_ms_with_buf,avg_decode_ms_without_buf,avg_push_block_ms,"
    "avg_frame_interval_ms,min_decode_ms,max_decode_ms,last_run";

// Заголовок по-пакетного CSV-лога
static const char* kPktHeader =
    "resolution,frame_no,pts,codec,source_width,source_height,"
    "decode_ms,push_block_ms,app_to_dec_ms,dec_to_conv_ms,conv_to_sink_ms,"
    "display_ms,frame_interval_ms,queue_depth,decoded_at";

// Уникальный идентификатор запуска: "YYYYMMDD-HHMMSS-ffffff" (с микросекундами).
static std::string runIdNow() {
    using namespace std::chrono;
    auto now = system_clock::now();
    time_t t = system_clock::to_time_t(now);
    auto us = duration_cast<microseconds>(now.time_since_epoch()).count() % 1000000;
    struct tm* tm = localtime(&t);
    char buf[40];
    strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", tm);
    char out[64];
    snprintf(out, sizeof(out), "%s-%06lld", buf, (long long)us);
    return out;
}

// Текущее время в формате "YYYY-MM-DD HH:MM:SS"
std::string SpeedLogger::timestamp() {
    time_t now = time(nullptr);
    struct tm* tm = localtime(&now);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm);
    return buf;
}

SpeedLogger& SpeedLogger::instance() {
    static SpeedLogger logger;
    return logger;
}

size_t SpeedLogger::size() const {
    std::lock_guard<std::mutex> lock(m_mtx);
    return m_rows.size();
}

// Разбор одной строки CSV сводной таблицы. false — пустая/заголовок/нечитаема.
bool SpeedLogger::parseLine(const std::string& line, DecodeSpeedRecord& out) {
    std::vector<std::string> f;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) f.push_back(cell);
    if (f.size() < 2 || f[0].empty() || f[0] == "resolution") return false;

    auto parseD = [](const std::string& s, double fallback) {
        try { return std::stod(s); } catch (...) { return fallback; }
    };
    auto parseI = [](const std::string& s, long long fallback) {
        try { return std::stoll(s); } catch (...) { return fallback; }
    };

    out = DecodeSpeedRecord();
    out.resolution = f[0];
    out.codec = f[1];
    if (f.size() > 2) out.srcW = (int)parseI(f[2], 0);
    if (f.size() > 3) out.srcH = (int)parseI(f[3], 0);
    if (f.size() > 4) out.totalFrames = parseI(f[4], 0);
    if (f.size() > 5) out.avgFps = parseD(f[5], 0.0);
    if (f.size() > 6) out.avgDecodeWithBufMs = parseD(f[6], -1.0);
    if (f.size() > 7) out.avgDecodeWithoutBufMs = parseD(f[7], -1.0);
    if (f.size() > 8) out.avgPushBlockMs = parseD(f[8], -1.0);
    if (f.size() > 9) out.avgFrameIntervalMs = parseD(f[9], -1.0);
    if (f.size() > 10) out.minDecodeMs = parseD(f[10], -1.0);
    if (f.size() > 11) out.maxDecodeMs = parseD(f[11], -1.0);
    if (f.size() > 12) out.lastRun = f[12];
    return true;
}

// Загрузка/подготовка каталога logs/
bool SpeedLogger::load(const std::string& dir) {
    std::lock_guard<std::mutex> lock(m_mtx);
    m_dir = dir;
    m_path = dir + "/decode_speed.csv";
    m_pktPath = dir + "/decode_packets.csv";

    // Создание каталога logs/ при отсутствии
    struct stat st;
    if (stat(dir.c_str(), &st) != 0) {
        if (mkdir(dir.c_str(), 0755) != 0 && stat(dir.c_str(), &st) != 0)
            return false;
    }

    m_rows.clear();
    std::ifstream in(m_path);
    if (!in.is_open()) return true;  // файла ещё нет — стартуем с пустой таблицей

    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (first) { first = false; continue; }  // пропуск заголовка
        DecodeSpeedRecord rec;
        if (parseLine(line, rec)) m_rows.push_back(rec);
    }
    return true;
}

// Запись сводной таблицы в CSV (через временный файл + переименование)
void SpeedLogger::writeCsvLocked() {
    if (m_path.empty()) return;
    std::string tmp = m_path + ".tmp";
    std::ofstream out(tmp);
    if (!out.is_open()) return;
    out << kCsvHeader << "\n";
    for (const auto& r : m_rows) {
        out << r.resolution << ','
            << r.codec << ','
            << r.srcW << ','
            << r.srcH << ','
            << r.totalFrames << ','
            << r.avgFps << ','
            << r.avgDecodeWithBufMs << ','
            << r.avgDecodeWithoutBufMs << ','
            << r.avgPushBlockMs << ','
            << r.avgFrameIntervalMs << ','
            << r.minDecodeMs << ','
            << r.maxDecodeMs << ','
            << r.lastRun << '\n';
    }
    out.close();
    if (rename(tmp.c_str(), m_path.c_str()) != 0) {
        // Переименование не удалось (например, другой ФС) — пишем напрямую
        std::ofstream direct(m_path);
        if (!direct.is_open()) return;
        direct << kCsvHeader << "\n";
        for (const auto& r : m_rows) {
            direct << r.resolution << ',' << r.codec << ',' << r.srcW << ','
                   << r.srcH << ',' << r.totalFrames << ',' << r.avgFps << ','
                   << r.avgDecodeWithBufMs << ',' << r.avgDecodeWithoutBufMs << ','
                   << r.avgPushBlockMs << ',' << r.avgFrameIntervalMs << ','
                   << r.minDecodeMs << ',' << r.maxDecodeMs << ',' << r.lastRun << '\n';
        }
    }
}

// Вставка новой строки или обновление существующей в сводной таблице
void SpeedLogger::update(const DecodeSpeedRecord& rec) {
    std::lock_guard<std::mutex> lock(m_mtx);
    for (auto& r : m_rows) {
        if (r.resolution == rec.resolution) {
            r = rec;
            writeCsvLocked();
            return;
        }
    }
    m_rows.push_back(rec);
    writeCsvLocked();
}

// ─── По-пакетный лог ─────────────────────────────────────────────────────────

// Начало сессии: создание временного файла, куда будут писаться пакеты
std::string SpeedLogger::beginRun() {
    std::lock_guard<std::mutex> lock(m_mtx);
    std::string rid = runIdNow();
    if (m_pktPath.empty()) return rid;  // каталог не инициализирован
    std::string tmp = m_dir + "/.packets_" + rid + ".tmp";
    m_runStreams[rid].open(tmp, std::ios::out | std::ios::trunc);
    m_runTemp[rid] = tmp;
    return rid;
}

// Запись одного пакета в текущую сессию (во временный файл)
void SpeedLogger::appendPacket(const std::string& runId, const PacketRecord& p) {
    std::lock_guard<std::mutex> lock(m_mtx);
    auto it = m_runStreams.find(runId);
    if (it == m_runStreams.end()) return;
    std::ofstream& out = it->second;
    out << p.resolution << ','
        << p.frameNo << ',' << p.pts << ',' << p.codec << ','
        << p.srcW << ',' << p.srcH << ','
        << p.decodeMs << ',' << p.pushBlockMs << ','
        << p.appToDecMs << ',' << p.decToConvMs << ',' << p.convToSinkMs << ','
        << p.displayMs << ',' << p.frameIntervalMs << ','
        << p.queueDepth << ',' << p.decodedAt << '\n';
}

// Завершение сессии: строки разрешения resolution в общем CSV заменяются
// данными этого запуска (обновление), новые разрешения — добавляются.
void SpeedLogger::endRun(const std::string& runId, const std::string& resolution) {
    std::lock_guard<std::mutex> lock(m_mtx);
    if (m_pktPath.empty()) return;
    auto it = m_runStreams.find(runId);
    if (it == m_runStreams.end()) return;
    it->second.close();

    std::string tmp = m_runTemp[runId];

    // Строки текущего запуска из временного файла
    std::vector<std::string> newRows;
    {
        std::ifstream tin(tmp);
        std::string line;
        while (std::getline(tin, line))
            if (!line.empty()) newRows.push_back(line);
    }

    // Чтение существующего CSV: заголовок + строки остальных разрешений.
    // Строки этого разрешения удаляются — будут заменены данными текущего запуска.
    std::vector<std::string> keep;
    bool headerPresent = false;
    {
        std::ifstream in(m_pktPath);
        std::string line;
        bool first = true;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            if (first) { headerPresent = true; keep.push_back(line); first = false; continue; }
            size_t comma = line.find(',');
            std::string res = (comma == std::string::npos) ? line : line.substr(0, comma);
            if (res != resolution) keep.push_back(line);
        }
    }
    if (!headerPresent) keep.insert(keep.begin(), kPktHeader);

    // Запись объединённого файла (временный + атомарная замена)
    std::string tmpOut = m_pktPath + ".tmp";
    {
        std::ofstream out(tmpOut);
        for (const auto& l : keep) out << l << '\n';
        for (const auto& l : newRows) out << l << '\n';
    }
    rename(tmpOut.c_str(), m_pktPath.c_str());

    // Очистка временного файла и сессии
    remove(tmp.c_str());
    m_runStreams.erase(runId);
    m_runTemp.erase(runId);
}