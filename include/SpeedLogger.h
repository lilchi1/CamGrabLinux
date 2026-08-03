// SpeedLogger.h — система логирования скорости аппаратного декодера (NVDEC).
// Результаты замеров скорости декодирования сохраняются в CSV в папке logs/:
//   1) logs/decode_packets.csv  — построчный (по-пакетный) лог: для каждого
//      пакета/кадра записано, за сколько мс он был декодирован и задержки
//      по элементам пайплайна (app_to_dec/dec_to_conv/conv_to_sink/display).
//      Один файл для всех разрешений; при повторном запуске того же разрешения
//      его строки обновляются (заменяются данными последнего запуска).
//   2) logs/decode_speed.csv    — сводная таблица по разрешениям (обновление
//      существующей строки либо добавление новой).
#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

// Одна строка сводной CSV-таблицы (logs/decode_speed.csv)
struct DecodeSpeedRecord {
    std::string resolution;              // ключ строки: "1920x1080" (выбранное окно)
    std::string codec;                   // "H.264" / "H.265" / "Unknown"
    int srcW = 0;                        // исходное разрешение декодируемого потока
    int srcH = 0;
    long long totalFrames = 0;           // число измеренных кадров за запуск
    double avgFps = 0.0;                 // средний FPS за запуск
    double avgDecodeWithBufMs = -1.0;    // средний decode_ms по всем кадрам (с буферизацией)
    double avgDecodeWithoutBufMs = -1.0; // средний decode_ms при queue_depth==0 (без буферизации)
    double avgPushBlockMs = -1.0;        // среднее время блокировки push (backpressure)
    double avgFrameIntervalMs = -1.0;    // средний интервал между кадрами на выходе декодера
    double minDecodeMs = -1.0;           // минимальный decode_ms
    double maxDecodeMs = -1.0;           // максимальный decode_ms
    std::string lastRun;                 // метка времени последнего замера "YYYY-MM-DD HH:MM:SS"
};

// Одна строка по-пакетного лога (logs/decode_packets.csv): конкретный пакет,
// время его декодирования и задержки по элементам пайплайна.
struct PacketRecord {
    std::string resolution;       // ключ: выбранное разрешение окна "1920x1080"
    long long frameNo = 0;        // порядковый номер кадра/пакета (с 1)
    int64_t pts = 0;              // PTS пакета (временная метка)
    std::string codec;            // "H.264" / "H.265" / "Unknown"
    int srcW = 0;                 // исходное разрешение потока
    int srcH = 0;
    double decodeMs = -1.0;       // время декодирования этого пакета, мс
    double pushBlockMs = -1.0;    // блокировка push для этого пакета, мс
    double appToDecMs = -1.0;     // appsrc → выход nvv4l2decoder, мс
    double decToConvMs = -1.0;    // nvv4l2decoder → выход nvvidconv, мс
    double convToSinkMs = -1.0;   // nvvidconv → вход appsink, мс
    double displayMs = -1.0;      // обработка кадра в колбэке (CUDA+X11), мс
    double frameIntervalMs = -1.0;// интервал с предыдущим кадром, мс
    int queueDepth = 0;           // глубина очереди (буферизация) на момент декодирования
    std::string decodedAt;        // метка времени "YYYY-MM-DD HH:MM:SS"
};

class SpeedLogger {
public:
    static SpeedLogger& instance();

    // Загрузка/подготовка каталога logs/ (создаётся при отсутствии).
    bool load(const std::string& dir);

    // ─── Сводная таблица по разрешениям (logs/decode_speed.csv) ──────────
    // Вставка новой строки или обновление существующей (по ключу resolution)
    // с немедленной записью файла. Потокобезопасна.
    void update(const DecodeSpeedRecord& rec);

    // Текущее число строк сводной таблицы.
    size_t size() const;

    // ─── По-пакетный лог (logs/decode_packets.csv) ────────────────────────
    // Начало сессии логирования одного потока камеры. Возвращает run_id,
    // который затем передаётся в appendPacket/endRun. Строки пишутся во
    // временный файл и вливаются в общий CSV при endRun.
    std::string beginRun();

    // Запись одного декодированного пакета в текущую сессию.
    void appendPacket(const std::string& runId, const PacketRecord& p);

    // Завершение сессии: строки разрешения resolution в общем CSV заменяются
    // данными текущего запуска (если разрешение ранее уже логировалось),
    // либо добавляются новым блоком (если разрешение новое).
    void endRun(const std::string& runId, const std::string& resolution);

    // Текущая метка времени "YYYY-MM-DD HH:MM:SS".
    static std::string timestamp();

private:
    SpeedLogger() = default;
    std::string m_dir;
    std::string m_path;    // logs/decode_speed.csv
    std::string m_pktPath; // logs/decode_packets.csv
    std::vector<DecodeSpeedRecord> m_rows;   // строки сводной таблицы
    std::map<std::string, std::ofstream> m_runStreams; // run_id -> временный файл
    std::map<std::string, std::string> m_runTemp;      // run_id -> путь временного файла
    mutable std::mutex m_mtx;

    bool parseLine(const std::string& line, DecodeSpeedRecord& out);
    void writeCsvLocked();
};