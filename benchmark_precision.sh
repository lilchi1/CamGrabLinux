#!/bin/bash
# benchmark_precision.sh — Сборка и тестирование FP32/FP16/INT8 движков YOLO
#
# Использование:
#   ./benchmark_precision.sh [rtsp_url] [num_frames]
#
# Требования:
#   - Собранный Build/rtsp_decoder
#   - engine файлы в yolo/ (yolo26n_fp32.engine, yolo26n_fp16.engine, yolo26n_int8.engine)
#   - labels: yolo/coco.names
#
# Пример:
#   ./benchmark_precision.sh rtsp://admin:pass@192.168.1.100:554/stream1 200

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_DIR="Build"
BINARY="$BUILD_DIR/rtsp_decoder"
LABELS="yolo/coco.names"
RTSP_URL="${1:-rtsp://admin:pass@192.168.1.100:554/stream1}"
NUM_FRAMES="${2:-200}"
LOG_DIR="benchmark_results"
mkdir -p "$LOG_DIR"

# Проверяем наличие бинарника
if [ ! -f "$BINARY" ]; then
    echo "Бинарник не найден: $BINARY"
    echo "Собираю..."
    cmake -B Build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    cmake --build Build -j$(nproc) 2>&1 | tail -5
fi

# Проверяем наличие движков
echo "=== Доступные engine файлы ==="
ls -lh yolo/*.engine 2>/dev/null || echo "Engine файлы не найдены в yolo/"

# Функция запуска бенчмарка для одной точности
run_benchmark() {
    local engine_path="$1"
    local precision="$2"
    local log_file="$LOG_DIR/benchmark_${precision}.csv"
    local summary_file="$LOG_DIR/benchmark_${precision}_summary.txt"
    
    echo ""
    echo "=== БЕНЧМАРК: $precision ==="
    echo "Engine: $engine_path"
    echo "RTSP: $RTSP_URL"
    echo "Кадров: $NUM_FRAMES"
    
    if [ ! -f "$engine_path" ]; then
        echo "ОШИБКА: engine не найден: $engine_path"
        echo "Пропуск $precision"
        return 1
    fi
    
    # Запуск в benchmark mode (--benchmark) + детекция
    # Ввод: URL камеры, затем exit после NUM_FRAMES кадров
    echo "Запуск бенчмарка..."
    
    # Используем timeout чтобы не зависнуть
    timeout 120 "$BINARY" \
        -b \
        -m "$engine_path" \
        -l "$LABELS" \
        -d cuda \
        -w 1200 -H 800 \
        --conf 0.35 --nms 0.45 \
        < <(echo "$RTSP_URL"; sleep 30; echo "exit") \
        2>&1 | tee "$summary_file"
    
    # Копируем CSV лог если есть
    if ls logs/pipeline_log_*.csv 1>/dev/null 2>&1; then
        cp logs/pipeline_log_0.csv "$log_file" 2>/dev/null || true
    fi
    
    echo "Результаты сохранены: $summary_file"
    echo "CSV лог: $log_file"
    
    # Извлекаем ключевые метрики
    echo ""
    echo "--- Ключевые метрики ($precision) ---"
    grep -E "(SYNC FPS|avg_total|avg_decode|avg_preprocess|avg_infer|DET:)" "$summary_file" 2>/dev/null | tail -10 || true
}

# Генерация engine файлов если нет
generate_engines() {
    echo ""
    echo "=== Генерация engine файлов ==="
    
    if [ ! -f "yolo/yolo26n.pt" ]; then
        echo "yolo26n.pt не найден — генерация engine невозможна"
        echo "Скачайте модель: https://github.com/ultralytics/assets/releases/"
        return 1
    fi
    
    echo "Генерация FP32 engine..."
    python3 yolo/convert_pt_to_engine.py \
        --pt yolo/yolo26n.pt \
        --precision fp32 \
        --engine yolo/yolo26n_fp32.engine \
        --workspace 0.5 2>&1 | tail -3
    
    echo "Генерация FP16 engine..."
    python3 yolo/convert_pt_to_engine.py \
        --pt yolo/yolo26n.pt \
        --precision fp16 \
        --engine yolo/yolo26n_fp16.engine \
        --workspace 0.5 2>&1 | tail -3
    
    echo "Генерация INT8 engine (требует калибровки)..."
    # Для INT8 нужен папка с изображениями
    if [ -d "yolo/calib_images" ]; then
        python3 yolo/convert_pt_to_engine.py \
            --pt yolo/yolo26n.pt \
            --precision int8 \
            --engine yolo/yolo26n_int8.engine \
            --calib-dir yolo/calib_images \
            --workspace 0.5 2>&1 | tail -3
    else
        echo "INT8: нет папки yolo/calib_images — пропуск"
        echo "Создайте yolo/calib_images/ с 100+ изображениями для калибровки"
    fi
    
    echo ""
    echo "=== Сгенерированные engine ==="
    ls -lh yolo/*.engine 2>/dev/null
}

# Основной режим
echo "=== CamGrabLinux Benchmark ==="
echo "Бинарник: $BINARY"
echo "Модель: yolo/yolo26n"
echo "Кадров для замера: $NUM_FRAMES"
echo "RTSP URL: $RTSP_URL"
echo ""

# Проверяем engine файлы
ENGINES_FOUND=0
for p in fp32 fp16 int8; do
    if [ -f "yolo/yolo26n_${p}.engine" ]; then
        echo "Найден: yolo/yolo26n_${p}.engine"
        ENGINES_FOUND=$((ENGINES_FOUND + 1))
    fi
done

if [ "$ENGINES_FOUND" -eq 0 ]; then
    echo "Engine файлы не найдены!"
    echo "Варианты:"
    echo "  1. Скопируйте engine файлы в yolo/"
    echo "  2. Сгенерируйте: ./benchmark_precision.sh generate"
    echo ""
    
    if [ "${1:-}" = "generate" ]; then
        generate_engines
    else
        echo "Запустите: $0 generate [url] [frames]"
    fi
    exit 0
fi

# Запуск бенчмарков
echo ""
echo "=== ЗАПУСК БЕНЧМАРКОВ ==="

RESULTS_FILE="$LOG_DIR/comparison.txt"
echo "=== СРАВНЕНИЕ ТОЧНОСТЕЙ ===" > "$RESULTS_FILE"
echo "Дата: $(date)" >> "$RESULTS_FILE"
echo "Кадров: $NUM_FRAMES" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

for precision in fp32 fp16 int8; do
    engine="yolo/yolo26n_${precision}.engine"
    if [ -f "$engine" ]; then
        run_benchmark "$engine" "$precision" 2>&1 | tee -a "$RESULTS_FILE"
    fi
done

echo ""
echo "=== ИТОГОВОЕ СРАВНЕНИЕ ==="
echo ""
echo "Результаты сохранены в: $LOG_DIR/"
echo ""
echo "Для ручного сравнения:"
echo "  cat $LOG_DIR/benchmark_fp32_summary.txt"
echo "  cat $LOG_DIR/benchmark_fp16_summary.txt"
echo "  cat $LOG_DIR/benchmark_int8_summary.txt"
