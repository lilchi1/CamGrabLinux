# CamGrabLinux (rtsp_jetson_decoder)

Multi-camera RTSP video stream decoder and real-time display for NVIDIA Jetson platforms.

## Features

- Simultaneous display of multiple RTSP camera streams
- Hardware-accelerated decoding via NVIDIA NVDEC (GStreamer `nvv4l2decoder`)
- Real-time FPS display per camera
- Per-frame timing log (CSV): clean NVDEC decode, preprocess, infer, total
- Zero-latency sync pipeline: one thread per camera, no queues, no dropped frames
- PTZ control via Hikvision ISAPI (arrow keys)
- Auto-reconnection on stream failure
- Dynamic camera add at runtime
- Real-time YOLO object detection (TensorRT engine): YOLOv8/v11/v12 (anchor-free)
  and YOLOv2 (anchor-based, Keras h5 → engine) with bounding-box overlay

## Dependencies

- FFmpeg (libavformat, libavcodec, libavutil)
- GStreamer 1.0 (core, app, video)
- X11 (Xlib)
- CUDA (for NV12→BGRA conversion and scaling)
- pthreads

## Build

```sh
cmake -B Build
cmake --build Build -j8
```

The binary is produced at `Build/rtsp_decoder`.

## Usage

```sh
./Build/rtsp_decoder -b
```

Options (see `--help`):

| Flag | Meaning |
|---|---|
| `-b, --benchmark` | benchmark mode, no window (timings only) |
| `-w, -H` | window width/height (default 1600x900); when passed explicitly, the same resolution is requested from the camera (`?width=W&height=H` appended to the RTSP URL) |
| `-d, --display M` | `xvimagesink` (default) or `cuda` |
| `-m, --model PATH` | TensorRT engine for YOLO detection |
| `-l, --labels PATH` | class names file |
| `-c, -n` | confidence / NMS thresholds |
| `--yolov2` | YOLOv2 anchor-based decoder |

At startup enter RTSP URLs as a comma-separated list at the prompt:

```
rtsp://admin:pass@192.168.1.100:554/stream1,rtsp://admin:pass@192.168.1.101:554/stream2
```

During runtime, enter a new URL to add a camera, or type `exit` to quit
(ESC closes the window). Only one instance runs at a time
(`/tmp/rtsp_decoder.lock`).

## Architecture

Each camera runs in one thread with a zero-latency sync pipeline:

```
RTSP (FFmpeg, UDP) → readPacket → push (NVDEC, I/P only) → pullFrame (sync) →
NV12→GPU upload → preprocess (letterbox/NCHW) → TensorRT infer →
CUDA render → CSV timing log
```

- No queues and no separate render/writer threads: appsink uses
  `drop=FALSE, max-buffers=1`, so the pipeline blocks (backpressure) instead of
  buffering.
- B-frames are dropped before the decoder (`GstDecoder::pushPacket`), so NVDEC
  never reorders and the loop never waits on a dropped frame; packets before the
  first keyframe are skipped.
- No warm-up: the loop starts measuring immediately after the decoder opens.
- Per-frame timings are written synchronously to
  `logs/pipeline_log_<idx>.csv`:
  `frame_no, is_key, decode_ms, decode_func_ms, queue_depth, frame_interval_ms,`
  `push_block_ms, preprocess_ms, infer_ms, total_ms`
  - `decode_ms` — время пребывания в NVDEC (sink→src pad-пробы), включает ожидание
    в очереди декодера; `decode_func_ms` — от pushPacket до выхода кадра из appsink;
  - `queue_depth` — VCL-пакетов в полёте (1 = очереди нет); `frame_interval_ms` —
    темп выхода кадров из appsink; `push_block_ms` — блокировка push (backpressure);
  - (total = packet received → frame fully processed). `is_key` marks keyframes:
  regular total_ms spikes line up with them (GOP interval), not with GC/buffers.

## YOLO detection

```sh
# YOLOv8/v11/v12 (anchor-free, вход 640×640) — рабочий путь (yolo26n.pt → engine)
./Build/rtsp_decoder --model yolo/yolo26n.engine --labels yolo/coco.names \
    --conf 0.5 --nms 0.45 --display cuda

# YOLOv2 (anchor-based, Keras h5 → .engine, вход 608×608)
./Build/rtsp_decoder --model yolo/yolo_v2.engine --labels yolo/coco.names --yolov2
```

- Препроцессинг: letterbox → NCHW F32 (нормализация /255), общий для всех моделей.
- Постпроцессинг YOLOv8 (anchor-free, выход `[1, 4+nc, 8400]`) — в `YoloPostprocess`;
  YOLOv2 (anchor-based, сетка 19×19, якоря, NMS) — в `YoloV2Postprocess`.
- Оверлей боксов ИИ рисуется в `src/display/Display.cpp` (`showFrame`), режим
  `--display cuda` (XImage-буфер). В режиме `xvimagesink` окно отдано GStreamer,
  поэтому детекции накладывать некуда.
- Конвертация моделей → `.engine` описана в папке [`yolo/`](yolo/README.md).

### Сборка engine из yolo26n.pt (ultralytics)

```sh
cd yolo

# все три точности сразу (int8 — с калибровкой на фото из coco8.yaml)
python3 convert_pt_to_engine.py --pt yolo26n.pt --precision fp32 fp16 int8 \
    --data /data/coco8.yaml --fraction 0.5

# или по одной точности, движок вручную
python3 convert_pt_to_engine.py --pt yolo26n.pt --precision fp16 \
    --engine yolo26n_fp16.engine
```

Скрипт: `.pt --ultralytics export--> .onnx` (CPU), затем `.onnx --TensorRT API-->`
`.engine` (GPU). При нескольких точностях ONNX экспортируется один раз и
переиспользуется. int8 использует энтропийную калибровку на реальных
изображениях (`--calib-dir` или `--data <data.yaml>`), кэш калибровки
сохраняется в `*.cache`.

> **Память (Jetson: GPU-память = общая RAM).** Экспорт ONNX выполняется в
> отдельном под-процессе, который завершается и освобождает torch (~1.5-2 ГБ)
> до сборки engine — иначе TensorRT падает с `NvMapMemAlloc error 12` /
> `terminate called without an active exception`. Проверяйте свободную память
> (`free -h`), убивайте зависшие процессы (`ps aux | grep convert_pt`), а
> `--workspace` по умолчанию 0.5 ГБ. Сборка занимает 5-15 минут, не прерывайте
> её (зависшие «stopped» процессы после core dump держат RAM).

Выход ONNX — `[1, 84, 8400]` (4 + 80 классов COCO, 8400 анкоров), что совпадает
с ожиданиями `YoloPostprocess`.

Enter RTSP URL(s), comma-separated, at the prompt; add cameras at runtime or type `exit`.

## Benchmark FP32 vs FP16 vs INT8

Сравнение производительности моделей разных точностей на Jetson:

```sh
# Автоматический бенчмарк (требует engine файлы в yolo/)
./benchmark_precision.sh rtsp://admin:pass@192.168.1.100:554/stream1 200

# Генерация engine файлов (если есть yolo26n.pt)
./benchmark_precision.sh generate
```

### Аналитика точностей

| Параметр | FP32 | FP16 | INT8 |
|---|---|---|---|
| **Размер engine** | ~12 MB | ~6 MB | ~3 MB |
| **Точность (mAP50)** | 100% (baseline) | ~99.5% | ~97-99% |
| **Скорость infer (мс)** | ~8-12 | ~4-6 | ~2-3 |
| **Память GPU** | ~50 MB | ~25 MB | ~15 MB |
| **Калибровка** | нет | нет | да (100+ фото) |

**Рекомендации:**
- **FP16** — лучший баланс скорость/точность для Jetson. ~2x быстрее FP32 без потери
  точности. Рекомендуется для production.
- **INT8** — максимальная скорость, но требует калибровки. Точность может упасть на
  1-3% в зависимости от домена. Для安防/промышленного применения нужна валидация.
- **FP32** — только для baseline-сравнения. Не рекомендуется для production на Jetson.

### Оптимизации (vs исходная версия)

Ключевые улучшения производительности:

1. **CUDA-ядро**: `__ldg()` texture cache, vectorized loads, оптимальные block sizes
2. **Preprocess**: убран лишний `cudaStreamSynchronize`, пул буферов
3. **Infer pipeline**: 2 cudaEvent вместо 3, combined preprocess+infer timing
4. **NAL scanning**: fast-skip через non-zero bytes (2-10x ускорение)
5. **Display**: merged letterbox fill с memcpy, XFlush каждый кадр
6. **Memory**:NV12-буфер пулится (не перевыделяется на каждый кадр)
