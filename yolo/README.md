# YOLO (yolo/)

Здесь лежат файлы для детекции объектов и их интеграция с проектом
`rtsp_jetson_decoder` (C++/CUDA/TensorRT).

## Содержимое

| Файл | Описание |
|------|----------|
| `yolo_model_complete.h5` | Обученная **Keras YOLOv2**-модель. Вход NHWC `[1,608,608,3]`, выход NHWC `[1,19,19,425]` = 5 якорей × (5 + 80 классов COCO). Есть кастомный слой `Lambda space_to_depth` (passthrough). |
| `main.py` | Проверка модели на одном изображении: загрузка h5 → предикт → YOLOv2-декод → рамки на `annotated.jpg`. Требует `tensorflow`. |
| `convert_h5_to_engine.py` | Конвейер **h5 → ONNX → TensorRT .engine** для запуска в C++-проекте (см. ниже). |
| `coco.names` | 80 имён классов COCO (для флага `--labels`). |
| `ultralytics/` | Python-пакет Ultralytics (экспорт YOLOv8/11 в .engine, обучение). Git-история удалена. |
| `txt.txt`, `gitattributes` | Оригинальные файлы из папки источника. |

> Большие файлы (`*.h5`, `*.onnx`, `*.engine`, `ultralytics/`) исключены из git
> через `.gitignore` в корне проекта.

## Что это за модель

Это **YOLOv2** (darknet19 + SpaceToDepth passthrough), а НЕ YOLOv8/11/12.
Поэтому в проекте для неё нужен отдельный постпроцессор (anchor-based):

- декод: `cx = (sigmoid(tx) + gx) * 32`, `cy = (sigmoid(ty) + gy) * 32`,
  `w = anchor_w * exp(tw)`, `h = anchor_h * exp(th)`, сетка `19×19`, stride `32`
- выход модели после конвертации: NCHW `[1, 425, 19, 19]` (каналы-major)
- 80 классов COCO, уверенность = `sigmoid(obj) * max(sigmoid(cls))`

## Как получить .engine (один раз, на машине с интернетом)

На этой Jetson нет интернета и не установлены `tensorflow`/`torch`, поэтому
конвертация выполняется на другой машине (или после установки зависимостей):

```sh
pip install tensorflow tf2onnx numpy opencv-python-headless onnxruntime

cd yolo
python3 convert_h5_to_engine.py --h5 yolo_model_complete.h5 \
    --engine yolo_v2.engine --fp16
```

Скрипт:
1. грузит h5 (с `custom_objects` для Lambda),
2. оборачивает модель: вход NCHW `[1,3,608,608]`, выход NCHW `[1,425,19,19]`,
3. экспортирует ONNX (tf2onnx) и проверяет форму,
4. собирает `.engine` через `trtexec --fp16`,
5. печатает I/O-тензоры engine для сверки с постпроцессором.

> **Важно про якоря.** По умолчанию используются COCO YAD2K (5 якорей в пикселях
> входа 608/416). Если модель обучалась с другими якорями, укажите их:
> `--anchors "w1,h1,w2,h2,..."`. Те же значения передавайте флагом `--v2-anchors`
> при запуске проекта.

## Запуск в проекте

```sh
cmake --build Build
./Build/rtsp_decoder --model yolo/yolo_v2.engine \
    --labels yolo/coco.names --yolov2 \
    --v2-grid 19 \
    --v2-anchors "18.33,21.68,59.98,66.0,106.83,175.18,252.25,112.89,312.66,293.38"
```

Флаги детекции:

| Флаг | Описание |
|------|----------|
| `-m, --model PATH` | путь к `.engine` (пусто = без детекции) |
| `-l, --labels PATH` | файл имён классов |
| `-c, --conf F` | порог уверенности (0.35) |
| `-n, --nms F` | порог NMS (0.45) |
| `--yolov2` | включить YOLOv2-декод (иначе YOLOv8/11/12 anchor-free) |
| `--v2-grid N` | сетка выхода (19 для 608×608) |
| `--v2-anchors L` | якоря, пары w,h в пикселях входа модели |
| `--in-size N` | размер входа модели (0 = авто: 640 для v8, `grid*32` для v2) |

## Как это устроено в коде

- `src/display/CudaKernels.cu` — ядро `yoloV2DecodeKernel` (декод на GPU,
  сигмоиды/экспоненты, маппинг в координаты кадра);
- `include/YoloV2Postprocess.h`, `src/inference/YoloV2Postprocess.cpp` — декод + NMS;
- `src/inference/InferPipeline.cpp` — `initV2()`, выбор постпроцессора в `run()`;
- `src/core/CameraThread.cpp` — подключение YOLOv2-пути по флагу `--yolov2`;
- `src/main.cpp` — CLI-флаги.

Препроцессинг общий (letterbox → NCHW F32 `/255`), поэтому модель и
YOLOv8/11/12-движки работают через один `CvcudaPreprocessor`.
