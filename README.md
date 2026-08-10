# CamGrabLinux (rtsp_jetson_decoder)

Multi-camera RTSP video stream decoder and real-time display for NVIDIA Jetson platforms.

## Features

- Simultaneous display of multiple RTSP camera streams
- Hardware-accelerated decoding via NVIDIA NVDEC (GStreamer `nvv4l2decoder`)
- Real-time FPS display per camera
- Optional per-frame decode speed logging (console + JSON)
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
cmake -B build
cmake --build build
```

The binary is produced at `build/rtsp_decoder`.

## Usage

```sh
./build/rtsp_decoder
```

At startup the program asks whether to log decode speed (decode_ms per frame).
Then enter RTSP URLs as a comma-separated list at the prompt:

```
rtsp://admin:pass@192.168.1.100:554/stream1,rtsp://admin:pass@192.168.1.101:554/stream2
```

During runtime, enter a new URL to add a camera, or type `exit` to quit.

## Architecture

Each camera runs in its own thread: RTSP read (FFmpeg) → NVDEC decode
(GStreamer `nvv4l2decoder`) → NV12→BGRA conversion and scaling (CUDA) → X11 window.

If decode speed logging is enabled, per-frame decode times are printed and
written to `decode_times_<idx>.json`.

## YOLO detection

```sh
# YOLOv8/v11/v12 (anchor-free, вход 640×640) — рабочий путь (yolo26n.pt → engine)
./build/rtsp_decoder --model yolo/yolo26n.engine --labels yolo/coco.names \
    --conf 0.5 --nms 0.45 --display cuda

# YOLOv2 (anchor-based, Keras h5 → .engine, вход 608×608)
./build/rtsp_decoder --model yolo/yolo_v2.engine --labels yolo/coco.names --yolov2
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
python3 -c "
from ultralytics import YOLO
YOLO('yolo26n.pt').export(format='onnx', imgsz=640, opset=13, end2end=False, nms=False)
"
/usr/src/tensorrt/bin/trtexec --onnx=yolo26n.onnx --saveEngine=yolo26n.engine --fp16
```

Выход ONNX — `[1, 84, 8400]` (4 + 80 классов COCO, 8400 анкоров), что совпадает
с ожиданиями `YoloPostprocess`.

Enter RTSP URL(s), comma-separated, at the prompt; add cameras at runtime or type `exit`.
