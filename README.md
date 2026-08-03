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
