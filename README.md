# CamGrabLinux (rtsp_jetson_decoder)

Multi-camera RTSP video stream decoder and real-time display for NVIDIA Jetson platforms.

## Features

- Simultaneous display of multiple RTSP camera streams
- Hardware-accelerated decoding via NVIDIA NVDEC (V4L2)
- Software decoding fallback via FFmpeg libavcodec
- Pipe-based MJPEG decoding via external ffmpeg process
- Real-time FPS display per camera
- Auto-reconnection on stream failure
- Dynamic camera add at runtime

## Dependencies

- FFmpeg (libavformat, libavcodec, libavutil, libswscale)
- OpenCV (core, imgcodecs, imgproc)
- X11 (Xlib)
- NVIDIA Jetson Multimedia API (`/usr/src/jetson_multimedia_api`)
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

Enter RTSP URLs as a comma-separated list at the prompt:
```
rtsp://admin:pass@192.168.1.100:554/stream1,rtsp://admin:pass@192.168.1.101:554/stream2
```

During runtime, enter a new URL to add a camera, or type `exit` to quit.

### Environment variables

| Variable | Effect |
|---|---|
| `PIPE_DECODER=1` | Use external ffmpeg process for MJPEG decoding |
| `USE_NVDEC=0` | Force software decoding (skip NVDEC) |

## Architecture

Each camera runs in its own thread with three decoding paths:
1. **PipeDecoder** (PIPE_DECODER=1) — spawns ffmpeg, reads MJPEG via pipe
2. **NvV4l2Decoder** (default) — hardware NVDEC via Jetson V4L2 API
3. **SwDecoder** (USE_NVDEC=0) — software FFmpeg libavcodec

All decoders produce NV12 frames rendered via X11.
