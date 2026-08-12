# Jetson RTSP Decoder — build configuration.
cmake_minimum_required(VERSION 3.10)
project(rtsp_jetson_decoder CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Оптимизация по умолчанию (если не задано пользователем)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Detect Jetson GPU architecture (sm_72 for Xavier, sm_87 for Orin)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    execute_process(COMMAND nvidia-smi --query-gpu=name --format=csv,noheader
                    OUTPUT_VARIABLE GPU_NAME ERROR_QUIET)
    if(GPU_NAME MATCHES "Orin")
        set(CMAKE_CUDA_ARCHITECTURES 87)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 72)
    endif()
endif()

# FFmpeg libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAVFORMAT REQUIRED libavformat)
pkg_check_modules(LIBAVCODEC  REQUIRED libavcodec)
pkg_check_modules(LIBAVUTIL   REQUIRED libavutil)

# GStreamer (аппаратный декодер NVDEC через nvv4l2decoder на JetPack 6)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
pkg_check_modules(GSTAPP     REQUIRED gstreamer-app-1.0)
pkg_check_modules(GSTVIDEO   REQUIRED gstreamer-video-1.0)

# CV-CUDA (препроцессинг для YOLO: NV12→RGB→Resize→NCHW).
# nvcv_types должен быть найден ДО cvcuda (cvcuda ссылается на его target).
set(cvcuda_DIR "/opt/nvidia/cvcuda0/lib/cmake/cvcuda")
set(nvcv_types_DIR "/opt/nvidia/cvcuda0/lib/cmake/nvcv_types")
find_package(nvcv_types CONFIG REQUIRED)
find_package(cvcuda CONFIG REQUIRED)

# TensorRT (запуск .engine YOLO)
find_library(NVINFER_LIB nvinfer PATHS /usr/lib/aarch64-linux-gnu REQUIRED)

# CUDA::cudart_static (требует target cvcuda)
find_package(CUDAToolkit REQUIRED)

add_executable(rtsp_decoder
    src/main.cpp
    src/core/Globals.cpp
    src/core/Log.cpp
    src/core/CameraThread.cpp
    src/rtsp/RtspReader.cpp
    src/decoder/GstDecoder.cpp

    src/display/Display.cpp
    src/display/CudaDisplay.cpp
    src/display/CudaKernels.cu

    src/inference/CvcudaPreprocessor.cpp
    src/inference/YoloPostprocess.cpp
    src/inference/YoloV2Postprocess.cpp
    src/inference/TensorRtInfer.cpp
    src/inference/InferPipeline.cpp
)

target_include_directories(rtsp_decoder PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LIBAVFORMAT_INCLUDE_DIRS}
    ${LIBAVCODEC_INCLUDE_DIRS}
    ${LIBAVUTIL_INCLUDE_DIRS}
    ${GSTREAMER_INCLUDE_DIRS}
    ${GSTAPP_INCLUDE_DIRS}
    ${GSTVIDEO_INCLUDE_DIRS}
    /usr/include/aarch64-linux-gnu
    /usr/local/cuda/include
)

target_compile_definitions(rtsp_decoder PRIVATE
    __linux__
    __STDC_CONSTANT_MACROS
    __STDC_LIMIT_MACROS
)

target_link_libraries(rtsp_decoder PRIVATE
    ${LIBAVFORMAT_LIBRARIES}
    ${LIBAVCODEC_LIBRARIES}
    ${LIBAVUTIL_LIBRARIES}
    ${GSTREAMER_LIBRARIES}
    ${GSTAPP_LIBRARIES}
    ${GSTVIDEO_LIBRARIES}
    cvcuda
    nvcv_types
    ${NVINFER_LIB}
    pthread
    dl
    X11
    cudart
)
target_link_directories(rtsp_decoder PRIVATE
    /usr/lib/aarch64-linux-gnu
    /usr/local/cuda/lib64
    /opt/nvidia/cvcuda0/lib
    ${LIBAVCODEC_LIBRARY_DIRS}
    ${LIBAVFORMAT_LIBRARY_DIRS}
    ${LIBAVUTIL_LIBRARY_DIRS}
)