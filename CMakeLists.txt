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

# Detect Jetson GPU architecture (default to sm_72 for Xavier, sm_87 for Orin)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 72)
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

add_executable(rtsp_decoder
    main.cpp
    func.cpp
    RtspReader.cpp
    GstDecoder.cpp

    Display.cpp
    CudaDisplay.cpp
    CudaKernels.cu
    PtzControl.cpp
)

target_include_directories(rtsp_decoder PRIVATE
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
    pthread
    dl
    X11
    cudart
)
target_link_directories(rtsp_decoder PRIVATE
    /usr/lib/aarch64-linux-gnu
    /usr/local/cuda/lib64
    ${LIBAVCODEC_LIBRARY_DIRS}
    ${LIBAVFORMAT_LIBRARY_DIRS}
    ${LIBAVUTIL_LIBRARY_DIRS}
)
