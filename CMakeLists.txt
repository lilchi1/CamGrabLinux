# Jetson RTSP Decoder — build configuration.
cmake_minimum_required(VERSION 3.10)
project(rtsp_jetson_decoder CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# FFmpeg libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAVFORMAT REQUIRED libavformat)
pkg_check_modules(LIBAVCODEC  REQUIRED libavcodec)
pkg_check_modules(LIBAVUTIL   REQUIRED libavutil)
pkg_check_modules(LIBSWSCALE  REQUIRED libswscale)

# OpenCV (for image writing in Display)
find_package(OpenCV REQUIRED core imgcodecs imgproc)

# Jetson Multimedia API paths
set(NVMULTIMEDIA_DIR /usr/src/jetson_multimedia_api)
set(NVIDIA_LIB_DIR /usr/lib/aarch64-linux-gnu/nvidia)
set(NVMM_COMMON_SRC_DIR ${NVMULTIMEDIA_DIR}/samples/common/classes)

# Try prebuilt libnvmm_common.a first; fall back to compiling sources directly
find_library(NVMM_COMMON_LIB nvmm_common
    PATHS ${NVIDIA_LIB_DIR}
)
if(NVMM_COMMON_LIB AND NOT NVMM_COMMON_LIB MATCHES "NOTFOUND")
    set(NVMM_COMMON_SOURCES "")
    message(STATUS "Found libnvmm_common.a: ${NVMM_COMMON_LIB}")
else()
    file(GLOB NVMM_COMMON_SOURCES
        ${NVMM_COMMON_SRC_DIR}/NvV4l2Element.cpp
        ${NVMM_COMMON_SRC_DIR}/NvV4l2ElementPlane.cpp
        ${NVMM_COMMON_SRC_DIR}/NvVideoDecoder.cpp
        ${NVMM_COMMON_SRC_DIR}/NvBuffer.cpp
        ${NVMM_COMMON_SRC_DIR}/NvBufSurface.cpp
        ${NVMM_COMMON_SRC_DIR}/NvElement.cpp
        ${NVMM_COMMON_SRC_DIR}/NvElementProfiler.cpp
        ${NVMM_COMMON_SRC_DIR}/NvUtils.cpp
        ${NVMM_COMMON_SRC_DIR}/NvLogging.cpp
    )
    if(NVMM_COMMON_SOURCES)
        message(STATUS "Using NVIDIA common sources from ${NVMM_COMMON_SRC_DIR}")
    else()
        message(WARNING "No NVIDIA common sources found at ${NVMM_COMMON_SRC_DIR}")
    endif()
endif()

add_executable(rtsp_decoder
    main.cpp
    func.cpp
    RtspReader.cpp
    SwDecoder.cpp
    NvV4l2Decoder.cpp
    PipeDecoder.cpp
    Display.cpp
    ${NVMM_COMMON_SOURCES}
)

target_include_directories(rtsp_decoder PRIVATE
    ${LIBAVFORMAT_INCLUDE_DIRS}
    ${LIBAVCODEC_INCLUDE_DIRS}
    ${LIBAVUTIL_INCLUDE_DIRS}
    ${LIBSWSCALE_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    /usr/include/aarch64-linux-gnu
    ${NVMULTIMEDIA_DIR}/include
)

target_compile_definitions(rtsp_decoder PRIVATE
    __linux__
    __STDC_CONSTANT_MACROS
    __STDC_LIMIT_MACROS
)

target_link_libraries(rtsp_decoder PRIVATE
    ${LIBAVFORMAT_LIBRARIES}
    ${LIBAVCODEC_LIBRARIES}
    ${LIBSWSCALE_LIBRARIES}
    ${LIBAVUTIL_LIBRARIES}
    ${OpenCV_LIBS}
    pthread
    X11
    nvv4l2
    nvbufsurface
    nvbufsurftransform
    v4l2
)
if(NVMM_COMMON_LIB AND NOT NVMM_COMMON_LIB MATCHES "NOTFOUND" AND NOT NVMM_COMMON_SOURCES)
    target_link_libraries(rtsp_decoder PRIVATE ${NVMM_COMMON_LIB})
endif()

target_link_directories(rtsp_decoder PRIVATE
    ${NVIDIA_LIB_DIR}
)
