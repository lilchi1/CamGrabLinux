cmake_minimum_required(VERSION 3.10)
project(rtsp_jetson_decoder CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAVFORMAT REQUIRED libavformat)
pkg_check_modules(LIBAVCODEC  REQUIRED libavcodec)
pkg_check_modules(LIBAVUTIL   REQUIRED libavutil)
pkg_check_modules(LIBSWSCALE  REQUIRED libswscale)

find_package(OpenCV REQUIRED core imgcodecs imgproc)

set(NVMULTIMEDIA_DIR /usr/src/jetson_multimedia_api)
set(NVIDIA_LIB_DIR /usr/lib/aarch64-linux-gnu/nvidia)

add_executable(rtsp_decoder
    main.cpp
    func.cpp
    RtspReader.cpp
    SwDecoder.cpp
    NvV4l2Decoder.cpp
    PipeDecoder.cpp
    Display.cpp
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
    /tmp/opencode/libnvmm_common.a
    nvv4l2
    nvbufsurface
    nvbufsurftransform
    v4l2
)

target_link_directories(rtsp_decoder PRIVATE
    ${NVIDIA_LIB_DIR}
)
