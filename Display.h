#pragma once

#include <cstdint>
#include <string>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>

extern "C" {
#include <libswscale/swscale.h>
}

class VideoDisplay {
public:
    VideoDisplay(const std::string& title, int width, int height);
    ~VideoDisplay();

    bool init();
    void showFrame(const uint8_t* yPlane, const uint8_t* uvPlane,
                   int width, int height, int strideY, int strideUV);
    bool shouldQuit() const { return m_shouldQuit; }
    int width() const { return m_width; }
    int height() const { return m_height; }

private:
    std::string m_title;
    int m_width;
    int m_height;

    ::Display* m_xDisplay;
    Window m_window;
    GC m_gc;
    XImage* m_image;
    uint8_t* m_rgbBuf;
    bool m_shouldQuit;

    SwsContext* m_swsCtx;

    void pollEvents();
    void convertNV12toRGB(const uint8_t* yPlane, const uint8_t* uvPlane,
                          int width, int height, int strideY, int strideUV,
                          uint8_t* rgb);
};
