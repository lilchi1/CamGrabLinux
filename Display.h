// DisplayWindow — X11 window showing NV12 frames using manual NV12→RGB + XPutImage.
#pragma once

#include <cstdint>
#include <string>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

class DisplayWindow {
public:
    DisplayWindow();
    ~DisplayWindow();

    bool open(const std::string& title, int width, int height);
    void close();
    void showFrame(uint8_t* yPlane, uint8_t* uvPlane,
                   int width, int height,
                   int strideY, int strideUV);

    bool isOpen() const { return m_window != 0; }
    int width() const { return m_width; }
    int height() const { return m_height; }

private:
    ::Display* m_display;
    Window m_window;
    GC m_gc;
    XImage* m_image;
    int m_width;
    int m_height;
    int m_bpp;  // bytes per pixel (from XImage bits_per_pixel / 8)
    uint8_t* m_rgbBuf;

    void nv12ToRgb(uint8_t* yPlane, uint8_t* uvPlane,
                   uint8_t* rgb, int w, int h,
                   int strideY, int strideUV);
};
