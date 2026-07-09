// DisplayWindow implementation — X11 image + NV12→RGB manual conversion.
#include "Display.h"
#include "headers.h"

DisplayWindow::DisplayWindow()
    : m_display(nullptr), m_window(0), m_gc(nullptr),
      m_image(nullptr), m_width(0), m_height(0), m_rgbBuf(nullptr) {}

DisplayWindow::~DisplayWindow() { close(); }

bool DisplayWindow::open(const std::string& title, int width, int height) {
    close();
    m_width = width;
    m_height = height;

    m_display = XOpenDisplay(nullptr);
    if (!m_display) return false;

    int screen = DefaultScreen(m_display);
    Window root = RootWindow(m_display, screen);

    m_window = XCreateSimpleWindow(m_display, root,
                                   0, 0, width, height, 1,
                                   BlackPixel(m_display, screen),
                                   WhitePixel(m_display, screen));
    if (!m_window) { close(); return false; }
    XStoreName(m_display, m_window, title.c_str());

    XSelectInput(m_display, m_window, ExposureMask | StructureNotifyMask);
    XMapWindow(m_display, m_window);

    m_gc = XCreateGC(m_display, m_window, 0, nullptr);

    // Find 24-bit TrueColor visual to guarantee 3 bytes/pixel
    XVisualInfo visInfo;
    if (!XMatchVisualInfo(m_display, screen, 24, TrueColor, &visInfo)) {
        close(); return false;
    }

    m_image = XCreateImage(m_display, visInfo.visual, visInfo.depth,
                           ZPixmap, 0, nullptr, width, height, 32, 0);
    if (!m_image) { close(); return false; }

    m_bpp = m_image->bits_per_pixel / 8;  // bytes per pixel (3 for 24-bit)
    m_image->data = new char[height * m_image->bytes_per_line];
    m_rgbBuf = new uint8_t[width * height * m_bpp];

    // Wait for window to appear
    XEvent e;
    while (true) {
        XNextEvent(m_display, &e);
        if (e.type == MapNotify) break;
    }

    return true;
}

void DisplayWindow::close() {
    if (m_rgbBuf) { delete[] m_rgbBuf; m_rgbBuf = nullptr; }
    if (m_image) {
        if (m_image->data) delete[] m_image->data;
        XDestroyImage(m_image);
        m_image = nullptr;
    }
    if (m_gc) { XFreeGC(m_display, m_gc); m_gc = nullptr; }
    if (m_window) { XDestroyWindow(m_display, m_window); m_window = 0; }
    if (m_display) { XCloseDisplay(m_display); m_display = nullptr; }
}

void DisplayWindow::showFrame(uint8_t* yPlane, uint8_t* uvPlane,
                              int width, int height,
                              int strideY, int strideUV) {
    if (!m_image || !m_rgbBuf) return;

    nv12ToRgb(yPlane, uvPlane, m_rgbBuf, width, height, strideY, strideUV);

    // Copy RGB to XImage (BGR order for X). m_bpp = 3 for 24-bit visual.
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int srcIdx = (y * width + x) * 3;
            int dstIdx = y * m_image->bytes_per_line + x * m_bpp;
            m_image->data[dstIdx + 0] = m_rgbBuf[srcIdx + 2];
            m_image->data[dstIdx + 1] = m_rgbBuf[srcIdx + 1];
            m_image->data[dstIdx + 2] = m_rgbBuf[srcIdx + 0];
        }
    }

    XPutImage(m_display, m_window, m_gc, m_image,
              0, 0, 0, 0, width, height);
    XFlush(m_display);
}

// NV12 → RGB24 conversion (BT.601 limited range).
// NV12: Y plane (stride bytes/row) + interleaved UV plane (same stride, half height).
void DisplayWindow::nv12ToRgb(uint8_t* yPlane, uint8_t* uvPlane,
                              uint8_t* rgb, int w, int h,
                              int strideY, int /*strideUV*/) {
    int stride = strideY;  // for NV12, UV stride == Y stride
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int yi = y * stride + x;
            int uvIdx = (y / 2) * stride + (x / 2) * 2;
            int Y = yPlane[yi];
            int U = uvPlane[uvIdx];
            int V = uvPlane[uvIdx + 1];

            int C = Y - 16;
            int D = U - 128;
            int E = V - 128;

            int R = (298 * C + 409 * E + 128) >> 8;
            int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
            int B = (298 * C + 516 * D + 128) >> 8;

            int outIdx = (y * w + x) * 3;
            rgb[outIdx + 0] = std::max(0, std::min(255, R));
            rgb[outIdx + 1] = std::max(0, std::min(255, G));
            rgb[outIdx + 2] = std::max(0, std::min(255, B));
        }
    }
}
