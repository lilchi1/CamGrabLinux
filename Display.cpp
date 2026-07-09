#include "Display.h"
#include <cstdio>
#include <cstdlib>

VideoDisplay::VideoDisplay(const std::string& title, int width, int height)
    : m_title(title)
    , m_width(width)
    , m_height(height)
    , m_xDisplay(nullptr)
    , m_window(0)
    , m_gc(nullptr)
    , m_image(nullptr)
    , m_rgbBuf(nullptr)
    , m_shouldQuit(false)
    , m_swsCtx(nullptr)
{
}

VideoDisplay::~VideoDisplay()
{
    if (m_swsCtx) sws_freeContext(m_swsCtx);
    if (m_image) XDestroyImage(m_image);
    if (m_rgbBuf) free(m_rgbBuf);
    if (m_gc && m_xDisplay) XFreeGC(m_xDisplay, m_gc);
    if (m_window && m_xDisplay) XDestroyWindow(m_xDisplay, m_window);
    if (m_xDisplay) XCloseDisplay(m_xDisplay);
}

bool VideoDisplay::init()
{
    m_xDisplay = XOpenDisplay(nullptr);
    if (!m_xDisplay) {
        fprintf(stderr, "VideoDisplay: Cannot open X display\n");
        return false;
    }

    int screen = DefaultScreen(m_xDisplay);
    Window root = RootWindow(m_xDisplay, screen);

    m_window = XCreateSimpleWindow(m_xDisplay, root,
                                    0, 0, m_width, m_height, 0,
                                    BlackPixel(m_xDisplay, screen),
                                    WhitePixel(m_xDisplay, screen));

    XSelectInput(m_xDisplay, m_window, ExposureMask | StructureNotifyMask |
                 KeyPressMask | ButtonPressMask | DestroyNotify);

    Atom wmDelete = XInternAtom(m_xDisplay, "WM_DELETE_WINDOW", True);
    XSetWMProtocols(m_xDisplay, m_window, &wmDelete, 1);

    XStoreName(m_xDisplay, m_window, m_title.c_str());
    XMapWindow(m_xDisplay, m_window);

    m_gc = XCreateGC(m_xDisplay, m_window, 0, nullptr);

    m_rgbBuf = (uint8_t*)malloc(m_width * m_height * 3);
    if (!m_rgbBuf) {
        fprintf(stderr, "VideoDisplay: malloc failed\n");
        return false;
    }

    char* imgData = (char*)malloc(m_width * m_height * 3);
    if (!imgData) {
        fprintf(stderr, "VideoDisplay: image data malloc failed\n");
        return false;
    }

    m_image = XCreateImage(m_xDisplay, DefaultVisual(m_xDisplay, screen),
                           24, ZPixmap, 0, imgData,
                           m_width, m_height, 32, 0);
    if (!m_image) {
        fprintf(stderr, "VideoDisplay: XCreateImage failed\n");
        free(imgData);
        return false;
    }

    m_swsCtx = sws_getContext(m_width, m_height, AV_PIX_FMT_NV12,
                               m_width, m_height, AV_PIX_FMT_RGB24,
                               SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!m_swsCtx) {
        fprintf(stderr, "VideoDisplay: sws_getContext failed\n");
        return false;
    }

    return true;
}

void VideoDisplay::convertNV12toRGB(const uint8_t* yPlane, const uint8_t* uvPlane,
                                     int width, int height, int strideY, int strideUV,
                                     uint8_t* rgb)
{
    if (strideY < width) strideY = width;
    if (strideUV < width) strideUV = width;
    int aligned = (width + 31) & ~31;
    if (strideY > aligned * 4) strideY = aligned;
    if (strideUV > aligned * 4) strideUV = aligned;

    const uint8_t* srcPlanes[2] = { yPlane, uvPlane };
    int srcStrides[2] = { strideY, strideUV };
    uint8_t* dstPlanes[1] = { rgb };
    int dstStrides[1] = { width * 3 };

    sws_scale(m_swsCtx, srcPlanes, srcStrides, 0, height,
              dstPlanes, dstStrides);
}

void VideoDisplay::pollEvents()
{
    XEvent ev;
    while (XPending(m_xDisplay) > 0) {
        XNextEvent(m_xDisplay, &ev);
        if (ev.type == DestroyNotify) {
            m_shouldQuit = true;
        } else if (ev.type == ClientMessage) {
            m_shouldQuit = true;
        } else if (ev.type == KeyPress) {
            KeySym key = XLookupKeysym(&ev.xkey, 0);
            if (key == XK_Escape || key == XK_q) {
                m_shouldQuit = true;
            }
        }
    }
}

void VideoDisplay::showFrame(const uint8_t* yPlane, const uint8_t* uvPlane,
                              int width, int height, int strideY, int strideUV)
{
    if (m_shouldQuit) return;

    pollEvents();

    if (width != m_width || height != m_height) {
        fprintf(stderr, "VideoDisplay: size mismatch display=%dx%d frame=%dx%d\n",
                m_width, m_height, width, height);
        return;
    }

    if (strideY < width / 2 || strideY > width * 16 || strideUV < width / 2 || strideUV > width * 16)
        fprintf(stderr, "VideoDisplay: suspect stride Y=%d UV=%d for w=%d h=%d\n",
                strideY, strideUV, width, height);

    convertNV12toRGB(yPlane, uvPlane, width, height, strideY, strideUV, m_rgbBuf);

    memcpy(m_image->data, m_rgbBuf, m_width * m_height * 3);

    XPutImage(m_xDisplay, m_window, m_gc, m_image,
              0, 0, 0, 0, m_width, m_height);
    XFlush(m_xDisplay);
}
