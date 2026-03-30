#include "display_x11.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cvh {
namespace {

struct ChannelMask
{
    unsigned long mask;
    int shift;
    int bits;
};

struct WindowState
{
    Window win;
    Atom wm_delete;
    int width;
    int height;
    int channels;
    std::vector<unsigned char> pixels;
};

inline int lowbit_shift(unsigned long mask)
{
    for (int i = 0; i < static_cast<int>(sizeof(unsigned long) * 8); ++i)
    {
        if (mask & (1UL << i))
        {
            return i;
        }
    }
    return 0;
}

inline int popcount_bits(unsigned long mask)
{
    int count = 0;
    while (mask)
    {
        count += static_cast<int>(mask & 1UL);
        mask >>= 1;
    }
    return count;
}

inline unsigned long pack_component(unsigned char v, const ChannelMask& cm)
{
    if (cm.bits <= 0 || cm.mask == 0)
    {
        return 0;
    }

    const unsigned long max_value = (1UL << cm.bits) - 1UL;
    const unsigned long scaled = (static_cast<unsigned long>(v) * max_value + 127UL) / 255UL;
    return (scaled << cm.shift) & cm.mask;
}

class x11_context
{
public:
    bool supported()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return open_locked();
    }

    int show_bgr(const char* winname, const unsigned char* bgrdata, int width, int height)
    {
        if (!winname || !bgrdata || width <= 0 || height <= 0)
        {
            return -1;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (!open_locked())
        {
            return -1;
        }

        WindowState* ws = get_or_create_window_locked(winname, width, height);
        if (!ws)
        {
            return -1;
        }

        ws->channels = 3;
        ws->pixels.assign(bgrdata, bgrdata + static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
        return redraw_window_locked(*ws);
    }

    int show_gray(const char* winname, const unsigned char* graydata, int width, int height)
    {
        if (!winname || !graydata || width <= 0 || height <= 0)
        {
            return -1;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (!open_locked())
        {
            return -1;
        }

        WindowState* ws = get_or_create_window_locked(winname, width, height);
        if (!ws)
        {
            return -1;
        }

        ws->channels = 1;
        ws->pixels.assign(graydata, graydata + static_cast<size_t>(width) * static_cast<size_t>(height));
        return redraw_window_locked(*ws);
    }

    int wait_key(int delay)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!open_locked() || windows_.empty())
        {
            if (delay > 0)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(delay));
            }
            return -1;
        }

        if (delay == 0)
        {
            while (true)
            {
                while (XPending(display_) > 0)
                {
                    XEvent event;
                    XNextEvent(display_, &event);
                    const int key = handle_event_locked(event);
                    if (key >= 0)
                    {
                        return key;
                    }
                }

                XEvent event;
                XNextEvent(display_, &event);
                const int key = handle_event_locked(event);
                if (key >= 0)
                {
                    return key;
                }
            }
        }

        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max(0, delay));
        while (std::chrono::steady_clock::now() < deadline)
        {
            while (XPending(display_) > 0)
            {
                XEvent event;
                XNextEvent(display_, &event);
                const int key = handle_event_locked(event);
                if (key >= 0)
                {
                    return key;
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return -1;
    }

private:
    bool open_locked()
    {
        if (display_)
        {
            return true;
        }

        const char* display_env = std::getenv("DISPLAY");
        if (!display_env || display_env[0] == '\0')
        {
            return false;
        }

        display_ = XOpenDisplay(nullptr);
        if (!display_)
        {
            return false;
        }

        screen_ = DefaultScreen(display_);
        visual_ = DefaultVisual(display_, screen_);
        depth_ = DefaultDepth(display_, screen_);
        if (!visual_ || depth_ <= 0)
        {
            XCloseDisplay(display_);
            display_ = nullptr;
            return false;
        }

        red_ = ChannelMask{visual_->red_mask, lowbit_shift(visual_->red_mask), popcount_bits(visual_->red_mask)};
        green_ = ChannelMask{visual_->green_mask, lowbit_shift(visual_->green_mask), popcount_bits(visual_->green_mask)};
        blue_ = ChannelMask{visual_->blue_mask, lowbit_shift(visual_->blue_mask), popcount_bits(visual_->blue_mask)};
        return true;
    }

    WindowState* get_or_create_window_locked(const char* winname, int width, int height)
    {
        auto it = windows_.find(winname);
        if (it == windows_.end())
        {
            WindowState ws{};
            ws.width = width;
            ws.height = height;
            ws.channels = 0;
            ws.win = XCreateSimpleWindow(display_,
                                         RootWindow(display_, screen_),
                                         0,
                                         0,
                                         static_cast<unsigned int>(width),
                                         static_cast<unsigned int>(height),
                                         1,
                                         BlackPixel(display_, screen_),
                                         WhitePixel(display_, screen_));
            if (!ws.win)
            {
                return nullptr;
            }

            ws.wm_delete = XInternAtom(display_, "WM_DELETE_WINDOW", False);
            if (ws.wm_delete != None)
            {
                XSetWMProtocols(display_, ws.win, &ws.wm_delete, 1);
            }

            XStoreName(display_, ws.win, winname);
            XSelectInput(display_, ws.win, ExposureMask | KeyPressMask | StructureNotifyMask);
            XMapRaised(display_, ws.win);
            XFlush(display_);

            it = windows_.emplace(winname, ws).first;
        }
        else if (it->second.width != width || it->second.height != height)
        {
            XResizeWindow(display_, it->second.win, static_cast<unsigned int>(width), static_cast<unsigned int>(height));
            it->second.width = width;
            it->second.height = height;
            XFlush(display_);
        }

        return &it->second;
    }

    int put_image_locked(const WindowState& ws, const unsigned char* data, int width, int height, int channels)
    {
        XImage* image = XCreateImage(display_,
                                     visual_,
                                     static_cast<unsigned int>(depth_),
                                     ZPixmap,
                                     0,
                                     nullptr,
                                     static_cast<unsigned int>(width),
                                     static_cast<unsigned int>(height),
                                     32,
                                     0);
        if (!image)
        {
            return -1;
        }

        const size_t stride = static_cast<size_t>(image->bytes_per_line);
        const size_t bytes = stride * static_cast<size_t>(height);
        image->data = reinterpret_cast<char*>(std::malloc(bytes));
        if (!image->data)
        {
            image->data = nullptr;
            XDestroyImage(image);
            return -1;
        }
        std::memset(image->data, 0, bytes);

        for (int y = 0; y < height; ++y)
        {
            const unsigned char* src_row = data + static_cast<size_t>(y) * static_cast<size_t>(width) * static_cast<size_t>(channels);
            for (int x = 0; x < width; ++x)
            {
                unsigned char r = 0;
                unsigned char g = 0;
                unsigned char b = 0;

                if (channels == 3)
                {
                    const unsigned char* p = src_row + static_cast<size_t>(x) * 3;
                    b = p[0];
                    g = p[1];
                    r = p[2];
                }
                else
                {
                    const unsigned char v = src_row[x];
                    r = v;
                    g = v;
                    b = v;
                }

                const unsigned long pixel = pack_component(r, red_) |
                                            pack_component(g, green_) |
                                            pack_component(b, blue_);
                XPutPixel(image, x, y, pixel);
            }
        }

        XPutImage(display_, ws.win, DefaultGC(display_, screen_), image, 0, 0, 0, 0, width, height);
        XFlush(display_);
        XDestroyImage(image);
        return 0;
    }

    int redraw_window_locked(const WindowState& ws)
    {
        if (ws.channels != 1 && ws.channels != 3)
        {
            return -1;
        }
        if (ws.pixels.empty())
        {
            return -1;
        }
        return put_image_locked(ws, ws.pixels.data(), ws.width, ws.height, ws.channels);
    }

    int handle_event_locked(const XEvent& event)
    {
        if (event.type == KeyPress)
        {
            char buf[8] = {0};
            KeySym key_sym = 0;
            const int n = XLookupString(const_cast<XKeyEvent*>(&event.xkey), buf, static_cast<int>(sizeof(buf)), &key_sym, nullptr);
            if (n > 0)
            {
                return static_cast<unsigned char>(buf[0]);
            }
            return static_cast<int>(key_sym);
        }
        else if (event.type == Expose)
        {
            auto it = find_window_by_id_locked(event.xexpose.window);
            if (it != windows_.end() && event.xexpose.count == 0)
            {
                (void)redraw_window_locked(it->second);
            }
        }

        if (event.type == ClientMessage)
        {
            auto it = find_window_by_id_locked(event.xclient.window);
            if (it != windows_.end())
            {
                const bool is_wm_delete = it->second.wm_delete != None &&
                                          static_cast<Atom>(event.xclient.data.l[0]) == it->second.wm_delete;
                if (is_wm_delete)
                {
                    XDestroyWindow(display_, it->second.win);
                    windows_.erase(it);
                }
            }
        }
        else if (event.type == DestroyNotify)
        {
            auto it = find_window_by_id_locked(event.xdestroywindow.window);
            if (it != windows_.end())
            {
                windows_.erase(it);
            }
        }

        return -1;
    }

    std::map<std::string, WindowState>::iterator find_window_by_id_locked(Window window)
    {
        return std::find_if(windows_.begin(),
                            windows_.end(),
                            [window](const std::pair<const std::string, WindowState>& item) {
                                return item.second.win == window;
                            });
    }

private:
    std::mutex mutex_;
    Display* display_ = nullptr;
    int screen_ = 0;
    Visual* visual_ = nullptr;
    int depth_ = 0;

    ChannelMask red_{};
    ChannelMask green_{};
    ChannelMask blue_{};

    std::map<std::string, WindowState> windows_;
};

x11_context& global_x11_context()
{
    static x11_context ctx;
    return ctx;
}

}  // namespace

bool display_x11::supported()
{
    return global_x11_context().supported();
}

int display_x11::show_bgr(const char* winname, const unsigned char* bgrdata, int width, int height)
{
    return global_x11_context().show_bgr(winname, bgrdata, width, height);
}

int display_x11::show_gray(const char* winname, const unsigned char* graydata, int width, int height)
{
    return global_x11_context().show_gray(winname, graydata, width, height);
}

int display_x11::wait_key(int delay)
{
    return global_x11_context().wait_key(delay);
}

}  // namespace cvh
