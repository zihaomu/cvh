#ifndef CVH_HIGHGUI_DETAIL_BACKEND_X11_HPP
#define CVH_HIGHGUI_DETAIL_BACKEND_X11_HPP

#include "backend_api.hpp"

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cvh {
namespace detail {

namespace x11_highgui {

struct ChannelMask
{
    unsigned long mask = 0;
    int shift = 0;
    int bits = 0;
};

struct Window
{
    ::Window handle = 0;
    Atom wm_delete = None;
    int width = 0;
    int height = 0;
    int flags = 0;
    HighguiImage image;
};

inline int lowbit_shift(unsigned long mask)
{
    for (int index = 0;
         index < static_cast<int>(sizeof(mask) * 8);
         ++index)
    {
        if ((mask & (1UL << index)) != 0)
        {
            return index;
        }
    }
    return 0;
}

inline int popcount(unsigned long mask)
{
    int count = 0;
    while (mask)
    {
        count += static_cast<int>(mask & 1UL);
        mask >>= 1;
    }
    return count;
}

inline unsigned long pack(
    uchar value,
    const ChannelMask& channel)
{
    if (channel.bits <= 0 || channel.mask == 0)
    {
        return 0;
    }
    const unsigned long maximum =
        (1UL << channel.bits) - 1UL;
    const unsigned long scaled =
        (static_cast<unsigned long>(value) * maximum +
         127UL) /
        255UL;
    return (scaled << channel.shift) & channel.mask;
}

}  // namespace x11_highgui

class HighguiBackend
{
public:
    ~HighguiBackend()
    {
        destroy_all_windows();
        if (display_)
        {
            XCloseDisplay(display_);
        }
    }

    const char* name() const
    {
        return "x11";
    }

    bool available()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return open_display();
    }

    bool named_window(const std::string& name, int flags)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return get_or_create_window(
                   name,
                   640,
                   480,
                   flags) != nullptr;
    }

    bool show_image(
        const std::string& name,
        const HighguiImage& image,
        int flags)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        x11_highgui::Window* window =
            get_or_create_window(
                name,
                image.width,
                image.height,
                flags);
        if (!window)
        {
            return false;
        }

        window->image = image;
        if ((window->flags & 1) != 0 &&
            (window->width != image.width ||
             window->height != image.height))
        {
            XResizeWindow(
                display_,
                window->handle,
                static_cast<unsigned int>(image.width),
                static_cast<unsigned int>(image.height));
            window->width = image.width;
            window->height = image.height;
        }
        return redraw(*window);
    }

    int wait_key(int delay)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!open_display() || windows_.empty())
        {
            lock.unlock();
            if (delay > 0)
            {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(delay));
            }
            return -1;
        }

        if (delay <= 0)
        {
            for (;;)
            {
                XEvent event{};
                XNextEvent(display_, &event);
                const int key = handle_event(event);
                if (key >= 0 || windows_.empty())
                {
                    return key;
                }
            }
        }

        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(delay);
        while (std::chrono::steady_clock::now() < deadline)
        {
            while (XPending(display_) > 0)
            {
                XEvent event{};
                XNextEvent(display_, &event);
                const int key = handle_event(event);
                if (key >= 0)
                {
                    return key;
                }
            }
            lock.unlock();
            std::this_thread::sleep_for(
                std::chrono::milliseconds(1));
            lock.lock();
        }
        return -1;
    }

    void destroy_window(const std::string& name)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = windows_.find(name);
        if (found == windows_.end())
        {
            return;
        }
        if (display_ && found->second.handle)
        {
            XDestroyWindow(
                display_,
                found->second.handle);
            XFlush(display_);
        }
        windows_.erase(found);
    }

    void destroy_all_windows()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (display_)
        {
            for (const auto& item : windows_)
            {
                if (item.second.handle)
                {
                    XDestroyWindow(
                        display_,
                        item.second.handle);
                }
            }
            XFlush(display_);
        }
        windows_.clear();
    }

private:
    bool open_display()
    {
        if (display_)
        {
            return true;
        }
        const char* display_name = std::getenv("DISPLAY");
        if (!display_name || display_name[0] == '\0')
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

        red_ = {
            visual_->red_mask,
            x11_highgui::lowbit_shift(visual_->red_mask),
            x11_highgui::popcount(visual_->red_mask)};
        green_ = {
            visual_->green_mask,
            x11_highgui::lowbit_shift(visual_->green_mask),
            x11_highgui::popcount(visual_->green_mask)};
        blue_ = {
            visual_->blue_mask,
            x11_highgui::lowbit_shift(visual_->blue_mask),
            x11_highgui::popcount(visual_->blue_mask)};
        return true;
    }

    x11_highgui::Window* get_or_create_window(
        const std::string& name,
        int width,
        int height,
        int flags)
    {
        if (!open_display())
        {
            return nullptr;
        }
        auto found = windows_.find(name);
        if (found != windows_.end())
        {
            return &found->second;
        }

        x11_highgui::Window window;
        window.width = width;
        window.height = height;
        window.flags = flags;
        window.handle = XCreateSimpleWindow(
            display_,
            RootWindow(display_, screen_),
            0,
            0,
            static_cast<unsigned int>(width),
            static_cast<unsigned int>(height),
            1,
            BlackPixel(display_, screen_),
            WhitePixel(display_, screen_));
        if (!window.handle)
        {
            return nullptr;
        }

        window.wm_delete =
            XInternAtom(
                display_,
                "WM_DELETE_WINDOW",
                False);
        if (window.wm_delete != None)
        {
            XSetWMProtocols(
                display_,
                window.handle,
                &window.wm_delete,
                1);
        }
        XStoreName(
            display_,
            window.handle,
            name.c_str());
        XSelectInput(
            display_,
            window.handle,
            ExposureMask |
                KeyPressMask |
                StructureNotifyMask);
        XMapRaised(display_, window.handle);
        XFlush(display_);
        found = windows_.emplace(name, window).first;
        return &found->second;
    }

    bool redraw(const x11_highgui::Window& window)
    {
        if (window.image.pixels.empty())
        {
            return true;
        }

        const int width = window.image.width;
        const int height = window.image.height;
        XImage* native_image = XCreateImage(
            display_,
            visual_,
            static_cast<unsigned int>(depth_),
            ZPixmap,
            0,
            nullptr,
            static_cast<unsigned int>(width),
            static_cast<unsigned int>(height),
            32,
            0);
        if (!native_image)
        {
            return false;
        }

        const size_t bytes =
            static_cast<size_t>(native_image->bytes_per_line) *
            static_cast<size_t>(height);
        native_image->data =
            static_cast<char*>(std::malloc(bytes));
        if (!native_image->data)
        {
            XDestroyImage(native_image);
            return false;
        }
        std::memset(native_image->data, 0, bytes);

        for (int y = 0; y < height; ++y)
        {
            const uchar* source =
                window.image.pixels.data() +
                static_cast<size_t>(y) *
                    static_cast<size_t>(width) *
                    static_cast<size_t>(window.image.channels);
            for (int x = 0; x < width; ++x)
            {
                const uchar* pixel =
                    source +
                    static_cast<size_t>(x) *
                        static_cast<size_t>(window.image.channels);
                const uchar blue =
                    window.image.channels == 1
                        ? pixel[0]
                        : pixel[0];
                const uchar green =
                    window.image.channels == 1
                        ? pixel[0]
                        : pixel[1];
                const uchar red =
                    window.image.channels == 1
                        ? pixel[0]
                        : pixel[2];
                const unsigned long native_pixel =
                    x11_highgui::pack(red, red_) |
                    x11_highgui::pack(green, green_) |
                    x11_highgui::pack(blue, blue_);
                XPutPixel(native_image, x, y, native_pixel);
            }
        }

        XPutImage(
            display_,
            window.handle,
            DefaultGC(display_, screen_),
            native_image,
            0,
            0,
            0,
            0,
            static_cast<unsigned int>(width),
            static_cast<unsigned int>(height));
        XFlush(display_);
        XDestroyImage(native_image);
        return true;
    }

    int handle_event(const XEvent& event)
    {
        if (event.type == KeyPress)
        {
            char buffer[8] = {};
            KeySym key_symbol = 0;
            const int count = XLookupString(
                const_cast<XKeyEvent*>(&event.xkey),
                buffer,
                static_cast<int>(sizeof(buffer)),
                &key_symbol,
                nullptr);
            return count > 0
                       ? static_cast<unsigned char>(buffer[0])
                       : static_cast<int>(key_symbol);
        }
        if (event.type == Expose)
        {
            auto found =
                find_window(event.xexpose.window);
            if (found != windows_.end() &&
                event.xexpose.count == 0)
            {
                (void)redraw(found->second);
            }
        }
        else if (event.type == ClientMessage)
        {
            auto found =
                find_window(event.xclient.window);
            if (found != windows_.end() &&
                found->second.wm_delete != None &&
                static_cast<Atom>(
                    event.xclient.data.l[0]) ==
                    found->second.wm_delete)
            {
                XDestroyWindow(
                    display_,
                    found->second.handle);
                windows_.erase(found);
            }
        }
        else if (event.type == DestroyNotify)
        {
            auto found =
                find_window(event.xdestroywindow.window);
            if (found != windows_.end())
            {
                windows_.erase(found);
            }
        }
        return -1;
    }

    std::map<std::string, x11_highgui::Window>::iterator
    find_window(::Window handle)
    {
        return std::find_if(
            windows_.begin(),
            windows_.end(),
            [handle](const auto& item) {
                return item.second.handle == handle;
            });
    }

    std::mutex mutex_;
    Display* display_ = nullptr;
    int screen_ = 0;
    Visual* visual_ = nullptr;
    int depth_ = 0;
    x11_highgui::ChannelMask red_;
    x11_highgui::ChannelMask green_;
    x11_highgui::ChannelMask blue_;
    std::map<std::string, x11_highgui::Window> windows_;
};

inline HighguiBackend& highgui_backend()
{
    static HighguiBackend backend;
    return backend;
}

}  // namespace detail
}  // namespace cvh

#endif  // CVH_HIGHGUI_DETAIL_BACKEND_X11_HPP
