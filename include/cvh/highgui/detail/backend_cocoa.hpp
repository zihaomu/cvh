#ifndef CVH_HIGHGUI_DETAIL_BACKEND_COCOA_HPP
#define CVH_HIGHGUI_DETAIL_BACKEND_COCOA_HPP

#include "backend_api.hpp"

#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <pthread.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace cvh {
namespace detail {

namespace cocoa_runtime {

template <class Result, class... Args>
inline Result send(id receiver, const char* selector, Args... args)
{
    using Function = Result (*)(id, SEL, Args...);
    const Function function =
        reinterpret_cast<Function>(objc_msgSend);
    return function(receiver, sel_registerName(selector), args...);
}

inline id class_object(const char* name)
{
    return reinterpret_cast<id>(objc_getClass(name));
}

inline id string(const std::string& value)
{
    return send<id>(
        class_object("NSString"),
        "stringWithUTF8String:",
        value.c_str());
}

inline id string(const char* value)
{
    return send<id>(
        class_object("NSString"),
        "stringWithUTF8String:",
        value);
}

struct Window
{
    id handle = nullptr;
    id image_view = nullptr;
    int width = 0;
    int height = 0;
    int flags = 0;
};

}  // namespace cocoa_runtime

class HighguiBackend
{
public:
    const char* name() const
    {
        return "cocoa-runtime";
    }

    bool available()
    {
        return on_main_thread() && ensure_application();
    }

    bool named_window(const std::string& name, int flags)
    {
        if (!available())
        {
            return false;
        }
        return get_or_create_window(name, 640, 480, flags) != nullptr;
    }

    bool show_image(
        const std::string& name,
        const HighguiImage& image,
        int flags)
    {
        if (!available())
        {
            return false;
        }

        cocoa_runtime::Window* window =
            get_or_create_window(
                name,
                image.width,
                image.height,
                flags);
        if (!window)
        {
            return false;
        }

        if ((window->flags & 1) != 0 &&
            (window->width != image.width ||
             window->height != image.height))
        {
            const CGSize size =
                CGSizeMake(image.width, image.height);
            cocoa_runtime::send<void>(
                window->handle,
                "setContentSize:",
                size);
            window->width = image.width;
            window->height = image.height;
        }
        if (!cocoa_runtime::send<BOOL>(
                window->handle,
                "isVisible"))
        {
            cocoa_runtime::send<void>(
                window->handle,
                "makeKeyAndOrderFront:",
                static_cast<id>(nullptr));
        }

        std::vector<uchar> rgba(
            static_cast<size_t>(image.width) *
            static_cast<size_t>(image.height) * 4);
        for (int y = 0; y < image.height; ++y)
        {
            const uchar* source =
                image.pixels.data() +
                static_cast<size_t>(y) *
                    static_cast<size_t>(image.width) *
                    static_cast<size_t>(image.channels);
            uchar* destination =
                rgba.data() +
                static_cast<size_t>(y) *
                    static_cast<size_t>(image.width) * 4;
            for (int x = 0; x < image.width; ++x)
            {
                const uchar* source_pixel =
                    source +
                    static_cast<size_t>(x) *
                        static_cast<size_t>(image.channels);
                uchar* destination_pixel =
                    destination + static_cast<size_t>(x) * 4;
                if (image.channels == 1)
                {
                    destination_pixel[0] = source_pixel[0];
                    destination_pixel[1] = source_pixel[0];
                    destination_pixel[2] = source_pixel[0];
                }
                else
                {
                    destination_pixel[0] = source_pixel[2];
                    destination_pixel[1] = source_pixel[1];
                    destination_pixel[2] = source_pixel[0];
                }
                destination_pixel[3] = 255;
            }
        }

        CFDataRef data = CFDataCreate(
            kCFAllocatorDefault,
            rgba.data(),
            static_cast<CFIndex>(rgba.size()));
        if (!data)
        {
            return false;
        }
        CGDataProviderRef provider =
            CGDataProviderCreateWithCFData(data);
        CGColorSpaceRef color_space =
            CGColorSpaceCreateDeviceRGB();
        const CGBitmapInfo bitmap_info =
            static_cast<CGBitmapInfo>(
                kCGBitmapByteOrder32Big |
                kCGImageAlphaPremultipliedLast);
        CGImageRef cg_image = nullptr;
        if (provider && color_space)
        {
            cg_image = CGImageCreate(
                static_cast<size_t>(image.width),
                static_cast<size_t>(image.height),
                8,
                32,
                static_cast<size_t>(image.width) * 4,
                color_space,
                bitmap_info,
                provider,
                nullptr,
                false,
                kCGRenderingIntentDefault);
        }

        bool success = false;
        if (cg_image)
        {
            id native_image = cocoa_runtime::send<id>(
                cocoa_runtime::send<id>(
                    cocoa_runtime::class_object("NSImage"),
                    "alloc"),
                "initWithCGImage:size:",
                cg_image,
                CGSizeMake(image.width, image.height));
            if (native_image)
            {
                cocoa_runtime::send<void>(
                    window->image_view,
                    "setImage:",
                    native_image);
                cocoa_runtime::send<void>(
                    window->handle,
                    "displayIfNeeded");
                cocoa_runtime::send<void>(
                    application_,
                    "updateWindows");
                cocoa_runtime::send<void>(
                    native_image,
                    "release");
                success = true;
            }
        }

        if (cg_image)
        {
            CGImageRelease(cg_image);
        }
        if (color_space)
        {
            CGColorSpaceRelease(color_space);
        }
        if (provider)
        {
            CGDataProviderRelease(provider);
        }
        CFRelease(data);
        return success;
    }

    int wait_key(int delay)
    {
        if (!available() || windows_.empty())
        {
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
                const int key = pump_event(
                    cocoa_runtime::send<id>(
                        cocoa_runtime::class_object("NSDate"),
                        "distantFuture"));
                if (key >= 0)
                {
                    return key;
                }
                if (windows_.empty())
                {
                    return -1;
                }
            }
        }

        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(delay);
        while (std::chrono::steady_clock::now() < deadline)
        {
            const auto remaining =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    deadline - std::chrono::steady_clock::now());
            const int step = static_cast<int>(
                std::max<int64_t>(
                    1,
                    std::min<int64_t>(remaining.count(), 10)));
            id until = cocoa_runtime::send<id>(
                cocoa_runtime::class_object("NSDate"),
                "dateWithTimeIntervalSinceNow:",
                static_cast<double>(step) / 1000.0);
            const int key = pump_event(until);
            if (key >= 0)
            {
                return key;
            }
        }
        return -1;
    }

    void destroy_window(const std::string& name)
    {
        if (!on_main_thread())
        {
            return;
        }
        const auto found = windows_.find(name);
        if (found == windows_.end())
        {
            return;
        }
        close_window(found->second);
        windows_.erase(found);
    }

    void destroy_all_windows()
    {
        if (!on_main_thread())
        {
            return;
        }
        for (auto& item : windows_)
        {
            close_window(item.second);
        }
        windows_.clear();
    }

private:
    static bool on_main_thread()
    {
        return pthread_main_np() != 0;
    }

    bool ensure_application()
    {
        if (application_)
        {
            return true;
        }
        application_ = cocoa_runtime::send<id>(
            cocoa_runtime::class_object("NSApplication"),
            "sharedApplication");
        if (!application_)
        {
            return false;
        }
        cocoa_runtime::send<void>(
            application_,
            "setActivationPolicy:",
            static_cast<long>(0));
        cocoa_runtime::send<void>(
            application_,
            "finishLaunching");
        cocoa_runtime::send<void>(
            application_,
            "activateIgnoringOtherApps:",
            static_cast<BOOL>(true));
        return true;
    }

    cocoa_runtime::Window* get_or_create_window(
        const std::string& name,
        int width,
        int height,
        int flags)
    {
        auto found = windows_.find(name);
        if (found != windows_.end())
        {
            return &found->second;
        }

        const CGRect frame =
            CGRectMake(0, 0, width, height);
        constexpr unsigned long style =
            (1UL << 0) |
            (1UL << 1) |
            (1UL << 2) |
            (1UL << 3);
        id window = cocoa_runtime::send<id>(
            cocoa_runtime::send<id>(
                cocoa_runtime::class_object("NSWindow"),
                "alloc"),
            "initWithContentRect:styleMask:backing:defer:",
            frame,
            style,
            static_cast<unsigned long>(2),
            static_cast<BOOL>(false));
        if (!window)
        {
            return nullptr;
        }

        cocoa_runtime::send<void>(
            window,
            "setReleasedWhenClosed:",
            static_cast<BOOL>(false));
        cocoa_runtime::send<void>(
            window,
            "setTitle:",
            cocoa_runtime::string(name));
        cocoa_runtime::send<void>(window, "center");

        id image_view = cocoa_runtime::send<id>(
            cocoa_runtime::send<id>(
                cocoa_runtime::class_object("NSImageView"),
                "alloc"),
            "initWithFrame:",
            frame);
        if (!image_view)
        {
            cocoa_runtime::send<void>(window, "release");
            return nullptr;
        }

        cocoa_runtime::send<void>(
            image_view,
            "setImageScaling:",
            static_cast<unsigned long>(3));
        cocoa_runtime::send<void>(
            image_view,
            "setAutoresizingMask:",
            static_cast<unsigned long>((1UL << 1) | (1UL << 4)));
        cocoa_runtime::send<void>(
            window,
            "setContentView:",
            image_view);
        cocoa_runtime::send<void>(
            image_view,
            "release");

        cocoa_runtime::Window state;
        state.handle = window;
        state.image_view = image_view;
        state.width = width;
        state.height = height;
        state.flags = flags;
        found = windows_.emplace(name, state).first;

        cocoa_runtime::send<void>(
            window,
            "makeKeyAndOrderFront:",
            static_cast<id>(nullptr));
        cocoa_runtime::send<void>(
            application_,
            "activateIgnoringOtherApps:",
            static_cast<BOOL>(true));
        return &found->second;
    }

    int pump_event(id until)
    {
        id event = cocoa_runtime::send<id>(
            application_,
            "nextEventMatchingMask:untilDate:inMode:dequeue:",
            std::numeric_limits<unsigned long long>::max(),
            until,
            cocoa_runtime::string("kCFRunLoopDefaultMode"),
            static_cast<BOOL>(true));
        if (!event)
        {
            cocoa_runtime::send<void>(
                application_,
                "updateWindows");
            return -1;
        }

        int key = -1;
        const unsigned long event_type =
            cocoa_runtime::send<unsigned long>(
                event,
                "type");
        if (event_type == 10)
        {
            id characters = cocoa_runtime::send<id>(
                event,
                "charactersIgnoringModifiers");
            if (characters &&
                cocoa_runtime::send<unsigned long>(
                    characters,
                    "length") > 0)
            {
                key = static_cast<int>(
                    cocoa_runtime::send<unsigned short>(
                        characters,
                        "characterAtIndex:",
                        static_cast<unsigned long>(0)) &
                    0xFF);
            }
            else
            {
                key = static_cast<int>(
                    cocoa_runtime::send<unsigned short>(
                        event,
                        "keyCode"));
            }
        }

        cocoa_runtime::send<void>(
            application_,
            "sendEvent:",
            event);
        cocoa_runtime::send<void>(
            application_,
            "updateWindows");
        remove_closed_windows();
        return key;
    }

    void remove_closed_windows()
    {
        for (auto iterator = windows_.begin();
             iterator != windows_.end();)
        {
            if (!cocoa_runtime::send<BOOL>(
                    iterator->second.handle,
                    "isVisible"))
            {
                cocoa_runtime::send<void>(
                    iterator->second.handle,
                    "release");
                iterator = windows_.erase(iterator);
            }
            else
            {
                ++iterator;
            }
        }
    }

    static void close_window(cocoa_runtime::Window& window)
    {
        if (!window.handle)
        {
            return;
        }
        cocoa_runtime::send<void>(
            window.handle,
            "orderOut:",
            static_cast<id>(nullptr));
        cocoa_runtime::send<void>(
            window.handle,
            "close");
        cocoa_runtime::send<void>(
            window.handle,
            "release");
        window.handle = nullptr;
        window.image_view = nullptr;
    }

    id application_ = nullptr;
    std::map<std::string, cocoa_runtime::Window> windows_;
};

inline HighguiBackend& highgui_backend()
{
    static HighguiBackend backend;
    return backend;
}

}  // namespace detail
}  // namespace cvh

#endif  // CVH_HIGHGUI_DETAIL_BACKEND_COCOA_HPP
