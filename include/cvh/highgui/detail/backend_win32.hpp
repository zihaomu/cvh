#ifndef CVH_HIGHGUI_DETAIL_BACKEND_WIN32_HPP
#define CVH_HIGHGUI_DETAIL_BACKEND_WIN32_HPP

#include "backend_api.hpp"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cvh {
namespace detail {

namespace win32_highgui {

struct Window
{
    HWND handle = nullptr;
    BITMAPINFO bitmap_info{};
    std::vector<uchar> pixels;
    int width = 0;
    int height = 0;
    int flags = 0;
};

inline LRESULT CALLBACK window_procedure(
    HWND handle,
    UINT message,
    WPARAM word,
    LPARAM parameter)
{
    Window* window = reinterpret_cast<Window*>(
        GetWindowLongPtrA(handle, GWLP_USERDATA));
    if (message == WM_NCCREATE)
    {
        const CREATESTRUCTA* create =
            reinterpret_cast<const CREATESTRUCTA*>(parameter);
        window = static_cast<Window*>(create->lpCreateParams);
        SetWindowLongPtrA(
            handle,
            GWLP_USERDATA,
            reinterpret_cast<LONG_PTR>(window));
        if (window)
        {
            window->handle = handle;
        }
    }

    if (!window)
    {
        return DefWindowProcA(handle, message, word, parameter);
    }

    switch (message)
    {
    case WM_ERASEBKGND:
        return 1;
    case WM_PAINT:
    {
        PAINTSTRUCT paint{};
        HDC device = BeginPaint(handle, &paint);
        RECT client{};
        GetClientRect(handle, &client);
        if (!window->pixels.empty())
        {
            StretchDIBits(
                device,
                0,
                0,
                client.right - client.left,
                client.bottom - client.top,
                0,
                0,
                window->width,
                window->height,
                window->pixels.data(),
                &window->bitmap_info,
                DIB_RGB_COLORS,
                SRCCOPY);
        }
        EndPaint(handle, &paint);
        return 0;
    }
    case WM_CLOSE:
        DestroyWindow(handle);
        return 0;
    case WM_NCDESTROY:
        window->handle = nullptr;
        SetWindowLongPtrA(handle, GWLP_USERDATA, 0);
        return DefWindowProcA(handle, message, word, parameter);
    default:
        return DefWindowProcA(handle, message, word, parameter);
    }
}

}  // namespace win32_highgui

class HighguiBackend
{
public:
    const char* name() const
    {
        return "win32";
    }

    bool available()
    {
        return ensure_window_class();
    }

    bool named_window(const std::string& name, int flags)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        return get_or_create_window(name, 640, 480, flags) != nullptr;
    }

    bool show_image(
        const std::string& name,
        const HighguiImage& image,
        int flags)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        win32_highgui::Window* window =
            get_or_create_window(
                name,
                image.width,
                image.height,
                flags);
        if (!window || !window->handle)
        {
            return false;
        }

        window->width = image.width;
        window->height = image.height;
        window->pixels.resize(
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
                window->pixels.data() +
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
                    destination_pixel[0] = source_pixel[0];
                    destination_pixel[1] = source_pixel[1];
                    destination_pixel[2] = source_pixel[2];
                }
                destination_pixel[3] = 255;
            }
        }

        ZeroMemory(
            &window->bitmap_info,
            sizeof(window->bitmap_info));
        BITMAPINFOHEADER& header =
            window->bitmap_info.bmiHeader;
        header.biSize = sizeof(BITMAPINFOHEADER);
        header.biWidth = image.width;
        header.biHeight = -image.height;
        header.biPlanes = 1;
        header.biBitCount = 32;
        header.biCompression = BI_RGB;

        if ((window->flags & 1) != 0)
        {
            resize_client(
                window->handle,
                image.width,
                image.height);
        }
        InvalidateRect(window->handle, nullptr, FALSE);
        UpdateWindow(window->handle);
        return true;
    }

    int wait_key(int delay)
    {
        if (windows_.empty())
        {
            if (delay > 0)
            {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(delay));
            }
            return -1;
        }

        const bool wait_forever = delay <= 0;
        const auto deadline =
            std::chrono::steady_clock::now() +
            std::chrono::milliseconds(std::max(0, delay));
        for (;;)
        {
            MSG message{};
            bool received = false;
            if (wait_forever)
            {
                const BOOL status =
                    GetMessageA(&message, nullptr, 0, 0);
                if (status <= 0)
                {
                    return -1;
                }
                received = true;
            }
            else
            {
                received =
                    PeekMessageA(
                        &message,
                        nullptr,
                        0,
                        0,
                        PM_REMOVE) != FALSE;
            }

            if (received)
            {
                if (message.message == WM_CHAR)
                {
                    return static_cast<int>(message.wParam);
                }
                TranslateMessage(&message);
                DispatchMessageA(&message);
                cleanup_closed_windows();
                if (windows_.empty())
                {
                    return -1;
                }
                continue;
            }

            if (!wait_forever &&
                std::chrono::steady_clock::now() >= deadline)
            {
                return -1;
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(1));
        }
    }

    void destroy_window(const std::string& name)
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        const auto found = windows_.find(name);
        if (found == windows_.end())
        {
            return;
        }
        if (found->second->handle)
        {
            DestroyWindow(found->second->handle);
        }
        windows_.erase(found);
    }

    void destroy_all_windows()
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        for (auto& item : windows_)
        {
            if (item.second->handle)
            {
                DestroyWindow(item.second->handle);
            }
        }
        windows_.clear();
    }

private:
    bool ensure_window_class()
    {
        if (window_class_)
        {
            return true;
        }

        WNDCLASSA window_class{};
        window_class.style = CS_HREDRAW | CS_VREDRAW;
        window_class.lpfnWndProc =
            win32_highgui::window_procedure;
        window_class.hInstance =
            GetModuleHandleA(nullptr);
        window_class.hCursor =
            LoadCursor(nullptr, IDC_ARROW);
        window_class.hbrBackground =
            reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1);
        window_class.lpszClassName =
            "cvh_highgui_window";
        window_class_ = RegisterClassA(&window_class);
        if (!window_class_ &&
            GetLastError() == ERROR_CLASS_ALREADY_EXISTS)
        {
            window_class_ = 1;
        }
        return window_class_ != 0;
    }

    win32_highgui::Window* get_or_create_window(
        const std::string& name,
        int width,
        int height,
        int flags)
    {
        if (!ensure_window_class())
        {
            return nullptr;
        }

        auto found = windows_.find(name);
        if (found != windows_.end() &&
            found->second->handle)
        {
            return found->second.get();
        }
        if (found != windows_.end())
        {
            windows_.erase(found);
        }

        std::unique_ptr<win32_highgui::Window> state(
            new win32_highgui::Window());
        state->width = width;
        state->height = height;
        state->flags = flags;
        win32_highgui::Window* state_pointer = state.get();

        RECT rectangle{0, 0, width, height};
        const DWORD style = WS_OVERLAPPEDWINDOW;
        AdjustWindowRect(&rectangle, style, FALSE);
        HWND handle = CreateWindowExA(
            0,
            "cvh_highgui_window",
            name.c_str(),
            style,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rectangle.right - rectangle.left,
            rectangle.bottom - rectangle.top,
            nullptr,
            nullptr,
            GetModuleHandleA(nullptr),
            state_pointer);
        if (!handle)
        {
            return nullptr;
        }

        state_pointer->handle = handle;
        windows_.emplace(name, std::move(state));
        ShowWindow(handle, SW_SHOWNORMAL);
        UpdateWindow(handle);
        return state_pointer;
    }

    static void resize_client(
        HWND handle,
        int width,
        int height)
    {
        RECT rectangle{0, 0, width, height};
        const DWORD style =
            static_cast<DWORD>(
                GetWindowLongPtrA(handle, GWL_STYLE));
        AdjustWindowRect(&rectangle, style, FALSE);
        SetWindowPos(
            handle,
            nullptr,
            0,
            0,
            rectangle.right - rectangle.left,
            rectangle.bottom - rectangle.top,
            SWP_NOMOVE | SWP_NOZORDER);
    }

    void cleanup_closed_windows()
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        for (auto iterator = windows_.begin();
             iterator != windows_.end();)
        {
            if (!iterator->second->handle)
            {
                iterator = windows_.erase(iterator);
            }
            else
            {
                ++iterator;
            }
        }
    }

    std::recursive_mutex mutex_;
    ATOM window_class_ = 0;
    std::map<
        std::string,
        std::unique_ptr<win32_highgui::Window>>
        windows_;
};

inline HighguiBackend& highgui_backend()
{
    static HighguiBackend backend;
    return backend;
}

}  // namespace detail
}  // namespace cvh

#endif  // CVH_HIGHGUI_DETAIL_BACKEND_WIN32_HPP
