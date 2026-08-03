#ifndef CVH_HIGHGUI_HIGHGUI_H
#define CVH_HIGHGUI_HIGHGUI_H

#include "detail/backend_select.hpp"

#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <thread>

namespace cvh {

enum WindowFlags
{
    WINDOW_NORMAL = 0x00000000,
    WINDOW_AUTOSIZE = 0x00000001
};

namespace detail {

struct HighguiRegistry
{
    std::mutex mutex;
    std::map<std::string, int> windows;
};

inline HighguiRegistry& highgui_registry()
{
    static HighguiRegistry registry;
    return registry;
}

inline void validate_window_name(
    const std::string& name,
    const char* function)
{
    if (name.empty())
    {
        CV_Error_(
            Error::StsBadArg,
            ("%s: window name must not be empty", function));
    }
}

inline void report_backend_unavailable(const char* function)
{
    CV_Error_(
        Error::StsError,
        ("%s: the cvh header-only HighGUI backend is unavailable "
         "on this host. Use cvh::highgui and the platform GUI "
         "dependency, or set CVH_HIGHGUI_HEADLESS=1 for tests.",
         function));
}

inline int registered_window_flags(
    const std::string& name,
    bool& found)
{
    HighguiRegistry& registry = highgui_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    const auto item = registry.windows.find(name);
    found = item != registry.windows.end();
    return found ? item->second : WINDOW_AUTOSIZE;
}

}  // namespace detail

inline void namedWindow(
    const std::string& winname,
    int flags = WINDOW_AUTOSIZE)
{
    detail::validate_window_name(winname, "namedWindow");
    if (flags != WINDOW_NORMAL &&
        flags != WINDOW_AUTOSIZE)
    {
        CV_Error_(
            Error::StsBadArg,
            ("namedWindow: unsupported flags=%d", flags));
    }

    detail::HighguiRegistry& registry =
        detail::highgui_registry();
    {
        std::lock_guard<std::mutex> lock(registry.mutex);
        const auto found = registry.windows.find(winname);
        if (found != registry.windows.end())
        {
            return;
        }
    }

    if (!detail::highgui_headless_mode() &&
        !detail::highgui_backend().named_window(
            winname,
            flags))
    {
        detail::report_backend_unavailable("namedWindow");
    }

    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.windows.emplace(winname, flags);
}

inline void imshow(
    const std::string& winname,
    const Mat& mat)
{
    detail::validate_window_name(winname, "imshow");
    detail::HighguiImage image =
        detail::prepare_highgui_image(mat);

    bool found = false;
    int flags =
        detail::registered_window_flags(winname, found);
    if (!found)
    {
        namedWindow(winname, WINDOW_AUTOSIZE);
        flags = WINDOW_AUTOSIZE;
    }

    if (detail::highgui_headless_mode())
    {
        return;
    }
    if (!detail::highgui_backend().show_image(
            winname,
            image,
            flags))
    {
        detail::report_backend_unavailable("imshow");
    }
}

inline int waitKey(int delay = 0)
{
    if (detail::highgui_headless_mode())
    {
        if (delay > 0)
        {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(delay));
        }
        return -1;
    }
    return detail::highgui_backend().wait_key(delay);
}

inline void destroyWindow(const std::string& winname)
{
    detail::validate_window_name(winname, "destroyWindow");
    if (!detail::highgui_headless_mode())
    {
        detail::highgui_backend().destroy_window(winname);
    }

    detail::HighguiRegistry& registry =
        detail::highgui_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.windows.erase(winname);
}

inline void destroyAllWindows()
{
    if (!detail::highgui_headless_mode())
    {
        detail::highgui_backend().destroy_all_windows();
    }

    detail::HighguiRegistry& registry =
        detail::highgui_registry();
    std::lock_guard<std::mutex> lock(registry.mutex);
    registry.windows.clear();
}

}  // namespace cvh

#endif  // CVH_HIGHGUI_HIGHGUI_H
