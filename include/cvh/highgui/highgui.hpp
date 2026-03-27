#ifndef CVH_HIGHGUI_HPP
#define CVH_HIGHGUI_HPP

#include "../core/mat.h"
#include "../imgcodecs/imgcodecs.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <string>
#include <thread>

namespace cvh {

#if defined(CVH_FULL)
CV_EXPORTS void register_highgui_backends();
#endif

namespace detail {

using ImshowFn = void (*)(const std::string&, const Mat&);
using WaitKeyFn = int (*)(int);

inline std::string sanitize_window_name(std::string name)
{
    if (name.empty())
    {
        return "cvh_imshow";
    }

    for (char& ch : name)
    {
        const bool alnum = std::isalnum(static_cast<unsigned char>(ch)) != 0;
        if (!alnum && ch != '_' && ch != '-' && ch != '.')
        {
            ch = '_';
        }
    }
    return name;
}

inline void imshow_fallback(const std::string& winname, const Mat& mat)
{
    CV_Assert(!mat.empty() && "imshow: source image can not be empty");
    CV_Assert(mat.depth() == CV_8U && "imshow: v1 supports CV_8U only");
    CV_Assert(mat.dims == 2 && "imshow: only 2D Mat is supported");

    const std::string filename = sanitize_window_name(winname) + ".png";
    if (!imwrite(filename, mat))
    {
        CV_Error_(Error::StsBadArg, ("imshow: failed to write fallback image '%s'", filename.c_str()));
    }

#if defined(_WIN32)
    const std::string cmd = "start \"\" \"" + filename + "\"";
    (void)std::system(cmd.c_str());
#elif defined(__linux__) && !defined(__ANDROID__)
    const std::string cmd = "xdg-open \"" + filename + "\" >/dev/null 2>&1 &";
    (void)std::system(cmd.c_str());
#endif
}

inline int waitkey_fallback(int delay)
{
    if (delay > 0)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    }
    return -1;
}

inline ImshowFn& imshow_dispatch()
{
    static ImshowFn fn = &imshow_fallback;
    return fn;
}

inline WaitKeyFn& waitkey_dispatch()
{
    static WaitKeyFn fn = &waitkey_fallback;
    return fn;
}

inline void register_imshow_backend(ImshowFn fn)
{
    if (fn)
    {
        imshow_dispatch() = fn;
    }
}

inline void register_waitkey_backend(WaitKeyFn fn)
{
    if (fn)
    {
        waitkey_dispatch() = fn;
    }
}

inline bool is_imshow_backend_registered()
{
    return imshow_dispatch() != &imshow_fallback;
}

inline bool is_waitkey_backend_registered()
{
    return waitkey_dispatch() != &waitkey_fallback;
}

inline void ensure_highgui_backends_registered_once()
{
#if defined(CVH_FULL)
    static bool initialized = []() {
        cvh::register_highgui_backends();
        return true;
    }();
    (void)initialized;
#endif
}

}  // namespace detail

inline void imshow(const std::string& winname, const Mat& mat)
{
    detail::ensure_highgui_backends_registered_once();
    detail::imshow_dispatch()(winname, mat);
}

inline int waitKey(int delay = 0)
{
    detail::ensure_highgui_backends_registered_once();
    return detail::waitkey_dispatch()(delay);
}

}  // namespace cvh

#endif  // CVH_HIGHGUI_HPP
