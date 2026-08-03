#ifndef CVH_HIGHGUI_DETAIL_BACKEND_STUB_HPP
#define CVH_HIGHGUI_DETAIL_BACKEND_STUB_HPP

#include "backend_api.hpp"

#include <chrono>
#include <string>
#include <thread>

namespace cvh {
namespace detail {

class HighguiBackend
{
public:
    const char* name() const
    {
        return "unavailable";
    }

    bool available() const
    {
        return false;
    }

    bool named_window(const std::string&, int)
    {
        return false;
    }

    bool show_image(const std::string&, const HighguiImage&, int)
    {
        return false;
    }

    int wait_key(int delay)
    {
        if (delay > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay));
        }
        return -1;
    }

    void destroy_window(const std::string&)
    {
    }

    void destroy_all_windows()
    {
    }
};

inline HighguiBackend& highgui_backend()
{
    static HighguiBackend backend;
    return backend;
}

}  // namespace detail
}  // namespace cvh

#endif  // CVH_HIGHGUI_DETAIL_BACKEND_STUB_HPP
