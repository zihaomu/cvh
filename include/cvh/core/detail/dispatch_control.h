#ifndef CVH_DISPATCH_CONTROL_H
#define CVH_DISPATCH_CONTROL_H

#include <atomic>

namespace cvh {
namespace cpu {

enum class DispatchMode
{
    Auto = 0,
    ScalarOnly,
    OpenCVUIOnly,
    NeonOnly,
    Avx2Only,
};

enum class DispatchTag
{
    Unknown = 0,
    Scalar,
    OpenCVUI,
    NEON,
    AVX2,
};

inline std::atomic<DispatchMode> g_dispatch_mode {DispatchMode::Auto};
inline thread_local DispatchTag g_last_dispatch_tag = DispatchTag::Unknown;

inline void set_dispatch_mode(DispatchMode mode)
{
    g_dispatch_mode.store(mode, std::memory_order_relaxed);
}

inline DispatchMode dispatch_mode()
{
    return g_dispatch_mode.load(std::memory_order_relaxed);
}

inline bool opencv_ui_allowed()
{
    const DispatchMode mode = dispatch_mode();
    return mode == DispatchMode::Auto ||
           mode == DispatchMode::OpenCVUIOnly;
}

inline void set_last_dispatch_tag(DispatchTag tag)
{
    g_last_dispatch_tag = tag;
}

inline void reset_last_dispatch_tag()
{
    g_last_dispatch_tag = DispatchTag::Unknown;
}

inline DispatchTag last_dispatch_tag()
{
    return g_last_dispatch_tag;
}

inline const char* dispatch_tag_name(DispatchTag tag)
{
    switch (tag)
    {
        case DispatchTag::Scalar:
            return "scalar";
        case DispatchTag::OpenCVUI:
            return "opencv_ui";
        case DispatchTag::NEON:
            return "neon";
        case DispatchTag::AVX2:
            return "avx2";
        case DispatchTag::Unknown:
        default:
            return "unknown";
    }
}

}  // namespace cpu
}  // namespace cvh

#endif  // CVH_DISPATCH_CONTROL_H
