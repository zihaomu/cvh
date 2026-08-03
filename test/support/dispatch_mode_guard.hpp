#pragma once

#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/simd/opencv_ui.h"

namespace cvh::test
{

class DispatchModeGuard
{
public:
    DispatchModeGuard()
        : previous_(cpu::dispatch_mode())
    {
    }

    explicit DispatchModeGuard(cpu::DispatchMode mode)
        : previous_(cpu::dispatch_mode())
    {
        cpu::set_dispatch_mode(mode);
    }

    ~DispatchModeGuard()
    {
        cpu::set_dispatch_mode(previous_);
    }

    DispatchModeGuard(const DispatchModeGuard&) = delete;
    DispatchModeGuard& operator=(const DispatchModeGuard&) = delete;

private:
    cpu::DispatchMode previous_;
};

inline constexpr bool fixed_width_opencv_ui_available()
{
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return true;
#else
    return false;
#endif
}

inline cpu::DispatchTag expected_fixed_width_dispatch_tag()
{
    return fixed_width_opencv_ui_available()
        ? cpu::DispatchTag::OpenCVUI
        : cpu::DispatchTag::Scalar;
}

template<typename T>
inline int fixed_width_opencv_ui_lanes()
{
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    using Vector = decltype(cv::vx_load(static_cast<const T*>(nullptr)));
    return cv::VTraits<Vector>::vlanes();
#else
    return 1;
#endif
}

template<typename T>
inline int accepted_fixed_width_test_length()
{
    return 2 * fixed_width_opencv_ui_lanes<T>() + 3;
}

}  // namespace cvh::test
