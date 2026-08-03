#ifndef CVH_IMGPROC_EQUALIZE_HIST_H
#define CVH_IMGPROC_EQUALIZE_HIST_H

#include "detail/common.h"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/simd/opencv_ui.h"

#include <array>

namespace cvh
{

inline void equalizeHist(const Mat& src, Mat& dst)
{
    if (src.empty() || src.dims != 2 || src.type() != CV_8UC1)
    {
        CV_Error(Error::StsBadArg, "equalizeHist expects non-empty CV_8UC1 source");
    }
    const Mat source = src;
    std::array<int, 256> histogram = {};
    for (int y = 0; y < source.size.p[0]; ++y)
    {
        const uchar* input =
            source.data + static_cast<size_t>(y) * source.step(0);
        int x = 0;
        for (; x + 3 < source.size.p[1]; x += 4)
        {
            ++histogram[input[x]];
            ++histogram[input[x + 1]];
            ++histogram[input[x + 2]];
            ++histogram[input[x + 3]];
        }
        for (; x < source.size.p[1]; ++x)
        {
            ++histogram[input[x]];
        }
    }

    int first = 0;
    while (first < 256 && histogram[static_cast<size_t>(first)] == 0)
    {
        ++first;
    }
    dst.create(source.shape(), source.type());
    if (first == 256)
    {
        dst.setTo(Scalar::all(0.0));
        return;
    }
    const int total = static_cast<int>(source.total());
    if (histogram[static_cast<size_t>(first)] == total)
    {
        dst.setTo(Scalar::all(first));
        return;
    }

    std::array<uchar, 256> lookup = {};
    const float scale =
        255.0f /
        static_cast<float>(
            total - histogram[static_cast<size_t>(first)]);
    int cumulative = 0;
    lookup[static_cast<size_t>(first)] = 0;
    for (int value = first + 1; value < 256; ++value)
    {
        cumulative += histogram[static_cast<size_t>(value)];
        lookup[static_cast<size_t>(value)] =
            saturate_cast<uchar>(
                static_cast<int>(std::lrint(
                    static_cast<float>(cumulative) * scale)));
    }

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_AVX_512VBMI)
    const bool use_ui =
        cpu::opencv_ui_allowed() &&
        source.size.p[1] >= cv::VTraits<cv::v_uint8>::vlanes();
#else
    const bool use_ui = false;
#endif
    for (int y = 0; y < source.size.p[0]; ++y)
    {
        const uchar* input =
            source.data + static_cast<size_t>(y) * source.step(0);
        uchar* output =
            dst.data + static_cast<size_t>(y) * dst.step(0);
        int x = 0;
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_AVX_512VBMI)
        if (use_ui)
        {
            const int lanes =
                cv::VTraits<cv::v_uint8>::vlanes();
            for (; x + lanes <= source.size.p[1]; x += lanes)
            {
                cv::vx_store(
                    output + x,
                    cv::v_lut(
                        lookup.data(),
                        cv::vx_load(input + x)));
            }
        }
#endif
        for (; x + 3 < source.size.p[1]; x += 4)
        {
            output[x] = lookup[input[x]];
            output[x + 1] = lookup[input[x + 1]];
            output[x + 2] = lookup[input[x + 2]];
            output[x + 3] = lookup[input[x + 3]];
        }
        for (; x < source.size.p[1]; ++x)
        {
            output[x] = lookup[input[x]];
        }
    }
    if (use_ui)
    {
        cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_EQUALIZE_HIST_H
