#ifndef CVH_IMGPROC_DETAIL_TEMPLATE_MATCH_UI_HPP
#define CVH_IMGPROC_DETAIL_TEMPLATE_MATCH_UI_HPP

#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/simd/opencv_ui.h"

#include <cstddef>

namespace cvh {
namespace detail {
namespace template_match_ui {

inline bool enabled()
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return cpu::opencv_ui_allowed();
#else
    return false;
#endif
}

inline bool can_dot_u8(int rows, int columns)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    // v_dotprod_expand accumulates four U8 products in each U32 lane. Keep
    // the accepted template area conservative so the lane accumulator cannot
    // overflow even for an all-255 input.
    constexpr std::size_t kMaximumSafeArea = 65536;
    return enabled() &&
           columns >= cv::VTraits<cv::v_uint8>::vlanes() &&
           static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns) <=
               kMaximumSafeArea;
#else
    (void)rows;
    (void)columns;
    return false;
#endif
}

inline bool can_dot_f32(int columns)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return enabled() && columns >= cv::VTraits<cv::v_float32>::vlanes();
#else
    (void)columns;
    return false;
#endif
}

inline double dot_u8(const unsigned char* image,
                     std::size_t image_step,
                     const unsigned char* templ,
                     std::size_t template_step,
                     int rows,
                     int columns)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const int lanes = cv::VTraits<cv::v_uint8>::vlanes();
    cv::v_uint32 sum = cv::vx_setzero_u32();
    double tail_sum = 0.0;
    for (int row = 0; row < rows; ++row)
    {
        const unsigned char* image_row = image +
            static_cast<std::size_t>(row) * image_step;
        const unsigned char* template_row = templ +
            static_cast<std::size_t>(row) * template_step;
        int column = 0;
        for (; column <= columns - lanes; column += lanes)
        {
            sum = cv::v_dotprod_expand(
                cv::vx_load(image_row + column),
                cv::vx_load(template_row + column),
                sum);
        }
        for (; column < columns; ++column)
        {
            tail_sum += static_cast<double>(image_row[column]) *
                        static_cast<double>(template_row[column]);
        }
    }
    return static_cast<double>(cv::v_reduce_sum(sum)) + tail_sum;
#else
    (void)image;
    (void)image_step;
    (void)templ;
    (void)template_step;
    (void)rows;
    (void)columns;
    return 0.0;
#endif
}

inline double dot_f32(const float* image,
                      std::size_t image_step,
                      const float* templ,
                      std::size_t template_step,
                      int rows,
                      int columns)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const int lanes = cv::VTraits<cv::v_float32>::vlanes();
    cv::v_float32 sum0 = cv::vx_setzero_f32();
    cv::v_float32 sum1 = cv::vx_setzero_f32();
    double tail_sum = 0.0;
    for (int row = 0; row < rows; ++row)
    {
        const float* image_row = reinterpret_cast<const float*>(
            reinterpret_cast<const unsigned char*>(image) +
            static_cast<std::size_t>(row) * image_step);
        const float* template_row = reinterpret_cast<const float*>(
            reinterpret_cast<const unsigned char*>(templ) +
            static_cast<std::size_t>(row) * template_step);
        int column = 0;
        for (; column <= columns - 2 * lanes; column += 2 * lanes)
        {
            sum0 = cv::v_fma(
                cv::vx_load(image_row + column),
                cv::vx_load(template_row + column),
                sum0);
            sum1 = cv::v_fma(
                cv::vx_load(image_row + column + lanes),
                cv::vx_load(template_row + column + lanes),
                sum1);
        }
        for (; column <= columns - lanes; column += lanes)
        {
            sum0 = cv::v_fma(
                cv::vx_load(image_row + column),
                cv::vx_load(template_row + column),
                sum0);
        }
        for (; column < columns; ++column)
        {
            tail_sum += static_cast<double>(image_row[column]) *
                        static_cast<double>(template_row[column]);
        }
    }
    return static_cast<double>(cv::v_reduce_sum(cv::v_add(sum0, sum1))) +
           tail_sum;
#else
    (void)image;
    (void)image_step;
    (void)templ;
    (void)template_step;
    (void)rows;
    (void)columns;
    return 0.0;
#endif
}

}  // namespace template_match_ui
}  // namespace detail
}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_TEMPLATE_MATCH_UI_HPP
