#ifndef CVH_CORE_DETAIL_CHANNELS_UI_HPP
#define CVH_CORE_DETAIL_CHANNELS_UI_HPP

#include "dispatch_control.h"
#include "../simd/opencv_ui.h"

#include <cstddef>

namespace cvh {
namespace detail {
namespace channels_ui {

inline bool enabled()
{
#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return cpu::dispatch_mode() != cpu::DispatchMode::ScalarOnly;
#else
    return false;
#endif
}

inline bool extract_u8(const uchar* src,
                       size_t src_step,
                       uchar* dst,
                       size_t dst_step,
                       size_t rows,
                       size_t pixels,
                       int channels,
                       int channel)
{
#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const size_t lanes = static_cast<size_t>(
        cv::VTraits<cv::v_uint8>::vlanes());
    if (!enabled() || pixels < lanes ||
        (channels != 3 && channels != 4))
    {
        return false;
    }
    for (size_t row = 0; row < rows; ++row)
    {
        const uchar* src_row = src + row * src_step;
        uchar* dst_row = dst + row * dst_step;
        size_t x = 0;
        if (channels == 3)
        {
            for (; x + lanes <= pixels; x += lanes)
            {
                cv::v_uint8 c0, c1, c2;
                cv::v_load_deinterleave(src_row + 3 * x, c0, c1, c2);
                const cv::v_uint8 values[] = {c0, c1, c2};
                cv::vx_store(dst_row + x, values[channel]);
            }
        }
        else
        {
            for (; x + lanes <= pixels; x += lanes)
            {
                cv::v_uint8 c0, c1, c2, c3;
                cv::v_load_deinterleave(
                    src_row + 4 * x, c0, c1, c2, c3);
                const cv::v_uint8 values[] = {c0, c1, c2, c3};
                cv::vx_store(dst_row + x, values[channel]);
            }
        }
        for (; x < pixels; ++x)
        {
            dst_row[x] = src_row[x * static_cast<size_t>(channels) +
                                 static_cast<size_t>(channel)];
        }
    }
    return true;
#else
    (void)src;
    (void)src_step;
    (void)dst;
    (void)dst_step;
    (void)rows;
    (void)pixels;
    (void)channels;
    (void)channel;
    return false;
#endif
}

inline bool insert_u8(const uchar* src,
                      size_t src_step,
                      uchar* dst,
                      size_t dst_step,
                      size_t rows,
                      size_t pixels,
                      int channels,
                      int channel)
{
#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const size_t lanes = static_cast<size_t>(
        cv::VTraits<cv::v_uint8>::vlanes());
    if (!enabled() || pixels < lanes ||
        (channels != 3 && channels != 4))
    {
        return false;
    }
    for (size_t row = 0; row < rows; ++row)
    {
        const uchar* src_row = src + row * src_step;
        uchar* dst_row = dst + row * dst_step;
        size_t x = 0;
        if (channels == 3)
        {
            for (; x + lanes <= pixels; x += lanes)
            {
                cv::v_uint8 c0, c1, c2;
                cv::v_load_deinterleave(dst_row + 3 * x, c0, c1, c2);
                cv::v_uint8 values[] = {c0, c1, c2};
                values[channel] = cv::vx_load(src_row + x);
                cv::v_store_interleave(
                    dst_row + 3 * x, values[0], values[1], values[2]);
            }
        }
        else
        {
            for (; x + lanes <= pixels; x += lanes)
            {
                cv::v_uint8 c0, c1, c2, c3;
                cv::v_load_deinterleave(
                    dst_row + 4 * x, c0, c1, c2, c3);
                cv::v_uint8 values[] = {c0, c1, c2, c3};
                values[channel] = cv::vx_load(src_row + x);
                cv::v_store_interleave(
                    dst_row + 4 * x,
                    values[0],
                    values[1],
                    values[2],
                    values[3]);
            }
        }
        for (; x < pixels; ++x)
        {
            dst_row[x * static_cast<size_t>(channels) +
                    static_cast<size_t>(channel)] = src_row[x];
        }
    }
    return true;
#else
    (void)src;
    (void)src_step;
    (void)dst;
    (void)dst_step;
    (void)rows;
    (void)pixels;
    (void)channels;
    (void)channel;
    return false;
#endif
}

inline bool reorder_u8(const uchar* src,
                       size_t src_step,
                       uchar* dst,
                       size_t dst_step,
                       size_t rows,
                       size_t pixels,
                       int channels,
                       const int* source_for_destination)
{
#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const size_t lanes = static_cast<size_t>(
        cv::VTraits<cv::v_uint8>::vlanes());
    if (!enabled() || pixels < lanes ||
        (channels != 3 && channels != 4))
    {
        return false;
    }
    for (size_t row = 0; row < rows; ++row)
    {
        const uchar* src_row = src + row * src_step;
        uchar* dst_row = dst + row * dst_step;
        size_t x = 0;
        if (channels == 3)
        {
            for (; x + lanes <= pixels; x += lanes)
            {
                cv::v_uint8 c0, c1, c2;
                cv::v_load_deinterleave(src_row + 3 * x, c0, c1, c2);
                const cv::v_uint8 values[] = {c0, c1, c2};
                cv::v_store_interleave(
                    dst_row + 3 * x,
                    values[source_for_destination[0]],
                    values[source_for_destination[1]],
                    values[source_for_destination[2]]);
            }
        }
        else
        {
            for (; x + lanes <= pixels; x += lanes)
            {
                cv::v_uint8 c0, c1, c2, c3;
                cv::v_load_deinterleave(
                    src_row + 4 * x, c0, c1, c2, c3);
                const cv::v_uint8 values[] = {c0, c1, c2, c3};
                cv::v_store_interleave(
                    dst_row + 4 * x,
                    values[source_for_destination[0]],
                    values[source_for_destination[1]],
                    values[source_for_destination[2]],
                    values[source_for_destination[3]]);
            }
        }
        for (; x < pixels; ++x)
        {
            for (int destination = 0; destination < channels; ++destination)
            {
                dst_row[x * static_cast<size_t>(channels) +
                        static_cast<size_t>(destination)] =
                    src_row[x * static_cast<size_t>(channels) +
                            static_cast<size_t>(
                                source_for_destination[destination])];
            }
        }
    }
    return true;
#else
    (void)src;
    (void)src_step;
    (void)dst;
    (void)dst_step;
    (void)rows;
    (void)pixels;
    (void)channels;
    (void)source_for_destination;
    return false;
#endif
}

}  // namespace channels_ui
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_CHANNELS_UI_HPP
