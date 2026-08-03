#ifndef CVH_CORE_DETAIL_COPY_UI_HPP
#define CVH_CORE_DETAIL_COPY_UI_HPP

#include "dispatch_control.h"
#include "../simd/opencv_ui.h"

#include <cstddef>

namespace cvh {
namespace detail {
namespace copy_ui {

inline bool enabled()
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return cpu::opencv_ui_allowed();
#else
    return false;
#endif
}

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)

inline void copy_masked_u8c1_row(const uchar* src,
                                 const uchar* mask,
                                 uchar* dst,
                                 size_t pixels)
{
    const size_t lanes = static_cast<size_t>(
        cv::VTraits<cv::v_uint8>::vlanes());
    const cv::v_uint8 zero = cv::vx_setzero_u8();
    size_t x = 0;
    for (; x + lanes <= pixels; x += lanes)
    {
        const cv::v_uint8 keep = cv::v_eq(cv::vx_load(mask + x), zero);
        cv::vx_store(
            dst + x,
            cv::v_select(keep, cv::vx_load(dst + x), cv::vx_load(src + x)));
    }
    for (; x < pixels; ++x)
    {
        if (mask[x] != 0)
        {
            dst[x] = src[x];
        }
    }
}

inline void copy_masked_u8c3_row(const uchar* src,
                                 const uchar* mask,
                                 uchar* dst,
                                 size_t pixels)
{
    const size_t lanes = static_cast<size_t>(
        cv::VTraits<cv::v_uint8>::vlanes());
    const cv::v_uint8 zero = cv::vx_setzero_u8();
    size_t x = 0;
    for (; x + lanes <= pixels; x += lanes)
    {
        const cv::v_uint8 keep = cv::v_eq(cv::vx_load(mask + x), zero);
        cv::v_uint8 src0, src1, src2;
        cv::v_uint8 dst0, dst1, dst2;
        cv::v_load_deinterleave(src + 3 * x, src0, src1, src2);
        cv::v_load_deinterleave(dst + 3 * x, dst0, dst1, dst2);
        cv::v_store_interleave(
            dst + 3 * x,
            cv::v_select(keep, dst0, src0),
            cv::v_select(keep, dst1, src1),
            cv::v_select(keep, dst2, src2));
    }
    for (; x < pixels; ++x)
    {
        if (mask[x] != 0)
        {
            const size_t offset = 3 * x;
            dst[offset] = src[offset];
            dst[offset + 1] = src[offset + 1];
            dst[offset + 2] = src[offset + 2];
        }
    }
}

inline void copy_masked_u8c4_row(const uchar* src,
                                 const uchar* mask,
                                 uchar* dst,
                                 size_t pixels)
{
    const size_t lanes = static_cast<size_t>(
        cv::VTraits<cv::v_uint8>::vlanes());
    const cv::v_uint8 zero = cv::vx_setzero_u8();
    size_t x = 0;
    for (; x + lanes <= pixels; x += lanes)
    {
        const cv::v_uint8 keep = cv::v_eq(cv::vx_load(mask + x), zero);
        cv::v_uint8 src0, src1, src2, src3;
        cv::v_uint8 dst0, dst1, dst2, dst3;
        cv::v_load_deinterleave(src + 4 * x, src0, src1, src2, src3);
        cv::v_load_deinterleave(dst + 4 * x, dst0, dst1, dst2, dst3);
        cv::v_store_interleave(
            dst + 4 * x,
            cv::v_select(keep, dst0, src0),
            cv::v_select(keep, dst1, src1),
            cv::v_select(keep, dst2, src2),
            cv::v_select(keep, dst3, src3));
    }
    for (; x < pixels; ++x)
    {
        if (mask[x] != 0)
        {
            const size_t offset = 4 * x;
            dst[offset] = src[offset];
            dst[offset + 1] = src[offset + 1];
            dst[offset + 2] = src[offset + 2];
            dst[offset + 3] = src[offset + 3];
        }
    }
}

#endif

inline bool copy_masked_u8_rows(const uchar* src,
                                size_t src_step,
                                const uchar* mask,
                                size_t mask_step,
                                uchar* dst,
                                size_t dst_step,
                                size_t pixels,
                                size_t rows,
                                int channels)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const size_t lanes = static_cast<size_t>(
        cv::VTraits<cv::v_uint8>::vlanes());
    if (!enabled() || pixels < lanes ||
        (channels != 1 && channels != 3 && channels != 4))
    {
        return false;
    }

    for (size_t row = 0; row < rows; ++row)
    {
        const uchar* src_row = src + row * src_step;
        const uchar* mask_row = mask + row * mask_step;
        uchar* dst_row = dst + row * dst_step;
        if (channels == 1)
        {
            copy_masked_u8c1_row(src_row, mask_row, dst_row, pixels);
        }
        else if (channels == 3)
        {
            copy_masked_u8c3_row(src_row, mask_row, dst_row, pixels);
        }
        else
        {
            copy_masked_u8c4_row(src_row, mask_row, dst_row, pixels);
        }
    }
    return true;
#else
    (void)src;
    (void)src_step;
    (void)mask;
    (void)mask_step;
    (void)dst;
    (void)dst_step;
    (void)pixels;
    (void)rows;
    (void)channels;
    return false;
#endif
}

}  // namespace copy_ui
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_COPY_UI_HPP
