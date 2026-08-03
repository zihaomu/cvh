#ifndef CVH_CORE_DETAIL_LAYOUT_UI_HPP
#define CVH_CORE_DETAIL_LAYOUT_UI_HPP

#include "dispatch_control.h"
#include "../simd/opencv_ui.h"

#include <cstddef>

namespace cvh {
namespace detail {
namespace layout_ui {

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)

template<typename Vector>
inline void flip_horizontal_single_rows(const uchar* src,
                                        size_t src_step,
                                        uchar* dst,
                                        size_t dst_step,
                                        size_t rows,
                                        size_t pixels,
                                        bool flip_vertical)
{
    using T = typename cv::VTraits<Vector>::lane_type;
    const size_t lanes = static_cast<size_t>(
        cv::VTraits<Vector>::vlanes());
    const size_t half = (pixels + 1) / 2;
    for (size_t row = 0; row < rows; ++row)
    {
        const size_t source_row = flip_vertical ? rows - 1 - row : row;
        const T* src_row = reinterpret_cast<const T*>(
            src + source_row * src_step);
        T* dst_row = reinterpret_cast<T*>(dst + row * dst_step);
        size_t x = 0;
        for (; x + lanes <= half; x += lanes)
        {
            const size_t right = pixels - x - lanes;
            const Vector left = cv::vx_load(src_row + x);
            const Vector right_values = cv::vx_load(src_row + right);
            cv::vx_store(dst_row + x, cv::v_reverse(right_values));
            cv::vx_store(dst_row + right, cv::v_reverse(left));
        }
        for (; x < half; ++x)
        {
            const size_t right = pixels - 1 - x;
            dst_row[x] = src_row[right];
            dst_row[right] = src_row[x];
        }
    }
}

template<typename Vector>
inline void flip_horizontal_c3_rows(const uchar* src,
                                    size_t src_step,
                                    uchar* dst,
                                    size_t dst_step,
                                    size_t rows,
                                    size_t pixels,
                                    bool flip_vertical)
{
    using T = typename cv::VTraits<Vector>::lane_type;
    const size_t lanes = static_cast<size_t>(
        cv::VTraits<Vector>::vlanes());
    const size_t half = (pixels + 1) / 2;
    for (size_t row = 0; row < rows; ++row)
    {
        const size_t source_row = flip_vertical ? rows - 1 - row : row;
        const T* src_row = reinterpret_cast<const T*>(
            src + source_row * src_step);
        T* dst_row = reinterpret_cast<T*>(dst + row * dst_step);
        size_t x = 0;
        for (; x + lanes <= half; x += lanes)
        {
            const size_t right = pixels - x - lanes;
            Vector l0, l1, l2;
            Vector r0, r1, r2;
            cv::v_load_deinterleave(src_row + 3 * x, l0, l1, l2);
            cv::v_load_deinterleave(src_row + 3 * right, r0, r1, r2);
            cv::v_store_interleave(
                dst_row + 3 * x,
                cv::v_reverse(r0),
                cv::v_reverse(r1),
                cv::v_reverse(r2));
            cv::v_store_interleave(
                dst_row + 3 * right,
                cv::v_reverse(l0),
                cv::v_reverse(l1),
                cv::v_reverse(l2));
        }
        for (; x < half; ++x)
        {
            const size_t right = pixels - 1 - x;
            for (size_t channel = 0; channel < 3; ++channel)
            {
                dst_row[3 * x + channel] =
                    src_row[3 * right + channel];
                dst_row[3 * right + channel] =
                    src_row[3 * x + channel];
            }
        }
    }
}

#endif

inline bool flip_horizontal_rows(const uchar* src,
                                 size_t src_step,
                                 uchar* dst,
                                 size_t dst_step,
                                 size_t rows,
                                 size_t pixels,
                                 size_t elem_size,
                                 bool flip_vertical)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    if (!cpu::opencv_ui_allowed())
    {
        return false;
    }
    switch (elem_size)
    {
        case 1:
            if (pixels >= static_cast<size_t>(cv::VTraits<cv::v_uint8>::vlanes()))
            {
                flip_horizontal_single_rows<cv::v_uint8>(
                    src, src_step, dst, dst_step, rows, pixels, flip_vertical);
                return true;
            }
            break;
        case 2:
            if (pixels >= static_cast<size_t>(cv::VTraits<cv::v_uint16>::vlanes()))
            {
                flip_horizontal_single_rows<cv::v_uint16>(
                    src, src_step, dst, dst_step, rows, pixels, flip_vertical);
                return true;
            }
            break;
        case 3:
            if (pixels >= static_cast<size_t>(cv::VTraits<cv::v_uint8>::vlanes()))
            {
                flip_horizontal_c3_rows<cv::v_uint8>(
                    src, src_step, dst, dst_step, rows, pixels, flip_vertical);
                return true;
            }
            break;
        case 4:
            if (pixels >= static_cast<size_t>(cv::VTraits<cv::v_uint32>::vlanes()))
            {
                flip_horizontal_single_rows<cv::v_uint32>(
                    src, src_step, dst, dst_step, rows, pixels, flip_vertical);
                return true;
            }
            break;
        case 6:
            if (pixels >= static_cast<size_t>(cv::VTraits<cv::v_uint16>::vlanes()))
            {
                flip_horizontal_c3_rows<cv::v_uint16>(
                    src, src_step, dst, dst_step, rows, pixels, flip_vertical);
                return true;
            }
            break;
        case 8:
            if (pixels >= static_cast<size_t>(cv::VTraits<cv::v_uint64>::vlanes()))
            {
                flip_horizontal_single_rows<cv::v_uint64>(
                    src, src_step, dst, dst_step, rows, pixels, flip_vertical);
                return true;
            }
            break;
        case 12:
            if (pixels >= static_cast<size_t>(cv::VTraits<cv::v_uint32>::vlanes()))
            {
                flip_horizontal_c3_rows<cv::v_uint32>(
                    src, src_step, dst, dst_step, rows, pixels, flip_vertical);
                return true;
            }
            break;
        default:
            break;
    }
#else
    (void)src;
    (void)src_step;
    (void)dst;
    (void)dst_step;
    (void)rows;
    (void)pixels;
    (void)elem_size;
    (void)flip_vertical;
#endif
    return false;
}

}  // namespace layout_ui
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_LAYOUT_UI_HPP
