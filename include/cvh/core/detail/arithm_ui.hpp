#ifndef CVH_CORE_DETAIL_ARITHM_UI_HPP
#define CVH_CORE_DETAIL_ARITHM_UI_HPP

#include "dispatch_control.h"
#include "../simd/opencv_ui.h"
#include "../saturate.h"

#include <cstddef>

namespace cvh {
namespace detail {
namespace arithm_ui {

inline bool enabled()
{
#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return cpu::dispatch_mode() != cpu::DispatchMode::ScalarOnly;
#else
    return false;
#endif
}

#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)

template<typename T, typename VectorOp, typename ScalarOp>
inline bool apply_binary_rows(const T* src1,
                              size_t src1_step,
                              const T* src2,
                              size_t src2_step,
                              T* dst,
                              size_t dst_step,
                              size_t row_scalars,
                              size_t rows,
                              VectorOp vector_op,
                              ScalarOp scalar_op)
{
    using Vec = decltype(cv::vx_load(src1));
    const size_t lanes = static_cast<size_t>(cv::VTraits<Vec>::vlanes());
    if (row_scalars < lanes)
    {
        return false;
    }

    for (size_t row = 0; row < rows; ++row)
    {
        const T* src1_row = reinterpret_cast<const T*>(
            reinterpret_cast<const uchar*>(src1) + row * src1_step);
        const T* src2_row = reinterpret_cast<const T*>(
            reinterpret_cast<const uchar*>(src2) + row * src2_step);
        T* dst_row = reinterpret_cast<T*>(
            reinterpret_cast<uchar*>(dst) + row * dst_step);

        size_t x = 0;
        for (; x + lanes <= row_scalars; x += lanes)
        {
            cv::vx_store(dst_row + x,
                         vector_op(cv::vx_load(src1_row + x),
                                   cv::vx_load(src2_row + x)));
        }
        for (; x < row_scalars; ++x)
        {
            dst_row[x] = saturate_cast<T>(scalar_op(src1_row[x], src2_row[x]));
        }
    }
    return true;
}

template<typename VectorOp, typename ScalarOp>
inline bool apply_binary_byte_rows(const uchar* src1,
                                   size_t src1_step,
                                   const uchar* src2,
                                   size_t src2_step,
                                   uchar* dst,
                                   size_t dst_step,
                                   size_t row_bytes,
                                   size_t rows,
                                   VectorOp vector_op,
                                   ScalarOp scalar_op)
{
    return apply_binary_rows(
        src1,
        src1_step,
        src2,
        src2_step,
        dst,
        dst_step,
        row_bytes,
        rows,
        vector_op,
        scalar_op);
}

template<typename VectorOp, typename ScalarOp>
inline bool apply_unary_byte_rows(const uchar* src,
                                  size_t src_step,
                                  uchar* dst,
                                  size_t dst_step,
                                  size_t row_bytes,
                                  size_t rows,
                                  VectorOp vector_op,
                                  ScalarOp scalar_op)
{
    const size_t lanes = static_cast<size_t>(cv::VTraits<cv::v_uint8>::vlanes());
    if (row_bytes < lanes)
    {
        return false;
    }

    for (size_t row = 0; row < rows; ++row)
    {
        const uchar* src_row = src + row * src_step;
        uchar* dst_row = dst + row * dst_step;

        size_t x = 0;
        for (; x + lanes <= row_bytes; x += lanes)
        {
            cv::vx_store(dst_row + x, vector_op(cv::vx_load(src_row + x)));
        }
        for (; x < row_bytes; ++x)
        {
            dst_row[x] = scalar_op(src_row[x]);
        }
    }
    return true;
}

#endif

}  // namespace arithm_ui
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_ARITHM_UI_HPP
