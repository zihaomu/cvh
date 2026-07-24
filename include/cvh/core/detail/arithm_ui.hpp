#ifndef CVH_CORE_DETAIL_ARITHM_UI_HPP
#define CVH_CORE_DETAIL_ARITHM_UI_HPP

#include "dispatch_control.h"
#include "../simd/opencv_ui.h"
#include "../saturate.h"

#include <cstddef>
#include <type_traits>
#include <vector>

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

template<typename T, typename VectorOp, typename ScalarOp>
inline bool apply_binary_scalar_rows(const T* src,
                                     size_t src_step,
                                     const T* scalar,
                                     int channels,
                                     T* dst,
                                     size_t dst_step,
                                     size_t row_scalars,
                                     size_t rows,
                                     bool scalar_first,
                                     VectorOp vector_op,
                                     ScalarOp scalar_op)
{
    using Vec = decltype(cv::vx_load(src));
    const size_t lanes = static_cast<size_t>(cv::VTraits<Vec>::vlanes());
    if (row_scalars < lanes)
    {
        return false;
    }

    std::vector<T> scalar_patterns(
        static_cast<size_t>(channels) * lanes);
    for (int phase = 0; phase < channels; ++phase)
    {
        for (size_t lane = 0; lane < lanes; ++lane)
        {
            const int channel =
                (phase + static_cast<int>(lane % static_cast<size_t>(channels))) %
                channels;
            scalar_patterns[static_cast<size_t>(phase) * lanes + lane] =
                scalar[channel];
        }
    }

    for (size_t row = 0; row < rows; ++row)
    {
        const T* src_row = reinterpret_cast<const T*>(
            reinterpret_cast<const uchar*>(src) + row * src_step);
        T* dst_row = reinterpret_cast<T*>(
            reinterpret_cast<uchar*>(dst) + row * dst_step);

        size_t x = 0;
        for (; x + lanes <= row_scalars; x += lanes)
        {
            const size_t phase = x % static_cast<size_t>(channels);
            const auto src_vector = cv::vx_load(src_row + x);
            const auto scalar_vector =
                cv::vx_load(scalar_patterns.data() + phase * lanes);
            cv::vx_store(
                dst_row + x,
                scalar_first
                    ? vector_op(scalar_vector, src_vector)
                    : vector_op(src_vector, scalar_vector));
        }
        for (; x < row_scalars; ++x)
        {
            const T scalar_value =
                scalar[x % static_cast<size_t>(channels)];
            dst_row[x] = saturate_cast<T>(
                scalar_first
                    ? scalar_op(scalar_value, src_row[x])
                    : scalar_op(src_row[x], scalar_value));
        }
    }
    return true;
}

template<typename VectorOp, typename ScalarOp>
inline bool apply_masked_binary_byte_rows(const uchar* src1,
                                          size_t src1_step,
                                          const uchar* src2,
                                          size_t src2_step,
                                          uchar* dst,
                                          size_t dst_step,
                                          const uchar* mask,
                                          size_t mask_step,
                                          size_t pixels,
                                          size_t pixel_bytes,
                                          size_t rows,
                                          VectorOp vector_op,
                                          ScalarOp scalar_op)
{
    const size_t lanes =
        static_cast<size_t>(cv::VTraits<cv::v_uint8>::vlanes());
    if (pixels < lanes || pixel_bytes == 0 || pixel_bytes > 4)
    {
        return false;
    }

    for (size_t row = 0; row < rows; ++row)
    {
        const uchar* src1_row = src1 + row * src1_step;
        const uchar* src2_row = src2 + row * src2_step;
        uchar* dst_row = dst + row * dst_step;
        const uchar* mask_row = mask + row * mask_step;

        size_t pixel = 0;
        for (; pixel + lanes <= pixels; pixel += lanes)
        {
            const auto not_selected =
                cv::v_eq(cv::vx_load(mask_row + pixel), cv::vx_setzero_u8());
            if (pixel_bytes == 1)
            {
                const auto value = vector_op(
                    cv::vx_load(src1_row + pixel),
                    cv::vx_load(src2_row + pixel));
                cv::vx_store(
                    dst_row + pixel,
                    cv::v_select(
                        not_selected,
                        cv::vx_load(dst_row + pixel),
                        value));
            }
            else if (pixel_bytes == 2)
            {
                cv::v_uint8 a0, a1, b0, b1, d0, d1;
                cv::v_load_deinterleave(
                    src1_row + pixel * 2, a0, a1);
                cv::v_load_deinterleave(
                    src2_row + pixel * 2, b0, b1);
                cv::v_load_deinterleave(
                    dst_row + pixel * 2, d0, d1);
                d0 = cv::v_select(not_selected, d0, vector_op(a0, b0));
                d1 = cv::v_select(not_selected, d1, vector_op(a1, b1));
                cv::v_store_interleave(dst_row + pixel * 2, d0, d1);
            }
            else if (pixel_bytes == 3)
            {
                cv::v_uint8 a0, a1, a2, b0, b1, b2, d0, d1, d2;
                cv::v_load_deinterleave(
                    src1_row + pixel * 3, a0, a1, a2);
                cv::v_load_deinterleave(
                    src2_row + pixel * 3, b0, b1, b2);
                cv::v_load_deinterleave(
                    dst_row + pixel * 3, d0, d1, d2);
                d0 = cv::v_select(not_selected, d0, vector_op(a0, b0));
                d1 = cv::v_select(not_selected, d1, vector_op(a1, b1));
                d2 = cv::v_select(not_selected, d2, vector_op(a2, b2));
                cv::v_store_interleave(
                    dst_row + pixel * 3, d0, d1, d2);
            }
            else
            {
                cv::v_uint8 a0, a1, a2, a3;
                cv::v_uint8 b0, b1, b2, b3;
                cv::v_uint8 d0, d1, d2, d3;
                cv::v_load_deinterleave(
                    src1_row + pixel * 4, a0, a1, a2, a3);
                cv::v_load_deinterleave(
                    src2_row + pixel * 4, b0, b1, b2, b3);
                cv::v_load_deinterleave(
                    dst_row + pixel * 4, d0, d1, d2, d3);
                d0 = cv::v_select(not_selected, d0, vector_op(a0, b0));
                d1 = cv::v_select(not_selected, d1, vector_op(a1, b1));
                d2 = cv::v_select(not_selected, d2, vector_op(a2, b2));
                d3 = cv::v_select(not_selected, d3, vector_op(a3, b3));
                cv::v_store_interleave(
                    dst_row + pixel * 4, d0, d1, d2, d3);
            }
        }
        for (; pixel < pixels; ++pixel)
        {
            if (mask_row[pixel] != 0)
            {
                const size_t offset = pixel * pixel_bytes;
                for (size_t byte = 0; byte < pixel_bytes; ++byte)
                {
                    dst_row[offset + byte] = scalar_op(
                        src1_row[offset + byte],
                        src2_row[offset + byte]);
                }
            }
        }
    }
    return true;
}

template<typename VectorOp, typename ScalarOp>
inline bool apply_masked_unary_byte_rows(const uchar* src,
                                         size_t src_step,
                                         uchar* dst,
                                         size_t dst_step,
                                         const uchar* mask,
                                         size_t mask_step,
                                         size_t pixels,
                                         size_t pixel_bytes,
                                         size_t rows,
                                         VectorOp vector_op,
                                         ScalarOp scalar_op)
{
    const size_t lanes =
        static_cast<size_t>(cv::VTraits<cv::v_uint8>::vlanes());
    if (pixels < lanes || pixel_bytes == 0 || pixel_bytes > 4)
    {
        return false;
    }

    for (size_t row = 0; row < rows; ++row)
    {
        const uchar* src_row = src + row * src_step;
        uchar* dst_row = dst + row * dst_step;
        const uchar* mask_row = mask + row * mask_step;

        size_t pixel = 0;
        for (; pixel + lanes <= pixels; pixel += lanes)
        {
            const auto not_selected =
                cv::v_eq(cv::vx_load(mask_row + pixel), cv::vx_setzero_u8());
            if (pixel_bytes == 1)
            {
                cv::vx_store(
                    dst_row + pixel,
                    cv::v_select(
                        not_selected,
                        cv::vx_load(dst_row + pixel),
                        vector_op(cv::vx_load(src_row + pixel))));
            }
            else if (pixel_bytes == 2)
            {
                cv::v_uint8 s0, s1, d0, d1;
                cv::v_load_deinterleave(
                    src_row + pixel * 2, s0, s1);
                cv::v_load_deinterleave(
                    dst_row + pixel * 2, d0, d1);
                d0 = cv::v_select(not_selected, d0, vector_op(s0));
                d1 = cv::v_select(not_selected, d1, vector_op(s1));
                cv::v_store_interleave(dst_row + pixel * 2, d0, d1);
            }
            else if (pixel_bytes == 3)
            {
                cv::v_uint8 s0, s1, s2, d0, d1, d2;
                cv::v_load_deinterleave(
                    src_row + pixel * 3, s0, s1, s2);
                cv::v_load_deinterleave(
                    dst_row + pixel * 3, d0, d1, d2);
                d0 = cv::v_select(not_selected, d0, vector_op(s0));
                d1 = cv::v_select(not_selected, d1, vector_op(s1));
                d2 = cv::v_select(not_selected, d2, vector_op(s2));
                cv::v_store_interleave(
                    dst_row + pixel * 3, d0, d1, d2);
            }
            else
            {
                cv::v_uint8 s0, s1, s2, s3;
                cv::v_uint8 d0, d1, d2, d3;
                cv::v_load_deinterleave(
                    src_row + pixel * 4, s0, s1, s2, s3);
                cv::v_load_deinterleave(
                    dst_row + pixel * 4, d0, d1, d2, d3);
                d0 = cv::v_select(not_selected, d0, vector_op(s0));
                d1 = cv::v_select(not_selected, d1, vector_op(s1));
                d2 = cv::v_select(not_selected, d2, vector_op(s2));
                d3 = cv::v_select(not_selected, d3, vector_op(s3));
                cv::v_store_interleave(
                    dst_row + pixel * 4, d0, d1, d2, d3);
            }
        }
        for (; pixel < pixels; ++pixel)
        {
            if (mask_row[pixel] != 0)
            {
                const size_t offset = pixel * pixel_bytes;
                for (size_t byte = 0; byte < pixel_bytes; ++byte)
                {
                    dst_row[offset + byte] =
                        scalar_op(src_row[offset + byte]);
                }
            }
        }
    }
    return true;
}

template<typename VectorOp, typename ScalarOp>
inline bool apply_binary_scalar_byte_rows(const uchar* src,
                                          size_t src_step,
                                          const uchar* scalar_pixel,
                                          size_t pixel_bytes,
                                          uchar* dst,
                                          size_t dst_step,
                                          const uchar* mask,
                                          size_t mask_step,
                                          size_t pixels,
                                          size_t rows,
                                          bool scalar_first,
                                          VectorOp vector_op,
                                          ScalarOp scalar_op)
{
    const size_t row_bytes = pixels * pixel_bytes;
    const size_t lanes =
        static_cast<size_t>(cv::VTraits<cv::v_uint8>::vlanes());
    if ((mask == nullptr && row_bytes < lanes) ||
        (mask != nullptr &&
         (pixels < lanes || pixel_bytes == 0 || pixel_bytes > 4)))
    {
        return false;
    }

    if (mask == nullptr)
    {
        std::vector<uchar> scalar_row(row_bytes);
        for (size_t x = 0; x < row_bytes; ++x)
        {
            scalar_row[x] = scalar_pixel[x % pixel_bytes];
        }

        for (size_t row = 0; row < rows; ++row)
        {
            const uchar* src_row = src + row * src_step;
            uchar* dst_row = dst + row * dst_step;
            size_t x = 0;
            for (; x + lanes <= row_bytes; x += lanes)
            {
                const auto src_vector = cv::vx_load(src_row + x);
                const auto scalar_vector =
                    cv::vx_load(scalar_row.data() + x);
                cv::vx_store(
                    dst_row + x,
                    scalar_first
                        ? vector_op(scalar_vector, src_vector)
                        : vector_op(src_vector, scalar_vector));
            }
            for (; x < row_bytes; ++x)
            {
                dst_row[x] =
                    scalar_first
                        ? scalar_op(scalar_row[x], src_row[x])
                        : scalar_op(src_row[x], scalar_row[x]);
            }
        }
        return true;
    }

    for (size_t row = 0; row < rows; ++row)
    {
        const uchar* src_row = src + row * src_step;
        uchar* dst_row = dst + row * dst_step;
        const uchar* mask_row = mask + row * mask_step;

        auto apply_plane = [&](const cv::v_uint8& src_vector,
                               uchar scalar_byte) {
            const auto scalar_vector = cv::vx_setall_u8(scalar_byte);
            return scalar_first
                       ? vector_op(scalar_vector, src_vector)
                       : vector_op(src_vector, scalar_vector);
        };

        size_t pixel = 0;
        for (; pixel + lanes <= pixels; pixel += lanes)
        {
            const auto not_selected =
                cv::v_eq(cv::vx_load(mask_row + pixel), cv::vx_setzero_u8());
            if (pixel_bytes == 1)
            {
                const auto value =
                    apply_plane(cv::vx_load(src_row + pixel), scalar_pixel[0]);
                cv::vx_store(
                    dst_row + pixel,
                    cv::v_select(
                        not_selected,
                        cv::vx_load(dst_row + pixel),
                        value));
            }
            else if (pixel_bytes == 2)
            {
                cv::v_uint8 s0, s1, d0, d1;
                cv::v_load_deinterleave(
                    src_row + pixel * 2, s0, s1);
                cv::v_load_deinterleave(
                    dst_row + pixel * 2, d0, d1);
                d0 = cv::v_select(
                    not_selected, d0, apply_plane(s0, scalar_pixel[0]));
                d1 = cv::v_select(
                    not_selected, d1, apply_plane(s1, scalar_pixel[1]));
                cv::v_store_interleave(dst_row + pixel * 2, d0, d1);
            }
            else if (pixel_bytes == 3)
            {
                cv::v_uint8 s0, s1, s2, d0, d1, d2;
                cv::v_load_deinterleave(
                    src_row + pixel * 3, s0, s1, s2);
                cv::v_load_deinterleave(
                    dst_row + pixel * 3, d0, d1, d2);
                d0 = cv::v_select(
                    not_selected, d0, apply_plane(s0, scalar_pixel[0]));
                d1 = cv::v_select(
                    not_selected, d1, apply_plane(s1, scalar_pixel[1]));
                d2 = cv::v_select(
                    not_selected, d2, apply_plane(s2, scalar_pixel[2]));
                cv::v_store_interleave(
                    dst_row + pixel * 3, d0, d1, d2);
            }
            else
            {
                cv::v_uint8 s0, s1, s2, s3;
                cv::v_uint8 d0, d1, d2, d3;
                cv::v_load_deinterleave(
                    src_row + pixel * 4, s0, s1, s2, s3);
                cv::v_load_deinterleave(
                    dst_row + pixel * 4, d0, d1, d2, d3);
                d0 = cv::v_select(
                    not_selected, d0, apply_plane(s0, scalar_pixel[0]));
                d1 = cv::v_select(
                    not_selected, d1, apply_plane(s1, scalar_pixel[1]));
                d2 = cv::v_select(
                    not_selected, d2, apply_plane(s2, scalar_pixel[2]));
                d3 = cv::v_select(
                    not_selected, d3, apply_plane(s3, scalar_pixel[3]));
                cv::v_store_interleave(
                    dst_row + pixel * 4, d0, d1, d2, d3);
            }
        }
        for (; pixel < pixels; ++pixel)
        {
            if (mask_row[pixel] == 0)
            {
                continue;
            }
            const size_t offset = pixel * pixel_bytes;
            for (size_t byte = 0; byte < pixel_bytes; ++byte)
            {
                dst_row[offset + byte] =
                    scalar_first
                        ? scalar_op(scalar_pixel[byte], src_row[offset + byte])
                        : scalar_op(src_row[offset + byte], scalar_pixel[byte]);
            }
        }
    }
    return true;
}

template<typename T>
inline size_t inrange_block_width()
{
    using Vec = decltype(cv::vx_load(static_cast<const T*>(nullptr)));
    if constexpr (sizeof(T) == 1)
    {
        return static_cast<size_t>(cv::VTraits<Vec>::vlanes());
    }
    else if constexpr (sizeof(T) == 2 || sizeof(T) == 4)
    {
        return static_cast<size_t>(cv::VTraits<Vec>::vlanes()) * 2;
    }
    return 0;
}

template<typename T>
inline void store_inrange_block(const T* src,
                                const T* lower,
                                const T* upper,
                                uchar* dst)
{
    using Vec = decltype(cv::vx_load(src));
    const size_t lanes = static_cast<size_t>(cv::VTraits<Vec>::vlanes());

    if constexpr (std::is_same<T, uchar>::value)
    {
        const auto values = cv::vx_load(src);
        const auto low = cv::vx_load(lower);
        const auto high = cv::vx_load(upper);
        cv::vx_store(dst, cv::v_and(cv::v_ge(values, low), cv::v_ge(high, values)));
    }
    else if constexpr (std::is_same<T, schar>::value)
    {
        const auto values = cv::vx_load(src);
        const auto low = cv::vx_load(lower);
        const auto high = cv::vx_load(upper);
        cv::vx_store(
            reinterpret_cast<schar*>(dst),
            cv::v_and(cv::v_ge(values, low), cv::v_ge(high, values)));
    }
    else if constexpr (std::is_same<T, ushort>::value)
    {
        const auto values0 = cv::vx_load(src);
        const auto low0 = cv::vx_load(lower);
        const auto high0 = cv::vx_load(upper);
        const auto values1 = cv::vx_load(src + lanes);
        const auto low1 = cv::vx_load(lower + lanes);
        const auto high1 = cv::vx_load(upper + lanes);
        cv::vx_store(
            dst,
            cv::v_pack(
                cv::v_and(cv::v_ge(values0, low0), cv::v_ge(high0, values0)),
                cv::v_and(cv::v_ge(values1, low1), cv::v_ge(high1, values1))));
    }
    else if constexpr (std::is_same<T, short>::value)
    {
        const auto values0 = cv::vx_load(src);
        const auto low0 = cv::vx_load(lower);
        const auto high0 = cv::vx_load(upper);
        const auto values1 = cv::vx_load(src + lanes);
        const auto low1 = cv::vx_load(lower + lanes);
        const auto high1 = cv::vx_load(upper + lanes);
        cv::vx_store(
            reinterpret_cast<schar*>(dst),
            cv::v_pack(
                cv::v_and(cv::v_ge(values0, low0), cv::v_ge(high0, values0)),
                cv::v_and(cv::v_ge(values1, low1), cv::v_ge(high1, values1))));
    }
    else if constexpr (std::is_same<T, int>::value)
    {
        const auto values0 = cv::vx_load(src);
        const auto low0 = cv::vx_load(lower);
        const auto high0 = cv::vx_load(upper);
        const auto values1 = cv::vx_load(src + lanes);
        const auto low1 = cv::vx_load(lower + lanes);
        const auto high1 = cv::vx_load(upper + lanes);
        cv::v_pack_store(
            dst,
            cv::v_reinterpret_as_u16(
                cv::v_pack(
                    cv::v_and(cv::v_ge(values0, low0), cv::v_ge(high0, values0)),
                    cv::v_and(cv::v_ge(values1, low1), cv::v_ge(high1, values1)))));
    }
    else if constexpr (std::is_same<T, uint>::value)
    {
        const auto values0 = cv::vx_load(src);
        const auto low0 = cv::vx_load(lower);
        const auto high0 = cv::vx_load(upper);
        const auto values1 = cv::vx_load(src + lanes);
        const auto low1 = cv::vx_load(lower + lanes);
        const auto high1 = cv::vx_load(upper + lanes);
        cv::v_pack_store(
            dst,
            cv::v_pack(
                cv::v_and(cv::v_ge(values0, low0), cv::v_ge(high0, values0)),
                cv::v_and(cv::v_ge(values1, low1), cv::v_ge(high1, values1))));
    }
    else if constexpr (std::is_same<T, float>::value)
    {
        const auto values0 = cv::vx_load(src);
        const auto low0 = cv::vx_load(lower);
        const auto high0 = cv::vx_load(upper);
        const auto values1 = cv::vx_load(src + lanes);
        const auto low1 = cv::vx_load(lower + lanes);
        const auto high1 = cv::vx_load(upper + lanes);
        cv::v_pack_store(
            dst,
            cv::v_pack(
                cv::v_and(
                    cv::v_reinterpret_as_u32(cv::v_ge(values0, low0)),
                    cv::v_reinterpret_as_u32(cv::v_ge(high0, values0))),
                cv::v_and(
                    cv::v_reinterpret_as_u32(cv::v_ge(values1, low1)),
                    cv::v_reinterpret_as_u32(cv::v_ge(high1, values1)))));
    }
}

inline void reduce_inrange_channels(const uchar* element_mask,
                                    uchar* pixel_mask,
                                    size_t pixels,
                                    int channels)
{
    for (size_t pixel = 0; pixel < pixels; ++pixel)
    {
        const size_t offset = pixel * static_cast<size_t>(channels);
        uchar combined = element_mask[offset];
        for (int channel = 1; channel < channels; ++channel)
        {
            combined = static_cast<uchar>(
                combined & element_mask[offset + static_cast<size_t>(channel)]);
        }
        pixel_mask[pixel] = combined;
    }
}

template<typename T>
inline bool apply_inrange_rows(const T* src,
                               size_t src_step,
                               const T* lower,
                               size_t lower_step,
                               const T* upper,
                               size_t upper_step,
                               uchar* dst,
                               size_t dst_step,
                               size_t pixels,
                               int channels,
                               size_t rows)
{
    const size_t block_width = inrange_block_width<T>();
    const size_t row_scalars = pixels * static_cast<size_t>(channels);
    if (block_width == 0 || row_scalars < block_width)
    {
        return false;
    }

    std::vector<uchar> element_mask(
        channels == 1 ? size_t(0) : row_scalars);

    for (size_t row = 0; row < rows; ++row)
    {
        const T* src_row = reinterpret_cast<const T*>(
            reinterpret_cast<const uchar*>(src) + row * src_step);
        const T* lower_row = reinterpret_cast<const T*>(
            reinterpret_cast<const uchar*>(lower) + row * lower_step);
        const T* upper_row = reinterpret_cast<const T*>(
            reinterpret_cast<const uchar*>(upper) + row * upper_step);
        uchar* dst_row = dst + row * dst_step;
        uchar* raw_mask = channels == 1 ? dst_row : element_mask.data();

        size_t x = 0;
        for (; x + block_width <= row_scalars; x += block_width)
        {
            store_inrange_block(
                src_row + x, lower_row + x, upper_row + x, raw_mask + x);
        }
        for (; x < row_scalars; ++x)
        {
            raw_mask[x] =
                lower_row[x] <= src_row[x] && src_row[x] <= upper_row[x]
                    ? static_cast<uchar>(255)
                    : static_cast<uchar>(0);
        }

        if (channels != 1)
        {
            reduce_inrange_channels(raw_mask, dst_row, pixels, channels);
        }
    }
    return true;
}

template<typename T>
inline bool apply_inrange_scalar_rows(const T* src,
                                      size_t src_step,
                                      const T* lower,
                                      const T* upper,
                                      uchar* dst,
                                      size_t dst_step,
                                      size_t pixels,
                                      int channels,
                                      size_t rows)
{
    const size_t block_width = inrange_block_width<T>();
    const size_t row_scalars = pixels * static_cast<size_t>(channels);
    if (block_width == 0 || row_scalars < block_width)
    {
        return false;
    }

    std::vector<T> lower_patterns(
        static_cast<size_t>(channels) * block_width);
    std::vector<T> upper_patterns(
        static_cast<size_t>(channels) * block_width);
    for (int phase = 0; phase < channels; ++phase)
    {
        for (size_t lane = 0; lane < block_width; ++lane)
        {
            const int channel =
                (phase + static_cast<int>(lane % static_cast<size_t>(channels))) %
                channels;
            lower_patterns[static_cast<size_t>(phase) * block_width + lane] =
                lower[channel];
            upper_patterns[static_cast<size_t>(phase) * block_width + lane] =
                upper[channel];
        }
    }

    std::vector<uchar> element_mask(
        channels == 1 ? size_t(0) : row_scalars);

    for (size_t row = 0; row < rows; ++row)
    {
        const T* src_row = reinterpret_cast<const T*>(
            reinterpret_cast<const uchar*>(src) + row * src_step);
        uchar* dst_row = dst + row * dst_step;
        uchar* raw_mask = channels == 1 ? dst_row : element_mask.data();

        size_t x = 0;
        for (; x + block_width <= row_scalars; x += block_width)
        {
            const size_t phase = x % static_cast<size_t>(channels);
            store_inrange_block(
                src_row + x,
                lower_patterns.data() + phase * block_width,
                upper_patterns.data() + phase * block_width,
                raw_mask + x);
        }
        for (; x < row_scalars; ++x)
        {
            const size_t channel = x % static_cast<size_t>(channels);
            raw_mask[x] =
                lower[channel] <= src_row[x] && src_row[x] <= upper[channel]
                    ? static_cast<uchar>(255)
                    : static_cast<uchar>(0);
        }

        if (channels != 1)
        {
            reduce_inrange_channels(raw_mask, dst_row, pixels, channels);
        }
    }
    return true;
}

#endif

}  // namespace arithm_ui
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_ARITHM_UI_HPP
