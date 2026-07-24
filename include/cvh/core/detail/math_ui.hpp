#ifndef CVH_CORE_DETAIL_MATH_UI_HPP
#define CVH_CORE_DETAIL_MATH_UI_HPP

#include "dispatch_control.h"
#include "../simd/opencv_ui.h"

#include <cmath>
#include <cstddef>
#include <cstring>

namespace cvh {
namespace math_detail {
namespace ui {

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

inline uchar convert_scale_abs_scalar_f32(float value, float alpha, float beta)
{
    const float transformed = std::fabs(std::fma(value, alpha, beta));
    if (!(transformed > 0.0f))
    {
        return 0;
    }
    if (transformed >= 255.0f)
    {
        return 255;
    }
    return static_cast<uchar>(std::nearbyint(transformed));
}

inline bool apply_convert_scale_abs_f32(const float* src,
                                        size_t src_step,
                                        uchar* dst,
                                        size_t dst_step,
                                        size_t row_scalars,
                                        size_t rows,
                                        float alpha,
                                        float beta)
{
    const size_t lanes = static_cast<size_t>(cv::VTraits<cv::v_float32>::vlanes());
    const size_t block = lanes * 4;
    if (row_scalars < block)
    {
        return false;
    }

    const cv::v_float32 v_alpha = cv::vx_setall_f32(alpha);
    const cv::v_float32 v_beta = cv::vx_setall_f32(beta);
    const cv::v_float32 v_limit = cv::vx_setall_f32(255.0f);
    for (size_t row = 0; row < rows; ++row)
    {
        const float* src_row = reinterpret_cast<const float*>(
            reinterpret_cast<const uchar*>(src) + row * src_step);
        uchar* dst_row = dst + row * dst_step;

        size_t x = 0;
        for (; x + block <= row_scalars; x += block)
        {
            const cv::v_int32 i0 = cv::v_round(cv::v_min(
                cv::v_abs(cv::v_fma(cv::vx_load(src_row + x), v_alpha, v_beta)),
                v_limit));
            const cv::v_int32 i1 = cv::v_round(cv::v_min(
                cv::v_abs(cv::v_fma(
                    cv::vx_load(src_row + x + lanes), v_alpha, v_beta)),
                v_limit));
            const cv::v_int32 i2 = cv::v_round(cv::v_min(
                cv::v_abs(cv::v_fma(
                    cv::vx_load(src_row + x + lanes * 2), v_alpha, v_beta)),
                v_limit));
            const cv::v_int32 i3 = cv::v_round(cv::v_min(
                cv::v_abs(cv::v_fma(
                    cv::vx_load(src_row + x + lanes * 3), v_alpha, v_beta)),
                v_limit));
            cv::vx_store(
                dst_row + x,
                cv::v_pack(
                    cv::v_pack_u(i0, i1),
                    cv::v_pack_u(i2, i3)));
        }
        for (; x < row_scalars; ++x)
        {
            dst_row[x] = convert_scale_abs_scalar_f32(
                src_row[x], alpha, beta);
        }
    }
    return true;
}

inline bool apply_f32_to_fp16(const float* src,
                              size_t src_step,
                              short* dst,
                              size_t dst_step,
                              size_t row_scalars,
                              size_t rows)
{
    static_assert(sizeof(cv::hfloat) == sizeof(short), "half layout mismatch");
    const size_t lanes = static_cast<size_t>(cv::VTraits<cv::v_float32>::vlanes());
    const size_t block = lanes * 4;
    if (row_scalars < lanes)
    {
        return false;
    }

    for (size_t row = 0; row < rows; ++row)
    {
        const float* src_row = reinterpret_cast<const float*>(
            reinterpret_cast<const uchar*>(src) + row * src_step);
        short* dst_row = reinterpret_cast<short*>(
            reinterpret_cast<uchar*>(dst) + row * dst_step);

        size_t x = 0;
        for (; x + block <= row_scalars; x += block)
        {
            cv::v_pack_store(
                reinterpret_cast<cv::hfloat*>(dst_row + x),
                cv::vx_load(src_row + x));
            cv::v_pack_store(
                reinterpret_cast<cv::hfloat*>(dst_row + x + lanes),
                cv::vx_load(src_row + x + lanes));
            cv::v_pack_store(
                reinterpret_cast<cv::hfloat*>(dst_row + x + lanes * 2),
                cv::vx_load(src_row + x + lanes * 2));
            cv::v_pack_store(
                reinterpret_cast<cv::hfloat*>(dst_row + x + lanes * 3),
                cv::vx_load(src_row + x + lanes * 3));
        }
        for (; x + lanes <= row_scalars; x += lanes)
        {
            cv::v_pack_store(
                reinterpret_cast<cv::hfloat*>(dst_row + x),
                cv::vx_load(src_row + x));
        }
        for (; x < row_scalars; ++x)
        {
            hfloat half(src_row[x]);
            std::memcpy(dst_row + x, half.get_ptr(), sizeof(short));
        }
    }
    return true;
}

inline bool apply_fp16_to_f32(const short* src,
                              size_t src_step,
                              float* dst,
                              size_t dst_step,
                              size_t row_scalars,
                              size_t rows)
{
    static_assert(sizeof(cv::hfloat) == sizeof(short), "half layout mismatch");
    const size_t lanes = static_cast<size_t>(cv::VTraits<cv::v_float32>::vlanes());
    const size_t block = lanes * 4;
    if (row_scalars < lanes)
    {
        return false;
    }

    for (size_t row = 0; row < rows; ++row)
    {
        const short* src_row = reinterpret_cast<const short*>(
            reinterpret_cast<const uchar*>(src) + row * src_step);
        float* dst_row = reinterpret_cast<float*>(
            reinterpret_cast<uchar*>(dst) + row * dst_step);

        size_t x = 0;
        for (; x + block <= row_scalars; x += block)
        {
            cv::vx_store(
                dst_row + x,
                cv::vx_load_expand(
                    reinterpret_cast<const cv::hfloat*>(src_row + x)));
            cv::vx_store(
                dst_row + x + lanes,
                cv::vx_load_expand(
                    reinterpret_cast<const cv::hfloat*>(src_row + x + lanes)));
            cv::vx_store(
                dst_row + x + lanes * 2,
                cv::vx_load_expand(
                    reinterpret_cast<const cv::hfloat*>(
                        src_row + x + lanes * 2)));
            cv::vx_store(
                dst_row + x + lanes * 3,
                cv::vx_load_expand(
                    reinterpret_cast<const cv::hfloat*>(
                        src_row + x + lanes * 3)));
        }
        for (; x + lanes <= row_scalars; x += lanes)
        {
            cv::vx_store(
                dst_row + x,
                cv::vx_load_expand(
                    reinterpret_cast<const cv::hfloat*>(src_row + x)));
        }
        for (; x < row_scalars; ++x)
        {
            hfloat half;
            std::memcpy(half.get_ptr(), src_row + x, sizeof(short));
            dst_row[x] = static_cast<float>(half);
        }
    }
    return true;
}

#else

inline bool apply_convert_scale_abs_f32(const float*,
                                        size_t,
                                        uchar*,
                                        size_t,
                                        size_t,
                                        size_t,
                                        float,
                                        float)
{
    return false;
}

inline bool apply_f32_to_fp16(const float*,
                              size_t,
                              short*,
                              size_t,
                              size_t,
                              size_t)
{
    return false;
}

inline bool apply_fp16_to_f32(const short*,
                              size_t,
                              float*,
                              size_t,
                              size_t,
                              size_t)
{
    return false;
}

#endif

}  // namespace ui
}  // namespace math_detail
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_MATH_UI_HPP
