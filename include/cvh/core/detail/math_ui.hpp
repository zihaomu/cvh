#ifndef CVH_CORE_DETAIL_MATH_UI_HPP
#define CVH_CORE_DETAIL_MATH_UI_HPP

#include "dispatch_control.h"
#include "../simd/opencv_ui.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

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

inline bool apply_patch_nans_f32(float* src,
                                size_t src_step,
                                size_t row_scalars,
                                size_t rows,
                                float replacement)
{
    const size_t lanes = static_cast<size_t>(cv::VTraits<cv::v_int32>::vlanes());
    const size_t block = lanes * 2;
    if (row_scalars < lanes)
    {
        return false;
    }

    const cv::v_int32 positive_mask = cv::vx_setall_s32(0x7fffffff);
    const cv::v_int32 exponent_mask = cv::vx_setall_s32(0x7f800000);
    const cv::v_float32 replacement_vector = cv::vx_setall_f32(replacement);
    for (size_t row = 0; row < rows; ++row)
    {
        float* src_row = reinterpret_cast<float*>(
            reinterpret_cast<uchar*>(src) + row * src_step);
        size_t x = 0;
        for (; x + block <= row_scalars; x += block)
        {
            const cv::v_float32 value0 = cv::vx_load(src_row + x);
            const cv::v_float32 value1 = cv::vx_load(src_row + x + lanes);
            const cv::v_int32 bits0 = cv::v_reinterpret_as_s32(value0);
            const cv::v_int32 bits1 = cv::v_reinterpret_as_s32(value1);
            const cv::v_int32 nan0 =
                cv::v_lt(exponent_mask, cv::v_and(bits0, positive_mask));
            const cv::v_int32 nan1 =
                cv::v_lt(exponent_mask, cv::v_and(bits1, positive_mask));
            if (cv::v_check_any(cv::v_or(nan0, nan1)))
            {
                cv::vx_store(
                    src_row + x,
                    cv::v_select(
                        cv::v_reinterpret_as_f32(nan0),
                        replacement_vector,
                        value0));
                cv::vx_store(
                    src_row + x + lanes,
                    cv::v_select(
                        cv::v_reinterpret_as_f32(nan1),
                        replacement_vector,
                        value1));
            }
        }
        for (; x + lanes <= row_scalars; x += lanes)
        {
            const cv::v_float32 value = cv::vx_load(src_row + x);
            const cv::v_int32 bits = cv::v_reinterpret_as_s32(value);
            const cv::v_int32 nan =
                cv::v_lt(exponent_mask, cv::v_and(bits, positive_mask));
            if (cv::v_check_any(nan))
            {
                cv::vx_store(
                    src_row + x,
                    cv::v_select(
                        cv::v_reinterpret_as_f32(nan),
                        replacement_vector,
                        value));
            }
        }
        for (; x < row_scalars; ++x)
        {
            if (std::isnan(src_row[x]))
            {
                src_row[x] = replacement;
            }
        }
    }
    return true;
}

inline bool apply_exp_f32(const float* src,
                          size_t src_step,
                          float* dst,
                          size_t dst_step,
                          size_t row_scalars,
                          size_t rows)
{
    const size_t lanes = static_cast<size_t>(cv::VTraits<cv::v_float32>::vlanes());
    if (row_scalars < lanes)
    {
        return false;
    }

    const cv::v_float32 lower = cv::vx_setall_f32(-80.0f);
    const cv::v_float32 upper = cv::vx_setall_f32(80.0f);
    for (size_t row = 0; row < rows; ++row)
    {
        const float* src_row = reinterpret_cast<const float*>(
            reinterpret_cast<const uchar*>(src) + row * src_step);
        float* dst_row = reinterpret_cast<float*>(
            reinterpret_cast<uchar*>(dst) + row * dst_step);
        size_t x = 0;
        for (; x + lanes <= row_scalars; x += lanes)
        {
            const cv::v_float32 value = cv::vx_load(src_row + x);
            const cv::v_float32 in_range =
                cv::v_and(cv::v_ge(value, lower), cv::v_le(value, upper));
            if (cv::v_check_all(in_range))
            {
                cv::vx_store(dst_row + x, cv::v_exp(value));
            }
            else
            {
                for (size_t lane = 0; lane < lanes; ++lane)
                {
                    dst_row[x + lane] = std::exp(src_row[x + lane]);
                }
            }
        }
        for (; x < row_scalars; ++x)
        {
            dst_row[x] = std::exp(src_row[x]);
        }
    }
    return true;
}

inline bool apply_log_f32(const float* src,
                          size_t src_step,
                          float* dst,
                          size_t dst_step,
                          size_t row_scalars,
                          size_t rows)
{
    const size_t lanes = static_cast<size_t>(cv::VTraits<cv::v_float32>::vlanes());
    if (row_scalars < lanes)
    {
        return false;
    }

    const cv::v_float32 minimum =
        cv::vx_setall_f32(std::numeric_limits<float>::min());
    const cv::v_float32 maximum =
        cv::vx_setall_f32(std::numeric_limits<float>::max());
    for (size_t row = 0; row < rows; ++row)
    {
        const float* src_row = reinterpret_cast<const float*>(
            reinterpret_cast<const uchar*>(src) + row * src_step);
        float* dst_row = reinterpret_cast<float*>(
            reinterpret_cast<uchar*>(dst) + row * dst_step);
        size_t x = 0;
        for (; x + lanes <= row_scalars; x += lanes)
        {
            const cv::v_float32 value = cv::vx_load(src_row + x);
            const cv::v_float32 in_range =
                cv::v_and(cv::v_ge(value, minimum), cv::v_le(value, maximum));
            if (cv::v_check_all(in_range))
            {
                cv::vx_store(dst_row + x, cv::v_log(value));
            }
            else
            {
                for (size_t lane = 0; lane < lanes; ++lane)
                {
                    dst_row[x + lane] = std::log(src_row[x + lane]);
                }
            }
        }
        for (; x < row_scalars; ++x)
        {
            dst_row[x] = std::log(src_row[x]);
        }
    }
    return true;
}

inline cv::v_float32 pow_integer_f32(cv::v_float32 base, int exponent)
{
    std::uint32_t remaining =
        exponent < 0
            ? static_cast<std::uint32_t>(-static_cast<std::int64_t>(exponent))
            : static_cast<std::uint32_t>(exponent);
    if (exponent < 0)
    {
        base = cv::v_div(cv::vx_setall_f32(1.0f), base);
    }

    cv::v_float32 result = cv::vx_setall_f32(1.0f);
    while (remaining != 0)
    {
        if ((remaining & 1U) != 0)
        {
            result = cv::v_mul(result, base);
        }
        remaining >>= 1U;
        if (remaining != 0)
        {
            base = cv::v_mul(base, base);
        }
    }
    return result;
}

inline bool apply_pow_f32(const float* src,
                          size_t src_step,
                          float* dst,
                          size_t dst_step,
                          size_t row_scalars,
                          size_t rows,
                          double power,
                          bool is_integer_power,
                          int integer_power)
{
    const size_t lanes = static_cast<size_t>(cv::VTraits<cv::v_float32>::vlanes());
    if (row_scalars < lanes)
    {
        return false;
    }

    const cv::v_float32 minimum =
        cv::vx_setall_f32(std::numeric_limits<float>::min());
    const cv::v_float32 maximum =
        cv::vx_setall_f32(std::numeric_limits<float>::max());
    const cv::v_float32 vector_power =
        cv::vx_setall_f32(static_cast<float>(power));
    for (size_t row = 0; row < rows; ++row)
    {
        const float* src_row = reinterpret_cast<const float*>(
            reinterpret_cast<const uchar*>(src) + row * src_step);
        float* dst_row = reinterpret_cast<float*>(
            reinterpret_cast<uchar*>(dst) + row * dst_step);
        size_t x = 0;
        for (; x + lanes <= row_scalars; x += lanes)
        {
            const cv::v_float32 value = cv::vx_load(src_row + x);
            if (is_integer_power)
            {
                cv::vx_store(
                    dst_row + x,
                    pow_integer_f32(value, integer_power));
                continue;
            }

            const cv::v_float32 is_positive_finite =
                cv::v_and(cv::v_ge(value, minimum), cv::v_le(value, maximum));
            if (cv::v_check_all(is_positive_finite))
            {
                cv::vx_store(
                    dst_row + x,
                    cv::v_exp(cv::v_mul(cv::v_log(value), vector_power)));
            }
            else
            {
                for (size_t lane = 0; lane < lanes; ++lane)
                {
                    dst_row[x + lane] = static_cast<float>(
                        std::pow(src_row[x + lane], power));
                }
            }
        }
        for (; x < row_scalars; ++x)
        {
            dst_row[x] = static_cast<float>(std::pow(src_row[x], power));
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

inline bool apply_patch_nans_f32(float*, size_t, size_t, size_t, float)
{
    return false;
}

inline bool apply_exp_f32(
    const float*, size_t, float*, size_t, size_t, size_t)
{
    return false;
}

inline bool apply_log_f32(
    const float*, size_t, float*, size_t, size_t, size_t)
{
    return false;
}

inline bool apply_pow_f32(
    const float*, size_t, float*, size_t, size_t, size_t, double, bool, int)
{
    return false;
}

#endif

}  // namespace ui
}  // namespace math_detail
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_MATH_UI_HPP
