#ifndef CVH_CORE_DETAIL_GEMM_UI_HPP
#define CVH_CORE_DETAIL_GEMM_UI_HPP

#include "dispatch_control.h"
#include "../simd/opencv_ui.h"

#include <cstddef>

namespace cvh {
namespace detail {
namespace gemm_ui {

inline bool enabled()
{
#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return cpu::dispatch_mode() != cpu::DispatchMode::ScalarOnly;
#else
    return false;
#endif
}

inline bool can_vectorize_nn(int n)
{
#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return enabled() &&
           n >= cv::VTraits<cv::v_float32>::vlanes();
#else
    (void)n;
    return false;
#endif
}

inline bool can_vectorize_nt(int k)
{
#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return enabled() &&
           k >= cv::VTraits<cv::v_float32>::vlanes();
#else
    (void)k;
    return false;
#endif
}

inline void kernel_nn_row_f32(const float* a,
                              const float* b,
                              float* c,
                              int n,
                              int k)
{
#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const int lanes = cv::VTraits<cv::v_float32>::vlanes();
    int col = 0;
    for (; col <= n - 4 * lanes; col += 4 * lanes)
    {
        cv::v_float32 sum0 = cv::vx_setzero_f32();
        cv::v_float32 sum1 = cv::vx_setzero_f32();
        cv::v_float32 sum2 = cv::vx_setzero_f32();
        cv::v_float32 sum3 = cv::vx_setzero_f32();
        for (int inner = 0; inner < k; ++inner)
        {
            const cv::v_float32 av = cv::vx_setall_f32(a[inner]);
            const float* b_row =
                b + static_cast<size_t>(inner) * static_cast<size_t>(n) +
                static_cast<size_t>(col);
            sum0 = cv::v_fma(av, cv::vx_load(b_row), sum0);
            sum1 = cv::v_fma(av, cv::vx_load(b_row + lanes), sum1);
            sum2 = cv::v_fma(av, cv::vx_load(b_row + 2 * lanes), sum2);
            sum3 = cv::v_fma(av, cv::vx_load(b_row + 3 * lanes), sum3);
        }
        cv::vx_store(c + col, sum0);
        cv::vx_store(c + col + lanes, sum1);
        cv::vx_store(c + col + 2 * lanes, sum2);
        cv::vx_store(c + col + 3 * lanes, sum3);
    }
    for (; col <= n - lanes; col += lanes)
    {
        cv::v_float32 sum = cv::vx_setzero_f32();
        for (int inner = 0; inner < k; ++inner)
        {
            const cv::v_float32 av = cv::vx_setall_f32(a[inner]);
            const float* b_row =
                b + static_cast<size_t>(inner) * static_cast<size_t>(n) +
                static_cast<size_t>(col);
            sum = cv::v_fma(av, cv::vx_load(b_row), sum);
        }
        cv::vx_store(c + col, sum);
    }
    for (; col < n; ++col)
    {
        float sum = 0.0f;
        for (int inner = 0; inner < k; ++inner)
        {
            sum += a[inner] *
                   b[static_cast<size_t>(inner) * static_cast<size_t>(n) +
                     static_cast<size_t>(col)];
        }
        c[col] = sum;
    }
#else
    (void)a;
    (void)b;
    (void)c;
    (void)n;
    (void)k;
#endif
}

inline float kernel_nt_dot_f32(const float* a,
                               const float* b,
                               int k)
{
#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const int lanes = cv::VTraits<cv::v_float32>::vlanes();
    cv::v_float32 sum0 = cv::vx_setzero_f32();
    cv::v_float32 sum1 = cv::vx_setzero_f32();
    int inner = 0;
    for (; inner <= k - 2 * lanes; inner += 2 * lanes)
    {
        sum0 = cv::v_fma(
            cv::vx_load(a + inner),
            cv::vx_load(b + inner),
            sum0);
        sum1 = cv::v_fma(
            cv::vx_load(a + inner + lanes),
            cv::vx_load(b + inner + lanes),
            sum1);
    }
    for (; inner <= k - lanes; inner += lanes)
    {
        sum0 = cv::v_fma(
            cv::vx_load(a + inner),
            cv::vx_load(b + inner),
            sum0);
    }
    float sum = cv::v_reduce_sum(cv::v_add(sum0, sum1));
    for (; inner < k; ++inner)
    {
        sum += a[inner] * b[inner];
    }
    return sum;
#else
    (void)a;
    (void)b;
    (void)k;
    return 0.0f;
#endif
}

}  // namespace gemm_ui
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_GEMM_UI_HPP
