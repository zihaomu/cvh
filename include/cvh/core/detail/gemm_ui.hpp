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
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return cpu::opencv_ui_allowed();
#else
    return false;
#endif
}

inline bool can_vectorize_nn(int n)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
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
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
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
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
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

template <int Rows>
inline void kernel_nn_block_4x2_f32(const float* a,
                                    const float* b,
                                    float* c,
                                    int n,
                                    int k)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const int lanes = cv::VTraits<cv::v_float32>::vlanes();
    int col = 0;
    for (; col <= n - 2 * lanes; col += 2 * lanes)
    {
        cv::v_float32 sum0[Rows];
        cv::v_float32 sum1[Rows];
        for (int row = 0; row < Rows; ++row)
        {
            sum0[row] = cv::vx_setzero_f32();
            sum1[row] = cv::vx_setzero_f32();
        }

        for (int inner = 0; inner < k; ++inner)
        {
            const float* b_row =
                b + static_cast<size_t>(inner) *
                        static_cast<size_t>(n) +
                static_cast<size_t>(col);
            const cv::v_float32 bv0 = cv::vx_load(b_row);
            const cv::v_float32 bv1 = cv::vx_load(b_row + lanes);
            for (int row = 0; row < Rows; ++row)
            {
                const cv::v_float32 av = cv::vx_setall_f32(
                    a[static_cast<size_t>(row) *
                          static_cast<size_t>(k) +
                      static_cast<size_t>(inner)]);
                sum0[row] = cv::v_fma(av, bv0, sum0[row]);
                sum1[row] = cv::v_fma(av, bv1, sum1[row]);
            }
        }

        for (int row = 0; row < Rows; ++row)
        {
            float* c_row =
                c + static_cast<size_t>(row) *
                        static_cast<size_t>(n) +
                static_cast<size_t>(col);
            cv::vx_store(c_row, sum0[row]);
            cv::vx_store(c_row + lanes, sum1[row]);
        }
    }

    for (; col <= n - lanes; col += lanes)
    {
        cv::v_float32 sum[Rows];
        for (int row = 0; row < Rows; ++row)
        {
            sum[row] = cv::vx_setzero_f32();
        }

        for (int inner = 0; inner < k; ++inner)
        {
            const cv::v_float32 bv = cv::vx_load(
                b + static_cast<size_t>(inner) *
                        static_cast<size_t>(n) +
                static_cast<size_t>(col));
            for (int row = 0; row < Rows; ++row)
            {
                const cv::v_float32 av = cv::vx_setall_f32(
                    a[static_cast<size_t>(row) *
                          static_cast<size_t>(k) +
                      static_cast<size_t>(inner)]);
                sum[row] = cv::v_fma(av, bv, sum[row]);
            }
        }

        for (int row = 0; row < Rows; ++row)
        {
            cv::vx_store(
                c + static_cast<size_t>(row) *
                        static_cast<size_t>(n) +
                    static_cast<size_t>(col),
                sum[row]);
        }
    }

    for (; col < n; ++col)
    {
        float sum[Rows] = {};
        for (int inner = 0; inner < k; ++inner)
        {
            const float bv =
                b[static_cast<size_t>(inner) *
                      static_cast<size_t>(n) +
                  static_cast<size_t>(col)];
            for (int row = 0; row < Rows; ++row)
            {
                sum[row] +=
                    a[static_cast<size_t>(row) *
                          static_cast<size_t>(k) +
                      static_cast<size_t>(inner)] *
                    bv;
            }
        }
        for (int row = 0; row < Rows; ++row)
        {
            c[static_cast<size_t>(row) *
                  static_cast<size_t>(n) +
              static_cast<size_t>(col)] = sum[row];
        }
    }
#else
    (void)a;
    (void)b;
    (void)c;
    (void)n;
    (void)k;
#endif
}

inline void kernel_nn_4x2_f32(const float* a,
                              const float* b,
                              float* c,
                              int m,
                              int n,
                              int k)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    int row = 0;
    for (; row <= m - 4; row += 4)
    {
        kernel_nn_block_4x2_f32<4>(
            a + static_cast<size_t>(row) * static_cast<size_t>(k),
            b,
            c + static_cast<size_t>(row) * static_cast<size_t>(n),
            n,
            k);
    }
    switch (m - row)
    {
        case 3:
            kernel_nn_block_4x2_f32<3>(
                a + static_cast<size_t>(row) * static_cast<size_t>(k),
                b,
                c + static_cast<size_t>(row) * static_cast<size_t>(n),
                n,
                k);
            break;
        case 2:
            kernel_nn_block_4x2_f32<2>(
                a + static_cast<size_t>(row) * static_cast<size_t>(k),
                b,
                c + static_cast<size_t>(row) * static_cast<size_t>(n),
                n,
                k);
            break;
        case 1:
            kernel_nn_block_4x2_f32<1>(
                a + static_cast<size_t>(row) * static_cast<size_t>(k),
                b,
                c + static_cast<size_t>(row) * static_cast<size_t>(n),
                n,
                k);
            break;
        default:
            break;
    }
#else
    (void)a;
    (void)b;
    (void)c;
    (void)m;
    (void)n;
    (void)k;
#endif
}

template <int Rows>
inline void kernel_nn_n1_block_f32(const float* a,
                                   const float* b,
                                   float* c,
                                   int k)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const int lanes = cv::VTraits<cv::v_float32>::vlanes();
    cv::v_float32 sum0[Rows];
    cv::v_float32 sum1[Rows];
    for (int row = 0; row < Rows; ++row)
    {
        sum0[row] = cv::vx_setzero_f32();
        sum1[row] = cv::vx_setzero_f32();
    }

    int inner = 0;
    for (; inner <= k - 2 * lanes; inner += 2 * lanes)
    {
        const cv::v_float32 bv0 = cv::vx_load(b + inner);
        const cv::v_float32 bv1 = cv::vx_load(b + inner + lanes);
        for (int row = 0; row < Rows; ++row)
        {
            const float* a_row =
                a + static_cast<size_t>(row) * static_cast<size_t>(k);
            sum0[row] =
                cv::v_fma(cv::vx_load(a_row + inner), bv0, sum0[row]);
            sum1[row] = cv::v_fma(
                cv::vx_load(a_row + inner + lanes), bv1, sum1[row]);
        }
    }
    for (; inner <= k - lanes; inner += lanes)
    {
        const cv::v_float32 bv = cv::vx_load(b + inner);
        for (int row = 0; row < Rows; ++row)
        {
            const float* a_row =
                a + static_cast<size_t>(row) * static_cast<size_t>(k);
            sum0[row] =
                cv::v_fma(cv::vx_load(a_row + inner), bv, sum0[row]);
        }
    }

    for (int row = 0; row < Rows; ++row)
    {
        float value =
            cv::v_reduce_sum(cv::v_add(sum0[row], sum1[row]));
        const float* a_row =
            a + static_cast<size_t>(row) * static_cast<size_t>(k);
        for (int tail = inner; tail < k; ++tail)
        {
            value += a_row[tail] * b[tail];
        }
        c[row] = value;
    }
#else
    (void)a;
    (void)b;
    (void)c;
    (void)k;
#endif
}

inline void kernel_nn_n1_4rows_f32(const float* a,
                                   const float* b,
                                   float* c,
                                   int m,
                                   int k)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    int row = 0;
    for (; row <= m - 4; row += 4)
    {
        kernel_nn_n1_block_f32<4>(
            a + static_cast<size_t>(row) * static_cast<size_t>(k),
            b,
            c + row,
            k);
    }
    switch (m - row)
    {
        case 3:
            kernel_nn_n1_block_f32<3>(
                a + static_cast<size_t>(row) * static_cast<size_t>(k),
                b,
                c + row,
                k);
            break;
        case 2:
            kernel_nn_n1_block_f32<2>(
                a + static_cast<size_t>(row) * static_cast<size_t>(k),
                b,
                c + row,
                k);
            break;
        case 1:
            kernel_nn_n1_block_f32<1>(
                a + static_cast<size_t>(row) * static_cast<size_t>(k),
                b,
                c + row,
                k);
            break;
        default:
            break;
    }
#else
    (void)a;
    (void)b;
    (void)c;
    (void)m;
    (void)k;
#endif
}

inline float kernel_nt_dot_f32(const float* a,
                               const float* b,
                               int k)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
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
