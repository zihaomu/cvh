#ifndef CVH_CORE_DETAIL_GEMM_AVX2_HPP
#define CVH_CORE_DETAIL_GEMM_AVX2_HPP

#include "cvh/core/detail/native_intrinsics.hpp"

#include <cstddef>

namespace cvh {
namespace detail {
namespace gemm_avx2 {

#if CVH_NATIVE_AVX2_COMPILED

template <int Rows>
CVH_TARGET_AVX2_FMA inline void kernel_nn_block_f32(
    const float* a,
    const float* b,
    float* c,
    int n,
    int k)
{
    constexpr int nr = 16;
    int col = 0;
    for (; col <= n - nr; col += nr)
    {
        __m256 sums[Rows][2];
        for (int row = 0; row < Rows; ++row)
        {
            sums[row][0] = _mm256_setzero_ps();
            sums[row][1] = _mm256_setzero_ps();
        }

        for (int inner = 0; inner < k; ++inner)
        {
            const float* b_row =
                b + static_cast<std::size_t>(inner) *
                        static_cast<std::size_t>(n) +
                static_cast<std::size_t>(col);
            const __m256 b0 = _mm256_loadu_ps(b_row);
            const __m256 b1 = _mm256_loadu_ps(b_row + 8);
            for (int row = 0; row < Rows; ++row)
            {
                const __m256 av = _mm256_broadcast_ss(
                    a + static_cast<std::size_t>(row) *
                            static_cast<std::size_t>(k) +
                    static_cast<std::size_t>(inner));
                sums[row][0] =
                    _mm256_fmadd_ps(av, b0, sums[row][0]);
                sums[row][1] =
                    _mm256_fmadd_ps(av, b1, sums[row][1]);
            }
        }

        for (int row = 0; row < Rows; ++row)
        {
            float* c_row =
                c + static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(n) +
                static_cast<std::size_t>(col);
            _mm256_storeu_ps(c_row, sums[row][0]);
            _mm256_storeu_ps(c_row + 8, sums[row][1]);
        }
    }

    for (; col <= n - 8; col += 8)
    {
        __m256 sums[Rows];
        for (int row = 0; row < Rows; ++row)
        {
            sums[row] = _mm256_setzero_ps();
        }
        for (int inner = 0; inner < k; ++inner)
        {
            const __m256 bv = _mm256_loadu_ps(
                b + static_cast<std::size_t>(inner) *
                        static_cast<std::size_t>(n) +
                static_cast<std::size_t>(col));
            for (int row = 0; row < Rows; ++row)
            {
                const __m256 av = _mm256_broadcast_ss(
                    a + static_cast<std::size_t>(row) *
                            static_cast<std::size_t>(k) +
                    static_cast<std::size_t>(inner));
                sums[row] = _mm256_fmadd_ps(av, bv, sums[row]);
            }
        }
        for (int row = 0; row < Rows; ++row)
        {
            _mm256_storeu_ps(
                c + static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(n) +
                    static_cast<std::size_t>(col),
                sums[row]);
        }
    }

    for (; col < n; ++col)
    {
        float sums[Rows] = {};
        for (int inner = 0; inner < k; ++inner)
        {
            const float bv =
                b[static_cast<std::size_t>(inner) *
                      static_cast<std::size_t>(n) +
                  static_cast<std::size_t>(col)];
            for (int row = 0; row < Rows; ++row)
            {
                sums[row] +=
                    a[static_cast<std::size_t>(row) *
                          static_cast<std::size_t>(k) +
                      static_cast<std::size_t>(inner)] *
                    bv;
            }
        }
        for (int row = 0; row < Rows; ++row)
        {
            c[static_cast<std::size_t>(row) *
                  static_cast<std::size_t>(n) +
              static_cast<std::size_t>(col)] = sums[row];
        }
    }
}

CVH_TARGET_AVX2_FMA inline bool kernel_nn_f32(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k)
{
    if (m <= 0 || n < 16 || k <= 0)
    {
        return false;
    }

    int row = 0;
    for (; row <= m - 6; row += 6)
    {
        kernel_nn_block_f32<6>(
            a + static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(k),
            b,
            c + static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(n),
            n,
            k);
    }
    if (row <= m - 4)
    {
        kernel_nn_block_f32<4>(
            a + static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(k),
            b,
            c + static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(n),
            n,
            k);
        row += 4;
    }
    switch (m - row)
    {
        case 3:
            kernel_nn_block_f32<3>(
                a + static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(k),
                b,
                c + static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(n),
                n,
                k);
            break;
        case 2:
            kernel_nn_block_f32<2>(
                a + static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(k),
                b,
                c + static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(n),
                n,
                k);
            break;
        case 1:
            kernel_nn_block_f32<1>(
                a + static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(k),
                b,
                c + static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(n),
                n,
                k);
            break;
        default:
            break;
    }
    _mm256_zeroupper();
    return true;
}

CVH_TARGET_AVX2_FMA inline bool kernel_nt_4x2_f32(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k)
{
    if (m <= 0 || n <= 0 || k < 8)
    {
        return false;
    }
    for (int row = 0; row < m; row += 4)
    {
        const int rows = (m - row) < 4 ? (m - row) : 4;
        for (int col = 0; col < n; col += 2)
        {
            const int cols = (n - col) < 2 ? (n - col) : 2;
            __m256 sums[4][2];
            for (int ri = 0; ri < rows; ++ri)
            {
                for (int ci = 0; ci < cols; ++ci)
                {
                    sums[ri][ci] = _mm256_setzero_ps();
                }
            }
            int inner = 0;
            for (; inner <= k - 8; inner += 8)
            {
                __m256 av[4];
                __m256 bv[2];
                for (int ri = 0; ri < rows; ++ri)
                {
                    av[ri] = _mm256_loadu_ps(
                        a + static_cast<std::size_t>(row + ri) *
                                static_cast<std::size_t>(k) +
                        static_cast<std::size_t>(inner));
                }
                for (int ci = 0; ci < cols; ++ci)
                {
                    bv[ci] = _mm256_loadu_ps(
                        b + static_cast<std::size_t>(col + ci) *
                                static_cast<std::size_t>(k) +
                        static_cast<std::size_t>(inner));
                }
                for (int ri = 0; ri < rows; ++ri)
                {
                    for (int ci = 0; ci < cols; ++ci)
                    {
                        sums[ri][ci] = _mm256_fmadd_ps(
                            av[ri], bv[ci], sums[ri][ci]);
                    }
                }
            }
            for (int ri = 0; ri < rows; ++ri)
            {
                const float* a_row =
                    a + static_cast<std::size_t>(row + ri) *
                            static_cast<std::size_t>(k);
                float* c_row =
                    c + static_cast<std::size_t>(row + ri) *
                            static_cast<std::size_t>(n);
                for (int ci = 0; ci < cols; ++ci)
                {
                    const __m128 lo =
                        _mm256_castps256_ps128(sums[ri][ci]);
                    const __m128 hi =
                        _mm256_extractf128_ps(sums[ri][ci], 1);
                    __m128 reduced = _mm_add_ps(lo, hi);
                    reduced = _mm_hadd_ps(reduced, reduced);
                    reduced = _mm_hadd_ps(reduced, reduced);
                    float value = _mm_cvtss_f32(reduced);
                    const float* b_row =
                        b + static_cast<std::size_t>(col + ci) *
                                static_cast<std::size_t>(k);
                    for (int tail = inner; tail < k; ++tail)
                    {
                        value += a_row[tail] * b_row[tail];
                    }
                    c_row[col + ci] = value;
                }
            }
        }
    }
    _mm256_zeroupper();
    return true;
}

#else

inline bool kernel_nn_f32(const float*,
                          const float*,
                          float*,
                          int,
                          int,
                          int)
{
    return false;
}

inline bool kernel_nt_4x2_f32(
    const float*, const float*, float*, int, int, int)
{
    return false;
}

#endif

}  // namespace gemm_avx2
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_GEMM_AVX2_HPP
