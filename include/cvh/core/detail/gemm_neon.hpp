#ifndef CVH_CORE_DETAIL_GEMM_NEON_HPP
#define CVH_CORE_DETAIL_GEMM_NEON_HPP

#include "cvh/core/detail/native_intrinsics.hpp"

#include <cstddef>

namespace cvh {
namespace detail {
namespace gemm_neon {

#if defined(__GNUC__) || defined(__clang__)
#define CVH_GEMM_NEON_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define CVH_GEMM_NEON_ALWAYS_INLINE inline
#endif

struct F32DirectA6x16Traits
{
    static constexpr int mr = 6;
    static constexpr int nr = 16;
    static constexpr int secondary_nr = 8;
    static constexpr int kr = 1;
    static constexpr int sr = 1;
    static constexpr int k_unroll = 4;
};

#if CVH_NATIVE_NEON_COMPILED

template <int Rows, int VectorCols, int Lane>
inline void fma_packed_b_lane_f32(
    float32x4_t (&sums)[Rows][VectorCols],
    const float32x4_t (&a_values)[Rows],
    const float* packed_b)
{
    for (int vector_col = 0; vector_col < VectorCols; ++vector_col)
    {
        const float32x4_t b_value =
            vld1q_f32(packed_b + vector_col * 4);
        for (int row = 0; row < Rows; ++row)
        {
            sums[row][vector_col] = vfmaq_laneq_f32(
                sums[row][vector_col],
                b_value,
                a_values[row],
                Lane);
        }
    }
}

template <int VectorCols, bool Accumulate>
inline void store_row_f32(const float32x4_t (&sums)[VectorCols],
                          float* c,
                          int valid_cols)
{
    const int width = VectorCols * 4;
    if (valid_cols == width)
    {
        for (int vector_col = 0; vector_col < VectorCols; ++vector_col)
        {
            vst1q_f32(c + vector_col * 4, sums[vector_col]);
        }
        return;
    }

    int col = 0;
    int vector_col = 0;
    for (; col + 4 <= valid_cols; col += 4, ++vector_col)
    {
        float32x4_t value = sums[vector_col];
        if (Accumulate)
        {
            value = vaddq_f32(value, vld1q_f32(c + col));
        }
        vst1q_f32(c + col, value);
    }
    if (col == valid_cols)
    {
        return;
    }

    const float32x4_t value = sums[vector_col];
    if (valid_cols - col >= 2)
    {
        c[col] = vgetq_lane_f32(value, 0) +
                 (Accumulate ? c[col] : 0.0f);
        c[col + 1] = vgetq_lane_f32(value, 1) +
                     (Accumulate ? c[col + 1] : 0.0f);
        col += 2;
    }
    if (col < valid_cols)
    {
        const int lane = col & 3;
        float lane_value = 0.0f;
        switch (lane)
        {
            case 1:
                lane_value = vgetq_lane_f32(value, 1);
                break;
            case 2:
                lane_value = vgetq_lane_f32(value, 2);
                break;
            case 3:
                lane_value = vgetq_lane_f32(value, 3);
                break;
            case 0:
            default:
                lane_value = vgetq_lane_f32(value, 0);
                break;
        }
        c[col] = lane_value + (Accumulate ? c[col] : 0.0f);
    }
}

template <int Rows,
          int VectorCols,
          bool Accumulate,
          bool FullWidth>
CVH_GEMM_NEON_ALWAYS_INLINE void kernel_f32_directa_packedb_u4(
    const float* a,
    std::size_t a_stride,
    const float* packed_b,
    int packed_b_stride,
    float* c,
    std::size_t c_stride,
    int k,
    int valid_cols)
{
    float32x4_t sums[Rows][VectorCols];
    const bool load_full_c = Accumulate && FullWidth;
    for (int row = 0; row < Rows; ++row)
    {
        float* c_row =
            c + static_cast<std::size_t>(row) * c_stride;
        for (int vector_col = 0;
             vector_col < VectorCols;
             ++vector_col)
        {
            sums[row][vector_col] =
                load_full_c
                    ? vld1q_f32(c_row + vector_col * 4)
                    : vdupq_n_f32(0.0f);
        }
    }

    int inner = 0;
    for (; inner <= k - 4; inner += 4)
    {
        float32x4_t a_values[Rows];
        for (int row = 0; row < Rows; ++row)
        {
            a_values[row] = vld1q_f32(
                a + static_cast<std::size_t>(row) * a_stride +
                static_cast<std::size_t>(inner));
        }

        const float* b0 =
            packed_b +
            static_cast<std::size_t>(inner) *
                static_cast<std::size_t>(packed_b_stride);
        fma_packed_b_lane_f32<Rows, VectorCols, 0>(
            sums, a_values, b0);
        fma_packed_b_lane_f32<Rows, VectorCols, 1>(
            sums, a_values, b0 + packed_b_stride);
        fma_packed_b_lane_f32<Rows, VectorCols, 2>(
            sums, a_values, b0 + 2 * packed_b_stride);
        fma_packed_b_lane_f32<Rows, VectorCols, 3>(
            sums, a_values, b0 + 3 * packed_b_stride);
    }

    for (; inner < k; ++inner)
    {
        const float* b_row =
            packed_b +
            static_cast<std::size_t>(inner) *
                static_cast<std::size_t>(packed_b_stride);
        for (int vector_col = 0;
             vector_col < VectorCols;
             ++vector_col)
        {
            const float32x4_t b_value =
                vld1q_f32(b_row + vector_col * 4);
            for (int row = 0; row < Rows; ++row)
            {
                const float a_value =
                    a[static_cast<std::size_t>(row) * a_stride +
                      static_cast<std::size_t>(inner)];
                sums[row][vector_col] = vfmaq_n_f32(
                    sums[row][vector_col], b_value, a_value);
            }
        }
    }

    if (FullWidth)
    {
        for (int row = 0; row < Rows; ++row)
        {
            float* c_row =
                c + static_cast<std::size_t>(row) * c_stride;
            for (int vector_col = 0;
                 vector_col < VectorCols;
                 ++vector_col)
            {
                vst1q_f32(
                    c_row + vector_col * 4,
                    sums[row][vector_col]);
            }
        }
    }
    else
    {
        for (int row = 0; row < Rows; ++row)
        {
            store_row_f32<VectorCols, Accumulate>(
                sums[row],
                c + static_cast<std::size_t>(row) * c_stride,
                valid_cols);
        }
    }
}

template <int Rows, bool Accumulate, bool FullWidth>
CVH_GEMM_NEON_ALWAYS_INLINE void kernel_f32_6x16_u4(
    const float* a,
    std::size_t a_stride,
    const float* packed_b,
    int packed_b_stride,
    float* c,
    std::size_t c_stride,
    int k,
    int valid_cols)
{
    kernel_f32_directa_packedb_u4<
        Rows, 4, Accumulate, FullWidth>(
        a,
        a_stride,
        packed_b,
        packed_b_stride,
        c,
        c_stride,
        k,
        valid_cols);
}

template <int Rows, bool Accumulate, bool FullWidth>
inline void kernel_f32_6x8_u4(const float* a,
                              std::size_t a_stride,
                              const float* packed_b,
                              int packed_b_stride,
                              float* c,
                              std::size_t c_stride,
                              int k,
                              int valid_cols)
{
    kernel_f32_directa_packedb_u4<
        Rows, 2, Accumulate, FullWidth>(
        a,
        a_stride,
        packed_b,
        packed_b_stride,
        c,
        c_stride,
        k,
        valid_cols);
}

template <int Rows>
inline void run_width_f32(const float* a,
                          std::size_t a_stride,
                          const float* packed_b,
                          int packed_b_stride,
                          float* c,
                          std::size_t c_stride,
                          int k,
                          int valid_cols,
                          bool accumulate)
{
    if (valid_cols == 16)
    {
        if (accumulate)
        {
            kernel_f32_6x16_u4<Rows, true, true>(
                a,
                a_stride,
                packed_b,
                packed_b_stride,
                c,
                c_stride,
                k,
                valid_cols);
        }
        else
        {
            kernel_f32_6x16_u4<Rows, false, true>(
                a,
                a_stride,
                packed_b,
                packed_b_stride,
                c,
                c_stride,
                k,
                valid_cols);
        }
        return;
    }

    if (valid_cols > 8)
    {
        if (accumulate)
        {
            kernel_f32_6x16_u4<Rows, true, false>(
                a,
                a_stride,
                packed_b,
                packed_b_stride,
                c,
                c_stride,
                k,
                valid_cols);
        }
        else
        {
            kernel_f32_6x16_u4<Rows, false, false>(
                a,
                a_stride,
                packed_b,
                packed_b_stride,
                c,
                c_stride,
                k,
                valid_cols);
        }
        return;
    }

    if (valid_cols == 8)
    {
        if (accumulate)
        {
            kernel_f32_6x8_u4<Rows, true, true>(
                a,
                a_stride,
                packed_b,
                packed_b_stride,
                c,
                c_stride,
                k,
                valid_cols);
        }
        else
        {
            kernel_f32_6x8_u4<Rows, false, true>(
                a,
                a_stride,
                packed_b,
                packed_b_stride,
                c,
                c_stride,
                k,
                valid_cols);
        }
        return;
    }

    if (accumulate)
    {
        kernel_f32_6x8_u4<Rows, true, false>(
            a,
            a_stride,
            packed_b,
            packed_b_stride,
            c,
            c_stride,
            k,
            valid_cols);
    }
    else
    {
        kernel_f32_6x8_u4<Rows, false, false>(
            a,
            a_stride,
            packed_b,
            packed_b_stride,
            c,
            c_stride,
            k,
            valid_cols);
    }
}

inline bool run_f32(const float* a,
                    std::size_t a_stride,
                    const float* packed_b,
                    int packed_b_stride,
                    float* c,
                    std::size_t c_stride,
                    int k,
                    int valid_rows,
                    int valid_cols,
                    bool accumulate)
{
    if (a == nullptr || packed_b == nullptr || c == nullptr ||
        k <= 0 || valid_rows <= 0 ||
        valid_rows > F32DirectA6x16Traits::mr ||
        valid_cols <= 0 ||
        valid_cols > F32DirectA6x16Traits::nr ||
        packed_b_stride < F32DirectA6x16Traits::nr)
    {
        return false;
    }

#define CVH_NEON_RUN_HEIGHT(ROWS)                                  \
    run_width_f32<ROWS>(                                           \
        a,                                                         \
        a_stride,                                                  \
        packed_b,                                                  \
        packed_b_stride,                                           \
        c,                                                         \
        c_stride,                                                  \
        k,                                                         \
        valid_cols,                                                \
        accumulate)

    switch (valid_rows)
    {
        case 6:
            CVH_NEON_RUN_HEIGHT(6);
            break;
        case 5:
            CVH_NEON_RUN_HEIGHT(5);
            break;
        case 4:
            CVH_NEON_RUN_HEIGHT(4);
            break;
        case 3:
            CVH_NEON_RUN_HEIGHT(3);
            break;
        case 2:
            CVH_NEON_RUN_HEIGHT(2);
            break;
        case 1:
            CVH_NEON_RUN_HEIGHT(1);
            break;
        default:
            return false;
    }
#undef CVH_NEON_RUN_HEIGHT
    return true;
}

#else

inline bool run_f32(const float*,
                    std::size_t,
                    const float*,
                    int,
                    float*,
                    std::size_t,
                    int,
                    int,
                    int,
                    bool)
{
    return false;
}

#endif

#undef CVH_GEMM_NEON_ALWAYS_INLINE

}  // namespace gemm_neon
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_GEMM_NEON_HPP
