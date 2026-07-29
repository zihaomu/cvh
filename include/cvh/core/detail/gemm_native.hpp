#ifndef CVH_CORE_DETAIL_GEMM_NATIVE_HPP
#define CVH_CORE_DETAIL_GEMM_NATIVE_HPP

#include "cvh/core/detail/cpu_features.hpp"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/detail/gemm_avx2.hpp"
#include "cvh/core/detail/gemm_neon.hpp"
#include "cvh/core/types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cvh {
namespace detail {
namespace gemm_native {

enum class Backend
{
    None = 0,
    Neon,
    Avx2,
};

constexpr int kNeonMr = 6;
constexpr int kNeonNr = 16;
constexpr int kNeonNc = 128;

inline cpu::DispatchTag dispatch_tag(Backend backend)
{
    switch (backend)
    {
        case Backend::Neon:
            return cpu::DispatchTag::NativeNEON;
        case Backend::Avx2:
            return cpu::DispatchTag::NativeAVX2;
        case Backend::None:
        default:
            return cpu::DispatchTag::Scalar;
    }
}

inline Backend best_auto_backend()
{
    const cpu::NativeIsa isa = cpu::best_auto_native_isa();
    switch (isa)
    {
        case cpu::NativeIsa::Neon:
            return Backend::Neon;
        case cpu::NativeIsa::Avx2:
            return Backend::Avx2;
        case cpu::NativeIsa::None:
        default:
            return Backend::None;
    }
}

inline Backend select_backend(int m, int n, int k)
{
    if (m <= 0 || n <= 0 || k <= 0)
    {
        return Backend::None;
    }

    const cpu::DispatchMode mode = cpu::dispatch_mode();
    if (mode == cpu::DispatchMode::ScalarOnly ||
        mode == cpu::DispatchMode::OpenCVUIOnly)
    {
        return Backend::None;
    }
    if (mode == cpu::DispatchMode::NeonOnly)
    {
        return cpu::native_neon_runtime_available()
                   ? Backend::Neon
                   : Backend::None;
    }
    if (mode == cpu::DispatchMode::Avx2Only)
    {
        return cpu::native_avx2_runtime_available() && n >= 16
                   ? Backend::Avx2
                   : Backend::None;
    }
    const std::uint64_t work =
        static_cast<std::uint64_t>(m) *
        static_cast<std::uint64_t>(n) *
        static_cast<std::uint64_t>(k);
    if (m < 2 || n < 8 || k < 8 || work < (1ULL << 15))
    {
        return Backend::None;
    }

    const Backend backend = best_auto_backend();
    return backend == Backend::Avx2 && n < 16
               ? Backend::None
               : backend;
}

inline std::size_t neon_packed_b_elements(int k, int n)
{
    const int panels = (n + kNeonNr - 1) / kNeonNr;
    return static_cast<std::size_t>(panels) *
           static_cast<std::size_t>(k) *
           static_cast<std::size_t>(kNeonNr);
}

inline std::size_t aligned_float_offset(
    const std::vector<float>& storage,
    std::size_t alignment = 64)
{
    if (storage.empty())
    {
        return 0;
    }
    const std::uintptr_t address =
        reinterpret_cast<std::uintptr_t>(storage.data());
    const std::uintptr_t aligned =
        (address + alignment - 1) & ~(alignment - 1);
    return static_cast<std::size_t>(
        (aligned - address) / sizeof(float));
}

template <typename WeightT>
inline void pack_neon_b(const WeightT* b,
                        float* packed_b,
                        int k,
                        int n,
                        bool transposed)
{
    for (int col = 0; col < n; col += kNeonNr)
    {
        const int valid_cols = std::min(kNeonNr, n - col);
        float* panel =
            packed_b +
            static_cast<std::size_t>(col / kNeonNr) *
                static_cast<std::size_t>(k) *
                static_cast<std::size_t>(kNeonNr);
        for (int inner = 0; inner < k; ++inner)
        {
            float* panel_row =
                panel +
                static_cast<std::size_t>(inner) *
                    static_cast<std::size_t>(kNeonNr);
            for (int lane = 0; lane < valid_cols; ++lane)
            {
                const std::size_t source_index =
                    transposed
                        ? static_cast<std::size_t>(col + lane) *
                                  static_cast<std::size_t>(k) +
                              static_cast<std::size_t>(inner)
                        : static_cast<std::size_t>(inner) *
                                  static_cast<std::size_t>(n) +
                              static_cast<std::size_t>(col + lane);
                panel_row[lane] =
                    static_cast<float>(b[source_index]);
            }
            std::fill(
                panel_row + valid_cols,
                panel_row + kNeonNr,
                0.0f);
        }
    }
}

inline bool run_neon_packed(const float* a,
                            const float* packed_b,
                            float* c,
                            int m,
                            int n,
                            int k)
{
    if (!cpu::native_neon_runtime_available() ||
        a == nullptr || packed_b == nullptr || c == nullptr)
    {
        return false;
    }

    for (int col_block = 0;
         col_block < n;
         col_block += kNeonNc)
    {
        const int col_end = std::min(n, col_block + kNeonNc);
        for (int row = 0; row < m; row += kNeonMr)
        {
            const int valid_rows = std::min(kNeonMr, m - row);
            for (int col = col_block;
                 col < col_end;
                 col += kNeonNr)
            {
                const int valid_cols =
                    std::min(kNeonNr, n - col);
                const float* panel =
                    packed_b +
                    static_cast<std::size_t>(col / kNeonNr) *
                        static_cast<std::size_t>(k) *
                        static_cast<std::size_t>(kNeonNr);
#if CVH_NATIVE_NEON_COMPILED
                if (valid_rows == kNeonMr &&
                    valid_cols == kNeonNr)
                {
                    gemm_neon::kernel_f32_6x16_u4<
                        kNeonMr, false, true>(
                        a + static_cast<std::size_t>(row) *
                                static_cast<std::size_t>(k),
                        static_cast<std::size_t>(k),
                        panel,
                        kNeonNr,
                        c + static_cast<std::size_t>(row) *
                                static_cast<std::size_t>(n) +
                            static_cast<std::size_t>(col),
                        static_cast<std::size_t>(n),
                        k,
                        valid_cols);
                    continue;
                }
#endif
                if (!gemm_neon::run_f32(
                        a + static_cast<std::size_t>(row) *
                                static_cast<std::size_t>(k),
                        static_cast<std::size_t>(k),
                        panel,
                        kNeonNr,
                        c + static_cast<std::size_t>(row) *
                                static_cast<std::size_t>(n) +
                            static_cast<std::size_t>(col),
                        static_cast<std::size_t>(n),
                        k,
                        valid_rows,
                        valid_cols,
                        false))
                {
                    return false;
                }
            }
        }
    }
    return true;
}

inline std::vector<float>& neon_b_workspace()
{
    static thread_local std::vector<float> workspace;
    return workspace;
}

template <typename WeightT>
inline bool run_neon(const float* a,
                     const WeightT* b,
                     float* c,
                     int m,
                     int n,
                     int k,
                     bool transposed_b)
{
    if (!cpu::native_neon_runtime_available())
    {
        return false;
    }
    std::vector<float>& packed = neon_b_workspace();
    packed.resize(neon_packed_b_elements(k, n) + 15);
    float* aligned_packed =
        packed.data() + aligned_float_offset(packed);
    pack_neon_b(b, aligned_packed, k, n, transposed_b);
    return run_neon_packed(a, aligned_packed, c, m, n, k);
}

inline bool run_nn_f32(Backend backend,
                       const float* a,
                       const float* b,
                       float* c,
                       int m,
                       int n,
                       int k)
{
    if (backend == Backend::Neon)
    {
        return run_neon(a, b, c, m, n, k, false);
    }
    if (backend == Backend::Avx2)
    {
        return gemm_avx2::kernel_nn_f32(a, b, c, m, n, k);
    }
    return false;
}

inline bool run_nt_f32(Backend backend,
                       const float* a,
                       const float* b,
                       float* c,
                       int m,
                       int n,
                       int k)
{
    if (backend == Backend::Neon)
    {
        return run_neon(a, b, c, m, n, k, true);
    }
    if (backend == Backend::Avx2)
    {
        return gemm_avx2::kernel_nt_4x2_f32(
            a, b, c, m, n, k);
    }
    return false;
}

}  // namespace gemm_native
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_GEMM_NATIVE_HPP
