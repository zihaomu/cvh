#ifndef CVH_CORE_DETAIL_CPU_FEATURES_HPP
#define CVH_CORE_DETAIL_CPU_FEATURES_HPP

#include "cvh/core/detail/isa_intrinsics.hpp"

#if CVH_DETAIL_HAVE_AVX2_KERNEL && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace cvh {
namespace cpu {

enum class Isa
{
    None = 0,
    Neon,
    Avx2,
};

inline constexpr bool neon_kernel_compiled()
{
    return CVH_DETAIL_HAVE_NEON_KERNEL != 0;
}

inline constexpr bool avx2_kernel_compiled()
{
    return CVH_DETAIL_HAVE_AVX2_KERNEL != 0;
}

inline bool neon_runtime_available()
{
    // Phase N2 targets AArch64 first. Advanced SIMD and FP FMA are part of
    // the AArch64 execution environment used by this specialized kernel.
    return neon_kernel_compiled();
}

inline bool avx2_fma_runtime_available()
{
#if CVH_DETAIL_HAVE_AVX2_KERNEL && (defined(__clang__) || defined(__GNUC__))
    static const bool available = []() {
#if defined(__GNUC__) && !defined(__clang__)
        __builtin_cpu_init();
#endif
        return __builtin_cpu_supports("avx2") &&
               __builtin_cpu_supports("fma");
    }();
    return available;
#elif CVH_DETAIL_HAVE_AVX2_KERNEL && defined(_MSC_VER)
    static const bool available = []() {
        int registers[4] = {};
        __cpuidex(registers, 1, 0);
        const bool osxsave = (registers[2] & (1 << 27)) != 0;
        const bool avx = (registers[2] & (1 << 28)) != 0;
        const bool fma = (registers[2] & (1 << 12)) != 0;
        if (!osxsave || !avx || !fma)
        {
            return false;
        }
        const unsigned __int64 xcr0 = _xgetbv(0);
        if ((xcr0 & 0x6) != 0x6)
        {
            return false;
        }
        __cpuidex(registers, 7, 0);
        return (registers[1] & (1 << 5)) != 0;
    }();
    return available;
#else
    return false;
#endif
}

inline Isa best_available_isa()
{
    if (neon_runtime_available())
    {
        return Isa::Neon;
    }
    if (avx2_fma_runtime_available())
    {
        return Isa::Avx2;
    }
    return Isa::None;
}

}  // namespace cpu
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_CPU_FEATURES_HPP
