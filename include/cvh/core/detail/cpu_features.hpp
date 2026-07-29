#ifndef CVH_CORE_DETAIL_CPU_FEATURES_HPP
#define CVH_CORE_DETAIL_CPU_FEATURES_HPP

#include "cvh/core/detail/native_intrinsics.hpp"

#if CVH_NATIVE_AVX2_COMPILED && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace cvh {
namespace cpu {

enum class NativeIsa
{
    None = 0,
    Neon,
    Avx2,
};

inline constexpr bool native_neon_compiled()
{
    return CVH_NATIVE_NEON_COMPILED != 0;
}

inline constexpr bool native_avx2_compiled()
{
    return CVH_NATIVE_AVX2_COMPILED != 0;
}

inline constexpr bool native_neon_auto_enabled()
{
    return CVH_ENABLE_NATIVE_NEON_AUTO != 0;
}

inline constexpr bool native_avx2_auto_enabled()
{
    return CVH_ENABLE_NATIVE_AVX2_AUTO != 0;
}

inline bool native_neon_runtime_available()
{
    // Phase N2 targets AArch64 first. Advanced SIMD and FP FMA are part of
    // the AArch64 execution environment used by this native kernel.
    return native_neon_compiled();
}

inline bool native_avx2_runtime_available()
{
#if CVH_NATIVE_AVX2_COMPILED && (defined(__clang__) || defined(__GNUC__))
    static const bool available = []() {
#if defined(__GNUC__) && !defined(__clang__)
        __builtin_cpu_init();
#endif
        return __builtin_cpu_supports("avx2") &&
               __builtin_cpu_supports("fma");
    }();
    return available;
#elif CVH_NATIVE_AVX2_COMPILED && defined(_MSC_VER)
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

inline NativeIsa best_auto_native_isa()
{
    if (native_neon_auto_enabled() &&
        native_neon_runtime_available())
    {
        return NativeIsa::Neon;
    }
    if (native_avx2_auto_enabled() &&
        native_avx2_runtime_available())
    {
        return NativeIsa::Avx2;
    }
    return NativeIsa::None;
}

}  // namespace cpu
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_CPU_FEATURES_HPP
