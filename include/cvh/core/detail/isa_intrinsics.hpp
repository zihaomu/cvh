#ifndef CVH_CORE_DETAIL_ISA_INTRINSICS_HPP
#define CVH_CORE_DETAIL_ISA_INTRINSICS_HPP

#include "cvh/detail/config.h"

#if CVH_ENABLE_OPTIMIZATION && \
    (defined(__aarch64__) || defined(_M_ARM64)) && \
    (defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64))
#define CVH_DETAIL_HAVE_NEON_KERNEL 1
#include <arm_neon.h>
#else
#define CVH_DETAIL_HAVE_NEON_KERNEL 0
#endif

#if CVH_ENABLE_OPTIMIZATION && \
    (defined(__i386__) || defined(__x86_64__) || \
     defined(_M_IX86) || defined(_M_X64))
#if defined(__clang__) || defined(__GNUC__)
#define CVH_DETAIL_HAVE_AVX2_KERNEL 1
#define CVH_DETAIL_TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
#include <immintrin.h>
#elif defined(_MSC_VER) && defined(__AVX2__)
#define CVH_DETAIL_HAVE_AVX2_KERNEL 1
#define CVH_DETAIL_TARGET_AVX2_FMA
#include <immintrin.h>
#else
#define CVH_DETAIL_HAVE_AVX2_KERNEL 0
#define CVH_DETAIL_TARGET_AVX2_FMA
#endif
#else
#define CVH_DETAIL_HAVE_AVX2_KERNEL 0
#define CVH_DETAIL_TARGET_AVX2_FMA
#endif

#endif  // CVH_CORE_DETAIL_ISA_INTRINSICS_HPP
