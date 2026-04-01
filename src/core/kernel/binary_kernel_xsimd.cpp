#include "binary_kernel_xsimd.h"
#include "cvh/core/detail/openmp_utils.h"
#include "cvh/core/define.h"

#include "xsimd/xsimd.hpp"

#include <array>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cvh {
namespace cpu {

namespace {

using Batch = xsimd::batch<float>;
constexpr size_t kLanes = Batch::size;

using Batch64 = xsimd::batch<double>;
constexpr size_t kLanes64 = Batch64::size;

using Batch32 = xsimd::batch<std::int32_t>;
constexpr size_t kLanes32 = Batch32::size;

using Batch16 = xsimd::batch<std::int16_t>;
constexpr size_t kLanes16 = Batch16::size;

using Batch8 = xsimd::batch<std::int8_t>;
constexpr size_t kLanes8 = Batch8::size;

inline float apply_scalar(BinaryKernelOp op, float lhs, float rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
        case BinaryKernelOp::Max:
            return lhs > rhs ? lhs : rhs;
        case BinaryKernelOp::Min:
            return lhs < rhs ? lhs : rhs;
        case BinaryKernelOp::Mean:
            return 0.5f * (lhs + rhs);
    }

    return 0.0f;
}

inline Batch apply_batch(BinaryKernelOp op, const Batch& lhs, const Batch& rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
        case BinaryKernelOp::Max:
            return xsimd::max(lhs, rhs);
        case BinaryKernelOp::Min:
            return xsimd::min(lhs, rhs);
        case BinaryKernelOp::Mean:
            return (lhs + rhs) * Batch(0.5f);
    }

    return Batch(0.0f);
}

inline bool apply_compare_scalar(CompareKernelOp op, float lhs, float rhs)
{
    switch (op)
    {
        case CompareKernelOp::Eq:
            return lhs == rhs;
        case CompareKernelOp::Gt:
            return lhs > rhs;
        case CompareKernelOp::Ge:
            return lhs >= rhs;
        case CompareKernelOp::Lt:
            return lhs < rhs;
        case CompareKernelOp::Le:
            return lhs <= rhs;
        case CompareKernelOp::Ne:
            return lhs != rhs;
    }

    return false;
}

inline Batch::batch_bool_type apply_compare_batch(CompareKernelOp op, const Batch& lhs, const Batch& rhs)
{
    switch (op)
    {
        case CompareKernelOp::Eq:
            return lhs == rhs;
        case CompareKernelOp::Gt:
            return lhs > rhs;
        case CompareKernelOp::Ge:
            return lhs >= rhs;
        case CompareKernelOp::Lt:
            return lhs < rhs;
        case CompareKernelOp::Le:
            return lhs <= rhs;
        case CompareKernelOp::Ne:
            return lhs != rhs;
    }

    return lhs != rhs;
}

// CV_16F (hfloat) variant - converts to float internally
inline void binary_broadcast_xsimd_hfloat_impl(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    // For hfloat (CV_16F), we convert to float, process, then convert back
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, inner, 1LL << 15, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const hfloat* lhs_row = reinterpret_cast<const hfloat*>(lhs) + outer_i * lhs_outer_stride;
        const hfloat* rhs_row = reinterpret_cast<const hfloat*>(rhs) + outer_i * rhs_outer_stride;
        hfloat* out_row = reinterpret_cast<hfloat*>(out) + outer_i * inner;

        // Temporary buffer for float conversion
        std::array<float, kLanes> lhs_buf;
        std::array<float, kLanes> rhs_buf;
        std::array<float, kLanes> out_buf;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            for (; inner_idx + kLanes <= inner; inner_idx += kLanes)
            {
                // Load hfloat and convert to float
                for (size_t lane = 0; lane < kLanes; ++lane)
                {
                    lhs_buf[lane] = lhs_inner_stride == 0 ? static_cast<float>(lhs_row[0])
                                                          : static_cast<float>(lhs_row[inner_idx + lane]);
                    rhs_buf[lane] = rhs_inner_stride == 0 ? static_cast<float>(rhs_row[0])
                                                          : static_cast<float>(rhs_row[inner_idx + lane]);
                }

                const Batch lhs_vec = Batch::load_unaligned(lhs_buf.data());
                const Batch rhs_vec = Batch::load_unaligned(rhs_buf.data());
                const Batch out_vec = apply_batch(op, lhs_vec, rhs_vec);
                out_vec.store_unaligned(out_buf.data());

                // Convert back to hfloat
                for (size_t lane = 0; lane < kLanes; ++lane)
                {
                    out_row[inner_idx + lane] = hfloat(out_buf[lane]);
                }
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const float lhs_val = lhs_inner_stride == 0 ? static_cast<float>(lhs_row[0])
                                                        : static_cast<float>(lhs_row[inner_idx * lhs_inner_stride]);
            const float rhs_val = rhs_inner_stride == 0 ? static_cast<float>(rhs_row[0])
                                                        : static_cast<float>(rhs_row[inner_idx * rhs_inner_stride]);
            out_row[inner_idx] = hfloat(apply_scalar(op, lhs_val, rhs_val));
        }
    }
}

}  // namespace

void binary_broadcast_xsimd(BinaryKernelOp op,
                            const float* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const float* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            float* out,
                            size_t outer,
                            size_t inner)
{
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, inner, 1LL << 15, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* lhs_row = lhs + outer_i * lhs_outer_stride;
        const float* rhs_row = rhs + outer_i * rhs_outer_stride;
        float* out_row = out + outer_i * inner;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            for (; inner_idx + kLanes <= inner; inner_idx += kLanes)
            {
                const Batch lhs_vec = lhs_inner_stride == 0
                    ? Batch(lhs_row[0])
                    : Batch::load_unaligned(lhs_row + inner_idx);
                const Batch rhs_vec = rhs_inner_stride == 0
                    ? Batch(rhs_row[0])
                    : Batch::load_unaligned(rhs_row + inner_idx);
                const Batch out_vec = apply_batch(op, lhs_vec, rhs_vec);
                out_vec.store_unaligned(out_row + inner_idx);
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const float lhs_val = lhs_row[inner_idx * lhs_inner_stride];
            const float rhs_val = rhs_row[inner_idx * rhs_inner_stride];
            out_row[inner_idx] = apply_scalar(op, lhs_val, rhs_val);
        }
    }
}

void binary_broadcast_xsimd_hfloat(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    binary_broadcast_xsimd_hfloat_impl(op, lhs, lhs_outer_stride, lhs_inner_stride,
                                        rhs, rhs_outer_stride, rhs_inner_stride,
                                        out, outer, inner);
}

inline double apply_scalar64(BinaryKernelOp op, double lhs, double rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
        case BinaryKernelOp::Max:
            return lhs > rhs ? lhs : rhs;
        case BinaryKernelOp::Min:
            return lhs < rhs ? lhs : rhs;
        case BinaryKernelOp::Mean:
            return 0.5 * (lhs + rhs);
    }

    return 0.0;
}

inline Batch64 apply_batch64(BinaryKernelOp op, const Batch64& lhs, const Batch64& rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
        case BinaryKernelOp::Max:
            return xsimd::max(lhs, rhs);
        case BinaryKernelOp::Min:
            return xsimd::min(lhs, rhs);
        case BinaryKernelOp::Mean:
            return (lhs + rhs) * Batch64(0.5);
    }

    return Batch64(0.0);
}

inline void binary_broadcast_xsimd_double_impl(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, inner, 1LL << 15, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const double* lhs_row = reinterpret_cast<const double*>(lhs) + outer_i * lhs_outer_stride;
        const double* rhs_row = reinterpret_cast<const double*>(rhs) + outer_i * rhs_outer_stride;
        double* out_row = reinterpret_cast<double*>(out) + outer_i * inner;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            for (; inner_idx + kLanes64 <= inner; inner_idx += kLanes64)
            {
                const Batch64 lhs_vec = lhs_inner_stride == 0
                    ? Batch64(lhs_row[0])
                    : Batch64::load_unaligned(lhs_row + inner_idx);
                const Batch64 rhs_vec = rhs_inner_stride == 0
                    ? Batch64(rhs_row[0])
                    : Batch64::load_unaligned(rhs_row + inner_idx);
                const Batch64 out_vec = apply_batch64(op, lhs_vec, rhs_vec);
                out_vec.store_unaligned(out_row + inner_idx);
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const double lhs_val = lhs_row[inner_idx * lhs_inner_stride];
            const double rhs_val = rhs_row[inner_idx * rhs_inner_stride];
            out_row[inner_idx] = apply_scalar64(op, lhs_val, rhs_val);
        }
    }
}

void binary_broadcast_xsimd_double(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    binary_broadcast_xsimd_double_impl(op, lhs, lhs_outer_stride, lhs_inner_stride,
                                        rhs, rhs_outer_stride, rhs_inner_stride,
                                        out, outer, inner);
}

// Integer scalar functions
inline std::int32_t apply_scalar_i32(BinaryKernelOp op, std::int32_t lhs, std::int32_t rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return rhs != 0 ? lhs / rhs : 0;
        case BinaryKernelOp::Max:
            return lhs > rhs ? lhs : rhs;
        case BinaryKernelOp::Min:
            return lhs < rhs ? lhs : rhs;
        case BinaryKernelOp::Mean:
            return (lhs + rhs) / 2;
        default:
            return 0;
    }
}

inline Batch32 apply_batch_i32(BinaryKernelOp op, const Batch32& lhs, const Batch32& rhs)
{
    switch (op)
    {
        case BinaryKernelOp::Add:
            return lhs + rhs;
        case BinaryKernelOp::Sub:
            return lhs - rhs;
        case BinaryKernelOp::Mul:
            return lhs * rhs;
        case BinaryKernelOp::Div:
            return lhs / rhs;
        case BinaryKernelOp::Max:
            return xsimd::max(lhs, rhs);
        case BinaryKernelOp::Min:
            return xsimd::min(lhs, rhs);
        case BinaryKernelOp::Mean:
            return (lhs + rhs) / Batch32(2);
        default:
            return Batch32(0);
    }
}

inline void binary_broadcast_xsimd_int32_impl(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, inner, 1LL << 15, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const std::int32_t* lhs_row = reinterpret_cast<const std::int32_t*>(lhs) + outer_i * lhs_outer_stride;
        const std::int32_t* rhs_row = reinterpret_cast<const std::int32_t*>(rhs) + outer_i * rhs_outer_stride;
        std::int32_t* out_row = reinterpret_cast<std::int32_t*>(out) + outer_i * inner;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            for (; inner_idx + kLanes32 <= inner; inner_idx += kLanes32)
            {
                const Batch32 lhs_vec = lhs_inner_stride == 0
                    ? Batch32(lhs_row[0])
                    : Batch32::load_unaligned(lhs_row + inner_idx);
                const Batch32 rhs_vec = rhs_inner_stride == 0
                    ? Batch32(rhs_row[0])
                    : Batch32::load_unaligned(rhs_row + inner_idx);
                const Batch32 out_vec = apply_batch_i32(op, lhs_vec, rhs_vec);
                out_vec.store_unaligned(out_row + inner_idx);
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const std::int32_t lhs_val = lhs_row[inner_idx * lhs_inner_stride];
            const std::int32_t rhs_val = rhs_row[inner_idx * rhs_inner_stride];
            out_row[inner_idx] = apply_scalar_i32(op, lhs_val, rhs_val);
        }
    }
}

void binary_broadcast_xsimd_int32(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    binary_broadcast_xsimd_int32_impl(op, lhs, lhs_outer_stride, lhs_inner_stride,
                                       rhs, rhs_outer_stride, rhs_inner_stride,
                                       out, outer, inner);
}

// Generic integer kernel for 8-bit and 16-bit types with saturation
template<typename T, typename BatchType, size_t kLanesInt>
inline void binary_broadcast_xsimd_int_impl(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, inner, 1LL << 15, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const T* lhs_row = reinterpret_cast<const T*>(lhs) + outer_i * lhs_outer_stride;
        const T* rhs_row = reinterpret_cast<const T*>(rhs) + outer_i * rhs_outer_stride;
        T* out_row = reinterpret_cast<T*>(out) + outer_i * inner;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            for (; inner_idx + kLanesInt <= inner; inner_idx += kLanesInt)
            {
                const BatchType lhs_vec = lhs_inner_stride == 0
                    ? BatchType(lhs_row[0])
                    : BatchType::load_unaligned(lhs_row + inner_idx);
                const BatchType rhs_vec = rhs_inner_stride == 0
                    ? BatchType(rhs_row[0])
                    : BatchType::load_unaligned(rhs_row + inner_idx);

                BatchType out_vec;
                switch (op)
                {
                    case BinaryKernelOp::Add:
                        out_vec = lhs_vec + rhs_vec;
                        break;
                    case BinaryKernelOp::Sub:
                        out_vec = lhs_vec - rhs_vec;
                        break;
                    case BinaryKernelOp::Mul:
                        out_vec = lhs_vec * rhs_vec;
                        break;
                    case BinaryKernelOp::Div:
                        out_vec = lhs_vec / rhs_vec;
                        break;
                    case BinaryKernelOp::Max:
                        out_vec = xsimd::max(lhs_vec, rhs_vec);
                        break;
                    case BinaryKernelOp::Min:
                        out_vec = xsimd::min(lhs_vec, rhs_vec);
                        break;
                    case BinaryKernelOp::Mean:
                        out_vec = (lhs_vec + rhs_vec) / BatchType(2);
                        break;
                    default:
                        out_vec = BatchType(0);
                }
                out_vec.store_unaligned(out_row + inner_idx);
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const T lhs_val = lhs_row[inner_idx * lhs_inner_stride];
            const T rhs_val = rhs_row[inner_idx * rhs_inner_stride];
            T result;
            switch (op)
            {
                case BinaryKernelOp::Add:
                    result = lhs_val + rhs_val;
                    break;
                case BinaryKernelOp::Sub:
                    result = lhs_val - rhs_val;
                    break;
                case BinaryKernelOp::Mul:
                    result = lhs_val * rhs_val;
                    break;
                case BinaryKernelOp::Div:
                    result = rhs_val != 0 ? lhs_val / rhs_val : 0;
                    break;
                case BinaryKernelOp::Max:
                    result = lhs_val > rhs_val ? lhs_val : rhs_val;
                    break;
                case BinaryKernelOp::Min:
                    result = lhs_val < rhs_val ? lhs_val : rhs_val;
                    break;
                case BinaryKernelOp::Mean:
                    result = static_cast<T>((static_cast<int>(lhs_val) + static_cast<int>(rhs_val)) / 2);
                    break;
                default:
                    result = 0;
            }
            out_row[inner_idx] = result;
        }
    }
}

void binary_broadcast_xsimd_int16(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    binary_broadcast_xsimd_int_impl<std::int16_t, Batch16, kLanes16>(
        op, lhs, lhs_outer_stride, lhs_inner_stride,
        rhs, rhs_outer_stride, rhs_inner_stride,
        out, outer, inner);
}

void binary_broadcast_xsimd_uint16(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    using BatchU16 = xsimd::batch<std::uint16_t>;
    constexpr size_t kLanesU16 = BatchU16::size;
    binary_broadcast_xsimd_int_impl<std::uint16_t, BatchU16, kLanesU16>(
        op, lhs, lhs_outer_stride, lhs_inner_stride,
        rhs, rhs_outer_stride, rhs_inner_stride,
        out, outer, inner);
}

void binary_broadcast_xsimd_int8(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    binary_broadcast_xsimd_int_impl<std::int8_t, Batch8, kLanes8>(
        op, lhs, lhs_outer_stride, lhs_inner_stride,
        rhs, rhs_outer_stride, rhs_inner_stride,
        out, outer, inner);
}

void binary_broadcast_xsimd_uint8(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner)
{
    using BatchU8 = xsimd::batch<std::uint8_t>;
    constexpr size_t kLanesU8 = BatchU8::size;
    binary_broadcast_xsimd_int_impl<std::uint8_t, BatchU8, kLanesU8>(
        op, lhs, lhs_outer_stride, lhs_inner_stride,
        rhs, rhs_outer_stride, rhs_inner_stride,
        out, outer, inner);
}

void compare_broadcast_xsimd(CompareKernelOp op,
                             const float* lhs,
                             size_t lhs_outer_stride,
                             size_t lhs_inner_stride,
                             const float* rhs,
                             size_t rhs_outer_stride,
                             size_t rhs_inner_stride,
                             std::uint8_t* out,
                             size_t outer,
                             size_t inner)
{
    const long long outer_ll = static_cast<long long>(outer);
#ifdef _OPENMP
#pragma omp parallel for if(should_parallelize_1d_loop(outer, inner, 1LL << 15, 2))
#endif
    for (long long outer_idx = 0; outer_idx < outer_ll; ++outer_idx)
    {
        const size_t outer_i = static_cast<size_t>(outer_idx);
        const float* lhs_row = lhs + outer_i * lhs_outer_stride;
        const float* rhs_row = rhs + outer_i * rhs_outer_stride;
        std::uint8_t* out_row = out + outer_i * inner;

        size_t inner_idx = 0;
        if (lhs_inner_stride <= 1 && rhs_inner_stride <= 1)
        {
            std::array<float, kLanes> tmp {};
            for (; inner_idx + kLanes <= inner; inner_idx += kLanes)
            {
                const Batch lhs_vec = lhs_inner_stride == 0
                    ? Batch(lhs_row[0])
                    : Batch::load_unaligned(lhs_row + inner_idx);
                const Batch rhs_vec = rhs_inner_stride == 0
                    ? Batch(rhs_row[0])
                    : Batch::load_unaligned(rhs_row + inner_idx);
                const Batch::batch_bool_type cmp_mask = apply_compare_batch(op, lhs_vec, rhs_vec);
                const Batch out_vec = xsimd::select(cmp_mask, Batch(255.0f), Batch(0.0f));
                out_vec.store_unaligned(tmp.data());

                for (size_t lane = 0; lane < kLanes; ++lane)
                {
                    out_row[inner_idx + lane] = static_cast<std::uint8_t>(tmp[lane]);
                }
            }
        }

        for (; inner_idx < inner; ++inner_idx)
        {
            const float lhs_val = lhs_row[inner_idx * lhs_inner_stride];
            const float rhs_val = rhs_row[inner_idx * rhs_inner_stride];
            out_row[inner_idx] = apply_compare_scalar(op, lhs_val, rhs_val) ? 255 : 0;
        }
    }
}

}  // namespace cpu
}  // namespace cvh