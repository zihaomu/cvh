#include "binary_kernel_xsimd.h"
#include "cvh/core/detail/openmp_utils.h"

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
