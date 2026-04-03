#ifndef CVH_BINARY_KERNEL_XSIMD_H
#define CVH_BINARY_KERNEL_XSIMD_H

#include <cstddef>
#include <cstdint>

namespace cvh {
namespace cpu {

enum class BinaryKernelOp
{
    Add = 0,
    Sub,
    Mul,
    Div,
    Max,
    Min,
    Mean,
};

enum class CompareKernelOp
{
    Eq = 0,
    Gt,
    Ge,
    Lt,
    Le,
    Ne,
};

void binary_broadcast_xsimd(BinaryKernelOp op,
                            const float* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const float* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            float* out,
                            size_t outer,
                            size_t inner);

// CV_16F (hfloat) variant - converts to float internally
void binary_broadcast_xsimd_hfloat(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner);

// CV_64F (double) variant
void binary_broadcast_xsimd_double(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner);

// CV_32S (int32) variant
void binary_broadcast_xsimd_int32(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner);

// CV_16S (int16) variant
void binary_broadcast_xsimd_int16(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner);

// CV_16U (uint16) variant
void binary_broadcast_xsimd_uint16(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner);

// CV_8S (int8) variant
void binary_broadcast_xsimd_int8(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner);

// CV_8U (uint8) variant
void binary_broadcast_xsimd_uint8(BinaryKernelOp op,
                            const void* lhs,
                            size_t lhs_outer_stride,
                            size_t lhs_inner_stride,
                            const void* rhs,
                            size_t rhs_outer_stride,
                            size_t rhs_inner_stride,
                            void* out,
                            size_t outer,
                            size_t inner);

void compare_broadcast_xsimd(CompareKernelOp op,
                             const float* lhs,
                             size_t lhs_outer_stride,
                             size_t lhs_inner_stride,
                             const float* rhs,
                             size_t rhs_outer_stride,
                             size_t rhs_inner_stride,
                             std::uint8_t* out,
                             size_t out_outer_stride,
                             size_t outer,
                             size_t inner);

// CV_16F (hfloat) compare variant - converts to float internally
void compare_broadcast_xsimd_hfloat(CompareKernelOp op,
                                    const void* lhs,
                                    size_t lhs_outer_stride,
                                    size_t lhs_inner_stride,
                                    const void* rhs,
                                    size_t rhs_outer_stride,
                                    size_t rhs_inner_stride,
                                    std::uint8_t* out,
                                    size_t out_outer_stride,
                                    size_t outer,
                                    size_t inner);

// Row-level helper for Mat-Scalar paths where scalar lanes differ per channel.
// `inner` is row element count (pixel_count * channels), `channels` is interleaved channel count.
void binary_scalar_channels_xsimd(BinaryKernelOp op,
                                  const float* src,
                                  size_t src_outer_stride,
                                  const float* scalar_lanes,
                                  int channels,
                                  float* out,
                                  size_t out_outer_stride,
                                  size_t outer,
                                  size_t inner,
                                  bool scalar_first);

void binary_scalar_channels_xsimd_hfloat(BinaryKernelOp op,
                                         const void* src,
                                         size_t src_outer_stride,
                                         const float* scalar_lanes,
                                         int channels,
                                         void* out,
                                         size_t out_outer_stride,
                                         size_t outer,
                                         size_t inner,
                                         bool scalar_first);

void compare_scalar_channels_xsimd(CompareKernelOp op,
                                   const float* src,
                                   size_t src_outer_stride,
                                   const float* scalar_lanes,
                                   int channels,
                                   std::uint8_t* out,
                                   size_t out_outer_stride,
                                   size_t outer,
                                   size_t inner,
                                   bool scalar_first);

void compare_scalar_channels_xsimd_hfloat(CompareKernelOp op,
                                          const void* src,
                                          size_t src_outer_stride,
                                          const float* scalar_lanes,
                                          int channels,
                                          std::uint8_t* out,
                                          size_t out_outer_stride,
                                          size_t outer,
                                          size_t inner,
                                          bool scalar_first);

}  // namespace cpu
}  // namespace cvh

#endif  // CVH_BINARY_KERNEL_XSIMD_H
