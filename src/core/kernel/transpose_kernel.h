#ifndef CVH_TRANSPOSE_KERNEL_H
#define CVH_TRANSPOSE_KERNEL_H

#include <cstddef>

namespace cvh {
namespace cpu {

void transpose2d_kernel_blocked(const unsigned char* src,
                                unsigned char* dst,
                                int rows,
                                int cols,
                                size_t elem_size1,
                                int channels);

}  // namespace cpu
}  // namespace cvh

#endif  // CVH_TRANSPOSE_KERNEL_H
