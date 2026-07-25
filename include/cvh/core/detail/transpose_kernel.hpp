#ifndef CVH_CORE_DETAIL_TRANSPOSE_KERNEL_HPP
#define CVH_CORE_DETAIL_TRANSPOSE_KERNEL_HPP

#include "cvh/core/parallel.h"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/detail/openmp_utils.h"
#include "cvh/core/simd/opencv_ui.h"
#include "cvh/core/system.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace cvh {
namespace cpu {

namespace {

#if CVH_ENABLE_OPENCV_INTRIN && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)

inline void transpose_u8_ui(const uchar* src,
                            uchar* dst,
                            int rows,
                            int cols)
{
    for (int col = 0; col <= cols - 16; col += 16)
    {
        int row = 0;
        for (; row <= rows - 16; row += 16)
        {
            cv::v_uint8x16 r0 = cv::v_load(src + (row + 0) * cols + col);
            cv::v_uint8x16 r1 = cv::v_load(src + (row + 1) * cols + col);
            cv::v_uint8x16 r2 = cv::v_load(src + (row + 2) * cols + col);
            cv::v_uint8x16 r3 = cv::v_load(src + (row + 3) * cols + col);
            cv::v_uint8x16 r4 = cv::v_load(src + (row + 4) * cols + col);
            cv::v_uint8x16 r5 = cv::v_load(src + (row + 5) * cols + col);
            cv::v_uint8x16 r6 = cv::v_load(src + (row + 6) * cols + col);
            cv::v_uint8x16 r7 = cv::v_load(src + (row + 7) * cols + col);
            cv::v_uint8x16 r8 = cv::v_load(src + (row + 8) * cols + col);
            cv::v_uint8x16 r9 = cv::v_load(src + (row + 9) * cols + col);
            cv::v_uint8x16 r10 = cv::v_load(src + (row + 10) * cols + col);
            cv::v_uint8x16 r11 = cv::v_load(src + (row + 11) * cols + col);
            cv::v_uint8x16 r12 = cv::v_load(src + (row + 12) * cols + col);
            cv::v_uint8x16 r13 = cv::v_load(src + (row + 13) * cols + col);
            cv::v_uint8x16 r14 = cv::v_load(src + (row + 14) * cols + col);
            cv::v_uint8x16 r15 = cv::v_load(src + (row + 15) * cols + col);

            cv::v_uint8x16 t0, t1, t2, t3, t4, t5, t6, t7;
            cv::v_uint8x16 t8, t9, t10, t11, t12, t13, t14, t15;
            cv::v_zip(r0, r1, t0, t1);
            cv::v_zip(r2, r3, t2, t3);
            cv::v_zip(r4, r5, t4, t5);
            cv::v_zip(r6, r7, t6, t7);
            cv::v_zip(r8, r9, t8, t9);
            cv::v_zip(r10, r11, t10, t11);
            cv::v_zip(r12, r13, t12, t13);
            cv::v_zip(r14, r15, t14, t15);

            cv::v_uint16x8 s0, s1, s2, s3, s4, s5, s6, s7;
            cv::v_uint16x8 s8, s9, s10, s11, s12, s13, s14, s15;
            cv::v_zip(cv::v_reinterpret_as_u16(t0), cv::v_reinterpret_as_u16(t2), s0, s1);
            cv::v_zip(cv::v_reinterpret_as_u16(t1), cv::v_reinterpret_as_u16(t3), s2, s3);
            cv::v_zip(cv::v_reinterpret_as_u16(t4), cv::v_reinterpret_as_u16(t6), s4, s5);
            cv::v_zip(cv::v_reinterpret_as_u16(t5), cv::v_reinterpret_as_u16(t7), s6, s7);
            cv::v_zip(cv::v_reinterpret_as_u16(t8), cv::v_reinterpret_as_u16(t10), s8, s9);
            cv::v_zip(cv::v_reinterpret_as_u16(t9), cv::v_reinterpret_as_u16(t11), s10, s11);
            cv::v_zip(cv::v_reinterpret_as_u16(t12), cv::v_reinterpret_as_u16(t14), s12, s13);
            cv::v_zip(cv::v_reinterpret_as_u16(t13), cv::v_reinterpret_as_u16(t15), s14, s15);

            cv::v_uint32x4 u0, u1, u2, u3, u4, u5, u6, u7;
            cv::v_uint32x4 u8, u9, u10, u11, u12, u13, u14, u15;
            cv::v_zip(cv::v_reinterpret_as_u32(s0), cv::v_reinterpret_as_u32(s4), u0, u1);
            cv::v_zip(cv::v_reinterpret_as_u32(s1), cv::v_reinterpret_as_u32(s5), u2, u3);
            cv::v_zip(cv::v_reinterpret_as_u32(s2), cv::v_reinterpret_as_u32(s6), u4, u5);
            cv::v_zip(cv::v_reinterpret_as_u32(s3), cv::v_reinterpret_as_u32(s7), u6, u7);
            cv::v_zip(cv::v_reinterpret_as_u32(s8), cv::v_reinterpret_as_u32(s12), u8, u9);
            cv::v_zip(cv::v_reinterpret_as_u32(s9), cv::v_reinterpret_as_u32(s13), u10, u11);
            cv::v_zip(cv::v_reinterpret_as_u32(s10), cv::v_reinterpret_as_u32(s14), u12, u13);
            cv::v_zip(cv::v_reinterpret_as_u32(s11), cv::v_reinterpret_as_u32(s15), u14, u15);

            const cv::v_uint32x4 v0 = cv::v_combine_low(u0, u8);
            const cv::v_uint32x4 v1 = cv::v_combine_high(u0, u8);
            const cv::v_uint32x4 v2 = cv::v_combine_low(u1, u9);
            const cv::v_uint32x4 v3 = cv::v_combine_high(u1, u9);
            const cv::v_uint32x4 v4 = cv::v_combine_low(u2, u10);
            const cv::v_uint32x4 v5 = cv::v_combine_high(u2, u10);
            const cv::v_uint32x4 v6 = cv::v_combine_low(u3, u11);
            const cv::v_uint32x4 v7 = cv::v_combine_high(u3, u11);
            const cv::v_uint32x4 v8 = cv::v_combine_low(u4, u12);
            const cv::v_uint32x4 v9 = cv::v_combine_high(u4, u12);
            const cv::v_uint32x4 v10 = cv::v_combine_low(u5, u13);
            const cv::v_uint32x4 v11 = cv::v_combine_high(u5, u13);
            const cv::v_uint32x4 v12 = cv::v_combine_low(u6, u14);
            const cv::v_uint32x4 v13 = cv::v_combine_high(u6, u14);
            const cv::v_uint32x4 v14 = cv::v_combine_low(u7, u15);
            const cv::v_uint32x4 v15 = cv::v_combine_high(u7, u15);

            cv::v_store(dst + (col + 0) * rows + row, cv::v_reinterpret_as_u8(v0));
            cv::v_store(dst + (col + 1) * rows + row, cv::v_reinterpret_as_u8(v1));
            cv::v_store(dst + (col + 2) * rows + row, cv::v_reinterpret_as_u8(v2));
            cv::v_store(dst + (col + 3) * rows + row, cv::v_reinterpret_as_u8(v3));
            cv::v_store(dst + (col + 4) * rows + row, cv::v_reinterpret_as_u8(v4));
            cv::v_store(dst + (col + 5) * rows + row, cv::v_reinterpret_as_u8(v5));
            cv::v_store(dst + (col + 6) * rows + row, cv::v_reinterpret_as_u8(v6));
            cv::v_store(dst + (col + 7) * rows + row, cv::v_reinterpret_as_u8(v7));
            cv::v_store(dst + (col + 8) * rows + row, cv::v_reinterpret_as_u8(v8));
            cv::v_store(dst + (col + 9) * rows + row, cv::v_reinterpret_as_u8(v9));
            cv::v_store(dst + (col + 10) * rows + row, cv::v_reinterpret_as_u8(v10));
            cv::v_store(dst + (col + 11) * rows + row, cv::v_reinterpret_as_u8(v11));
            cv::v_store(dst + (col + 12) * rows + row, cv::v_reinterpret_as_u8(v12));
            cv::v_store(dst + (col + 13) * rows + row, cv::v_reinterpret_as_u8(v13));
            cv::v_store(dst + (col + 14) * rows + row, cv::v_reinterpret_as_u8(v14));
            cv::v_store(dst + (col + 15) * rows + row, cv::v_reinterpret_as_u8(v15));
        }
        for (; row < rows; ++row)
        {
            for (int k = 0; k < 16; ++k)
            {
                dst[(col + k) * rows + row] = src[row * cols + col + k];
            }
        }
    }
    const int vector_cols = cols & -16;
    for (int col = vector_cols; col < cols; ++col)
    {
        for (int row = 0; row < rows; ++row)
        {
            dst[col * rows + row] = src[row * cols + col];
        }
    }
}

inline void transpose_u16_ui(const ushort* src,
                             ushort* dst,
                             int rows,
                             int cols)
{
    for (int col = 0; col <= cols - 8; col += 8)
    {
        int row = 0;
        for (; row <= rows - 8; row += 8)
        {
            cv::v_uint16x8 r0 = cv::v_load(src + (row + 0) * cols + col);
            cv::v_uint16x8 r1 = cv::v_load(src + (row + 1) * cols + col);
            cv::v_uint16x8 r2 = cv::v_load(src + (row + 2) * cols + col);
            cv::v_uint16x8 r3 = cv::v_load(src + (row + 3) * cols + col);
            cv::v_uint16x8 r4 = cv::v_load(src + (row + 4) * cols + col);
            cv::v_uint16x8 r5 = cv::v_load(src + (row + 5) * cols + col);
            cv::v_uint16x8 r6 = cv::v_load(src + (row + 6) * cols + col);
            cv::v_uint16x8 r7 = cv::v_load(src + (row + 7) * cols + col);
            cv::v_uint16x8 t0, t1, t2, t3, t4, t5, t6, t7;
            cv::v_zip(r0, r1, t0, t1);
            cv::v_zip(r2, r3, t2, t3);
            cv::v_zip(r4, r5, t4, t5);
            cv::v_zip(r6, r7, t6, t7);
            cv::v_uint32x4 u0, u1, u2, u3, u4, u5, u6, u7;
            cv::v_zip(cv::v_reinterpret_as_u32(t0), cv::v_reinterpret_as_u32(t4), u0, u1);
            cv::v_zip(cv::v_reinterpret_as_u32(t1), cv::v_reinterpret_as_u32(t5), u2, u3);
            cv::v_zip(cv::v_reinterpret_as_u32(t2), cv::v_reinterpret_as_u32(t6), u4, u5);
            cv::v_zip(cv::v_reinterpret_as_u32(t3), cv::v_reinterpret_as_u32(t7), u6, u7);
            cv::v_uint32x4 v0, v1, v2, v3, v4, v5, v6, v7;
            cv::v_zip(u0, u4, v0, v1);
            cv::v_zip(u1, u5, v2, v3);
            cv::v_zip(u2, u6, v4, v5);
            cv::v_zip(u3, u7, v6, v7);
            cv::v_store(dst + (col + 0) * rows + row, cv::v_reinterpret_as_u16(v0));
            cv::v_store(dst + (col + 1) * rows + row, cv::v_reinterpret_as_u16(v1));
            cv::v_store(dst + (col + 2) * rows + row, cv::v_reinterpret_as_u16(v2));
            cv::v_store(dst + (col + 3) * rows + row, cv::v_reinterpret_as_u16(v3));
            cv::v_store(dst + (col + 4) * rows + row, cv::v_reinterpret_as_u16(v4));
            cv::v_store(dst + (col + 5) * rows + row, cv::v_reinterpret_as_u16(v5));
            cv::v_store(dst + (col + 6) * rows + row, cv::v_reinterpret_as_u16(v6));
            cv::v_store(dst + (col + 7) * rows + row, cv::v_reinterpret_as_u16(v7));
        }
        for (; row < rows; ++row)
        {
            for (int k = 0; k < 8; ++k)
            {
                dst[(col + k) * rows + row] = src[row * cols + col + k];
            }
        }
    }
    const int vector_cols = cols & -8;
    for (int col = vector_cols; col < cols; ++col)
    {
        for (int row = 0; row < rows; ++row)
        {
            dst[col * rows + row] = src[row * cols + col];
        }
    }
}

inline void transpose_u32_ui(const uint* src,
                             uint* dst,
                             int rows,
                             int cols)
{
    for (int col = 0; col <= cols - 4; col += 4)
    {
        int row = 0;
        for (; row <= rows - 4; row += 4)
        {
            const cv::v_uint32x4 r0 = cv::v_load(src + (row + 0) * cols + col);
            const cv::v_uint32x4 r1 = cv::v_load(src + (row + 1) * cols + col);
            const cv::v_uint32x4 r2 = cv::v_load(src + (row + 2) * cols + col);
            const cv::v_uint32x4 r3 = cv::v_load(src + (row + 3) * cols + col);
            cv::v_uint32x4 o0, o1, o2, o3;
            cv::v_transpose4x4(r0, r1, r2, r3, o0, o1, o2, o3);
            cv::v_store(dst + (col + 0) * rows + row, o0);
            cv::v_store(dst + (col + 1) * rows + row, o1);
            cv::v_store(dst + (col + 2) * rows + row, o2);
            cv::v_store(dst + (col + 3) * rows + row, o3);
        }
        for (; row < rows; ++row)
        {
            for (int k = 0; k < 4; ++k)
            {
                dst[(col + k) * rows + row] = src[row * cols + col + k];
            }
        }
    }
    const int vector_cols = cols & -4;
    for (int col = vector_cols; col < cols; ++col)
    {
        for (int row = 0; row < rows; ++row)
        {
            dst[col * rows + row] = src[row * cols + col];
        }
    }
}

#endif

template <class RowBlockFn>
void for_each_row_block(int rows, int cols, int tile, RowBlockFn&& fn)
{
    const int row_blocks = (rows + tile - 1) / tile;
    const bool do_parallel = should_parallelize_1d_loop(
        static_cast<size_t>(row_blocks),
        static_cast<size_t>(tile) * static_cast<size_t>(cols),
        1LL << 14,
        2);

    if (!do_parallel)
    {
        for (int block_idx = 0; block_idx < row_blocks; ++block_idx)
        {
            const int row0 = block_idx * tile;
            fn(row0);
        }
        return;
    }

    cvh::parallel_for_(
        cvh::Range(0, row_blocks),
        [&](const cvh::Range& range) {
            for (int block_idx = range.start; block_idx < range.end; ++block_idx)
            {
                const int row0 = block_idx * tile;
                fn(row0);
            }
        },
        static_cast<double>(row_blocks));
}

template<typename T>
void transpose2d_tiled(const unsigned char* src_raw, unsigned char* dst_raw, int rows, int cols)
{
    const T* src = reinterpret_cast<const T*>(src_raw);
    T* dst = reinterpret_cast<T*>(dst_raw);

    constexpr int TILE = 32;
    for_each_row_block(rows, cols, TILE, [&](int row0) {
        const int row1 = std::min(row0 + TILE, rows);
        for (int col0 = 0; col0 < cols; col0 += TILE)
        {
            const int col1 = std::min(col0 + TILE, cols);
            for (int row = row0; row < row1; ++row)
            {
                for (int col = col0; col < col1; ++col)
                {
                    dst[static_cast<size_t>(col) * rows + row] = src[static_cast<size_t>(row) * cols + col];
                }
            }
        }
    });
}


template<size_t Bytes>
struct FixedPixel
{
    unsigned char data[Bytes];
};

inline void transpose2d_memcpy_fallback(const unsigned char* src,
                                        unsigned char* dst,
                                        int rows,
                                        int cols,
                                        size_t elem_size)
{
    constexpr int TILE = 32;
    for_each_row_block(rows, cols, TILE, [&](int row0) {
        const int row1 = std::min(row0 + TILE, rows);
        for (int col0 = 0; col0 < cols; col0 += TILE)
        {
            const int col1 = std::min(col0 + TILE, cols);
            for (int row = row0; row < row1; ++row)
            {
                for (int col = col0; col < col1; ++col)
                {
                    std::memcpy(dst + (static_cast<size_t>(col) * rows + row) * elem_size,
                                src + (static_cast<size_t>(row) * cols + col) * elem_size,
                                elem_size);
                }
            }
        }
    });
}

}  // namespace

inline void transpose2d_kernel_blocked(const unsigned char* src,
                                       unsigned char* dst,
                                       int rows,
                                       int cols,
                                       size_t elem_size1,
                                       int channels)
{
    if (rows <= 0 || cols <= 0 || elem_size1 == 0 || channels <= 0)
    {
        return;
    }

    const size_t elem_size = elem_size1 * static_cast<size_t>(channels);
    set_last_dispatch_tag(DispatchTag::Scalar);

#if CVH_ENABLE_OPENCV_INTRIN && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    if (dispatch_mode() != DispatchMode::ScalarOnly)
    {
        if (elem_size == 1 && rows >= 16 && cols >= 16)
        {
            transpose_u8_ui(src, dst, rows, cols);
            set_last_dispatch_tag(DispatchTag::OpenCVUI);
            return;
        }
        if (elem_size == 2 && rows >= 8 && cols >= 8)
        {
            transpose_u16_ui(
                reinterpret_cast<const ushort*>(src),
                reinterpret_cast<ushort*>(dst),
                rows,
                cols);
            set_last_dispatch_tag(DispatchTag::OpenCVUI);
            return;
        }
        if (elem_size == 4 && rows >= 4 && cols >= 4)
        {
            transpose_u32_ui(
                reinterpret_cast<const uint*>(src),
                reinterpret_cast<uint*>(dst),
                rows,
                cols);
            set_last_dispatch_tag(DispatchTag::OpenCVUI);
            return;
        }
    }
#endif

    // Fixed-size pixel fallback avoids per-element memcpy call overhead for
    // common multi-channel layouts not representable as 1/2/4/8-byte lanes.
    switch (elem_size)
    {
        case 3:
            transpose2d_tiled<FixedPixel<3>>(src, dst, rows, cols);
            return;
        case 1:
            transpose2d_tiled<FixedPixel<1>>(src, dst, rows, cols);
            return;
        case 2:
            transpose2d_tiled<FixedPixel<2>>(src, dst, rows, cols);
            return;
        case 4:
            transpose2d_tiled<FixedPixel<4>>(src, dst, rows, cols);
            return;
        case 8:
            transpose2d_tiled<FixedPixel<8>>(src, dst, rows, cols);
            return;
        case 6:
            transpose2d_tiled<FixedPixel<6>>(src, dst, rows, cols);
            return;
        case 12:
            transpose2d_tiled<FixedPixel<12>>(src, dst, rows, cols);
            return;
        case 16:
            transpose2d_tiled<FixedPixel<16>>(src, dst, rows, cols);
            return;
        default:
            transpose2d_memcpy_fallback(src, dst, rows, cols, elem_size);
            return;
    }
}

}  // namespace cpu
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_TRANSPOSE_KERNEL_HPP
