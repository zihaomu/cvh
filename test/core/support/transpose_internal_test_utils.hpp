#pragma once

#include "test/support/dispatch_mode_guard.hpp"
#include "cvh.h"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/detail/transpose_kernel.hpp"
#include "gtest/gtest.h"

#include <cstring>
#include <vector>

using namespace cvh;

namespace
{

using DispatchModeGuard = cvh::test::DispatchModeGuard;

void fill_with_byte_pattern(std::vector<uchar>& bytes)
{
    for (size_t index = 0; index < bytes.size(); ++index)
    {
        bytes[index] =
            static_cast<uchar>((index * 131u + 17u) & 0xFFu);
    }
}

void transpose2d_reference_bytes(
    const uchar* source,
    uchar* destination,
    int rows,
    int cols,
    size_t element_size)
{
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            std::memcpy(
                destination +
                    (static_cast<size_t>(col) * rows + row) *
                        element_size,
                source +
                    (static_cast<size_t>(row) * cols + col) *
                        element_size,
                element_size);
        }
    }
}

void expect_transpose_kernel_bytes_equal(
    int rows,
    int cols,
    size_t element_size,
    cpu::DispatchMode mode)
{
    const size_t byte_count =
        static_cast<size_t>(rows) *
        static_cast<size_t>(cols) *
        element_size;
    std::vector<uchar> source(byte_count);
    std::vector<uchar> destination(byte_count);
    std::vector<uchar> expected(byte_count);
    fill_with_byte_pattern(source);
    transpose2d_reference_bytes(
        source.data(),
        expected.data(),
        rows,
        cols,
        element_size);

    DispatchModeGuard guard(mode);
    cpu::reset_last_dispatch_tag();
    cpu::transpose2d_kernel_blocked(
        source.data(),
        destination.data(),
        rows,
        cols,
        element_size,
        1);

    cpu::DispatchTag expected_tag = cpu::DispatchTag::Scalar;
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const size_t block =
        element_size == 1 ? 16 : (element_size == 2 ? 8 : 4);
    if (mode == cpu::DispatchMode::Auto &&
        element_size <= 4 &&
        rows >= static_cast<int>(block) &&
        cols >= static_cast<int>(block))
    {
        expected_tag = cpu::DispatchTag::OpenCVUI;
    }
#endif
    ASSERT_EQ(cpu::last_dispatch_tag(), expected_tag);
    ASSERT_EQ(destination, expected);
}

}  // namespace
