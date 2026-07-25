#pragma once

#include "test/support/dispatch_mode_guard.hpp"
#include "cvh.h"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/detail/math_ui.hpp"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace cvh;

namespace
{

using DispatchModeGuard = cvh::test::DispatchModeGuard;

void expect_mat_bytes_equal(const Mat& actual, const Mat& expected)
{
    ASSERT_EQ(actual.type(), expected.type());
    ASSERT_EQ(actual.shape(), expected.shape());
    const size_t row_bytes =
        static_cast<size_t>(actual.size[1]) * actual.elemSize();
    for (int row = 0; row < actual.size[0]; ++row)
    {
        const uchar* actual_row =
            actual.data + static_cast<size_t>(row) * actual.step(0);
        const uchar* expected_row =
            expected.data + static_cast<size_t>(row) * expected.step(0);
        for (size_t byte = 0; byte < row_bytes; ++byte)
        {
            ASSERT_EQ(actual_row[byte], expected_row[byte])
                << "row=" << row << ", byte=" << byte;
        }
    }
}

void expect_float_mat_near(const Mat& actual,
                           const Mat& expected,
                           float relative_tolerance)
{
    ASSERT_EQ(actual.type(), expected.type());
    ASSERT_EQ(actual.shape(), expected.shape());
    ASSERT_EQ(actual.depth(), CV_32F);
    const size_t row_scalars =
        static_cast<size_t>(actual.size[1]) *
        static_cast<size_t>(actual.channels());
    for (int row = 0; row < actual.size[0]; ++row)
    {
        const float* actual_row = reinterpret_cast<const float*>(
            actual.data + static_cast<size_t>(row) * actual.step(0));
        const float* expected_row = reinterpret_cast<const float*>(
            expected.data + static_cast<size_t>(row) * expected.step(0));
        for (size_t index = 0; index < row_scalars; ++index)
        {
            const float actual_value = actual_row[index];
            const float expected_value = expected_row[index];
            if (std::isnan(expected_value))
            {
                EXPECT_TRUE(std::isnan(actual_value))
                    << "row=" << row << ", index=" << index;
            }
            else if (std::isinf(expected_value))
            {
                EXPECT_EQ(actual_value, expected_value)
                    << "row=" << row << ", index=" << index;
            }
            else
            {
                const float tolerance =
                    relative_tolerance * std::max(1.0f, std::fabs(expected_value));
                EXPECT_NEAR(actual_value, expected_value, tolerance)
                    << "row=" << row << ", index=" << index;
            }
        }
    }
}

std::uint16_t read_half_bits(const Mat& mat, int x)
{
    std::uint16_t bits = 0;
    std::memcpy(&bits, &mat.at<short>(0, x), sizeof(bits));
    return bits;
}

}  // namespace
