#pragma once

#include "cvh.h"
#include "gtest/gtest.h"

#include <cstring>
#include <vector>

using namespace cvh;

namespace
{

void fill_incrementing_bytes(Mat& mat, unsigned char start = 0)
{
    ASSERT_TRUE(mat.isContinuous());
    const size_t bytes = mat.total() * mat.elemSize();
    for (size_t i = 0; i < bytes; ++i)
    {
        mat.data[i] = static_cast<unsigned char>(start + i);
    }
}

void expect_same_bytes(const Mat& expected, const Mat& actual)
{
    ASSERT_EQ(actual.shape(), expected.shape());
    ASSERT_EQ(actual.type(), expected.type());
    ASSERT_EQ(expected.dims, 2);
    const size_t row_bytes =
        static_cast<size_t>(expected.size[1]) * expected.elemSize();
    for (int row = 0; row < expected.size[0]; ++row)
    {
        const uchar* expected_row =
            expected.data + static_cast<size_t>(row) * expected.step(0);
        const uchar* actual_row =
            actual.data + static_cast<size_t>(row) * actual.step(0);
        for (size_t byte = 0; byte < row_bytes; ++byte)
        {
            EXPECT_EQ(actual_row[byte], expected_row[byte])
                << "row=" << row << ", byte=" << byte;
        }
    }
}

}  // namespace
