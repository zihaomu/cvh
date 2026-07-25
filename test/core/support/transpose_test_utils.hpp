#pragma once

#include "cvh.h"
#include "gtest/gtest.h"

using namespace cvh;

namespace
{

void fill_with_byte_pattern(Mat& matrix)
{
    const size_t byte_count =
        matrix.total() *
        static_cast<size_t>(CV_ELEM_SIZE(matrix.type()));
    for (size_t index = 0; index < byte_count; ++index)
    {
        matrix.data[index] =
            static_cast<uchar>((index * 131u + 17u) & 0xFFu);
    }
}

void expect_transpose2d_bytes_equal(
    const Mat& source,
    const Mat& destination)
{
    ASSERT_EQ(source.dims, 2);
    ASSERT_EQ(destination.dims, 2);
    ASSERT_EQ(destination.type(), source.type());
    ASSERT_EQ(destination.size[0], source.size[1]);
    ASSERT_EQ(destination.size[1], source.size[0]);

    const int rows = source.size[0];
    const int cols = source.size[1];
    const size_t element_bytes =
        static_cast<size_t>(CV_ELEM_SIZE(source.type()));
    const size_t source_row_bytes =
        static_cast<size_t>(cols) * element_bytes;
    const size_t destination_row_bytes =
        static_cast<size_t>(rows) * element_bytes;

    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            const size_t source_offset =
                static_cast<size_t>(row) * source_row_bytes +
                static_cast<size_t>(col) * element_bytes;
            const size_t destination_offset =
                static_cast<size_t>(col) * destination_row_bytes +
                static_cast<size_t>(row) * element_bytes;
            for (size_t byte = 0; byte < element_bytes; ++byte)
            {
                ASSERT_EQ(
                    destination.data[destination_offset + byte],
                    source.data[source_offset + byte])
                    << "row=" << row
                    << ", col=" << col
                    << ", byte=" << byte;
            }
        }
    }
}

void expect_transpose_last2_3d_bytes_equal(
    const Mat& source,
    const Mat& destination)
{
    ASSERT_EQ(source.dims, 3);
    ASSERT_EQ(destination.dims, 3);
    ASSERT_EQ(destination.type(), source.type());
    ASSERT_EQ(destination.size[0], source.size[0]);
    ASSERT_EQ(destination.size[1], source.size[2]);
    ASSERT_EQ(destination.size[2], source.size[1]);

    const int batch = source.size[0];
    const int rows = source.size[1];
    const int cols = source.size[2];
    const size_t element_bytes =
        static_cast<size_t>(CV_ELEM_SIZE(source.type()));
    const size_t plane_bytes =
        static_cast<size_t>(rows) *
        static_cast<size_t>(cols) *
        element_bytes;

    for (int batch_index = 0;
         batch_index < batch;
         ++batch_index)
    {
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
            {
                const size_t source_offset =
                    static_cast<size_t>(batch_index) * plane_bytes +
                    (static_cast<size_t>(row) *
                         static_cast<size_t>(cols) +
                     static_cast<size_t>(col)) *
                        element_bytes;
                const size_t destination_offset =
                    static_cast<size_t>(batch_index) * plane_bytes +
                    (static_cast<size_t>(col) *
                         static_cast<size_t>(rows) +
                     static_cast<size_t>(row)) *
                        element_bytes;
                for (size_t byte = 0;
                     byte < element_bytes;
                     ++byte)
                {
                    ASSERT_EQ(
                        destination.data[destination_offset + byte],
                        source.data[source_offset + byte])
                        << "batch=" << batch_index
                        << ", row=" << row
                        << ", col=" << col
                        << ", byte=" << byte;
                }
            }
        }
    }
}

}  // namespace
