#pragma once

#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstring>

using namespace cvh;

namespace {

void fill_u8(Mat& matrix)
{
    for (int row = 0; row < matrix.size[0]; ++row)
    {
        for (int col = 0; col < matrix.size[1]; ++col)
        {
            for (int channel = 0; channel < matrix.channels(); ++channel)
            {
                matrix.at<uchar>(row, col, channel) =
                    static_cast<uchar>(
                        (row * 37 + col * 19 + channel * 53) & 255);
            }
        }
    }
}

void make_maps(Mat& map_x, Mat& map_y, int rows, int cols)
{
    map_x.create({rows, cols}, CV_32FC1);
    map_y.create({rows, cols}, CV_32FC1);
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            map_x.at<float>(row, col) =
                static_cast<float>(col) + 0.28125f;
            map_y.at<float>(row, col) =
                static_cast<float>(row) - 0.34375f;
        }
    }
}

int max_u8_difference(const Mat& first, const Mat& second)
{
    EXPECT_EQ(first.type(), second.type());
    EXPECT_EQ(first.shape(), second.shape());
    int maximum = 0;
    for (int row = 0; row < first.size[0]; ++row)
    {
        const uchar* first_row =
            first.data + static_cast<size_t>(row) * first.step(0);
        const uchar* second_row =
            second.data + static_cast<size_t>(row) * second.step(0);
        const size_t count =
            static_cast<size_t>(first.size[1]) * first.channels();
        for (size_t index = 0; index < count; ++index)
        {
            maximum = std::max(
                maximum,
                std::abs(
                    static_cast<int>(first_row[index]) -
                    static_cast<int>(second_row[index])));
        }
    }
    return maximum;
}

Mat identity_perspective(int depth)
{
    Mat matrix({3, 3}, CV_MAKETYPE(depth, 1));
    matrix.setTo(Scalar::all(0.0));
    if (depth == CV_32F)
    {
        matrix.at<float>(0, 0) = 1.0f;
        matrix.at<float>(1, 1) = 1.0f;
        matrix.at<float>(2, 2) = 1.0f;
    }
    else
    {
        matrix.at<double>(0, 0) = 1.0;
        matrix.at<double>(1, 1) = 1.0;
        matrix.at<double>(2, 2) = 1.0;
    }
    return matrix;
}

}  // namespace
