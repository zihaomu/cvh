#pragma once

#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace cvh;

namespace
{

void fill_pattern(Mat& mat)
{
    for (int y = 0; y < mat.size.p[0]; ++y)
    {
        for (int x = 0; x < mat.size.p[1]; ++x)
        {
            for (int ch = 0; ch < mat.channels(); ++ch)
            {
                mat.at<uchar>(y, x, ch) =
                    static_cast<uchar>((17 * y + 23 * x + 31 * ch) & 255);
            }
        }
    }
}

Mat naive_stack_blur_u8(const Mat& src, Size ksize)
{
    Mat dst(src.shape(), src.type());
    const int radius_x = ksize.width / 2;
    const int radius_y = ksize.height / 2;
    const std::int64_t divisor =
        static_cast<std::int64_t>(radius_x + 1) *
        static_cast<std::int64_t>(radius_x + 1) *
        static_cast<std::int64_t>(radius_y + 1) *
        static_cast<std::int64_t>(radius_y + 1);
    for (int y = 0; y < src.size.p[0]; ++y)
    {
        for (int x = 0; x < src.size.p[1]; ++x)
        {
            for (int channel = 0;
                 channel < src.channels();
                 ++channel)
            {
                std::int64_t sum = 0;
                for (int ky = -radius_y;
                     ky <= radius_y;
                     ++ky)
                {
                    const int source_y =
                        std::clamp(
                            y + ky, 0, src.size.p[0] - 1);
                    const int weight_y =
                        radius_y + 1 - std::abs(ky);
                    for (int kx = -radius_x;
                         kx <= radius_x;
                         ++kx)
                    {
                        const int source_x =
                            std::clamp(
                                x + kx, 0, src.size.p[1] - 1);
                        const int weight_x =
                            radius_x + 1 - std::abs(kx);
                        sum +=
                            static_cast<std::int64_t>(
                                weight_x * weight_y) *
                            src.at<uchar>(
                                source_y,
                                source_x,
                                channel);
                    }
                }
                dst.at<uchar>(y, x, channel) =
                    saturate_cast<uchar>(
                        static_cast<double>(sum) /
                        static_cast<double>(divisor));
            }
        }
    }
    return dst;
}

}  // namespace
