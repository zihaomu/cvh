#pragma once

#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace cvh;

namespace
{

double l1_u8(const Mat& src)
{
    const size_t count = src.total() * static_cast<size_t>(src.channels());
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        sum += static_cast<double>(src.data[i]);
    }
    return sum;
}

int max_abs_diff_u8(const Mat& a, const Mat& b)
{
    if (a.type() != b.type() || a.size[0] != b.size[0] || a.size[1] != b.size[1])
    {
        return 255;
    }
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    int max_diff = 0;
    for (size_t i = 0; i < count; ++i)
    {
        const int diff = std::abs(static_cast<int>(a.data[i]) - static_cast<int>(b.data[i]));
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }
    return max_diff;
}

float max_abs_diff_f32(const Mat& a, const Mat& b)
{
    if (a.type() != b.type() || a.size[0] != b.size[0] || a.size[1] != b.size[1])
    {
        return 1e9f;
    }
    CV_Assert(a.depth() == CV_32F);
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    const float* pa = reinterpret_cast<const float*>(a.data);
    const float* pb = reinterpret_cast<const float*>(b.data);
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        const float diff = std::abs(pa[i] - pb[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }
    return max_diff;
}

Mat mat_u8(int rows, int cols, const std::vector<int>& values)
{
    CV_Assert(static_cast<int>(values.size()) == rows * cols);
    Mat out({rows, cols}, CV_8UC1);
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            out.at<uchar>(y, x) = static_cast<uchar>(values[static_cast<size_t>(y * cols + x)]);
        }
    }
    return out;
}

Mat transpose_u8(const Mat& src)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert(src.dims == 2);

    Mat out({src.size[1], src.size[0]}, src.type());
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<uchar>(x, y) = src.at<uchar>(y, x);
        }
    }
    return out;
}

Mat resize_reference_linear_u8(const Mat& src, Size dsize, double fx, double fy)
{
    CV_Assert(!src.empty());
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = dsize.width > 0 ? dsize.width : std::max(1, static_cast<int>(std::lround(src_cols * fx)));
    const int dst_rows = dsize.height > 0 ? dsize.height : std::max(1, static_cast<int>(std::lround(src_rows * fy)));
    CV_Assert(dst_cols > 0 && dst_rows > 0);

    Mat out({dst_rows, dst_cols}, src.type());
    const int cn = src.channels();

    const float scale_x = static_cast<float>(src_cols) / static_cast<float>(dst_cols);
    const float scale_y = static_cast<float>(src_rows) / static_cast<float>(dst_rows);

    for (int y = 0; y < dst_rows; ++y)
    {
        const float fy_src = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(fy_src)), 0, src_rows - 1);
        const int y1 = std::min(y0 + 1, src_rows - 1);
        const float wy = fy_src - static_cast<float>(y0);

        for (int x = 0; x < dst_cols; ++x)
        {
            const float fx_src = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(fx_src)), 0, src_cols - 1);
            const int x1 = std::min(x0 + 1, src_cols - 1);
            const float wx = fx_src - static_cast<float>(x0);

            for (int c = 0; c < cn; ++c)
            {
                const float p00 = static_cast<float>(src.at<uchar>(y0, x0, c));
                const float p01 = static_cast<float>(src.at<uchar>(y0, x1, c));
                const float p10 = static_cast<float>(src.at<uchar>(y1, x0, c));
                const float p11 = static_cast<float>(src.at<uchar>(y1, x1, c));

                const float top = p00 + (p01 - p00) * wx;
                const float bot = p10 + (p11 - p10) * wx;
                out.at<uchar>(y, x, c) = saturate_cast<uchar>(top + (bot - top) * wy);
            }
        }
    }

    return out;
}

Mat resize_reference_nearest_u8(const Mat& src, Size dsize, double fx, double fy)
{
    CV_Assert(!src.empty());
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = dsize.width > 0 ? dsize.width : std::max(1, static_cast<int>(std::lround(src_cols * fx)));
    const int dst_rows = dsize.height > 0 ? dsize.height : std::max(1, static_cast<int>(std::lround(src_rows * fy)));
    CV_Assert(dst_cols > 0 && dst_rows > 0);

    Mat out({dst_rows, dst_cols}, src.type());
    const int cn = src.channels();
    for (int y = 0; y < dst_rows; ++y)
    {
        const int sy = std::min(src_rows - 1, (y * src_rows) / dst_rows);
        for (int x = 0; x < dst_cols; ++x)
        {
            const int sx = std::min(src_cols - 1, (x * src_cols) / dst_cols);
            for (int c = 0; c < cn; ++c)
            {
                out.at<uchar>(y, x, c) = src.at<uchar>(sy, sx, c);
            }
        }
    }
    return out;
}

Mat resize_reference_nearest_exact_u8(const Mat& src, Size dsize, double fx, double fy)
{
    CV_Assert(!src.empty());
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = dsize.width > 0 ? dsize.width : std::max(1, static_cast<int>(std::lround(src_cols * fx)));
    const int dst_rows = dsize.height > 0 ? dsize.height : std::max(1, static_cast<int>(std::lround(src_rows * fy)));
    CV_Assert(dst_cols > 0 && dst_rows > 0);

    Mat out({dst_rows, dst_cols}, src.type());
    const int cn = src.channels();
    const int64_t ifx = ((static_cast<int64_t>(src_cols) << 16) + dst_cols / 2) / dst_cols;
    const int64_t ifx0 = ifx / 2 - (src_cols % 2);
    const int64_t ify = ((static_cast<int64_t>(src_rows) << 16) + dst_rows / 2) / dst_rows;
    const int64_t ify0 = ify / 2 - (src_rows % 2);

    for (int y = 0; y < dst_rows; ++y)
    {
        const int sy = std::clamp(static_cast<int>((ify * y + ify0) >> 16), 0, src_rows - 1);
        for (int x = 0; x < dst_cols; ++x)
        {
            const int sx = std::clamp(static_cast<int>((ifx * x + ifx0) >> 16), 0, src_cols - 1);
            for (int c = 0; c < cn; ++c)
            {
                out.at<uchar>(y, x, c) = src.at<uchar>(sy, sx, c);
            }
        }
    }
    return out;
}

Mat resize_reference_linear_f32(const Mat& src, Size dsize, double fx, double fy)
{
    CV_Assert(!src.empty());
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_32F);

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = dsize.width > 0 ? dsize.width : std::max(1, static_cast<int>(std::lround(src_cols * fx)));
    const int dst_rows = dsize.height > 0 ? dsize.height : std::max(1, static_cast<int>(std::lround(src_rows * fy)));
    CV_Assert(dst_cols > 0 && dst_rows > 0);

    Mat out({dst_rows, dst_cols}, src.type());
    const int cn = src.channels();

    const float scale_x = static_cast<float>(src_cols) / static_cast<float>(dst_cols);
    const float scale_y = static_cast<float>(src_rows) / static_cast<float>(dst_rows);

    for (int y = 0; y < dst_rows; ++y)
    {
        const float fy_src = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(fy_src)), 0, src_rows - 1);
        const int y1 = std::min(y0 + 1, src_rows - 1);
        const float wy = fy_src - static_cast<float>(y0);

        for (int x = 0; x < dst_cols; ++x)
        {
            const float fx_src = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(fx_src)), 0, src_cols - 1);
            const int x1 = std::min(x0 + 1, src_cols - 1);
            const float wx = fx_src - static_cast<float>(x0);

            for (int c = 0; c < cn; ++c)
            {
                const float p00 = src.at<float>(y0, x0, c);
                const float p01 = src.at<float>(y0, x1, c);
                const float p10 = src.at<float>(y1, x0, c);
                const float p11 = src.at<float>(y1, x1, c);

                const float top = p00 + (p01 - p00) * wx;
                const float bot = p10 + (p11 - p10) * wx;
                out.at<float>(y, x, c) = top + (bot - top) * wy;
            }
        }
    }

    return out;
}

Mat resize_reference_nearest_f32(const Mat& src, Size dsize, double fx, double fy)
{
    CV_Assert(!src.empty());
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_32F);

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = dsize.width > 0 ? dsize.width : std::max(1, static_cast<int>(std::lround(src_cols * fx)));
    const int dst_rows = dsize.height > 0 ? dsize.height : std::max(1, static_cast<int>(std::lround(src_rows * fy)));
    CV_Assert(dst_cols > 0 && dst_rows > 0);

    Mat out({dst_rows, dst_cols}, src.type());
    const int cn = src.channels();
    for (int y = 0; y < dst_rows; ++y)
    {
        const int sy = std::min(src_rows - 1, (y * src_rows) / dst_rows);
        for (int x = 0; x < dst_cols; ++x)
        {
            const int sx = std::min(src_cols - 1, (x * src_cols) / dst_cols);
            for (int c = 0; c < cn; ++c)
            {
                out.at<float>(y, x, c) = src.at<float>(sy, sx, c);
            }
        }
    }
    return out;
}

Mat resize_reference_nearest_exact_f32(const Mat& src, Size dsize, double fx, double fy)
{
    CV_Assert(!src.empty());
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_32F);

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = dsize.width > 0 ? dsize.width : std::max(1, static_cast<int>(std::lround(src_cols * fx)));
    const int dst_rows = dsize.height > 0 ? dsize.height : std::max(1, static_cast<int>(std::lround(src_rows * fy)));
    CV_Assert(dst_cols > 0 && dst_rows > 0);

    Mat out({dst_rows, dst_cols}, src.type());
    const int cn = src.channels();
    const int64_t ifx = ((static_cast<int64_t>(src_cols) << 16) + dst_cols / 2) / dst_cols;
    const int64_t ifx0 = ifx / 2 - (src_cols % 2);
    const int64_t ify = ((static_cast<int64_t>(src_rows) << 16) + dst_rows / 2) / dst_rows;
    const int64_t ify0 = ify / 2 - (src_rows % 2);

    for (int y = 0; y < dst_rows; ++y)
    {
        const int sy = std::clamp(static_cast<int>((ify * y + ify0) >> 16), 0, src_rows - 1);
        for (int x = 0; x < dst_cols; ++x)
        {
            const int sx = std::clamp(static_cast<int>((ifx * x + ifx0) >> 16), 0, src_cols - 1);
            for (int c = 0; c < cn; ++c)
            {
                out.at<float>(y, x, c) = src.at<float>(sy, sx, c);
            }
        }
    }
    return out;
}

}  // namespace
