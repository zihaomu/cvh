#pragma once

#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <vector>

using namespace cvh;

namespace
{

int normalize_border_type(int borderType)
{
    return borderType & (~BORDER_ISOLATED);
}

int border_interpolate_ref(int p, int len, int borderType)
{
    if (static_cast<unsigned>(p) < static_cast<unsigned>(len))
    {
        return p;
    }

    if (borderType == BORDER_CONSTANT)
    {
        return -1;
    }
    if (borderType == BORDER_REPLICATE)
    {
        return p < 0 ? 0 : (len - 1);
    }
    if (borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101)
    {
        if (len == 1)
        {
            return 0;
        }
        const int delta = borderType == BORDER_REFLECT_101 ? 1 : 0;
        while (p < 0 || p >= len)
        {
            if (p < 0)
            {
                p = -p - 1 + delta;
            }
            else
            {
                p = len - 1 - (p - len) - delta;
            }
        }
        return p;
    }
    return -1;
}

void fill_u8_pattern(Mat& src)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.dims == 2);
    const int rows = src.size[0];
    const int cols = src.size[1];
    const int cn = src.channels();

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            for (int c = 0; c < cn; ++c)
            {
                src.at<uchar>(y, x, c) = static_cast<uchar>((y * 31 + x * 17 + c * 13) & 0xFF);
            }
        }
    }
}

Mat morph_reference_u8(const Mat& src, bool is_erode, int borderType, const Scalar& borderValue)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.dims == 2);
    const int rows = src.size[0];
    const int cols = src.size[1];
    const int cn = src.channels();
    const size_t src_step = src.step(0);

    Mat dst({rows, cols}, src.type());
    const size_t dst_step = dst.step(0);
    const int border = normalize_border_type(borderType);

    for (int y = 0; y < rows; ++y)
    {
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            uchar* out_px = dst_row + static_cast<size_t>(x) * cn;
            for (int c = 0; c < cn; ++c)
            {
                int best = is_erode ? 255 : 0;
                for (int ky = -1; ky <= 1; ++ky)
                {
                    for (int kx = -1; kx <= 1; ++kx)
                    {
                        const int sy = border_interpolate_ref(y + ky, rows, border);
                        const int sx = border_interpolate_ref(x + kx, cols, border);
                        int value = 0;
                        if (sy < 0 || sx < 0)
                        {
                            value = saturate_cast<uchar>(borderValue.val[c]);
                        }
                        else
                        {
                            const uchar* src_row = src.data + static_cast<size_t>(sy) * src_step;
                            value = src_row[static_cast<size_t>(sx) * cn + c];
                        }

                        if (is_erode)
                        {
                            best = std::min(best, value);
                        }
                        else
                        {
                            best = std::max(best, value);
                        }
                    }
                }
                out_px[c] = static_cast<uchar>(best);
            }
        }
    }

    return dst;
}

Mat sobel_reference_u8_to_f32(const Mat& src, int dx, int dy, int borderType)
{
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.dims == 2);
    CV_Assert((dx == 1 && dy == 0) || (dx == 0 && dy == 1));

    const int rows = src.size[0];
    const int cols = src.size[1];
    const int cn = src.channels();
    const size_t src_step = src.step(0);
    const int border = normalize_border_type(borderType);

    Mat dst({rows, cols}, CV_MAKETYPE(CV_32F, cn));
    const size_t dst_step = dst.step(0);

    for (int y = 0; y < rows; ++y)
    {
        float* dst_row = reinterpret_cast<float*>(dst.data + static_cast<size_t>(y) * dst_step);
        for (int x = 0; x < cols; ++x)
        {
            const int y0 = border_interpolate_ref(y - 1, rows, border);
            const int y1 = border_interpolate_ref(y, rows, border);
            const int y2 = border_interpolate_ref(y + 1, rows, border);
            const int x0 = border_interpolate_ref(x - 1, cols, border);
            const int x1 = border_interpolate_ref(x, cols, border);
            const int x2 = border_interpolate_ref(x + 1, cols, border);

            for (int c = 0; c < cn; ++c)
            {
                const auto sample = [&](int sy, int sx) -> float {
                    const uchar* src_row = src.data + static_cast<size_t>(sy) * src_step;
                    return static_cast<float>(src_row[static_cast<size_t>(sx) * cn + c]);
                };

                float value = 0.0f;
                if (dx == 1)
                {
                    value = (sample(y0, x2) + 2.0f * sample(y1, x2) + sample(y2, x2)) -
                            (sample(y0, x0) + 2.0f * sample(y1, x0) + sample(y2, x0));
                }
                else
                {
                    value = (sample(y2, x0) + 2.0f * sample(y2, x1) + sample(y2, x2)) -
                            (sample(y0, x0) + 2.0f * sample(y0, x1) + sample(y0, x2));
                }
                dst_row[static_cast<size_t>(x) * cn + c] = value;
            }
        }
    }

    return dst;
}

int max_abs_diff_u8(const Mat& a, const Mat& b)
{
    CV_Assert(a.type() == b.type());
    CV_Assert(a.total() == b.total());
    CV_Assert(a.channels() == b.channels());
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    int max_diff = 0;
    for (size_t i = 0; i < count; ++i)
    {
        max_diff = std::max(max_diff, std::abs(static_cast<int>(a.data[i]) - static_cast<int>(b.data[i])));
    }
    return max_diff;
}

float max_abs_diff_f32(const Mat& a, const Mat& b)
{
    CV_Assert(a.type() == b.type());
    CV_Assert(a.total() == b.total());
    CV_Assert(a.channels() == b.channels());
    const float* pa = reinterpret_cast<const float*>(a.data);
    const float* pb = reinterpret_cast<const float*>(b.data);
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        max_diff = std::max(max_diff, std::fabs(pa[i] - pb[i]));
    }
    return max_diff;
}

int max_abs_diff_s16(const Mat& a, const Mat& b)
{
    CV_Assert(a.type() == b.type());
    CV_Assert(a.total() == b.total());
    CV_Assert(a.channels() == b.channels());
    CV_Assert(a.depth() == CV_16S);
    const short* pa = reinterpret_cast<const short*>(a.data);
    const short* pb = reinterpret_cast<const short*>(b.data);
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    int max_diff = 0;
    for (size_t i = 0; i < count; ++i)
    {
        max_diff = std::max(max_diff, std::abs(static_cast<int>(pa[i]) - static_cast<int>(pb[i])));
    }
    return max_diff;
}

std::uint32_t lcg_next(std::uint32_t state)
{
    return state * 1664525u + 1013904223u;
}

void fill_u8_lcg(Mat& src, std::uint32_t seed)
{
    CV_Assert(src.depth() == CV_8U);
    const size_t count = src.total() * static_cast<size_t>(src.channels());
    for (size_t i = 0; i < count; ++i)
    {
        seed = lcg_next(seed);
        src.data[i] = static_cast<uchar>((seed >> 24) & 0xFFu);
    }
}

}  // namespace
