#pragma once

#include "cvh.h"
#include "gtest/gtest.h"

#include <cstdlib>
#include <type_traits>

using namespace cvh;

namespace
{

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
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    float max_diff = 0.0f;
    const float* ap = reinterpret_cast<const float*>(a.data);
    const float* bp = reinterpret_cast<const float*>(b.data);
    for (size_t i = 0; i < count; ++i)
    {
        const float diff = std::abs(ap[i] - bp[i]);
        if (diff > max_diff)
        {
            max_diff = diff;
        }
    }
    return max_diff;
}

Mat color3_to_gray_reference_u8(const Mat& src, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);

    constexpr int kB = 7471;
    constexpr int kG = 38470;
    constexpr int kR = 19595;
    constexpr int kRound = 1 << 15;

    Mat out({src.size[0], src.size[1]}, CV_8UC1);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const int b = src.at<uchar>(y, x, rgb_order ? 2 : 0);
            const int g = src.at<uchar>(y, x, 1);
            const int r = src.at<uchar>(y, x, rgb_order ? 0 : 2);
            out.at<uchar>(y, x) = static_cast<uchar>((kB * b + kG * g + kR * r + kRound) >> 16);
        }
    }
    return out;
}

Mat bgr2gray_reference_u8(const Mat& src)
{
    return color3_to_gray_reference_u8(src, false);
}

Mat rgb2gray_reference_u8(const Mat& src)
{
    return color3_to_gray_reference_u8(src, true);
}

Mat color3_to_gray_reference_f32(const Mat& src, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_32F);
    CV_Assert(src.channels() == 3);

    Mat out({src.size[0], src.size[1]}, CV_32FC1);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const float b = src.at<float>(y, x, rgb_order ? 2 : 0);
            const float g = src.at<float>(y, x, 1);
            const float r = src.at<float>(y, x, rgb_order ? 0 : 2);
            out.at<float>(y, x) = 0.114f * b + 0.587f * g + 0.299f * r;
        }
    }
    return out;
}

Mat bgr2gray_reference_f32(const Mat& src)
{
    return color3_to_gray_reference_f32(src, false);
}

Mat rgb2gray_reference_f32(const Mat& src)
{
    return color3_to_gray_reference_f32(src, true);
}

Mat gray2bgr_reference_u8(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);

    Mat out({src.size[0], src.size[1]}, CV_8UC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const uchar g = src.at<uchar>(y, x);
            out.at<uchar>(y, x, 0) = g;
            out.at<uchar>(y, x, 1) = g;
            out.at<uchar>(y, x, 2) = g;
        }
    }
    return out;
}

Mat gray2bgr_reference_f32(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_32F);
    CV_Assert(src.channels() == 1);

    Mat out({src.size[0], src.size[1]}, CV_32FC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const float g = src.at<float>(y, x);
            out.at<float>(y, x, 0) = g;
            out.at<float>(y, x, 1) = g;
            out.at<float>(y, x, 2) = g;
        }
    }
    return out;
}

template <typename T>
Mat bgr2rgb_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 3);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 0);
        }
    }
    return out;
}

template <typename T>
Mat bgr2bgra_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 3);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;
    const T alpha = std::is_same_v<T, uchar> ? static_cast<T>(255) : static_cast<T>(1.0f);
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 0);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 3) = alpha;
        }
    }
    return out;
}

template <typename T>
Mat bgra2bgr_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 4);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 0);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 2);
        }
    }
    return out;
}

template <typename T>
Mat rgb2rgba_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 3);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;
    const T alpha = std::is_same_v<T, uchar> ? static_cast<T>(255) : static_cast<T>(1.0f);
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 0);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 3) = alpha;
        }
    }
    return out;
}

template <typename T>
Mat rgba2rgb_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 4);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 0);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 2);
        }
    }
    return out;
}

template <typename T>
Mat bgr2rgba_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 3);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;
    const T alpha = std::is_same_v<T, uchar> ? static_cast<T>(255) : static_cast<T>(1.0f);
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 0);
            out.at<T>(y, x, 3) = alpha;
        }
    }
    return out;
}

template <typename T>
Mat rgba2bgr_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 4);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 0);
        }
    }
    return out;
}

template <typename T>
Mat rgb2bgra_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 3);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;
    const T alpha = std::is_same_v<T, uchar> ? static_cast<T>(255) : static_cast<T>(1.0f);
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 0);
            out.at<T>(y, x, 3) = alpha;
        }
    }
    return out;
}

template <typename T>
Mat bgra2rgb_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 4);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 0);
        }
    }
    return out;
}

template <typename T>
Mat swap_rb_4ch_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 4);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            out.at<T>(y, x, 0) = src.at<T>(y, x, 2);
            out.at<T>(y, x, 1) = src.at<T>(y, x, 1);
            out.at<T>(y, x, 2) = src.at<T>(y, x, 0);
            out.at<T>(y, x, 3) = src.at<T>(y, x, 3);
        }
    }
    return out;
}

template <typename T>
Mat gray2bgra_reference(const Mat& src)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 1);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC4 : CV_32FC4;
    const T alpha = std::is_same_v<T, uchar> ? static_cast<T>(255) : static_cast<T>(1.0f);
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const T g = src.at<T>(y, x);
            out.at<T>(y, x, 0) = g;
            out.at<T>(y, x, 1) = g;
            out.at<T>(y, x, 2) = g;
            out.at<T>(y, x, 3) = alpha;
        }
    }
    return out;
}

template <typename T>
Mat color4_to_gray_reference(const Mat& src, bool rgba_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 4);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC1 : CV_32FC1;
    Mat out({src.size[0], src.size[1]}, out_type);

    if constexpr (std::is_same_v<T, uchar>)
    {
        constexpr int kB = 7471;
        constexpr int kG = 38470;
        constexpr int kR = 19595;
        constexpr int kRound = 1 << 15;

        for (int y = 0; y < src.size[0]; ++y)
        {
            for (int x = 0; x < src.size[1]; ++x)
            {
                const int b = src.at<uchar>(y, x, rgba_order ? 2 : 0);
                const int g = src.at<uchar>(y, x, 1);
                const int r = src.at<uchar>(y, x, rgba_order ? 0 : 2);
                out.at<uchar>(y, x) = static_cast<uchar>((kB * b + kG * g + kR * r + kRound) >> 16);
            }
        }
        return out;
    }

    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const float b = src.at<float>(y, x, rgba_order ? 2 : 0);
            const float g = src.at<float>(y, x, 1);
            const float r = src.at<float>(y, x, rgba_order ? 0 : 2);
            out.at<float>(y, x) = 0.114f * b + 0.587f * g + 0.299f * r;
        }
    }
    return out;
}

template <typename T>
constexpr float yuv_delta_reference()
{
    return std::is_same_v<T, uchar> ? 128.0f : 0.5f;
}

template <typename T>
Mat color3_to_yuv_reference(const Mat& src, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 3);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    const float delta = yuv_delta_reference<T>();
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const float r = static_cast<float>(src.at<T>(y, x, rgb_order ? 0 : 2));
            const float g = static_cast<float>(src.at<T>(y, x, 1));
            const float b = static_cast<float>(src.at<T>(y, x, rgb_order ? 2 : 0));
            const float yy = 0.299f * r + 0.587f * g + 0.114f * b;
            const float uu = 0.492f * (b - yy) + delta;
            const float vv = 0.877f * (r - yy) + delta;

            if constexpr (std::is_same_v<T, uchar>)
            {
                out.at<uchar>(y, x, 0) = saturate_cast<uchar>(yy);
                out.at<uchar>(y, x, 1) = saturate_cast<uchar>(uu);
                out.at<uchar>(y, x, 2) = saturate_cast<uchar>(vv);
            }
            else
            {
                out.at<float>(y, x, 0) = yy;
                out.at<float>(y, x, 1) = uu;
                out.at<float>(y, x, 2) = vv;
            }
        }
    }
    return out;
}

template <typename T>
Mat yuv_to_color3_reference(const Mat& src, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.channels() == 3);

    const int out_type = std::is_same_v<T, uchar> ? CV_8UC3 : CV_32FC3;
    const float delta = yuv_delta_reference<T>();
    Mat out({src.size[0], src.size[1]}, out_type);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            const float yy = static_cast<float>(src.at<T>(y, x, 0));
            const float uu = static_cast<float>(src.at<T>(y, x, 1)) - delta;
            const float vv = static_cast<float>(src.at<T>(y, x, 2)) - delta;

            const float b = yy + 2.032f * uu;
            const float g = yy - 0.395f * uu - 0.581f * vv;
            const float r = yy + 1.140f * vv;

            if constexpr (std::is_same_v<T, uchar>)
            {
                out.at<uchar>(y, x, rgb_order ? 0 : 2) = saturate_cast<uchar>(r);
                out.at<uchar>(y, x, 1) = saturate_cast<uchar>(g);
                out.at<uchar>(y, x, rgb_order ? 2 : 0) = saturate_cast<uchar>(b);
            }
            else
            {
                out.at<float>(y, x, rgb_order ? 0 : 2) = r;
                out.at<float>(y, x, 1) = g;
                out.at<float>(y, x, rgb_order ? 2 : 0) = b;
            }
        }
    }
    return out;
}

inline uchar color3_to_yuv_limited_u8(int bb, int gg, int rr, int channel);

inline uchar yuv420_limited_to_u8(int yy, int uu, int vv, int channel)
{
    const int c = std::max(yy - 16, 0);
    const int d = uu - 128;
    const int e = vv - 128;

    if (channel == 0)
    {
        return saturate_cast<uchar>((298 * c + 516 * d + 128) >> 8);
    }
    if (channel == 1)
    {
        return saturate_cast<uchar>((298 * c - 100 * d - 208 * e + 128) >> 8);
    }
    return saturate_cast<uchar>((298 * c + 409 * e + 128) >> 8);
}

Mat yuv420sp_to_color3_reference_u8(const Mat& src, bool nv21_layout, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert((src.size[0] % 3) == 0);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0] * 2 / 3;
    const int cols = src.size[1];
    Mat out({rows, cols}, CV_8UC3);

    for (int y = 0; y < rows; ++y)
    {
        const int uv_y = rows + y / 2;
        for (int x = 0; x < cols; ++x)
        {
            const int uv_x = x & ~1;
            const int yy = static_cast<int>(src.at<uchar>(y, x));
            const int first = static_cast<int>(src.at<uchar>(uv_y, uv_x + 0));
            const int second = static_cast<int>(src.at<uchar>(uv_y, uv_x + 1));
            const int uu = nv21_layout ? second : first;
            const int vv = nv21_layout ? first : second;

            const uchar b = yuv420_limited_to_u8(yy, uu, vv, 0);
            const uchar g = yuv420_limited_to_u8(yy, uu, vv, 1);
            const uchar r = yuv420_limited_to_u8(yy, uu, vv, 2);

            out.at<uchar>(y, x, rgb_order ? 0 : 2) = r;
            out.at<uchar>(y, x, 1) = g;
            out.at<uchar>(y, x, rgb_order ? 2 : 0) = b;
        }
    }

    return out;
}

Mat color3_to_yuv420sp_reference_u8(const Mat& src, bool rgb_order, bool nv21_layout)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);
    CV_Assert((src.size[0] % 2) == 0);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0];
    const int cols = src.size[1];
    Mat out({rows * 3 / 2, cols}, CV_8UC1);

    for (int y = 0; y < rows; y += 2)
    {
        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;

            for (int dy = 0; dy < 2; ++dy)
            {
                for (int dx = 0; dx < 2; ++dx)
                {
                    const int yy_y = y + dy;
                    const int yy_x = x + dx;
                    const int bb = static_cast<int>(src.at<uchar>(yy_y, yy_x, rgb_order ? 2 : 0));
                    const int gg = static_cast<int>(src.at<uchar>(yy_y, yy_x, 1));
                    const int rr = static_cast<int>(src.at<uchar>(yy_y, yy_x, rgb_order ? 0 : 2));
                    const uchar yy = color3_to_yuv_limited_u8(bb, gg, rr, 0);

                    out.at<uchar>(yy_y, yy_x) = yy;
                    sum_b += bb;
                    sum_g += gg;
                    sum_r += rr;
                }
            }

            const int avg_b = (sum_b + 2) >> 2;
            const int avg_g = (sum_g + 2) >> 2;
            const int avg_r = (sum_r + 2) >> 2;
            const uchar uu = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);
            const int uv_y = rows + y / 2;

            out.at<uchar>(uv_y, x + 0) = nv21_layout ? vv : uu;
            out.at<uchar>(uv_y, x + 1) = nv21_layout ? uu : vv;
        }
    }

    return out;
}

inline uchar yuv420p_plane_byte_at_u8(const Mat& src, int rows, int cols, int plane_offset, int plane_index)
{
    const int logical_offset = plane_offset + plane_index;
    return src.at<uchar>(rows + logical_offset / cols, logical_offset % cols);
}

inline void set_yuv420p_plane_byte_u8(Mat& dst, int rows, int cols, int plane_offset, int plane_index, uchar value)
{
    const int logical_offset = plane_offset + plane_index;
    dst.at<uchar>(rows + logical_offset / cols, logical_offset % cols) = value;
}

Mat color3_to_yuv420p_reference_u8(const Mat& src, bool rgb_order, bool yv12_layout)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);
    CV_Assert((src.size[0] % 2) == 0);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0];
    const int cols = src.size[1];
    const int uv_size = rows * cols / 4;
    const int u_plane_offset = yv12_layout ? uv_size : 0;
    const int v_plane_offset = yv12_layout ? 0 : uv_size;
    Mat out({rows * 3 / 2, cols}, CV_8UC1);

    for (int y = 0; y < rows; y += 2)
    {
        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;

            for (int dy = 0; dy < 2; ++dy)
            {
                for (int dx = 0; dx < 2; ++dx)
                {
                    const int yy_y = y + dy;
                    const int yy_x = x + dx;
                    const int bb = static_cast<int>(src.at<uchar>(yy_y, yy_x, rgb_order ? 2 : 0));
                    const int gg = static_cast<int>(src.at<uchar>(yy_y, yy_x, 1));
                    const int rr = static_cast<int>(src.at<uchar>(yy_y, yy_x, rgb_order ? 0 : 2));
                    const uchar yy = color3_to_yuv_limited_u8(bb, gg, rr, 0);

                    out.at<uchar>(yy_y, yy_x) = yy;
                    sum_b += bb;
                    sum_g += gg;
                    sum_r += rr;
                }
            }

            const int avg_b = (sum_b + 2) >> 2;
            const int avg_g = (sum_g + 2) >> 2;
            const int avg_r = (sum_r + 2) >> 2;
            const uchar uu = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);
            const int chroma_index = (y / 2) * (cols / 2) + (x / 2);

            set_yuv420p_plane_byte_u8(out, rows, cols, u_plane_offset, chroma_index, uu);
            set_yuv420p_plane_byte_u8(out, rows, cols, v_plane_offset, chroma_index, vv);
        }
    }

    return out;
}

inline uchar yuv444p_plane_byte_at_u8(const Mat& src, int rows, int cols, int plane_offset, int plane_index)
{
    const int logical_offset = plane_offset + plane_index;
    return src.at<uchar>(rows + logical_offset / cols, logical_offset % cols);
}

inline void set_yuv444p_plane_byte_u8(Mat& dst, int rows, int cols, int plane_offset, int plane_index, uchar value)
{
    const int logical_offset = plane_offset + plane_index;
    dst.at<uchar>(rows + logical_offset / cols, logical_offset % cols) = value;
}

Mat yuv420p_to_color3_reference_u8(const Mat& src, bool yv12_layout, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert((src.size[0] % 3) == 0);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0] * 2 / 3;
    const int cols = src.size[1];
    const int uv_size = rows * cols / 4;
    const int u_plane_offset = yv12_layout ? uv_size : 0;
    const int v_plane_offset = yv12_layout ? 0 : uv_size;

    Mat out({rows, cols}, CV_8UC3);
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(src.at<uchar>(y, x));
            const int chroma_index = (y / 2) * (cols / 2) + (x / 2);
            const int uu = static_cast<int>(yuv420p_plane_byte_at_u8(src, rows, cols, u_plane_offset, chroma_index));
            const int vv = static_cast<int>(yuv420p_plane_byte_at_u8(src, rows, cols, v_plane_offset, chroma_index));

            const uchar b = yuv420_limited_to_u8(yy, uu, vv, 0);
            const uchar g = yuv420_limited_to_u8(yy, uu, vv, 1);
            const uchar r = yuv420_limited_to_u8(yy, uu, vv, 2);

            out.at<uchar>(y, x, rgb_order ? 0 : 2) = r;
            out.at<uchar>(y, x, 1) = g;
            out.at<uchar>(y, x, rgb_order ? 2 : 0) = b;
        }
    }

    return out;
}

Mat yuv444p_to_color3_reference_u8(const Mat& src, bool yv24_layout, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert((src.size[0] % 3) == 0);

    const int rows = src.size[0] / 3;
    const int cols = src.size[1];
    const int plane_size = rows * cols;
    const int u_plane_offset = yv24_layout ? plane_size : 0;
    const int v_plane_offset = yv24_layout ? 0 : plane_size;

    Mat out({rows, cols}, CV_8UC3);
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(src.at<uchar>(y, x));
            const int chroma_index = y * cols + x;
            const int uu = static_cast<int>(yuv444p_plane_byte_at_u8(src, rows, cols, u_plane_offset, chroma_index));
            const int vv = static_cast<int>(yuv444p_plane_byte_at_u8(src, rows, cols, v_plane_offset, chroma_index));

            const uchar b = yuv420_limited_to_u8(yy, uu, vv, 0);
            const uchar g = yuv420_limited_to_u8(yy, uu, vv, 1);
            const uchar r = yuv420_limited_to_u8(yy, uu, vv, 2);

            out.at<uchar>(y, x, rgb_order ? 0 : 2) = r;
            out.at<uchar>(y, x, 1) = g;
            out.at<uchar>(y, x, rgb_order ? 2 : 0) = b;
        }
    }

    return out;
}

Mat yuv422sp_to_color3_reference_u8(const Mat& src, bool nv61_layout, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert((src.size[0] % 2) == 0);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0] / 2;
    const int cols = src.size[1];
    Mat out({rows, cols}, CV_8UC3);

    for (int y = 0; y < rows; ++y)
    {
        const int uv_y = rows + y;
        for (int x = 0; x < cols; ++x)
        {
            const int uv_x = x & ~1;
            const int yy = static_cast<int>(src.at<uchar>(y, x));
            const int first = static_cast<int>(src.at<uchar>(uv_y, uv_x + 0));
            const int second = static_cast<int>(src.at<uchar>(uv_y, uv_x + 1));
            const int uu = nv61_layout ? second : first;
            const int vv = nv61_layout ? first : second;

            const uchar b = yuv420_limited_to_u8(yy, uu, vv, 0);
            const uchar g = yuv420_limited_to_u8(yy, uu, vv, 1);
            const uchar r = yuv420_limited_to_u8(yy, uu, vv, 2);

            out.at<uchar>(y, x, rgb_order ? 0 : 2) = r;
            out.at<uchar>(y, x, 1) = g;
            out.at<uchar>(y, x, rgb_order ? 2 : 0) = b;
        }
    }

    return out;
}

inline uchar yuv444sp_plane_byte_at_u8(const Mat& src, int rows, int cols, int plane_index)
{
    return src.at<uchar>(rows + plane_index / cols, plane_index % cols);
}

inline void set_yuv444sp_plane_byte_u8(Mat& dst, int rows, int cols, int plane_index, uchar value)
{
    dst.at<uchar>(rows + plane_index / cols, plane_index % cols) = value;
}

inline uchar yuv422sp_plane_byte_at_u8(const Mat& src, int rows, int cols, int plane_index)
{
    return src.at<uchar>(rows + plane_index / cols, plane_index % cols);
}

inline void set_yuv422sp_plane_byte_u8(Mat& dst, int rows, int cols, int plane_index, uchar value)
{
    dst.at<uchar>(rows + plane_index / cols, plane_index % cols) = value;
}

inline uchar color3_to_yuv_limited_u8(int bb, int gg, int rr, int channel)
{
    if (channel == 0)
    {
        return saturate_cast<uchar>(((66 * rr + 129 * gg + 25 * bb + 128) >> 8) + 16);
    }
    if (channel == 1)
    {
        return saturate_cast<uchar>(((-38 * rr - 74 * gg + 112 * bb + 128) >> 8) + 128);
    }
    return saturate_cast<uchar>(((112 * rr - 94 * gg - 18 * bb + 128) >> 8) + 128);
}

Mat yuv444sp_to_color3_reference_u8(const Mat& src, bool nv42_layout, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 1);
    CV_Assert((src.size[0] % 3) == 0);

    const int rows = src.size[0] / 3;
    const int cols = src.size[1];
    Mat out({rows, cols}, CV_8UC3);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int yy = static_cast<int>(src.at<uchar>(y, x));
            const int base = y * (cols * 2) + x * 2;
            const int uu = static_cast<int>(yuv444sp_plane_byte_at_u8(src, rows, cols, base + (nv42_layout ? 1 : 0)));
            const int vv = static_cast<int>(yuv444sp_plane_byte_at_u8(src, rows, cols, base + (nv42_layout ? 0 : 1)));

            const uchar b = yuv420_limited_to_u8(yy, uu, vv, 0);
            const uchar g = yuv420_limited_to_u8(yy, uu, vv, 1);
            const uchar r = yuv420_limited_to_u8(yy, uu, vv, 2);

            out.at<uchar>(y, x, rgb_order ? 0 : 2) = r;
            out.at<uchar>(y, x, 1) = g;
            out.at<uchar>(y, x, rgb_order ? 2 : 0) = b;
        }
    }

    return out;
}

Mat color3_to_yuv444sp_reference_u8(const Mat& src, bool rgb_order, bool nv42_layout)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);

    const int rows = src.size[0];
    const int cols = src.size[1];
    Mat out({rows * 3, cols}, CV_8UC1);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int bb = static_cast<int>(src.at<uchar>(y, x, rgb_order ? 2 : 0));
            const int gg = static_cast<int>(src.at<uchar>(y, x, 1));
            const int rr = static_cast<int>(src.at<uchar>(y, x, rgb_order ? 0 : 2));
            const uchar yy = color3_to_yuv_limited_u8(bb, gg, rr, 0);
            const uchar uu = color3_to_yuv_limited_u8(bb, gg, rr, 1);
            const uchar vv = color3_to_yuv_limited_u8(bb, gg, rr, 2);
            const int base = y * (cols * 2) + x * 2;

            out.at<uchar>(y, x) = yy;
            set_yuv444sp_plane_byte_u8(out, rows, cols, base + 0, nv42_layout ? vv : uu);
            set_yuv444sp_plane_byte_u8(out, rows, cols, base + 1, nv42_layout ? uu : vv);
        }
    }

    return out;
}

Mat color3_to_yuv444p_reference_u8(const Mat& src, bool rgb_order, bool yv24_layout)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);

    const int rows = src.size[0];
    const int cols = src.size[1];
    const int plane_size = rows * cols;
    Mat out({rows * 3, cols}, CV_8UC1);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int bb = static_cast<int>(src.at<uchar>(y, x, rgb_order ? 2 : 0));
            const int gg = static_cast<int>(src.at<uchar>(y, x, 1));
            const int rr = static_cast<int>(src.at<uchar>(y, x, rgb_order ? 0 : 2));
            const uchar yy = color3_to_yuv_limited_u8(bb, gg, rr, 0);
            const uchar uu = color3_to_yuv_limited_u8(bb, gg, rr, 1);
            const uchar vv = color3_to_yuv_limited_u8(bb, gg, rr, 2);
            const int chroma_index = y * cols + x;

            out.at<uchar>(y, x) = yy;
            set_yuv444p_plane_byte_u8(out, rows, cols, yv24_layout ? plane_size : 0, chroma_index, uu);
            set_yuv444p_plane_byte_u8(out, rows, cols, yv24_layout ? 0 : plane_size, chroma_index, vv);
        }
    }

    return out;
}

Mat color3_to_yuv422sp_reference_u8(const Mat& src, bool rgb_order, bool nv61_layout)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0];
    const int cols = src.size[1];
    Mat out({rows * 2, cols}, CV_8UC1);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;

            for (int dx = 0; dx < 2; ++dx)
            {
                const int xx = x + dx;
                const int bb = static_cast<int>(src.at<uchar>(y, xx, rgb_order ? 2 : 0));
                const int gg = static_cast<int>(src.at<uchar>(y, xx, 1));
                const int rr = static_cast<int>(src.at<uchar>(y, xx, rgb_order ? 0 : 2));
                const uchar yy = color3_to_yuv_limited_u8(bb, gg, rr, 0);

                out.at<uchar>(y, xx) = yy;
                sum_b += bb;
                sum_g += gg;
                sum_r += rr;
            }

            const int avg_b = (sum_b + 1) >> 1;
            const int avg_g = (sum_g + 1) >> 1;
            const int avg_r = (sum_r + 1) >> 1;
            const uchar uu = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);
            const int base = y * cols + x;

            set_yuv422sp_plane_byte_u8(out, rows, cols, base + 0, nv61_layout ? vv : uu);
            set_yuv422sp_plane_byte_u8(out, rows, cols, base + 1, nv61_layout ? uu : vv);
        }
    }

    return out;
}

inline void set_yuv422_packed_pair_u8(Mat& dst, int y, int pair_x, bool uyvy_layout, uchar yy0, uchar yy1, uchar uu, uchar vv)
{
    CV_Assert(dst.dims == 2);
    CV_Assert(dst.depth() == CV_8U);
    CV_Assert(dst.channels() == 2);
    CV_Assert((pair_x % 2) == 0);
    CV_Assert(pair_x >= 0 && pair_x + 1 < dst.size[1]);

    if (uyvy_layout)
    {
        dst.at<uchar>(y, pair_x + 0, 0) = uu;
        dst.at<uchar>(y, pair_x + 0, 1) = yy0;
        dst.at<uchar>(y, pair_x + 1, 0) = vv;
        dst.at<uchar>(y, pair_x + 1, 1) = yy1;
        return;
    }

    dst.at<uchar>(y, pair_x + 0, 0) = yy0;
    dst.at<uchar>(y, pair_x + 0, 1) = uu;
    dst.at<uchar>(y, pair_x + 1, 0) = yy1;
    dst.at<uchar>(y, pair_x + 1, 1) = vv;
}

Mat color3_to_yuv422packed_reference_u8(const Mat& src, bool rgb_order, bool uyvy_layout)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 3);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0];
    const int cols = src.size[1];
    Mat out({rows, cols}, CV_8UC2);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;
            uchar yy[2] = {0, 0};

            for (int dx = 0; dx < 2; ++dx)
            {
                const int xx = x + dx;
                const int bb = static_cast<int>(src.at<uchar>(y, xx, rgb_order ? 2 : 0));
                const int gg = static_cast<int>(src.at<uchar>(y, xx, 1));
                const int rr = static_cast<int>(src.at<uchar>(y, xx, rgb_order ? 0 : 2));

                yy[dx] = color3_to_yuv_limited_u8(bb, gg, rr, 0);
                sum_b += bb;
                sum_g += gg;
                sum_r += rr;
            }

            const int avg_b = (sum_b + 1) >> 1;
            const int avg_g = (sum_g + 1) >> 1;
            const int avg_r = (sum_r + 1) >> 1;
            const uchar uu = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 1);
            const uchar vv = color3_to_yuv_limited_u8(avg_b, avg_g, avg_r, 2);

            set_yuv422_packed_pair_u8(out, y, x, uyvy_layout, yy[0], yy[1], uu, vv);
        }
    }

    return out;
}

Mat yuv422packed_to_color3_reference_u8(const Mat& src, bool uyvy_layout, bool rgb_order)
{
    CV_Assert(src.dims == 2);
    CV_Assert(src.depth() == CV_8U);
    CV_Assert(src.channels() == 2);
    CV_Assert((src.size[1] % 2) == 0);

    const int rows = src.size[0];
    const int cols = src.size[1];
    Mat out({rows, cols}, CV_8UC3);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int pair_x = x & ~1;
            const int yy = static_cast<int>(src.at<uchar>(y, x, uyvy_layout ? 1 : 0));
            const int uu = static_cast<int>(src.at<uchar>(y, pair_x + 0, uyvy_layout ? 0 : 1));
            const int vv = static_cast<int>(src.at<uchar>(y, pair_x + 1, uyvy_layout ? 0 : 1));

            const uchar b = yuv420_limited_to_u8(yy, uu, vv, 0);
            const uchar g = yuv420_limited_to_u8(yy, uu, vv, 1);
            const uchar r = yuv420_limited_to_u8(yy, uu, vv, 2);

            out.at<uchar>(y, x, rgb_order ? 0 : 2) = r;
            out.at<uchar>(y, x, 1) = g;
            out.at<uchar>(y, x, rgb_order ? 2 : 0) = b;
        }
    }

    return out;
}

}  // namespace
