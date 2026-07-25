#pragma once

#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace cvh;

namespace
{

std::string fixture_path(const std::string& relative)
{
    return std::string(M_ROOT_PATH) + "/test/imgproc/data/opencv_extra/" + relative;
}

void require_fixture(const std::string& relative)
{
    ASSERT_TRUE(std::filesystem::exists(fixture_path(relative)))
        << "Missing fixture: " << fixture_path(relative);
}

int border_replicate(int p, int len)
{
    if (p < 0)
    {
        return 0;
    }
    if (p >= len)
    {
        return len - 1;
    }
    return p;
}

std::vector<int> sobel_kernel_1d(int aperture, bool derivative)
{
    if (aperture == 3)
    {
        return derivative ? std::vector<int>{-1, 0, 1} : std::vector<int>{1, 2, 1};
    }
    if (aperture == 5)
    {
        return derivative ? std::vector<int>{-1, -2, 0, 2, 1} : std::vector<int>{1, 4, 6, 4, 1};
    }
    return std::vector<int>();
}

void sobel_reference_u8_to_s16(const Mat& src, Mat& dx, Mat& dy, int aperture)
{
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(aperture == 3 || aperture == 5);

    const std::vector<int> der = sobel_kernel_1d(aperture, true);
    const std::vector<int> smooth = sobel_kernel_1d(aperture, false);
    const int radius = aperture / 2;

    const int rows = src.size[0];
    const int cols = src.size[1];
    const size_t src_step = src.step(0);

    dx.create(std::vector<int>{rows, cols}, CV_16SC1);
    dy.create(std::vector<int>{rows, cols}, CV_16SC1);
    const size_t dx_step = dx.step(0);
    const size_t dy_step = dy.step(0);

    for (int y = 0; y < rows; ++y)
    {
        short* dx_row = reinterpret_cast<short*>(dx.data + static_cast<size_t>(y) * dx_step);
        short* dy_row = reinterpret_cast<short*>(dy.data + static_cast<size_t>(y) * dy_step);
        for (int x = 0; x < cols; ++x)
        {
            int gx = 0;
            int gy = 0;
            for (int ky = 0; ky < aperture; ++ky)
            {
                const int sy = border_replicate(y + ky - radius, rows);
                const uchar* src_row = src.data + static_cast<size_t>(sy) * src_step;
                for (int kx = 0; kx < aperture; ++kx)
                {
                    const int sx = border_replicate(x + kx - radius, cols);
                    const int p = src_row[sx];
                    gx += p * smooth[static_cast<size_t>(ky)] * der[static_cast<size_t>(kx)];
                    gy += p * der[static_cast<size_t>(ky)] * smooth[static_cast<size_t>(kx)];
                }
            }
            dx_row[x] = saturate_cast<short>(gx);
            dy_row[x] = saturate_cast<short>(gy);
        }
    }
}

Mat canny_reference_from_derivatives(const Mat& dx,
                                     const Mat& dy,
                                     double threshold1,
                                     double threshold2,
                                     bool L2gradient)
{
    CV_Assert(dx.type() == CV_16SC1 && dy.type() == CV_16SC1);
    CV_Assert(dx.size[0] == dy.size[0] && dx.size[1] == dy.size[1]);

    const int rows = dx.size[0];
    const int cols = dx.size[1];
    const size_t dx_step = dx.step(0);
    const size_t dy_step = dy.step(0);

    const float low_threshold = static_cast<float>(std::min(threshold1, threshold2));
    const float high_threshold = static_cast<float>(std::max(threshold1, threshold2));
    const double tan_pi_8 = std::tan(CV_PI / 8.0);
    const double tan_3pi_8 = std::tan(CV_PI * 3.0 / 8.0);

    Mat mag({rows, cols}, CV_32FC1);
    const size_t mag_step = mag.step(0);
    for (int y = 0; y < rows; ++y)
    {
        const short* dx_row = reinterpret_cast<const short*>(dx.data + static_cast<size_t>(y) * dx_step);
        const short* dy_row = reinterpret_cast<const short*>(dy.data + static_cast<size_t>(y) * dy_step);
        float* mag_row = reinterpret_cast<float*>(mag.data + static_cast<size_t>(y) * mag_step);
        for (int x = 0; x < cols; ++x)
        {
            const int gx = dx_row[x];
            const int gy = dy_row[x];
            if (L2gradient)
            {
                mag_row[x] = static_cast<float>(std::sqrt(static_cast<double>(gx) * gx + static_cast<double>(gy) * gy));
            }
            else
            {
                mag_row[x] = static_cast<float>(std::abs(gx) + std::abs(gy));
            }
        }
    }

    for (int y = 0; y < rows; ++y)
    {
        const short* dx_row = reinterpret_cast<const short*>(dx.data + static_cast<size_t>(y) * dx_step);
        const short* dy_row = reinterpret_cast<const short*>(dy.data + static_cast<size_t>(y) * dy_step);
        float* mag_row = reinterpret_cast<float*>(mag.data + static_cast<size_t>(y) * mag_step);
        for (int x = 0; x < cols; ++x)
        {
            const float a = mag_row[x];
            if (a <= low_threshold)
            {
                continue;
            }

            const int gx = dx_row[x];
            const int gy = dy_row[x];
            const double tg = gx ? static_cast<double>(gy) / gx : DBL_MAX * (gy >= 0 ? 1.0 : -1.0);

            int x1 = 0;
            int y1 = 0;
            int x2 = 0;
            int y2 = 0;
            if (std::abs(tg) < tan_pi_8)
            {
                y1 = y;
                y2 = y;
                x1 = x + 1;
                x2 = x - 1;
            }
            else if (tan_pi_8 <= tg && tg <= tan_3pi_8)
            {
                y1 = y + 1;
                y2 = y - 1;
                x1 = x + 1;
                x2 = x - 1;
            }
            else if (-tan_3pi_8 <= tg && tg <= -tan_pi_8)
            {
                y1 = y - 1;
                y2 = y + 1;
                x1 = x + 1;
                x2 = x - 1;
            }
            else
            {
                x1 = x;
                x2 = x;
                y1 = y + 1;
                y2 = y - 1;
            }

            float b = 0.0f;
            float c = 0.0f;
            if (static_cast<unsigned>(x1) < static_cast<unsigned>(cols) &&
                static_cast<unsigned>(y1) < static_cast<unsigned>(rows))
            {
                const float* row = reinterpret_cast<const float*>(mag.data + static_cast<size_t>(y1) * mag_step);
                b = std::fabs(row[x1]);
            }
            if (static_cast<unsigned>(x2) < static_cast<unsigned>(cols) &&
                static_cast<unsigned>(y2) < static_cast<unsigned>(rows))
            {
                const float* row = reinterpret_cast<const float*>(mag.data + static_cast<size_t>(y2) * mag_step);
                c = std::fabs(row[x2]);
            }

            if (!((a > b || (a == b && ((x1 == x + 1 && y1 == y) || (x1 == x && y1 == y + 1)))) && a > c))
            {
                mag_row[x] = -a;
            }
        }
    }

    Mat dst({rows, cols}, CV_8UC1);
    const size_t dst_step = dst.step(0);
    for (int y = 0; y < rows; ++y)
    {
        uchar* dst_row = dst.data + static_cast<size_t>(y) * dst_step;
        std::fill(dst_row, dst_row + cols, static_cast<uchar>(0));
    }

    static const int kOffsets[8][2] = {
        {1, 0}, {1, -1}, {0, -1}, {-1, -1},
        {-1, 0}, {-1, 1}, {0, 1}, {1, 1},
    };

    std::vector<Point> stack;
    stack.reserve(static_cast<size_t>(rows) * static_cast<size_t>(cols) / 8 + 8);

    for (int y = 0; y < rows; ++y)
    {
        const float* mag_row = reinterpret_cast<const float*>(mag.data + static_cast<size_t>(y) * mag_step);
        for (int x = 0; x < cols; ++x)
        {
            if (mag_row[x] <= high_threshold)
            {
                continue;
            }

            uchar* dst_seed_row = dst.data + static_cast<size_t>(y) * dst_step;
            if (dst_seed_row[x] != 0)
            {
                continue;
            }

            dst_seed_row[x] = 255;
            stack.push_back(Point(x, y));

            while (!stack.empty())
            {
                const Point p = stack.back();
                stack.pop_back();

                for (int k = 0; k < 8; ++k)
                {
                    const int nx = p.x + kOffsets[k][0];
                    const int ny = p.y + kOffsets[k][1];
                    if (static_cast<unsigned>(nx) >= static_cast<unsigned>(cols) ||
                        static_cast<unsigned>(ny) >= static_cast<unsigned>(rows))
                    {
                        continue;
                    }

                    uchar* dst_row = dst.data + static_cast<size_t>(ny) * dst_step;
                    if (dst_row[nx] != 0)
                    {
                        continue;
                    }

                    const float* mag_n_row = reinterpret_cast<const float*>(mag.data + static_cast<size_t>(ny) * mag_step);
                    if (mag_n_row[nx] > low_threshold)
                    {
                        dst_row[nx] = 255;
                        stack.push_back(Point(nx, ny));
                    }
                }
            }
        }
    }

    return dst;
}

Mat canny_reference(const Mat& src,
                    double threshold1,
                    double threshold2,
                    int apertureSize,
                    bool L2gradient)
{
    Mat dx;
    Mat dy;
    sobel_reference_u8_to_s16(src, dx, dy, apertureSize);
    return canny_reference_from_derivatives(dx, dy, threshold1, threshold2, L2gradient);
}

int max_abs_diff_u8(const Mat& a, const Mat& b)
{
    CV_Assert(a.type() == b.type());
    CV_Assert(a.size[0] == b.size[0] && a.size[1] == b.size[1]);
    const size_t count = a.total() * static_cast<size_t>(a.channels());
    int max_diff = 0;
    for (size_t i = 0; i < count; ++i)
    {
        max_diff = std::max(max_diff, std::abs(static_cast<int>(a.data[i]) - static_cast<int>(b.data[i])));
    }
    return max_diff;
}

void fill_u8_pattern(Mat& src, std::uint32_t seed)
{
    CV_Assert(src.type() == CV_8UC1);
    const size_t count = src.total();
    for (size_t i = 0; i < count; ++i)
    {
        seed = seed * 1664525u + 1013904223u;
        src.data[i] = static_cast<uchar>((seed >> 24) & 0xFFu);
    }
}

}  // namespace
