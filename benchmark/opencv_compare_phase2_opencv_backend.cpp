#include "opencv_compare_phase2_benchmark.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace cvh_bench_compare {
namespace {

volatile double g_phase2_opencv_sink = 0.0;

inline void p2_compiler_barrier()
{
#if defined(_MSC_VER)
    _ReadWriteBarrier();
#else
    __asm__ __volatile__("" : : : "memory");
#endif
}

std::uint32_t p2_lcg_next(std::uint32_t state)
{
    return state * 1664525u + 1013904223u;
}

cv::Mat p2_make_mat(int rows, int cols, int type)
{
    return cv::Mat(rows, cols, type);
}

void p2_fill_u8(cv::Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int scalars_per_row = mat.cols * mat.channels();
    for (int row = 0; row < mat.rows; ++row)
    {
        unsigned char* output = mat.ptr<unsigned char>(row);
        for (int index = 0; index < scalars_per_row; ++index)
        {
            state = p2_lcg_next(state);
            output[index] = static_cast<unsigned char>(
                (state >> 24) ^
                static_cast<std::uint32_t>(index + row * 17));
        }
    }
}

void p2_fill_f32(cv::Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int scalars_per_row = mat.cols * mat.channels();
    for (int row = 0; row < mat.rows; ++row)
    {
        float* output = mat.ptr<float>(row);
        for (int index = 0; index < scalars_per_row; ++index)
        {
            state = p2_lcg_next(state);
            output[index] = static_cast<float>(
                static_cast<int>(state & 0xFFFFu) - 32768) /
                4096.0f;
        }
    }
}

void p2_fill_region_mask(cv::Mat& mask)
{
    for (int row = 0; row < mask.rows; ++row)
    {
        for (int col = 0; col < mask.cols; ++col)
        {
            mask.at<unsigned char>(row, col) =
                ((row * 17 + col * 29) % 23) < 3 ? 255 : 0;
        }
    }
}

void p2_fill_contour_mask(cv::Mat& mask)
{
    mask = 0;
    for (int y = 4; y + 12 < mask.rows; y += 20)
    {
        for (int x = 4; x + 12 < mask.cols; x += 20)
        {
            for (int yy = y; yy < y + 12; ++yy)
            {
                for (int xx = x; xx < x + 12; ++xx)
                {
                    mask.at<unsigned char>(yy, xx) = 255;
                }
            }
        }
    }
}

std::vector<cv::Point> p2_make_shape_points(int count)
{
    std::vector<cv::Point> points;
    points.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index)
    {
        points.emplace_back(
            index,
            (index * 37 + (index % 11) * 19) % 1009);
    }
    return points;
}

double p2_checksum(const cv::Mat& mat)
{
    if (mat.empty())
    {
        return 0.0;
    }
    const int scalar_count = mat.cols * mat.channels();
    const int stride = std::max(1, scalar_count / 64);
    double value = static_cast<double>(mat.total());
    for (int row = 0; row < mat.rows; ++row)
    {
        if (mat.depth() == CV_8U)
        {
            const unsigned char* data = mat.ptr<unsigned char>(row);
            for (int index = 0; index < scalar_count; index += stride)
            {
                value += static_cast<double>(data[index]);
            }
        }
        else if (mat.depth() == CV_32S)
        {
            const int* data = mat.ptr<int>(row);
            for (int index = 0; index < scalar_count; index += stride)
            {
                value += static_cast<double>(data[index]);
            }
        }
        else if (mat.depth() == CV_32F)
        {
            const float* data = mat.ptr<float>(row);
            for (int index = 0; index < scalar_count; index += stride)
            {
                value += static_cast<double>(data[index]);
            }
        }
        else if (mat.depth() == CV_64F)
        {
            const double* data = mat.ptr<double>(row);
            for (int index = 0; index < scalar_count; index += stride)
            {
                value += data[index];
            }
        }
    }
    return value;
}

double p2_checksum_points(const std::vector<cv::Point>& points)
{
    double value = static_cast<double>(points.size());
    for (const cv::Point& point : points)
    {
        value += static_cast<double>(point.x) * 0.5 +
            static_cast<double>(point.y) * 0.25;
    }
    return value;
}

double p2_checksum_contours(
    const std::vector<std::vector<cv::Point>>& contours)
{
    double value = static_cast<double>(contours.size());
    for (const auto& contour : contours)
    {
        value += p2_checksum_points(contour);
    }
    return value;
}

void p2_calc_hist(const cv::Mat& image, cv::Mat& histogram)
{
    const int channel = 0;
    const int histogram_size = 256;
    const float range[] = {0.0f, 256.0f};
    const float* ranges[] = {range};
    cv::calcHist(
        &image,
        1,
        &channel,
        cv::Mat(),
        histogram,
        1,
        &histogram_size,
        ranges,
        true,
        false);
}

template<typename RunFn, typename ProbeFn>
double p2_measure_ms(RunFn&& run,
                     ProbeFn&& probe,
                     int warmup,
                     int iters,
                     int repeats)
{
    for (int index = 0; index < warmup; ++index)
    {
        p2_compiler_barrier();
        run();
        p2_compiler_barrier();
    }
    double best_ms = std::numeric_limits<double>::max();
    for (int repeat = 0; repeat < repeats; ++repeat)
    {
        const auto begin = std::chrono::steady_clock::now();
        for (int index = 0; index < iters; ++index)
        {
            p2_compiler_barrier();
            run();
            p2_compiler_barrier();
        }
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration_cast<
                std::chrono::duration<double, std::milli>>(end - begin)
                .count();
        best_ms = std::min(
            best_ms, elapsed_ms / static_cast<double>(iters));
        g_phase2_opencv_sink += static_cast<double>(probe());
    }
    return best_ms;
}

}  // namespace

#define P2_BENCH_FUNCTION bench_opencv_phase2
#define P2_NAMESPACE cv
#define P2_MAT cv::Mat
#define P2_POINT_TYPE cv::Point
#include "opencv_compare_phase2_cases.inl"
#undef P2_POINT_TYPE
#undef P2_MAT
#undef P2_NAMESPACE
#undef P2_BENCH_FUNCTION

}  // namespace cvh_bench_compare
