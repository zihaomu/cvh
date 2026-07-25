#ifndef CVH_IMGPROC_INTEGRAL_H
#define CVH_IMGPROC_INTEGRAL_H

#include "detail/common.h"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/simd/opencv_ui.h"

#include <climits>
#include <cstdint>
#include <cstring>
#include <vector>

namespace cvh
{

inline void integral(const Mat& src, Mat& sum, int sdepth = -1)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        CV_Error(Error::StsBadArg, "integral currently expects non-empty 2D CV_8U src");
    }
    const int output_depth = sdepth < 0 ? CV_32S : CV_MAT_DEPTH(sdepth);
    if (output_depth != CV_32S && output_depth != CV_64F)
    {
        CV_Error(Error::StsBadArg, "integral output depth must be CV_32S or CV_64F");
    }
    const Mat source = src.data == sum.data ? src.clone() : src;
    const int rows = source.size.p[0];
    const int cols = source.size.p[1];
    const int channels = source.channels();
    sum.create(
        {rows + 1, cols + 1}, CV_MAKETYPE(output_depth, channels));
    const size_t output_scalar_size =
        output_depth == CV_32S ? sizeof(int) : sizeof(double);
    std::memset(
        sum.data,
        0,
        static_cast<size_t>(cols + 1) *
            static_cast<size_t>(channels) * output_scalar_size);

#if CVH_ENABLE_OPENCV_INTRIN && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const bool can_use_ui =
        cpu::dispatch_mode() != cpu::DispatchMode::ScalarOnly &&
        output_depth == CV_32S && channels == 1 &&
        static_cast<std::int64_t>(rows) *
                static_cast<std::int64_t>(cols) * 255 <=
            INT_MAX;
    if (can_use_ui)
    {
        for (int y = 0; y < rows; ++y)
        {
            const uchar* source_row =
                source.data + static_cast<size_t>(y) * source.step(0);
            const int* previous_row = reinterpret_cast<const int*>(
                sum.data + static_cast<size_t>(y) * sum.step(0));
            int* output_row = reinterpret_cast<int*>(
                sum.data + static_cast<size_t>(y + 1) * sum.step(0));
            output_row[0] = 0;
            int row_sum = 0;
            int x = 0;
            for (; x + 4 <= cols; x += 4)
            {
                int prefixes[4];
                row_sum += source_row[x + 0];
                prefixes[0] = row_sum;
                row_sum += source_row[x + 1];
                prefixes[1] = row_sum;
                row_sum += source_row[x + 2];
                prefixes[2] = row_sum;
                row_sum += source_row[x + 3];
                prefixes[3] = row_sum;
                cv::v_store(
                    output_row + x + 1,
                    cv::v_add(
                        cv::v_load(prefixes),
                        cv::v_load(previous_row + x + 1)));
            }
            for (; x < cols; ++x)
            {
                row_sum += source_row[x];
                output_row[x + 1] = row_sum + previous_row[x + 1];
            }
        }
        cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
        return;
    }
#endif

    std::vector<std::int64_t> row_sums(static_cast<size_t>(channels), 0);
    if (output_depth == CV_32S)
    {
        for (int y = 0; y < rows; ++y)
        {
            std::fill(row_sums.begin(), row_sums.end(), 0);
            const uchar* source_row =
                source.data + static_cast<size_t>(y) * source.step(0);
            const int* previous_row = reinterpret_cast<const int*>(
                sum.data + static_cast<size_t>(y) * sum.step(0));
            int* output_row = reinterpret_cast<int*>(
                sum.data + static_cast<size_t>(y + 1) * sum.step(0));
            std::fill(output_row, output_row + channels, 0);
            for (int x = 0; x < cols; ++x)
            {
                for (int ch = 0; ch < channels; ++ch)
                {
                    row_sums[static_cast<size_t>(ch)] +=
                        source_row[
                            static_cast<size_t>(x) * channels +
                            static_cast<size_t>(ch)];
                    const std::int64_t value =
                        row_sums[static_cast<size_t>(ch)] +
                        previous_row[
                            static_cast<size_t>(x + 1) * channels +
                            static_cast<size_t>(ch)];
                    output_row[
                        static_cast<size_t>(x + 1) * channels +
                        static_cast<size_t>(ch)] = static_cast<int>(value);
                }
            }
        }
        return;
    }

    for (int y = 0; y < rows; ++y)
    {
        std::fill(row_sums.begin(), row_sums.end(), 0);
        const uchar* source_row =
            source.data + static_cast<size_t>(y) * source.step(0);
        const double* previous_row = reinterpret_cast<const double*>(
            sum.data + static_cast<size_t>(y) * sum.step(0));
        double* output_row = reinterpret_cast<double*>(
            sum.data + static_cast<size_t>(y + 1) * sum.step(0));
        std::fill(output_row, output_row + channels, 0.0);
        for (int x = 0; x < cols; ++x)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                row_sums[static_cast<size_t>(ch)] +=
                    source_row[
                        static_cast<size_t>(x) * channels +
                        static_cast<size_t>(ch)];
                output_row[
                    static_cast<size_t>(x + 1) * channels +
                    static_cast<size_t>(ch)] =
                    static_cast<double>(
                        row_sums[static_cast<size_t>(ch)]) +
                    previous_row[
                        static_cast<size_t>(x + 1) * channels +
                        static_cast<size_t>(ch)];
            }
        }
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_INTEGRAL_H
