#ifndef CVH_IMGPROC_STACK_BLUR_H
#define CVH_IMGPROC_STACK_BLUR_H

#include "../core/detail/dispatch_control.h"
#include "detail/common.h"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace cvh
{
namespace stack_blur_detail
{

inline thread_local const char* g_last_stack_blur_algorithm_path =
    "stack_blur_generic";

inline const char* last_stack_blur_algorithm_path()
{
    return g_last_stack_blur_algorithm_path;
}

inline void run_u8_k5(const Mat& src, Mat& dst)
{
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int channels = src.channels();
    const int row_width = cols * channels;
    constexpr int divisor = 81;
    std::vector<std::int32_t> horizontal(
        static_cast<size_t>(rows) *
        static_cast<size_t>(row_width));
    for (int y = 0; y < rows; ++y)
    {
        const uchar* input =
            src.data + static_cast<size_t>(y) * src.step(0);
        std::int32_t* output =
            horizontal.data() + static_cast<size_t>(y) * row_width;
        const int interior_begin = std::min(row_width, 2 * channels);
        const int interior_end = std::max(
            interior_begin,
            (cols - 2) * channels);
        for (int index = 0; index < interior_begin; ++index)
        {
            const int x = index / channels;
            const int channel = index % channels;
            const auto read = [&](int source_x) {
                return static_cast<std::int32_t>(
                    input[static_cast<size_t>(
                              std::clamp(source_x, 0, cols - 1)) *
                              channels +
                          channel]);
            };
            output[index] =
                read(x - 2) + 2 * read(x - 1) + 3 * read(x) +
                2 * read(x + 1) + read(x + 2);
        }
        for (int index = interior_begin;
             index < interior_end;
             ++index)
        {
            output[index] =
                input[index - 2 * channels] +
                2 * input[index - channels] +
                3 * input[index] +
                2 * input[index + channels] +
                input[index + 2 * channels];
        }
        for (int index = interior_end; index < row_width; ++index)
        {
            const int x = index / channels;
            const int channel = index % channels;
            const auto read = [&](int source_x) {
                return static_cast<std::int32_t>(
                    input[static_cast<size_t>(
                              std::clamp(source_x, 0, cols - 1)) *
                              channels +
                          channel]);
            };
            output[index] =
                read(x - 2) + 2 * read(x - 1) + 3 * read(x) +
                2 * read(x + 1) + read(x + 2);
        }
    }

    for (int y = 0; y < rows; ++y)
    {
        const auto row = [&](int source_y) {
            return horizontal.data() +
                static_cast<size_t>(
                    std::clamp(source_y, 0, rows - 1)) *
                    row_width;
        };
        const std::int32_t* row0 = row(y - 2);
        const std::int32_t* row1 = row(y - 1);
        const std::int32_t* row2 = row(y);
        const std::int32_t* row3 = row(y + 1);
        const std::int32_t* row4 = row(y + 2);
        uchar* output =
            dst.data + static_cast<size_t>(y) * dst.step(0);
        for (int index = 0; index < row_width; ++index)
        {
            const std::int32_t sum =
                row0[index] + 2 * row1[index] + 3 * row2[index] +
                2 * row3[index] + row4[index];
            output[index] = static_cast<uchar>(
                (sum + divisor / 2) / divisor);
        }
    }
}

template<typename Accumulator>
inline void run_u8_accumulator(const Mat& src, Mat& dst, Size ksize)
{
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int channels = src.channels();
    const int radius_x = ksize.width / 2;
    const int radius_y = ksize.height / 2;
    const int row_width = cols * channels;
    const Accumulator divisor_x =
        static_cast<Accumulator>(radius_x + 1) *
        static_cast<Accumulator>(radius_x + 1);
    const Accumulator divisor_y =
        static_cast<Accumulator>(radius_y + 1) *
        static_cast<Accumulator>(radius_y + 1);
    const Accumulator divisor = divisor_x * divisor_y;
    std::vector<Accumulator> temporary(
        static_cast<size_t>(rows) *
        static_cast<size_t>(row_width));
    for (int y = 0; y < rows; ++y)
    {
        const uchar* input =
            src.data + static_cast<size_t>(y) * src.step(0);
        Accumulator* output =
            temporary.data() +
            static_cast<size_t>(y) * static_cast<size_t>(row_width);
        for (int channel = 0; channel < channels; ++channel)
        {
            const auto read = [&](int x) -> Accumulator {
                const int source_x =
                    std::clamp(x, 0, cols - 1);
                return input[
                    static_cast<size_t>(source_x) *
                        static_cast<size_t>(channels) +
                    static_cast<size_t>(channel)];
            };
            Accumulator weighted_sum = 0;
            Accumulator left_sum = 0;
            Accumulator right_sum = 0;
            for (int offset = -radius_x;
                 offset <= radius_x;
                 ++offset)
            {
                weighted_sum +=
                    static_cast<Accumulator>(
                        radius_x + 1 - std::abs(offset)) *
                    read(offset);
            }
            for (int offset = -radius_x;
                 offset <= 0;
                 ++offset)
            {
                left_sum += read(offset);
            }
            for (int offset = 1;
                 offset <= radius_x + 1;
                 ++offset)
            {
                right_sum += read(offset);
            }

            for (int x = 0; x < cols; ++x)
            {
                output[x * channels + channel] = weighted_sum;
                if (x + 1 == cols)
                {
                    continue;
                }
                weighted_sum += right_sum - left_sum;
                const std::int64_t next_center = read(x + 1);
                left_sum +=
                    next_center - read(x - radius_x);
                right_sum +=
                    read(x + radius_x + 2) - next_center;
            }
        }
    }

    std::vector<Accumulator> weighted_sum(
        static_cast<size_t>(row_width), 0);
    std::vector<Accumulator> left_sum(
        static_cast<size_t>(row_width), 0);
    std::vector<Accumulator> right_sum(
        static_cast<size_t>(row_width), 0);
    const auto temporary_row = [&](int y) {
        const int source_y =
            std::clamp(y, 0, rows - 1);
        return temporary.data() +
               static_cast<size_t>(source_y) *
                   static_cast<size_t>(row_width);
    };
    for (int offset = -radius_y;
         offset <= radius_y;
         ++offset)
    {
        const Accumulator* input = temporary_row(offset);
        const Accumulator weight =
            radius_y + 1 - std::abs(offset);
        for (int index = 0; index < row_width; ++index)
        {
            weighted_sum[static_cast<size_t>(index)] +=
                weight * input[index];
        }
    }
    for (int offset = -radius_y; offset <= 0; ++offset)
    {
        const Accumulator* input = temporary_row(offset);
        for (int index = 0; index < row_width; ++index)
        {
            left_sum[static_cast<size_t>(index)] += input[index];
        }
    }
    for (int offset = 1;
         offset <= radius_y + 1;
         ++offset)
    {
        const Accumulator* input = temporary_row(offset);
        for (int index = 0; index < row_width; ++index)
        {
            right_sum[static_cast<size_t>(index)] += input[index];
        }
    }

    for (int y = 0; y < rows; ++y)
    {
        uchar* output =
            dst.data + static_cast<size_t>(y) * dst.step(0);
        for (int index = 0; index < row_width; ++index)
        {
            output[index] = static_cast<uchar>(
                (weighted_sum[static_cast<size_t>(index)] +
                 divisor / 2) /
                divisor);
        }
        if (y + 1 == rows)
        {
            continue;
        }
        const Accumulator* next_center =
            temporary_row(y + 1);
        const Accumulator* leaving_left =
            temporary_row(y - radius_y);
        const Accumulator* entering_right =
            temporary_row(y + radius_y + 2);
        for (int index = 0; index < row_width; ++index)
        {
            const size_t position = static_cast<size_t>(index);
            weighted_sum[position] +=
                right_sum[position] - left_sum[position];
            left_sum[position] +=
                next_center[index] - leaving_left[index];
            right_sum[position] +=
                entering_right[index] - next_center[index];
        }
    }
}

inline void run_u8(const Mat& src, Mat& dst, Size ksize)
{
    const long double radius_x = ksize.width / 2;
    const long double radius_y = ksize.height / 2;
    const long double divisor =
        (radius_x + 1) * (radius_x + 1) *
        (radius_y + 1) * (radius_y + 1);
    if (divisor <=
        static_cast<long double>(
            std::numeric_limits<std::int32_t>::max()) /
            255)
    {
        run_u8_accumulator<std::int32_t>(src, dst, ksize);
        return;
    }
    run_u8_accumulator<std::int64_t>(src, dst, ksize);
}

template<typename T>
inline T cast_value(double value)
{
    if constexpr (std::is_same<T, uchar>::value)
    {
        return saturate_cast<uchar>(value);
    }
    return static_cast<T>(value);
}

template<typename T>
inline void run(const Mat& src, Mat& dst, Size ksize)
{
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int channels = src.channels();
    const int radius_x = ksize.width / 2;
    const int radius_y = ksize.height / 2;
    const double divisor_x =
        static_cast<double>((radius_x + 1) * (radius_x + 1));
    const double divisor_y =
        static_cast<double>((radius_y + 1) * (radius_y + 1));
    std::vector<double> temporary(
        static_cast<size_t>(rows) * cols * channels, 0.0);

    for (int y = 0; y < rows; ++y)
    {
        const T* input = reinterpret_cast<const T*>(
            src.data + static_cast<size_t>(y) * src.step(0));
        for (int x = 0; x < cols; ++x)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                double accumulator = 0.0;
                for (int k = -radius_x; k <= radius_x; ++k)
                {
                    const int source_x = std::clamp(x + k, 0, cols - 1);
                    const int weight = radius_x + 1 - std::abs(k);
                    accumulator +=
                        weight *
                        static_cast<double>(
                            input[static_cast<size_t>(source_x) * channels +
                                  static_cast<size_t>(ch)]);
                }
                temporary[
                    (static_cast<size_t>(y) * cols +
                     static_cast<size_t>(x)) *
                        channels +
                    static_cast<size_t>(ch)] =
                    accumulator / divisor_x;
            }
        }
    }

    for (int y = 0; y < rows; ++y)
    {
        T* output = reinterpret_cast<T*>(
            dst.data + static_cast<size_t>(y) * dst.step(0));
        for (int x = 0; x < cols; ++x)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                double accumulator = 0.0;
                for (int k = -radius_y; k <= radius_y; ++k)
                {
                    const int source_y = std::clamp(y + k, 0, rows - 1);
                    const int weight = radius_y + 1 - std::abs(k);
                    accumulator +=
                        weight *
                        temporary[
                            (static_cast<size_t>(source_y) * cols +
                             static_cast<size_t>(x)) *
                                channels +
                            static_cast<size_t>(ch)];
                }
                output[static_cast<size_t>(x) * channels +
                       static_cast<size_t>(ch)] =
                    cast_value<T>(accumulator / divisor_y);
            }
        }
    }
}

}  // namespace stack_blur_detail

inline void stackBlur(const Mat& src, Mat& dst, Size ksize)
{
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    stack_blur_detail::g_last_stack_blur_algorithm_path =
        "stack_blur_generic";
    if (src.empty() || src.dims != 2 ||
        (src.depth() != CV_8U && src.depth() != CV_32F) ||
        (src.channels() != 1 && src.channels() != 3 &&
         src.channels() != 4))
    {
        CV_Error(Error::StsBadArg, "stackBlur unsupported source");
    }
    if (ksize.width <= 0 || ksize.height <= 0 ||
        (ksize.width & 1) == 0 || (ksize.height & 1) == 0)
    {
        CV_Error(Error::StsBadSize, "stackBlur ksize must be positive and odd");
    }
    const Mat source = src.data == dst.data ? src.clone() : src;
    dst.create(source.shape(), source.type());
    if (source.depth() == CV_8U)
    {
        if (ksize.width == 5 && ksize.height == 5)
        {
            stack_blur_detail::g_last_stack_blur_algorithm_path =
                "stack_blur_u8_k5_typed";
            stack_blur_detail::run_u8_k5(source, dst);
        }
        else
        {
            stack_blur_detail::g_last_stack_blur_algorithm_path =
                "stack_blur_u8_rolling";
            stack_blur_detail::run_u8(source, dst, ksize);
        }
    }
    else
    {
        stack_blur_detail::g_last_stack_blur_algorithm_path =
            "stack_blur_f32_generic";
        stack_blur_detail::run<float>(source, dst, ksize);
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_STACK_BLUR_H
