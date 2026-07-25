#ifndef CVH_IMGPROC_SQR_BOX_FILTER_H
#define CVH_IMGPROC_SQR_BOX_FILTER_H

#include "../core/basic_op.h"
#include "box_filter.h"
#include "detail/common.h"

#include <cstdint>
#include <vector>

namespace cvh
{
namespace sqr_box_detail
{

inline double read_value(const Mat& src, int y, int x, int channel)
{
    const size_t index =
        static_cast<size_t>(x) * src.channels() +
        static_cast<size_t>(channel);
    const uchar* row = src.data + static_cast<size_t>(y) * src.step(0);
    if (src.depth() == CV_8U)
    {
        return row[index];
    }
    return reinterpret_cast<const float*>(row)[index];
}

inline void write_value(Mat& dst,
                        int y,
                        int x,
                        int channel,
                        double value)
{
    const size_t index =
        static_cast<size_t>(x) * dst.channels() +
        static_cast<size_t>(channel);
    uchar* row = dst.data + static_cast<size_t>(y) * dst.step(0);
    if (dst.depth() == CV_8U)
    {
        row[index] = saturate_cast<uchar>(value);
    }
    else if (dst.depth() == CV_32F)
    {
        reinterpret_cast<float*>(row)[index] = static_cast<float>(value);
    }
    else
    {
        reinterpret_cast<double*>(row)[index] = value;
    }
}

inline bool squared_box_u8_wide(const Mat& src,
                                Mat& dst,
                                int output_depth,
                                Size ksize,
                                Point anchor,
                                bool normalize,
                                int border_type)
{
    if (src.depth() != CV_8U ||
        (output_depth != CV_8U && output_depth != CV_64F))
    {
        return false;
    }
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int channels = src.channels();
    const int row_elements = cols * channels;
    const int right = ksize.width - anchor.x - 1;
    const int bottom = ksize.height - anchor.y - 1;
    const std::vector<int> x_map = detail::build_extended_index_map(
        cols, anchor.x, right, border_type);
    const std::vector<int> y_map = detail::build_extended_index_map(
        rows, anchor.y, bottom, border_type);
    std::vector<std::int64_t> row_sums(
        static_cast<size_t>(rows) * row_elements);

    for (int y = 0; y < rows; ++y)
    {
        const uchar* source =
            src.data + static_cast<size_t>(y) * src.step(0);
        std::int64_t* output =
            row_sums.data() + static_cast<size_t>(y) * row_elements;
        for (int channel = 0; channel < channels; ++channel)
        {
            std::int64_t sum = 0;
            for (int kernel_x = 0;
                 kernel_x < ksize.width;
                 ++kernel_x)
            {
                const int source_x =
                    x_map[static_cast<size_t>(kernel_x)];
                if (source_x >= 0)
                {
                    const std::int64_t value =
                        source[
                            static_cast<size_t>(source_x) * channels +
                            channel];
                    sum += value * value;
                }
            }
            output[channel] = sum;
            for (int x = 1; x < cols; ++x)
            {
                const int add_x = x_map[
                    static_cast<size_t>(x + ksize.width - 1)];
                if (add_x >= 0)
                {
                    const std::int64_t value =
                        source[
                            static_cast<size_t>(add_x) * channels +
                            channel];
                    sum += value * value;
                }
                const int subtract_x =
                    x_map[static_cast<size_t>(x - 1)];
                if (subtract_x >= 0)
                {
                    const std::int64_t value =
                        source[
                            static_cast<size_t>(subtract_x) * channels +
                            channel];
                    sum -= value * value;
                }
                output[static_cast<size_t>(x) * channels + channel] =
                    sum;
            }
        }
    }

    dst.create(
        src.shape(), CV_MAKETYPE(output_depth, channels));
    std::vector<std::int64_t> accumulated(
        static_cast<size_t>(row_elements),
        0);
    for (int kernel_y = 0;
         kernel_y < ksize.height;
         ++kernel_y)
    {
        const int source_y =
            y_map[static_cast<size_t>(kernel_y)];
        if (source_y < 0)
        {
            continue;
        }
        const std::int64_t* source =
            row_sums.data() +
            static_cast<size_t>(source_y) * row_elements;
        for (int index = 0; index < row_elements; ++index)
        {
            accumulated[static_cast<size_t>(index)] += source[index];
        }
    }

    const double scale = normalize
        ? 1.0 /
              static_cast<double>(ksize.width * ksize.height)
        : 1.0;
    for (int y = 0; y < rows; ++y)
    {
        if (output_depth == CV_64F)
        {
            double* output = reinterpret_cast<double*>(
                dst.data + static_cast<size_t>(y) * dst.step(0));
            for (int index = 0; index < row_elements; ++index)
            {
                output[index] =
                    static_cast<double>(
                        accumulated[static_cast<size_t>(index)]) *
                    scale;
            }
        }
        else
        {
            uchar* output =
                dst.data + static_cast<size_t>(y) * dst.step(0);
            for (int index = 0; index < row_elements; ++index)
            {
                output[index] = saturate_cast<uchar>(
                    static_cast<double>(
                        accumulated[static_cast<size_t>(index)]) *
                    scale);
            }
        }
        if (y + 1 >= rows)
        {
            continue;
        }
        const int subtract_y =
            y_map[static_cast<size_t>(y)];
        const int add_y =
            y_map[static_cast<size_t>(y + ksize.height)];
        const std::int64_t* subtract_row =
            subtract_y >= 0
                ? row_sums.data() +
                      static_cast<size_t>(subtract_y) * row_elements
                : nullptr;
        const std::int64_t* add_row =
            add_y >= 0
                ? row_sums.data() +
                      static_cast<size_t>(add_y) * row_elements
                : nullptr;
        for (int index = 0; index < row_elements; ++index)
        {
            if (subtract_row)
            {
                accumulated[static_cast<size_t>(index)] -=
                    subtract_row[index];
            }
            if (add_row)
            {
                accumulated[static_cast<size_t>(index)] +=
                    add_row[index];
            }
        }
    }
    return true;
}

}  // namespace sqr_box_detail

inline void sqrBoxFilter(const Mat& src,
                         Mat& dst,
                         int ddepth,
                         Size ksize,
                         Point anchor = Point(-1, -1),
                         bool normalize = true,
                         int borderType = BORDER_DEFAULT)
{
    if (src.empty() || src.dims != 2 ||
        (src.depth() != CV_8U && src.depth() != CV_32F) ||
        (src.channels() != 1 && src.channels() != 3 && src.channels() != 4))
    {
        CV_Error(Error::StsBadArg, "sqrBoxFilter unsupported src");
    }
    if (ksize.width <= 0 || ksize.height <= 0)
    {
        CV_Error(Error::StsBadSize, "sqrBoxFilter invalid ksize");
    }
    if (anchor.x < 0) anchor.x = ksize.width / 2;
    if (anchor.y < 0) anchor.y = ksize.height / 2;
    if (anchor.x < 0 || anchor.x >= ksize.width ||
        anchor.y < 0 || anchor.y >= ksize.height)
    {
        CV_Error(Error::StsOutOfRange, "sqrBoxFilter invalid anchor");
    }
    const int border_type = detail::normalize_border_type(borderType);
    if (!detail::is_supported_filter_border(border_type))
    {
        CV_Error(Error::StsBadArg, "sqrBoxFilter unsupported border");
    }
    const int output_depth =
        ddepth < 0 ? src.depth() : CV_MAT_DEPTH(ddepth);
    if (output_depth != CV_8U && output_depth != CV_32F &&
        output_depth != CV_64F)
    {
        CV_Error(Error::StsBadArg, "sqrBoxFilter unsupported ddepth");
    }

    const Mat source = src.data == dst.data ? src.clone() : src;
    if (sqr_box_detail::squared_box_u8_wide(
            source,
            dst,
            output_depth,
            ksize,
            anchor,
            normalize,
            border_type))
    {
        return;
    }
    if (output_depth == CV_32F)
    {
        Mat source_f32;
        if (source.depth() == CV_32F)
        {
            source_f32 = source;
        }
        else
        {
            source.convertTo(
                source_f32,
                CV_MAKETYPE(CV_32F, source.channels()));
        }

        Mat squared;
        multiply(source_f32, source_f32, squared);
        Mat filtered;
        boxFilter(
            squared,
            filtered,
            CV_32F,
            ksize,
            anchor,
            normalize,
            border_type);
        dst = filtered;
        return;
    }

    dst.create(
        source.shape(), CV_MAKETYPE(output_depth, source.channels()));
    const int area = ksize.width * ksize.height;
    const double scale = normalize ? 1.0 / area : 1.0;
    for (int y = 0; y < source.size.p[0]; ++y)
    {
        for (int x = 0; x < source.size.p[1]; ++x)
        {
            for (int ch = 0; ch < source.channels(); ++ch)
            {
                long double accumulator = 0.0L;
                for (int ky = 0; ky < ksize.height; ++ky)
                {
                    const int source_y = detail::border_interpolate(
                        y + ky - anchor.y,
                        source.size.p[0],
                        border_type);
                    if (source_y < 0)
                    {
                        continue;
                    }
                    for (int kx = 0; kx < ksize.width; ++kx)
                    {
                        const int source_x = detail::border_interpolate(
                            x + kx - anchor.x,
                            source.size.p[1],
                            border_type);
                        if (source_x < 0)
                        {
                            continue;
                        }
                        const double value = sqr_box_detail::read_value(
                            source, source_y, source_x, ch);
                        accumulator +=
                            static_cast<long double>(value) * value;
                    }
                }
                sqr_box_detail::write_value(
                    dst,
                    y,
                    x,
                    ch,
                    static_cast<double>(accumulator) * scale);
            }
        }
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_SQR_BOX_FILTER_H
