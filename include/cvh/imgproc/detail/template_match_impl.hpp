#ifndef CVH_IMGPROC_DETAIL_TEMPLATE_MATCH_IMPL_HPP
#define CVH_IMGPROC_DETAIL_TEMPLATE_MATCH_IMPL_HPP

#include "../template_match.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace cvh
{
namespace detail
{

inline double template_value(const Mat& matrix, int row, int column)
{
    return matrix.depth() == CV_8U
        ? static_cast<double>(matrix.at<uchar>(row, column))
        : static_cast<double>(matrix.at<float>(row, column));
}

}  // namespace detail

inline void matchTemplate(const Mat& image, const Mat& templ, Mat& result, int method)
{
    if (image.empty() || templ.empty() || image.dims != 2 || templ.dims != 2 ||
        image.channels() != 1 || templ.channels() != 1 || image.type() != templ.type() ||
        (image.depth() != CV_8U && image.depth() != CV_32F))
    {
        CV_Error(Error::StsUnsupportedFormat, "matchTemplate supports equal-type 2D U8/F32 C1 inputs");
    }
    if (templ.size[0] > image.size[0] || templ.size[1] > image.size[1])
    {
        CV_Error(Error::StsUnmatchedSizes, "matchTemplate template must not exceed image");
    }
    if (method != TM_SQDIFF && method != TM_SQDIFF_NORMED &&
        method != TM_CCORR && method != TM_CCORR_NORMED)
    {
        CV_Error(Error::StsBadFlag, "matchTemplate method is unsupported in P2-P0");
    }

    const Mat source_image = result.data == image.data ? image.clone() : image;
    const Mat source_template = result.data == templ.data ? templ.clone() : templ;
    const int output_rows = source_image.size[0] - source_template.size[0] + 1;
    const int output_columns = source_image.size[1] - source_template.size[1] + 1;
    result.create({output_rows, output_columns}, CV_32FC1);

    double template_sum_squared = 0.0;
    for (int row = 0; row < source_template.size[0]; ++row)
    {
        for (int column = 0; column < source_template.size[1]; ++column)
        {
            const double value = detail::template_value(source_template, row, column);
            template_sum_squared += value * value;
        }
    }

    for (int output_row = 0; output_row < output_rows; ++output_row)
    {
        for (int output_column = 0; output_column < output_columns; ++output_column)
        {
            double correlation = 0.0;
            double image_sum_squared = 0.0;
            double squared_difference = 0.0;
            for (int row = 0; row < source_template.size[0]; ++row)
            {
                for (int column = 0; column < source_template.size[1]; ++column)
                {
                    const double image_value = detail::template_value(
                        source_image, output_row + row, output_column + column);
                    const double template_pixel = detail::template_value(
                        source_template, row, column);
                    correlation += image_value * template_pixel;
                    image_sum_squared += image_value * image_value;
                    const double difference = image_value - template_pixel;
                    squared_difference += difference * difference;
                }
            }

            double value = method == TM_SQDIFF || method == TM_SQDIFF_NORMED
                ? squared_difference
                : correlation;
            if (method == TM_SQDIFF_NORMED || method == TM_CCORR_NORMED)
            {
                const double denominator = std::sqrt(image_sum_squared * template_sum_squared);
                if (std::fabs(value) < denominator)
                {
                    value /= denominator;
                }
                else if (std::fabs(value) < denominator * 1.125)
                {
                    value = value > 0.0 ? 1.0 : -1.0;
                }
                else
                {
                    value = method == TM_SQDIFF_NORMED ? 1.0 : 0.0;
                }
            }
            result.at<float>(output_row, output_column) = static_cast<float>(value);
        }
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_TEMPLATE_MATCH_IMPL_HPP
