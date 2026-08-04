#ifndef CVH_IMGPROC_DETAIL_TEMPLATE_MATCH_IMPL_HPP
#define CVH_IMGPROC_DETAIL_TEMPLATE_MATCH_IMPL_HPP

#include "../template_match.h"
#include "template_match_ui.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace cvh
{
namespace detail
{

template<typename T>
inline const T* template_match_row(const Mat& matrix, int row)
{
    return reinterpret_cast<const T*>(
        matrix.data + static_cast<std::size_t>(row) * matrix.step(0));
}

template<typename T>
inline double template_energy(const Mat& templ)
{
    double energy = 0.0;
    for (int row = 0; row < templ.size[0]; ++row)
    {
        const T* values = template_match_row<T>(templ, row);
        for (int column = 0; column < templ.size[1]; ++column)
        {
            const double value = static_cast<double>(values[column]);
            energy += value * value;
        }
    }
    return energy;
}

template<typename T>
inline std::vector<double> squared_integral(const Mat& image)
{
    const int integral_columns = image.size[1] + 1;
    std::vector<double> integral(
        static_cast<std::size_t>(image.size[0] + 1) *
            static_cast<std::size_t>(integral_columns),
        0.0);
    for (int row = 0; row < image.size[0]; ++row)
    {
        const T* source = template_match_row<T>(image, row);
        const double* previous = integral.data() +
            static_cast<std::size_t>(row) * integral_columns;
        double* current = integral.data() +
            static_cast<std::size_t>(row + 1) * integral_columns;
        double row_sum = 0.0;
        for (int column = 0; column < image.size[1]; ++column)
        {
            const double value = static_cast<double>(source[column]);
            row_sum += value * value;
            current[column + 1] = previous[column + 1] + row_sum;
        }
    }
    return integral;
}

inline double squared_window_sum(const std::vector<double>& integral,
                                 int integral_columns,
                                 int row,
                                 int column,
                                 int rows,
                                 int columns)
{
    const std::size_t top = static_cast<std::size_t>(row) * integral_columns;
    const std::size_t bottom =
        static_cast<std::size_t>(row + rows) * integral_columns;
    return integral[bottom + column + columns] -
           integral[top + column + columns] -
           integral[bottom + column] +
           integral[top + column];
}

template<typename T>
inline double template_correlation_scalar(const Mat& image,
                                          const Mat& templ,
                                          int output_row,
                                          int output_column)
{
    double correlation = 0.0;
    for (int row = 0; row < templ.size[0]; ++row)
    {
        const T* image_values = template_match_row<T>(image, output_row + row) +
            output_column;
        const T* template_values = template_match_row<T>(templ, row);
        for (int column = 0; column < templ.size[1]; ++column)
        {
            correlation += static_cast<double>(image_values[column]) *
                           static_cast<double>(template_values[column]);
        }
    }
    return correlation;
}

template<typename T>
inline double template_correlation(const Mat& image,
                                   const Mat& templ,
                                   int output_row,
                                   int output_column,
                                   bool use_ui)
{
    if (use_ui)
    {
        const T* image_values = template_match_row<T>(image, output_row) +
            output_column;
        const T* template_values = template_match_row<T>(templ, 0);
        if constexpr (std::is_same<T, uchar>::value)
        {
            return template_match_ui::dot_u8(
                image_values,
                image.step(0),
                template_values,
                templ.step(0),
                templ.size[0],
                templ.size[1]);
        }
        else
        {
            return template_match_ui::dot_f32(
                image_values,
                image.step(0),
                template_values,
                templ.step(0),
                templ.size[0],
                templ.size[1]);
        }
    }
    return template_correlation_scalar<T>(
        image, templ, output_row, output_column);
}

template<typename T>
inline void match_template_typed(const Mat& image,
                                 const Mat& templ,
                                 Mat& result,
                                 int method,
                                 bool use_ui)
{
    const bool needs_energy = method != TM_CCORR;
    const double templ_energy = needs_energy ? template_energy<T>(templ) : 0.0;
    const std::vector<double> image_squared = needs_energy
        ? squared_integral<T>(image)
        : std::vector<double>();
    const int integral_columns = image.size[1] + 1;
    const int output_rows = result.size[0];
    const int output_columns = result.size[1];

    for (int output_row = 0; output_row < output_rows; ++output_row)
    {
        float* output = reinterpret_cast<float*>(
            result.data + static_cast<std::size_t>(output_row) * result.step(0));
        for (int output_column = 0; output_column < output_columns; ++output_column)
        {
            const double correlation = template_correlation<T>(
                image, templ, output_row, output_column, use_ui);
            if (method == TM_CCORR)
            {
                output[output_column] = static_cast<float>(correlation);
                continue;
            }

            const double image_energy = squared_window_sum(
                image_squared,
                integral_columns,
                output_row,
                output_column,
                templ.size[0],
                templ.size[1]);
            double value = correlation;
            if (method == TM_SQDIFF || method == TM_SQDIFF_NORMED)
            {
                value = std::max(
                    image_energy - 2.0 * correlation + templ_energy,
                    0.0);
            }

            if (method == TM_SQDIFF_NORMED || method == TM_CCORR_NORMED)
            {
                const double denominator =
                    std::sqrt(image_energy * templ_energy);
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
            output[output_column] = static_cast<float>(value);
        }
    }
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

    bool use_ui = false;
    if (source_image.depth() == CV_8U)
    {
        use_ui = detail::template_match_ui::can_dot_u8(
            source_template.size[0], source_template.size[1]);
    }
    else
    {
        use_ui = detail::template_match_ui::can_dot_f32(
            source_template.size[1]);
    }
    cpu::set_last_dispatch_tag(
        use_ui ? cpu::DispatchTag::OpenCVUI : cpu::DispatchTag::Scalar);

    if (source_image.depth() == CV_8U)
    {
        detail::match_template_typed<uchar>(
            source_image, source_template, result, method, use_ui);
    }
    else
    {
        detail::match_template_typed<float>(
            source_image, source_template, result, method, use_ui);
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_TEMPLATE_MATCH_IMPL_HPP
