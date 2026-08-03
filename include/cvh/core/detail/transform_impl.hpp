#ifndef CVH_CORE_DETAIL_TRANSFORM_IMPL_HPP
#define CVH_CORE_DETAIL_TRANSFORM_IMPL_HPP

#include "../transform.h"

#include <cmath>
#include <limits>
#include <vector>

namespace cvh
{
namespace detail
{

inline double transform_matrix_value(const Mat& matrix, int row, int column)
{
    return matrix.depth() == CV_32F
        ? static_cast<double>(matrix.at<float>(row, column))
        : matrix.at<double>(row, column);
}

inline void validate_transform_matrix(const Mat& matrix, const char* function_name)
{
    if (matrix.empty() || matrix.dims != 2 || matrix.channels() != 1 ||
        (matrix.depth() != CV_32F && matrix.depth() != CV_64F))
    {
        CV_Error_(Error::StsBadArg,
                  ("%s expects a non-empty 2D F32/F64 C1 matrix", function_name));
    }
}

template<typename T>
inline void transform_typed(const Mat& src, Mat& dst, const Mat& matrix, int destination_channels)
{
    const int source_channels = src.channels();
    const bool affine = matrix.size[1] == source_channels + 1;
    for (int row = 0; row < src.size[0]; ++row)
    {
        const T* source = reinterpret_cast<const T*>(src.data + static_cast<size_t>(row) * src.step(0));
        T* destination = reinterpret_cast<T*>(dst.data + static_cast<size_t>(row) * dst.step(0));
        for (int column = 0; column < src.size[1]; ++column)
        {
            const T* pixel = source + static_cast<size_t>(column) * source_channels;
            T* output = destination + static_cast<size_t>(column) * destination_channels;
            for (int output_channel = 0; output_channel < destination_channels; ++output_channel)
            {
                double value = affine
                    ? transform_matrix_value(matrix, output_channel, source_channels)
                    : 0.0;
                for (int input_channel = 0; input_channel < source_channels; ++input_channel)
                {
                    value += static_cast<double>(pixel[input_channel]) *
                             transform_matrix_value(matrix, output_channel, input_channel);
                }
                output[output_channel] = static_cast<T>(value);
            }
        }
    }
}

template<typename T>
inline void perspective_transform_typed(const Mat& src, Mat& dst, const Mat& matrix)
{
    const int channels = src.channels();
    const double epsilon = static_cast<double>(std::numeric_limits<float>::epsilon());
    for (int row = 0; row < src.size[0]; ++row)
    {
        const T* source = reinterpret_cast<const T*>(src.data + static_cast<size_t>(row) * src.step(0));
        T* destination = reinterpret_cast<T*>(dst.data + static_cast<size_t>(row) * dst.step(0));
        for (int column = 0; column < src.size[1]; ++column)
        {
            const T* pixel = source + static_cast<size_t>(column) * channels;
            T* output = destination + static_cast<size_t>(column) * channels;
            double w = transform_matrix_value(matrix, channels, channels);
            for (int input_channel = 0; input_channel < channels; ++input_channel)
            {
                w += static_cast<double>(pixel[input_channel]) *
                     transform_matrix_value(matrix, channels, input_channel);
            }
            if (!(std::fabs(w) > epsilon))
            {
                for (int output_channel = 0; output_channel < channels; ++output_channel)
                {
                    output[output_channel] = static_cast<T>(0);
                }
                continue;
            }
            const double scale = 1.0 / w;
            for (int output_channel = 0; output_channel < channels; ++output_channel)
            {
                double value = transform_matrix_value(matrix, output_channel, channels);
                for (int input_channel = 0; input_channel < channels; ++input_channel)
                {
                    value += static_cast<double>(pixel[input_channel]) *
                             transform_matrix_value(matrix, output_channel, input_channel);
                }
                output[output_channel] = static_cast<T>(value * scale);
            }
        }
    }
}

}  // namespace detail

inline void transform(const Mat& src, Mat& dst, const Mat& matrix)
{
    if (src.empty() || src.dims != 2 ||
        (src.depth() != CV_32F && src.depth() != CV_64F) ||
        src.channels() < 1 || src.channels() > 4)
    {
        CV_Error(Error::StsBadArg, "transform supports non-empty 2D F32/F64 C1-C4 src");
    }
    detail::validate_transform_matrix(matrix, "transform");
    const int destination_channels = matrix.size[0];
    if (destination_channels < 1 || destination_channels > 4 ||
        (matrix.size[1] != src.channels() && matrix.size[1] != src.channels() + 1))
    {
        CV_Error(Error::StsUnmatchedSizes, "transform matrix shape is incompatible with src channels");
    }

    const Mat source = src.data == dst.data ? src.clone() : src;
    dst.create(source.dims, source.size.p, CV_MAKETYPE(source.depth(), destination_channels));
    if (source.depth() == CV_32F)
    {
        detail::transform_typed<float>(source, dst, matrix, destination_channels);
    }
    else
    {
        detail::transform_typed<double>(source, dst, matrix, destination_channels);
    }
}

inline void perspectiveTransform(const Mat& src, Mat& dst, const Mat& matrix)
{
    if (src.empty() || src.dims != 2 ||
        (src.depth() != CV_32F && src.depth() != CV_64F) ||
        (src.channels() != 2 && src.channels() != 3) ||
        (src.size[0] != 1 && src.size[1] != 1))
    {
        CV_Error(Error::StsBadArg, "perspectiveTransform supports F32/F64 C2/C3 point vectors");
    }
    detail::validate_transform_matrix(matrix, "perspectiveTransform");
    const int matrix_size = src.channels() + 1;
    if (matrix.size[0] != matrix_size || matrix.size[1] != matrix_size)
    {
        CV_Error(Error::StsUnmatchedSizes, "perspectiveTransform matrix shape is incompatible with src channels");
    }

    const Mat source = src.data == dst.data ? src.clone() : src;
    dst.create(source.dims, source.size.p, source.type());
    if (source.depth() == CV_32F)
    {
        detail::perspective_transform_typed<float>(source, dst, matrix);
    }
    else
    {
        detail::perspective_transform_typed<double>(source, dst, matrix);
    }
}

}  // namespace cvh

#endif  // CVH_CORE_DETAIL_TRANSFORM_IMPL_HPP
