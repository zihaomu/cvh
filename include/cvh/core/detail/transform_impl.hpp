#ifndef CVH_CORE_DETAIL_TRANSFORM_IMPL_HPP
#define CVH_CORE_DETAIL_TRANSFORM_IMPL_HPP

#include "../transform.h"
#include "dispatch_control.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace cvh
{
namespace detail
{

struct TransformCoefficients
{
    static constexpr int stride = 5;
    std::array<double, 4 * stride> values {};

    double at(int row, int column) const
    {
        return values[static_cast<std::size_t>(row) * stride + column];
    }
};

inline void validate_transform_matrix(const Mat& matrix, const char* function_name)
{
    if (matrix.empty() || matrix.dims != 2 || matrix.channels() != 1 ||
        (matrix.depth() != CV_32F && matrix.depth() != CV_64F))
    {
        CV_Error_(Error::StsBadArg,
                  ("%s expects a non-empty 2D F32/F64 C1 matrix", function_name));
    }
}

inline TransformCoefficients pack_transform_coefficients(const Mat& matrix)
{
    TransformCoefficients coefficients;
    for (int row = 0; row < matrix.size[0]; ++row)
    {
        if (matrix.depth() == CV_32F)
        {
            const float* source = reinterpret_cast<const float*>(
                matrix.data + static_cast<std::size_t>(row) * matrix.step(0));
            for (int column = 0; column < matrix.size[1]; ++column)
            {
                coefficients.values[
                    static_cast<std::size_t>(row) * TransformCoefficients::stride +
                    column] = static_cast<double>(source[column]);
            }
        }
        else
        {
            const double* source = reinterpret_cast<const double*>(
                matrix.data + static_cast<std::size_t>(row) * matrix.step(0));
            for (int column = 0; column < matrix.size[1]; ++column)
            {
                coefficients.values[
                    static_cast<std::size_t>(row) * TransformCoefficients::stride +
                    column] = source[column];
            }
        }
    }
    return coefficients;
}

template<int Channels, typename T>
inline double transform_dot(const T* pixel,
                            const TransformCoefficients& coefficients,
                            int output_channel,
                            double initial)
{
    double value = initial;
    if constexpr (Channels >= 1)
    {
        value += static_cast<double>(pixel[0]) *
                 coefficients.at(output_channel, 0);
    }
    if constexpr (Channels >= 2)
    {
        value += static_cast<double>(pixel[1]) *
                 coefficients.at(output_channel, 1);
    }
    if constexpr (Channels >= 3)
    {
        value += static_cast<double>(pixel[2]) *
                 coefficients.at(output_channel, 2);
    }
    if constexpr (Channels >= 4)
    {
        value += static_cast<double>(pixel[3]) *
                 coefficients.at(output_channel, 3);
    }
    return value;
}

template<typename T, int SourceChannels, bool Affine>
inline void transform_span(const T* source,
                           T* destination,
                           std::size_t pixel_count,
                           const TransformCoefficients& coefficients,
                           int destination_channels)
{
    for (std::size_t index = 0; index < pixel_count; ++index)
    {
        const T* pixel = source + index * SourceChannels;
        T* output = destination + index * destination_channels;
        for (int output_channel = 0;
             output_channel < destination_channels;
             ++output_channel)
        {
            const double initial = Affine
                ? coefficients.at(output_channel, SourceChannels)
                : 0.0;
            output[output_channel] = static_cast<T>(
                transform_dot<SourceChannels>(
                    pixel, coefficients, output_channel, initial));
        }
    }
}

template<typename T, int SourceChannels, bool Affine>
inline void transform_typed(const Mat& src,
                            Mat& dst,
                            const TransformCoefficients& coefficients,
                            int destination_channels)
{
    if (src.isContinuous() && dst.isContinuous())
    {
        transform_span<T, SourceChannels, Affine>(
            reinterpret_cast<const T*>(src.data),
            reinterpret_cast<T*>(dst.data),
            src.total(),
            coefficients,
            destination_channels);
        return;
    }

    for (int row = 0; row < src.size[0]; ++row)
    {
        const T* source = reinterpret_cast<const T*>(
            src.data + static_cast<std::size_t>(row) * src.step(0));
        T* destination = reinterpret_cast<T*>(
            dst.data + static_cast<std::size_t>(row) * dst.step(0));
        transform_span<T, SourceChannels, Affine>(
            source,
            destination,
            static_cast<std::size_t>(src.size[1]),
            coefficients,
            destination_channels);
    }
}

template<typename T, bool Affine>
inline void transform_source_dispatch(
    const Mat& src,
    Mat& dst,
    const TransformCoefficients& coefficients,
    int destination_channels)
{
    switch (src.channels())
    {
        case 1:
            transform_typed<T, 1, Affine>(
                src, dst, coefficients, destination_channels);
            break;
        case 2:
            transform_typed<T, 2, Affine>(
                src, dst, coefficients, destination_channels);
            break;
        case 3:
            transform_typed<T, 3, Affine>(
                src, dst, coefficients, destination_channels);
            break;
        case 4:
            transform_typed<T, 4, Affine>(
                src, dst, coefficients, destination_channels);
            break;
        default:
            break;
    }
}

template<typename T, int Channels>
inline void perspective_transform_span(
    const T* source,
    T* destination,
    std::size_t pixel_count,
    const TransformCoefficients& coefficients)
{
    const double epsilon =
        static_cast<double>(std::numeric_limits<float>::epsilon());
    for (std::size_t index = 0; index < pixel_count; ++index)
    {
        const T* pixel = source + index * Channels;
        T* output = destination + index * Channels;
        const double w = transform_dot<Channels>(
            pixel,
            coefficients,
            Channels,
            coefficients.at(Channels, Channels));
        if (!(std::fabs(w) > epsilon))
        {
            for (int output_channel = 0;
                 output_channel < Channels;
                 ++output_channel)
            {
                output[output_channel] = static_cast<T>(0);
            }
            continue;
        }

        const double scale = 1.0 / w;
        for (int output_channel = 0;
             output_channel < Channels;
             ++output_channel)
        {
            output[output_channel] = static_cast<T>(
                transform_dot<Channels>(
                    pixel,
                    coefficients,
                    output_channel,
                    coefficients.at(output_channel, Channels)) *
                scale);
        }
    }
}

template<typename T, int Channels>
inline void perspective_transform_typed(
    const Mat& src,
    Mat& dst,
    const TransformCoefficients& coefficients)
{
    if (src.isContinuous() && dst.isContinuous())
    {
        perspective_transform_span<T, Channels>(
            reinterpret_cast<const T*>(src.data),
            reinterpret_cast<T*>(dst.data),
            src.total(),
            coefficients);
        return;
    }

    for (int row = 0; row < src.size[0]; ++row)
    {
        const T* source = reinterpret_cast<const T*>(
            src.data + static_cast<std::size_t>(row) * src.step(0));
        T* destination = reinterpret_cast<T*>(
            dst.data + static_cast<std::size_t>(row) * dst.step(0));
        perspective_transform_span<T, Channels>(
            source,
            destination,
            static_cast<std::size_t>(src.size[1]),
            coefficients);
    }
}

template<typename T>
inline void perspective_transform_dispatch(
    const Mat& src,
    Mat& dst,
    const TransformCoefficients& coefficients)
{
    if (src.channels() == 2)
    {
        perspective_transform_typed<T, 2>(src, dst, coefficients);
    }
    else
    {
        perspective_transform_typed<T, 3>(src, dst, coefficients);
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
    const detail::TransformCoefficients coefficients =
        detail::pack_transform_coefficients(matrix);
    const bool affine = matrix.size[1] == source.channels() + 1;
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    if (source.depth() == CV_32F)
    {
        if (affine)
        {
            detail::transform_source_dispatch<float, true>(
                source, dst, coefficients, destination_channels);
        }
        else
        {
            detail::transform_source_dispatch<float, false>(
                source, dst, coefficients, destination_channels);
        }
    }
    else if (affine)
    {
        detail::transform_source_dispatch<double, true>(
            source, dst, coefficients, destination_channels);
    }
    else
    {
        detail::transform_source_dispatch<double, false>(
            source, dst, coefficients, destination_channels);
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
    const detail::TransformCoefficients coefficients =
        detail::pack_transform_coefficients(matrix);
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    if (source.depth() == CV_32F)
    {
        detail::perspective_transform_dispatch<float>(source, dst, coefficients);
    }
    else
    {
        detail::perspective_transform_dispatch<double>(source, dst, coefficients);
    }
}

}  // namespace cvh

#endif  // CVH_CORE_DETAIL_TRANSFORM_IMPL_HPP
