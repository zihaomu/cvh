#ifndef CVH_IMGPROC_DETAIL_GEOMETRIC_SAMPLING_HPP
#define CVH_IMGPROC_DETAIL_GEOMETRIC_SAMPLING_HPP

#include "common.h"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <type_traits>

namespace cvh {
namespace detail {

template<typename T>
inline T geometric_cast(double value)
{
    if constexpr (std::is_same<T, uchar>::value)
    {
        return saturate_cast<uchar>(value);
    }
    return static_cast<T>(value);
}

template<typename T>
inline T geometric_border_value(const Scalar& value, int channel)
{
    const double scalar = value.val[channel < 4 ? channel : 3];
    return geometric_cast<T>(scalar);
}

template<typename T>
inline T geometric_read(const Mat& src,
                        int y,
                        int x,
                        int channel,
                        int border_type,
                        const Scalar& border_value)
{
    const int rows = src.size[0];
    const int cols = src.size[1];
    if (static_cast<unsigned>(y) < static_cast<unsigned>(rows) &&
        static_cast<unsigned>(x) < static_cast<unsigned>(cols))
    {
        const T* row = reinterpret_cast<const T*>(
            src.data + static_cast<size_t>(y) * src.step(0));
        return row[static_cast<size_t>(x) * src.channels() + channel];
    }
    if (border_type == BORDER_CONSTANT)
    {
        return geometric_border_value<T>(border_value, channel);
    }
    const int source_y = border_interpolate(y, rows, border_type);
    const int source_x = border_interpolate(x, cols, border_type);
    if (source_y < 0 || source_x < 0)
    {
        return geometric_border_value<T>(border_value, channel);
    }
    const T* row = reinterpret_cast<const T*>(
        src.data + static_cast<size_t>(source_y) * src.step(0));
    return row[
        static_cast<size_t>(source_x) * src.channels() + channel];
}

template<typename T>
inline void geometric_write_nearest(const Mat& src,
                                    T* destination,
                                    int source_x,
                                    int source_y,
                                    int border_type,
                                    const Scalar& border_value)
{
    if (static_cast<unsigned>(source_y) <
            static_cast<unsigned>(src.size[0]) &&
        static_cast<unsigned>(source_x) <
            static_cast<unsigned>(src.size[1]))
    {
        const T* source = reinterpret_cast<const T*>(
            src.data + static_cast<size_t>(source_y) * src.step(0)) +
            static_cast<size_t>(source_x) * src.channels();
        std::memcpy(
            destination,
            source,
            static_cast<size_t>(src.channels()) * sizeof(T));
        return;
    }
    for (int channel = 0; channel < src.channels(); ++channel)
    {
        destination[channel] = geometric_read<T>(
            src,
            source_y,
            source_x,
            channel,
            border_type,
            border_value);
    }
}

template<typename SourceT, typename DestinationT>
inline void geometric_write_linear_as(const Mat& src,
                                      DestinationT* destination,
                                      int source_x,
                                      int source_y,
                                      double fraction_x,
                                      double fraction_y,
                                      int border_type,
                                      const Scalar& border_value)
{
    for (int channel = 0; channel < src.channels(); ++channel)
    {
        const double top_left = geometric_read<SourceT>(
            src,
            source_y,
            source_x,
            channel,
            border_type,
            border_value);
        const double top_right = geometric_read<SourceT>(
            src,
            source_y,
            source_x + 1,
            channel,
            border_type,
            border_value);
        const double bottom_left = geometric_read<SourceT>(
            src,
            source_y + 1,
            source_x,
            channel,
            border_type,
            border_value);
        const double bottom_right = geometric_read<SourceT>(
            src,
            source_y + 1,
            source_x + 1,
            channel,
            border_type,
            border_value);
        const double top =
            top_left + (top_right - top_left) * fraction_x;
        const double bottom =
            bottom_left + (bottom_right - bottom_left) * fraction_x;
        destination[channel] = geometric_cast<DestinationT>(
            top + (bottom - top) * fraction_y);
    }
}

template<typename T>
inline void geometric_write_linear(const Mat& src,
                                   T* destination,
                                   int source_x,
                                   int source_y,
                                   double fraction_x,
                                   double fraction_y,
                                   int border_type,
                                   const Scalar& border_value)
{
    geometric_write_linear_as<T, T>(
        src,
        destination,
        source_x,
        source_y,
        fraction_x,
        fraction_y,
        border_type,
        border_value);
}

template<int Channels>
inline void geometric_write_linear_f32_interior(
    const Mat& src,
    float* destination,
    int source_x,
    int source_y,
    double fraction_x,
    double fraction_y)
{
    const float* row0 = reinterpret_cast<const float*>(
        src.data + static_cast<size_t>(source_y) * src.step(0));
    const float* row1 = reinterpret_cast<const float*>(
        src.data + static_cast<size_t>(source_y + 1) * src.step(0));
    const float* top_left =
        row0 + static_cast<size_t>(source_x) * Channels;
    const float* bottom_left =
        row1 + static_cast<size_t>(source_x) * Channels;
    for (int channel = 0; channel < Channels; ++channel)
    {
        const double top =
            static_cast<double>(top_left[channel]) +
            (static_cast<double>(top_left[Channels + channel]) -
             static_cast<double>(top_left[channel])) *
                fraction_x;
        const double bottom =
            static_cast<double>(bottom_left[channel]) +
            (static_cast<double>(bottom_left[Channels + channel]) -
             static_cast<double>(bottom_left[channel])) *
                fraction_x;
        destination[channel] = static_cast<float>(
            top + (bottom - top) * fraction_y);
    }
}

inline void geometric_write_linear_f32(
    const Mat& src,
    float* destination,
    int source_x,
    int source_y,
    double fraction_x,
    double fraction_y,
    int border_type,
    const Scalar& border_value)
{
    if (static_cast<unsigned>(source_y) <
            static_cast<unsigned>(src.size[0] - 1) &&
        static_cast<unsigned>(source_x) <
            static_cast<unsigned>(src.size[1] - 1))
    {
        switch (src.channels())
        {
        case 1:
            geometric_write_linear_f32_interior<1>(
                src,
                destination,
                source_x,
                source_y,
                fraction_x,
                fraction_y);
            return;
        case 3:
            geometric_write_linear_f32_interior<3>(
                src,
                destination,
                source_x,
                source_y,
                fraction_x,
                fraction_y);
            return;
        case 4:
            geometric_write_linear_f32_interior<4>(
                src,
                destination,
                source_x,
                source_y,
                fraction_x,
                fraction_y);
            return;
        default:
            break;
        }
    }
    geometric_write_linear(
        src,
        destination,
        source_x,
        source_y,
        fraction_x,
        fraction_y,
        border_type,
        border_value);
}

inline int geometric_resolve_index(int index,
                                   int length,
                                   int border_type)
{
    if (static_cast<unsigned>(index) <
        static_cast<unsigned>(length))
    {
        return index;
    }
    return border_type == BORDER_CONSTANT
        ? -1
        : border_interpolate(index, length, border_type);
}

inline void geometric_fixed_linear_coordinate(double coordinate,
                                              int& integer,
                                              int& fraction)
{
    const long scaled = std::lrint(
        coordinate * static_cast<double>(INTER_TAB_SIZE));
    integer = static_cast<int>(std::floor(
        static_cast<double>(scaled) /
        static_cast<double>(INTER_TAB_SIZE)));
    fraction = static_cast<int>(
        scaled - static_cast<long>(integer) * INTER_TAB_SIZE);
}

inline ushort geometric_pack_linear_fraction(int fraction_x,
                                             int fraction_y)
{
    return static_cast<ushort>(
        fraction_x + fraction_y * INTER_TAB_SIZE);
}

template<int Channels>
inline void geometric_write_linear_u8_fixed_interior(
    const Mat& src,
    uchar* destination,
    int source_x,
    int source_y,
    int fraction_x,
    int fraction_y)
{
    const uchar* row0 =
        src.data + static_cast<size_t>(source_y) * src.step(0);
    const uchar* row1 = row0 + src.step(0);
    const uchar* top_left =
        row0 + static_cast<size_t>(source_x) * Channels;
    const uchar* bottom_left =
        row1 + static_cast<size_t>(source_x) * Channels;
    const int inverse_x = INTER_TAB_SIZE - fraction_x;
    const int inverse_y = INTER_TAB_SIZE - fraction_y;
    const int weight00 = inverse_x * inverse_y;
    const int weight01 = fraction_x * inverse_y;
    const int weight10 = inverse_x * fraction_y;
    const int weight11 = fraction_x * fraction_y;
    for (int channel = 0; channel < Channels; ++channel)
    {
        destination[channel] = saturate_cast<uchar>(
            (top_left[channel] * weight00 +
             top_left[Channels + channel] * weight01 +
             bottom_left[channel] * weight10 +
             bottom_left[Channels + channel] * weight11 +
             INTER_TAB_SIZE2 / 2) /
            INTER_TAB_SIZE2);
    }
}

inline void geometric_write_linear_u8_fixed(
    const Mat& src,
    uchar* destination,
    int source_x,
    int source_y,
    int fraction_x,
    int fraction_y,
    int border_type,
    const Scalar& border_value)
{
    if (static_cast<unsigned>(source_y) <
            static_cast<unsigned>(src.size[0] - 1) &&
        static_cast<unsigned>(source_x) <
            static_cast<unsigned>(src.size[1] - 1))
    {
        switch (src.channels())
        {
        case 1:
            geometric_write_linear_u8_fixed_interior<1>(
                src,
                destination,
                source_x,
                source_y,
                fraction_x,
                fraction_y);
            return;
        case 3:
            geometric_write_linear_u8_fixed_interior<3>(
                src,
                destination,
                source_x,
                source_y,
                fraction_x,
                fraction_y);
            return;
        case 4:
            geometric_write_linear_u8_fixed_interior<4>(
                src,
                destination,
                source_x,
                source_y,
                fraction_x,
                fraction_y);
            return;
        default:
            break;
        }
    }
    const int x0 = geometric_resolve_index(
        source_x, src.size[1], border_type);
    const int x1 = geometric_resolve_index(
        source_x + 1, src.size[1], border_type);
    const int y0 = geometric_resolve_index(
        source_y, src.size[0], border_type);
    const int y1 = geometric_resolve_index(
        source_y + 1, src.size[0], border_type);
    const uchar* row0 =
        y0 >= 0
            ? src.data + static_cast<size_t>(y0) * src.step(0)
            : nullptr;
    const uchar* row1 =
        y1 >= 0
            ? src.data + static_cast<size_t>(y1) * src.step(0)
            : nullptr;
    const int channels = src.channels();
    const int inverse_x = INTER_TAB_SIZE - fraction_x;
    const int inverse_y = INTER_TAB_SIZE - fraction_y;
    const int weight00 = inverse_x * inverse_y;
    const int weight01 = fraction_x * inverse_y;
    const int weight10 = inverse_x * fraction_y;
    const int weight11 = fraction_x * fraction_y;
    for (int channel = 0; channel < channels; ++channel)
    {
        const int border = saturate_cast<uchar>(
            border_value.val[channel < 4 ? channel : 3]);
        const int top_left =
            row0 && x0 >= 0
                ? row0[static_cast<size_t>(x0) * channels + channel]
                : border;
        const int top_right =
            row0 && x1 >= 0
                ? row0[static_cast<size_t>(x1) * channels + channel]
                : border;
        const int bottom_left =
            row1 && x0 >= 0
                ? row1[static_cast<size_t>(x0) * channels + channel]
                : border;
        const int bottom_right =
            row1 && x1 >= 0
                ? row1[static_cast<size_t>(x1) * channels + channel]
                : border;
        destination[channel] = saturate_cast<uchar>(
            (top_left * weight00 +
             top_right * weight01 +
             bottom_left * weight10 +
             bottom_right * weight11 +
             INTER_TAB_SIZE2 / 2) /
            INTER_TAB_SIZE2);
    }
}

template<typename CoordinateT>
inline bool geometric_try_linear_u8_fixed_contiguous_row(
    const Mat& src,
    uchar* destination,
    const CoordinateT* coordinates,
    const ushort* fractions,
    int count)
{
    const int channels = src.channels();
    if ((channels != 1 && channels != 3 && channels != 4) || count <= 0)
    {
        return false;
    }
    const int source_x = static_cast<int>(coordinates[0]);
    const int source_y = static_cast<int>(coordinates[1]);
    const ushort fraction = fractions[0];
    if (static_cast<unsigned>(source_y) >=
            static_cast<unsigned>(src.size[0] - 1) ||
        source_x < 0 || source_x + count >= src.size[1])
    {
        return false;
    }
    for (int index = 1; index < count; ++index)
    {
        if (static_cast<int>(coordinates[index * 2]) !=
                source_x + index ||
            static_cast<int>(coordinates[index * 2 + 1]) != source_y ||
            fractions[index] != fraction)
        {
            return false;
        }
    }
    const int fraction_x = fraction & (INTER_TAB_SIZE - 1);
    const int fraction_y = fraction >> INTER_BITS;
    const int inverse_x = INTER_TAB_SIZE - fraction_x;
    const int inverse_y = INTER_TAB_SIZE - fraction_y;
    const int weight00 = inverse_x * inverse_y;
    const int weight01 = fraction_x * inverse_y;
    const int weight10 = inverse_x * fraction_y;
    const int weight11 = fraction_x * fraction_y;
    const uchar* top =
        src.data + static_cast<size_t>(source_y) * src.step(0) +
        static_cast<size_t>(source_x) * channels;
    const uchar* bottom = top + src.step(0);
    const int scalar_count = count * channels;
    for (int index = 0; index < scalar_count; ++index)
    {
        destination[index] = static_cast<uchar>(
            (top[index] * weight00 +
             top[index + channels] * weight01 +
             bottom[index] * weight10 +
             bottom[index + channels] * weight11 +
             INTER_TAB_SIZE2 / 2) /
            INTER_TAB_SIZE2);
    }
    return true;
}

template<typename CoordinateT>
inline void geometric_write_linear_u8_fixed_row(
    const Mat& src,
    uchar* destination,
    const CoordinateT* coordinates,
    const ushort* fractions,
    int count,
    int border_type,
    const Scalar& border_value)
{
    if (geometric_try_linear_u8_fixed_contiguous_row(
            src,
            destination,
            coordinates,
            fractions,
            count))
    {
        return;
    }
    const int channels = src.channels();
    for (int index = 0; index < count; ++index)
    {
        const ushort fraction = fractions[index];
        geometric_write_linear_u8_fixed(
            src,
            destination + static_cast<size_t>(index) * channels,
            static_cast<int>(coordinates[index * 2]),
            static_cast<int>(coordinates[index * 2 + 1]),
            fraction & (INTER_TAB_SIZE - 1),
            fraction >> INTER_BITS,
            border_type,
            border_value);
    }
}

inline void geometric_write_linear_u8(
    const Mat& src,
    uchar* destination,
    int source_x,
    int source_y,
    double fraction_x,
    double fraction_y,
    int border_type,
    const Scalar& border_value)
{
    const int x0 = geometric_resolve_index(
        source_x, src.size[1], border_type);
    const int x1 = geometric_resolve_index(
        source_x + 1, src.size[1], border_type);
    const int y0 = geometric_resolve_index(
        source_y, src.size[0], border_type);
    const int y1 = geometric_resolve_index(
        source_y + 1, src.size[0], border_type);
    const uchar* row0 =
        y0 >= 0
            ? src.data + static_cast<size_t>(y0) * src.step(0)
            : nullptr;
    const uchar* row1 =
        y1 >= 0
            ? src.data + static_cast<size_t>(y1) * src.step(0)
            : nullptr;
    const int channels = src.channels();
    const double weight00 =
        (1.0 - fraction_x) * (1.0 - fraction_y);
    const double weight01 =
        fraction_x * (1.0 - fraction_y);
    const double weight10 =
        (1.0 - fraction_x) * fraction_y;
    const double weight11 = fraction_x * fraction_y;
    for (int channel = 0; channel < channels; ++channel)
    {
        const double border =
            border_value.val[channel < 4 ? channel : 3];
        const double top_left =
            row0 && x0 >= 0
                ? row0[static_cast<size_t>(x0) * channels + channel]
                : border;
        const double top_right =
            row0 && x1 >= 0
                ? row0[static_cast<size_t>(x1) * channels + channel]
                : border;
        const double bottom_left =
            row1 && x0 >= 0
                ? row1[static_cast<size_t>(x0) * channels + channel]
                : border;
        const double bottom_right =
            row1 && x1 >= 0
                ? row1[static_cast<size_t>(x1) * channels + channel]
                : border;
        destination[channel] = saturate_cast<uchar>(
            top_left * weight00 +
            top_right * weight01 +
            bottom_left * weight10 +
            bottom_right * weight11);
    }
}

inline void geometric_linear_coordinate(double coordinate,
                                        bool quantize,
                                        int& integer,
                                        double& fraction)
{
    if (!quantize)
    {
        integer = static_cast<int>(std::floor(coordinate));
        fraction = coordinate - static_cast<double>(integer);
        return;
    }
    const long scaled = std::lrint(
        coordinate * static_cast<double>(INTER_TAB_SIZE));
    integer = static_cast<int>(std::floor(
        static_cast<double>(scaled) /
        static_cast<double>(INTER_TAB_SIZE)));
    fraction =
        static_cast<double>(
            scaled - static_cast<long>(integer) * INTER_TAB_SIZE) /
        static_cast<double>(INTER_TAB_SIZE);
}

template<typename T>
inline void geometric_write_coordinate(const Mat& src,
                                       T* destination,
                                       double source_x,
                                       double source_y,
                                       int interpolation,
                                       int border_type,
                                       const Scalar& border_value,
                                       bool quantize_linear,
                                       bool nearest_even)
{
    if (interpolation == INTER_NEAREST)
    {
        const int nearest_x = nearest_even
            ? static_cast<int>(std::lrint(source_x))
            : static_cast<int>(std::floor(source_x + 0.5));
        const int nearest_y = nearest_even
            ? static_cast<int>(std::lrint(source_y))
            : static_cast<int>(std::floor(source_y + 0.5));
        geometric_write_nearest(
            src,
            destination,
            nearest_x,
            nearest_y,
            border_type,
            border_value);
        return;
    }
    int integer_x = 0;
    int integer_y = 0;
    double fraction_x = 0.0;
    double fraction_y = 0.0;
    geometric_linear_coordinate(
        source_x,
        quantize_linear,
        integer_x,
        fraction_x);
    geometric_linear_coordinate(
        source_y,
        quantize_linear,
        integer_y,
        fraction_y);
    geometric_write_linear(
        src,
        destination,
        integer_x,
        integer_y,
        fraction_x,
        fraction_y,
        border_type,
        border_value);
}

}  // namespace detail
}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_GEOMETRIC_SAMPLING_HPP
