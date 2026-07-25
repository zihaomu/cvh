#ifndef CVH_IMGPROC_REMAP_H
#define CVH_IMGPROC_REMAP_H

#include "convert_maps.h"
#include "detail/geometric_sampling.hpp"

#include <algorithm>

namespace cvh {
namespace detail {

template<typename T>
inline void remap_typed(const Mat& source,
                        Mat& destination,
                        const Mat& map1,
                        const Mat& map2,
                        int interpolation,
                        int border_type,
                        const Scalar& border_value)
{
    const bool fixed = map1.type() == CV_16SC2;
    for (int row = 0; row < map1.size[0]; ++row)
    {
        T* output = reinterpret_cast<T*>(
            destination.data +
            static_cast<size_t>(row) * destination.step(0));
        const short* fixed_coordinates =
            fixed
                ? reinterpret_cast<const short*>(
                      map1.data +
                      static_cast<size_t>(row) * map1.step(0))
                : nullptr;
        const ushort* fixed_fractions =
            fixed && !map2.empty()
                ? reinterpret_cast<const ushort*>(
                      map2.data +
                      static_cast<size_t>(row) * map2.step(0))
                : nullptr;
        const float* float_map1 =
            !fixed
                ? reinterpret_cast<const float*>(
                      map1.data +
                      static_cast<size_t>(row) * map1.step(0))
                : nullptr;
        const float* float_map2 =
            !fixed && map1.type() == CV_32FC1
                ? reinterpret_cast<const float*>(
                      map2.data +
                      static_cast<size_t>(row) * map2.step(0))
                : nullptr;
        if constexpr (std::is_same<T, uchar>::value)
        {
            if (fixed && interpolation == INTER_LINEAR &&
                fixed_fractions)
            {
                geometric_write_linear_u8_fixed_row(
                    source,
                    output,
                    fixed_coordinates,
                    fixed_fractions,
                    map1.size[1],
                    border_type,
                    border_value);
                continue;
            }
            if (!fixed && interpolation == INTER_LINEAR)
            {
                constexpr int block_size = 64;
                int coordinates[block_size * 2];
                ushort fractions[block_size];
                for (int block_start = 0;
                     block_start < map1.size[1];
                     block_start += block_size)
                {
                    const int count = std::min(
                        block_size,
                        map1.size[1] - block_start);
                    for (int index = 0; index < count; ++index)
                    {
                        const int col = block_start + index;
                        const double source_x =
                            map1.type() == CV_32FC2
                                ? float_map1[col * 2]
                                : float_map1[col];
                        const double source_y =
                            map1.type() == CV_32FC2
                                ? float_map1[col * 2 + 1]
                                : float_map2[col];
                        int fraction_x = 0;
                        int fraction_y = 0;
                        geometric_fixed_linear_coordinate(
                            source_x,
                            coordinates[index * 2],
                            fraction_x);
                        geometric_fixed_linear_coordinate(
                            source_y,
                            coordinates[index * 2 + 1],
                            fraction_y);
                        fractions[index] =
                            geometric_pack_linear_fraction(
                                fraction_x,
                                fraction_y);
                    }
                    geometric_write_linear_u8_fixed_row(
                        source,
                        output +
                            static_cast<size_t>(block_start) *
                                source.channels(),
                        coordinates,
                        fractions,
                        count,
                        border_type,
                        border_value);
                }
                continue;
            }
        }
        for (int col = 0; col < map1.size[1]; ++col)
        {
            T* pixel =
                output + static_cast<size_t>(col) * source.channels();
            if (fixed)
            {
                const int integer_x =
                    fixed_coordinates[col * 2];
                const int integer_y =
                    fixed_coordinates[col * 2 + 1];
                const ushort fraction = !fixed_fractions
                    ? 0
                    : static_cast<ushort>(
                          fixed_fractions[col] &
                          (INTER_TAB_SIZE2 - 1));
                const int fraction_x =
                    fraction & (INTER_TAB_SIZE - 1);
                const int fraction_y =
                    fraction >> INTER_BITS;
                if (interpolation == INTER_NEAREST)
                {
                    geometric_write_nearest(
                        source,
                        pixel,
                        integer_x +
                            (fraction_x >= INTER_TAB_SIZE / 2),
                        integer_y +
                            (fraction_y >= INTER_TAB_SIZE / 2),
                        border_type,
                        border_value);
                }
                else
                {
                    if constexpr (std::is_same<T, uchar>::value)
                    {
                        geometric_write_linear_u8_fixed(
                            source,
                            pixel,
                            integer_x,
                            integer_y,
                            fraction_x,
                            fraction_y,
                            border_type,
                            border_value);
                    }
                    else
                    {
                        geometric_write_linear(
                            source,
                            pixel,
                            integer_x,
                            integer_y,
                            static_cast<double>(fraction_x) /
                                INTER_TAB_SIZE,
                            static_cast<double>(fraction_y) /
                                INTER_TAB_SIZE,
                            border_type,
                            border_value);
                    }
                }
                continue;
            }

            const double source_x =
                map1.type() == CV_32FC2
                    ? float_map1[col * 2]
                    : float_map1[col];
            const double source_y =
                map1.type() == CV_32FC2
                    ? float_map1[col * 2 + 1]
                    : float_map2[col];
            if constexpr (std::is_same<T, uchar>::value)
            {
                if (interpolation == INTER_LINEAR)
                {
                    int integer_x = 0;
                    int integer_y = 0;
                    int fraction_x = 0;
                    int fraction_y = 0;
                    geometric_fixed_linear_coordinate(
                        source_x,
                        integer_x,
                        fraction_x);
                    geometric_fixed_linear_coordinate(
                        source_y,
                        integer_y,
                        fraction_y);
                    geometric_write_linear_u8_fixed(
                        source,
                        pixel,
                        integer_x,
                        integer_y,
                        fraction_x,
                        fraction_y,
                        border_type,
                        border_value);
                    continue;
                }
            }
            geometric_write_coordinate(
                source,
                pixel,
                source_x,
                source_y,
                interpolation,
                border_type,
                border_value,
                true,
                true);
        }
    }
}

}  // namespace detail

inline void remap(const Mat& src,
                  Mat& dst,
                  const Mat& map1,
                  const Mat& map2,
                  int interpolation,
                  int borderMode = BORDER_CONSTANT,
                  const Scalar& borderValue = Scalar())
{
    if (src.empty() || src.dims != 2)
    {
        CV_Error(Error::StsBadArg, "remap expects a non-empty 2D source");
    }
    detail::validate_map_input(map1, map2);
    if (interpolation != INTER_NEAREST &&
        interpolation != INTER_LINEAR)
    {
        CV_Error(
            Error::StsBadFlag,
            "remap supports INTER_NEAREST and INTER_LINEAR only");
    }
    const int border_type = detail::normalize_border_type(borderMode);
    if (!detail::is_supported_filter_border(border_type))
    {
        CV_Error(
            Error::StsBadFlag,
            "remap supports constant, replicate, reflect, and reflect-101 borders");
    }
    if ((src.depth() != CV_8U && src.depth() != CV_32F) ||
        (src.channels() != 1 &&
         src.channels() != 3 &&
         src.channels() != 4))
    {
        CV_Error(
            Error::StsUnsupportedFormat,
            "remap supports U8/F32 C1/C3/C4 source");
    }

    const Mat source =
        src.data == dst.data ? src.clone() : src;
    const Mat first =
        map1.data == dst.data ? map1.clone() : map1;
    const Mat second =
        !map2.empty() && map2.data == dst.data
            ? map2.clone()
            : map2;
    dst.create(
        {first.size[0], first.size[1]},
        source.type());
    if (source.depth() == CV_8U)
    {
        detail::remap_typed<uchar>(
            source,
            dst,
            first,
            second,
            interpolation,
            border_type,
            borderValue);
    }
    else
    {
        detail::remap_typed<float>(
            source,
            dst,
            first,
            second,
            interpolation,
            border_type,
            borderValue);
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_REMAP_H
