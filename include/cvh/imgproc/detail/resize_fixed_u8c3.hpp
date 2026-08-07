#ifndef CVH_IMGPROC_DETAIL_RESIZE_FIXED_U8C3_HPP
#define CVH_IMGPROC_DETAIL_RESIZE_FIXED_U8C3_HPP

#include "fastpath_common.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace cvh
{
namespace detail
{
namespace resize_fixed_u8c3
{

constexpr int coordinate_fraction_bits = 16;
constexpr int interpolation_fraction_bits = 8;
constexpr std::uint64_t coordinate_half =
    std::uint64_t{1} << (coordinate_fraction_bits - 1);
constexpr std::uint64_t interpolation_rounding_bias =
    std::uint64_t{1} <<
    (coordinate_fraction_bits - interpolation_fraction_bits - 1);

struct AxisCoordinate
{
    int first = 0;
    int second = 0;
    std::uint16_t fraction = 0;
};

struct FlatBlock
{
    std::size_t source_byte_base = 0;
    std::array<uchar, 16> left_index = {};
    std::array<std::uint16_t, 16> x_fraction = {};
};

struct Maps
{
    std::vector<AxisCoordinate> x;
    std::vector<AxisCoordinate> y;
    std::vector<FlatBlock> blocks;

    std::size_t vector_output_bytes() const
    {
        return blocks.size() * 16;
    }
};

inline bool is_exact_three_quarter_shape(
    int src_rows,
    int src_cols,
    int dst_rows,
    int dst_cols)
{
    return src_rows >= 2 && src_cols >= 2 && dst_rows > 0 && dst_cols > 0 &&
           static_cast<std::int64_t>(dst_rows) ==
               static_cast<std::int64_t>(src_rows) * 3 / 4 &&
           static_cast<std::int64_t>(dst_cols) ==
               static_cast<std::int64_t>(src_cols) * 3 / 4;
}

inline std::uint64_t rounded_divide(
    std::uint64_t numerator,
    std::uint64_t denominator)
{
    CV_Assert(denominator != 0);
    const std::uint64_t quotient = numerator / denominator;
    const std::uint64_t remainder = numerator % denominator;
    return quotient +
           static_cast<std::uint64_t>(
               remainder >= denominator - denominator / 2);
}

// Computes round((((output + 0.5) * source_size / destination_size) - 0.5)
//                * 2^16 + 0.5 / 2^8) without a potentially overflowing
// 64-bit multiply by 2^16. The product before the shift fits in uint64_t for
// the positive int dimensions accepted by Mat.
inline std::int64_t aligned_coordinate(
    int output,
    int source_size,
    int destination_size)
{
    CV_Assert(output >= 0 && output < destination_size);
    CV_Assert(source_size > 0 && destination_size > 0);

    const std::uint64_t twice_output_plus_one =
        static_cast<std::uint64_t>(output) * 2 + 1;
    const std::uint64_t unshifted_product =
        twice_output_plus_one * static_cast<std::uint64_t>(source_size);
    const std::uint64_t whole =
        unshifted_product / static_cast<std::uint64_t>(destination_size);
    const std::uint64_t remainder =
        unshifted_product % static_cast<std::uint64_t>(destination_size);
    const std::uint64_t scaled =
        (whole << (coordinate_fraction_bits - 1)) +
        rounded_divide(
            remainder << (coordinate_fraction_bits - 1),
            static_cast<std::uint64_t>(destination_size));

    return static_cast<std::int64_t>(scaled) -
           static_cast<std::int64_t>(coordinate_half) +
           static_cast<std::int64_t>(interpolation_rounding_bias);
}

inline AxisCoordinate build_axis_coordinate(
    int output,
    int source_size,
    int destination_size)
{
    const std::int64_t coordinate =
        aligned_coordinate(output, source_size, destination_size);
    if (coordinate <= 0 || source_size == 1)
    {
        return AxisCoordinate{0, std::min(1, source_size - 1), 0};
    }

    const std::uint64_t positive_coordinate =
        static_cast<std::uint64_t>(coordinate);
    const std::uint64_t source_index =
        positive_coordinate >> coordinate_fraction_bits;
    if (source_index >= static_cast<std::uint64_t>(source_size - 1))
    {
        return AxisCoordinate{source_size - 1, source_size - 1, 0};
    }

    const std::uint64_t fraction_mask =
        (std::uint64_t{1} << coordinate_fraction_bits) - 1;
    return AxisCoordinate{
        static_cast<int>(source_index),
        static_cast<int>(source_index + 1),
        static_cast<std::uint16_t>(
            (positive_coordinate & fraction_mask) >>
            (coordinate_fraction_bits - interpolation_fraction_bits))};
}

inline uchar lerp_u8(uchar first, uchar second, std::uint16_t fraction)
{
    const int accumulator =
        (static_cast<int>(first) << interpolation_fraction_bits) +
        (static_cast<int>(second) - static_cast<int>(first)) *
            static_cast<int>(fraction) +
        (1 << (interpolation_fraction_bits - 1));
    return static_cast<uchar>(accumulator >> interpolation_fraction_bits);
}

inline uchar bilinear_u8(
    uchar top_left,
    uchar top_right,
    uchar bottom_left,
    uchar bottom_right,
    std::uint16_t x_fraction,
    std::uint16_t y_fraction)
{
    const uchar left = lerp_u8(top_left, bottom_left, y_fraction);
    const uchar right = lerp_u8(top_right, bottom_right, y_fraction);
    return lerp_u8(left, right, x_fraction);
}

inline Maps build_maps(
    int src_rows,
    int src_cols,
    int dst_rows,
    int dst_cols)
{
    CV_Assert(src_rows > 0 && src_cols > 0);
    CV_Assert(dst_rows > 0 && dst_cols > 0);

    Maps maps;
    maps.x.resize(static_cast<std::size_t>(dst_cols));
    maps.y.resize(static_cast<std::size_t>(dst_rows));
    for (int x = 0; x < dst_cols; ++x)
    {
        maps.x[static_cast<std::size_t>(x)] =
            build_axis_coordinate(x, src_cols, dst_cols);
    }
    for (int y = 0; y < dst_rows; ++y)
    {
        maps.y[static_cast<std::size_t>(y)] =
            build_axis_coordinate(y, src_rows, dst_rows);
    }

    const std::size_t source_bytes =
        static_cast<std::size_t>(src_cols) * 3;
    const std::size_t output_bytes =
        static_cast<std::size_t>(dst_cols) * 3;
    if (source_bytes < 32 || output_bytes < 16)
    {
        return maps;
    }

    maps.blocks.reserve(output_bytes / 16);
    for (std::size_t output_base = 0;
         output_base + 16 <= output_bytes;
         output_base += 16)
    {
        const std::size_t first_pixel = output_base / 3;
        const std::size_t first_channel = output_base % 3;
        const AxisCoordinate& first_coordinate = maps.x[first_pixel];
        const std::size_t first_source =
            static_cast<std::size_t>(first_coordinate.first) * 3 +
            first_channel;

        FlatBlock block;
        block.source_byte_base =
            std::min(first_source, source_bytes - 32);
        bool valid = true;
        for (std::size_t lane = 0; lane < 16; ++lane)
        {
            const std::size_t output_element = output_base + lane;
            const std::size_t pixel = output_element / 3;
            const std::size_t channel = output_element % 3;
            const AxisCoordinate& coordinate = maps.x[pixel];
            const std::size_t left =
                static_cast<std::size_t>(coordinate.first) * 3 + channel;
            const std::size_t right =
                static_cast<std::size_t>(coordinate.second) * 3 + channel;
            if (left < block.source_byte_base ||
                right >= block.source_byte_base + 32)
            {
                valid = false;
                break;
            }
            block.left_index[lane] = static_cast<uchar>(
                left - block.source_byte_base);
            block.x_fraction[lane] = coordinate.fraction;
        }
        if (!valid)
        {
            break;
        }
        maps.blocks.push_back(block);
    }
    return maps;
}

inline uchar interpolate_output_byte(
    const uchar* top,
    const uchar* bottom,
    const Maps& maps,
    std::size_t output_element,
    std::uint16_t y_fraction)
{
    const std::size_t pixel = output_element / 3;
    const std::size_t channel = output_element % 3;
    const AxisCoordinate& horizontal = maps.x[pixel];
    const std::size_t left =
        static_cast<std::size_t>(horizontal.first) * 3 + channel;
    const std::size_t right =
        static_cast<std::size_t>(horizontal.second) * 3 + channel;
    return bilinear_u8(
        top[left],
        top[right],
        bottom[left],
        bottom[right],
        horizontal.fraction,
        y_fraction);
}

inline void resize_linear_scalar_reference(
    const Mat& src,
    Mat& dst,
    int dst_rows,
    int dst_cols)
{
    CV_Assert(!src.empty() && src.dims == 2 && src.type() == CV_8UC3);
    CV_Assert(dst_rows > 0 && dst_cols > 0);

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const Maps maps = build_maps(
        src_rows, src_cols, dst_rows, dst_cols);

    dst.create(std::vector<int>{dst_rows, dst_cols}, CV_8UC3);
    const bool do_parallel = should_parallelize_resize(dst_rows, dst_cols, 3);
    parallel_for_index_if(do_parallel, dst_rows, [&](int y) {
        const AxisCoordinate& vertical = maps.y[static_cast<std::size_t>(y)];
        const uchar* top = src.data +
            static_cast<std::size_t>(vertical.first) * src.step(0);
        const uchar* bottom = src.data +
            static_cast<std::size_t>(vertical.second) * src.step(0);
        uchar* output = dst.data + static_cast<std::size_t>(y) * dst.step(0);

        for (int x = 0; x < dst_cols; ++x)
        {
            const AxisCoordinate& horizontal = maps.x[static_cast<std::size_t>(x)];
            const int left = horizontal.first * 3;
            const int right = horizontal.second * 3;
            const int destination = x * 3;
            for (int channel = 0; channel < 3; ++channel)
            {
                output[destination + channel] = bilinear_u8(
                    top[left + channel],
                    top[right + channel],
                    bottom[left + channel],
                    bottom[right + channel],
                    horizontal.fraction,
                    vertical.fraction);
            }
        }
    });
}

}  // namespace resize_fixed_u8c3
}  // namespace detail
}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_RESIZE_FIXED_U8C3_HPP
