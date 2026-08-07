#ifndef CVH_CORE_DETAIL_INRANGE_NEON_HPP
#define CVH_CORE_DETAIL_INRANGE_NEON_HPP

#include "cpu_features.hpp"
#include "../mat.h"

#include <cstddef>

namespace cvh {
namespace detail {
namespace inrange_neon {

inline bool direct_neon_allowed()
{
    const cpu::DispatchMode mode = cpu::dispatch_mode();
    return cpu::neon_runtime_available() &&
           (mode == cpu::DispatchMode::Auto ||
            mode == cpu::DispatchMode::NeonOnly);
}

#if CVH_DETAIL_HAVE_NEON_KERNEL

inline uint8x16_t inclusive_mask(uint8x16_t values,
                                 uint8x16_t lower,
                                 uint8x16_t upper)
{
    return vandq_u8(
        vcgeq_u8(values, lower),
        vcgeq_u8(upper, values));
}

inline bool apply_u8_scalar_bounds(const Mat& src,
                                   const uchar lower[4],
                                   const uchar upper[4],
                                   Mat& dst)
{
    if (!direct_neon_allowed() || src.depth() != CV_8U ||
        (src.channels() != 1 && src.channels() != 3 &&
         src.channels() != 4))
    {
        return false;
    }
    const std::size_t rows = src.dims > 1
        ? static_cast<std::size_t>(src.size.p[0])
        : 1;
    const std::size_t pixels = src.dims > 1
        ? src.total(1, src.dims)
        : src.total();
    if (pixels < 16)
    {
        return false;
    }
    const std::size_t src_step = src.dims > 1
        ? src.step(0)
        : pixels * src.elemSize();
    const std::size_t dst_step = dst.dims > 1
        ? dst.step(0)
        : pixels;
    const int channels = src.channels();
    uint8x16_t lower_vectors[4];
    uint8x16_t upper_vectors[4];
    for (int channel = 0; channel < channels; ++channel)
    {
        lower_vectors[channel] = vdupq_n_u8(lower[channel]);
        upper_vectors[channel] = vdupq_n_u8(upper[channel]);
    }

    for (std::size_t row = 0; row < rows; ++row)
    {
        const uchar* source = src.data + row * src_step;
        uchar* destination = dst.data + row * dst_step;
        std::size_t pixel = 0;
        if (channels == 1)
        {
            for (; pixel + 16 <= pixels; pixel += 16)
            {
                vst1q_u8(
                    destination + pixel,
                    inclusive_mask(
                        vld1q_u8(source + pixel),
                        lower_vectors[0],
                        upper_vectors[0]));
            }
        }
        else if (channels == 3)
        {
            for (; pixel + 16 <= pixels; pixel += 16)
            {
                const uint8x16x3_t values =
                    vld3q_u8(source + pixel * 3);
                uint8x16_t mask = inclusive_mask(
                    values.val[0], lower_vectors[0], upper_vectors[0]);
                mask = vandq_u8(
                    mask,
                    inclusive_mask(
                        values.val[1], lower_vectors[1], upper_vectors[1]));
                mask = vandq_u8(
                    mask,
                    inclusive_mask(
                        values.val[2], lower_vectors[2], upper_vectors[2]));
                vst1q_u8(destination + pixel, mask);
            }
        }
        else
        {
            for (; pixel + 16 <= pixels; pixel += 16)
            {
                const uint8x16x4_t values =
                    vld4q_u8(source + pixel * 4);
                uint8x16_t mask = inclusive_mask(
                    values.val[0], lower_vectors[0], upper_vectors[0]);
                for (int channel = 1; channel < 4; ++channel)
                {
                    mask = vandq_u8(
                        mask,
                        inclusive_mask(
                            values.val[channel],
                            lower_vectors[channel],
                            upper_vectors[channel]));
                }
                vst1q_u8(destination + pixel, mask);
            }
        }
        for (; pixel < pixels; ++pixel)
        {
            bool inside = true;
            const std::size_t offset =
                pixel * static_cast<std::size_t>(channels);
            for (int channel = 0; channel < channels; ++channel)
            {
                const uchar value =
                    source[offset + static_cast<std::size_t>(channel)];
                inside = inside && lower[channel] <= value &&
                         value <= upper[channel];
            }
            destination[pixel] = inside ? static_cast<uchar>(255)
                                        : static_cast<uchar>(0);
        }
    }
    return true;
}

#else

inline bool apply_u8_scalar_bounds(
    const Mat&, const uchar[4], const uchar[4], Mat&)
{
    return false;
}

#endif

}  // namespace inrange_neon
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_INRANGE_NEON_HPP
