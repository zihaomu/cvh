#ifndef CVH_CORE_DETAIL_REDUCTION_NEON_HPP
#define CVH_CORE_DETAIL_REDUCTION_NEON_HPP

#include "cpu_features.hpp"
#include "../mat.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace cvh {
namespace detail {
namespace reduction_neon {

struct NormResult
{
    long double accumulator = 0.0L;
    double maximum = 0.0;
    bool has_nan = false;
};

struct StableStatistics
{
    long double means[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    long double m2[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    std::size_t count = 0;
};

inline bool direct_neon_allowed()
{
    const cpu::DispatchMode mode = cpu::dispatch_mode();
    return cpu::neon_runtime_available() &&
           (mode == cpu::DispatchMode::Auto ||
            mode == cpu::DispatchMode::NeonOnly);
}

#if CVH_DETAIL_HAVE_NEON_KERNEL

inline float64x2_t convert_low_f64(float32x4_t value)
{
    return vcvt_f64_f32(vget_low_f32(value));
}

inline float64x2_t convert_high_f64(float32x4_t value)
{
    return vcvt_high_f64_f32(value);
}

inline bool vector_has_nan(float64x2_t value)
{
    const uint64x2_t valid = vceqq_f64(value, value);
    return vgetq_lane_u64(valid, 0) == 0 ||
           vgetq_lane_u64(valid, 1) == 0;
}

inline void accumulate_norm_vector(float32x4_t first,
                                   float32x4_t second,
                                   bool difference,
                                   int norm_type,
                                   float64x2_t& low_accumulator,
                                   float64x2_t& high_accumulator,
                                   bool& has_nan)
{
    float64x2_t low = convert_low_f64(first);
    float64x2_t high = convert_high_f64(first);
    if (difference)
    {
        low = vsubq_f64(low, convert_low_f64(second));
        high = vsubq_f64(high, convert_high_f64(second));
    }
    has_nan = has_nan || vector_has_nan(low) || vector_has_nan(high);
    if (norm_type == NORM_INF)
    {
        low_accumulator = vmaxnmq_f64(low_accumulator, vabsq_f64(low));
        high_accumulator = vmaxnmq_f64(high_accumulator, vabsq_f64(high));
    }
    else if (norm_type == NORM_L1)
    {
        low_accumulator = vaddq_f64(low_accumulator, vabsq_f64(low));
        high_accumulator = vaddq_f64(high_accumulator, vabsq_f64(high));
    }
    else
    {
        low_accumulator = vfmaq_f64(low_accumulator, low, low);
        high_accumulator = vfmaq_f64(high_accumulator, high, high);
    }
}

inline void accumulate_norm_scalar(double first,
                                   double second,
                                   bool difference,
                                   int norm_type,
                                   NormResult& result)
{
    const double value = difference ? first - second : first;
    const double magnitude = std::fabs(value);
    result.has_nan = result.has_nan || std::isnan(magnitude);
    if (norm_type == NORM_INF)
    {
        result.maximum = std::max(result.maximum, magnitude);
    }
    else if (norm_type == NORM_L1)
    {
        result.accumulator += magnitude;
    }
    else
    {
        result.accumulator +=
            static_cast<long double>(value) * value;
    }
}

inline bool norm_f32_f64(const Mat& first,
                         const Mat* second,
                         int norm_type,
                         NormResult& result)
{
    if (!direct_neon_allowed() || first.depth() != CV_32F ||
        first.channels() != 1 || first.empty() ||
        (second != nullptr &&
         (second->type() != first.type() ||
          second->shape() != first.shape())))
    {
        return false;
    }
    const bool flatten = first.isContinuous() &&
                         (second == nullptr || second->isContinuous());
    const std::size_t rows = flatten
        ? 1
        : (first.dims > 1
               ? static_cast<std::size_t>(first.size.p[0])
               : 1);
    const std::size_t row_values = flatten
        ? first.total()
        : (first.dims > 1 ? first.total(1, first.dims) : first.total());
    if (row_values < 4)
    {
        return false;
    }
    const std::size_t first_step = flatten
        ? first.total() * first.elemSize()
        : (first.dims > 1 ? first.step(0) : row_values * sizeof(float));
    const std::size_t second_step = second == nullptr
        ? 0
        : (flatten
               ? second->total() * second->elemSize()
               : (second->dims > 1
                      ? second->step(0)
                      : row_values * sizeof(float)));

    float64x2_t low_accumulators[4] = {
        vdupq_n_f64(0.0), vdupq_n_f64(0.0),
        vdupq_n_f64(0.0), vdupq_n_f64(0.0)};
    float64x2_t high_accumulators[4] = {
        vdupq_n_f64(0.0), vdupq_n_f64(0.0),
        vdupq_n_f64(0.0), vdupq_n_f64(0.0)};
    bool has_nan = false;
    const bool difference = second != nullptr;
    for (std::size_t row = 0; row < rows; ++row)
    {
        const float* first_row = reinterpret_cast<const float*>(
            first.data + row * first_step);
        const float* second_row = difference
            ? reinterpret_cast<const float*>(second->data + row * second_step)
            : nullptr;
        std::size_t x = 0;
        for (; x + 16 <= row_values; x += 16)
        {
            for (int block = 0; block < 4; ++block)
            {
                const float32x4_t first_values =
                    vld1q_f32(first_row + x + static_cast<std::size_t>(block) * 4);
                const float32x4_t second_values = difference
                    ? vld1q_f32(
                          second_row + x + static_cast<std::size_t>(block) * 4)
                    : vdupq_n_f32(0.0f);
                accumulate_norm_vector(
                    first_values,
                    second_values,
                    difference,
                    norm_type,
                    low_accumulators[block],
                    high_accumulators[block],
                    has_nan);
            }
        }
        for (; x + 4 <= row_values; x += 4)
        {
            accumulate_norm_vector(
                vld1q_f32(first_row + x),
                difference ? vld1q_f32(second_row + x)
                           : vdupq_n_f32(0.0f),
                difference,
                norm_type,
                low_accumulators[0],
                high_accumulators[0],
                has_nan);
        }
        for (; x < row_values; ++x)
        {
            accumulate_norm_scalar(
                first_row[x],
                difference ? second_row[x] : 0.0,
                difference,
                norm_type,
                result);
        }
    }

    result.has_nan = result.has_nan || has_nan;
    if (norm_type == NORM_INF)
    {
        for (int block = 0; block < 4; ++block)
        {
            result.maximum = std::max(
                result.maximum,
                vmaxvq_f64(low_accumulators[block]));
            result.maximum = std::max(
                result.maximum,
                vmaxvq_f64(high_accumulators[block]));
        }
    }
    else
    {
        double vector_sum = 0.0;
        for (int block = 0; block < 4; ++block)
        {
            vector_sum += vaddvq_f64(low_accumulators[block]);
            vector_sum += vaddvq_f64(high_accumulators[block]);
        }
        result.accumulator += static_cast<long double>(vector_sum);
    }
    return true;
}

inline bool norm_f32(const Mat& first,
                     const Mat* second,
                     int norm_type,
                     NormResult& result)
{
    if (!direct_neon_allowed() || first.depth() != CV_32F ||
        first.channels() != 1 || first.empty() ||
        (second != nullptr &&
         (second->type() != first.type() ||
          second->shape() != first.shape())))
    {
        return false;
    }
    const bool flatten = first.isContinuous() &&
                         (second == nullptr || second->isContinuous());
    const std::size_t rows = flatten
        ? 1
        : (first.dims > 1
               ? static_cast<std::size_t>(first.size.p[0])
               : 1);
    const std::size_t row_values = flatten
        ? first.total()
        : (first.dims > 1 ? first.total(1, first.dims) : first.total());
    if (row_values < 4)
    {
        return false;
    }
    const std::size_t first_step = flatten
        ? first.total() * first.elemSize()
        : (first.dims > 1 ? first.step(0) : row_values * sizeof(float));
    const std::size_t second_step = second == nullptr
        ? 0
        : (flatten
               ? second->total() * second->elemSize()
               : (second->dims > 1
                      ? second->step(0)
                      : row_values * sizeof(float)));
    const bool difference = second != nullptr;
    if (norm_type == NORM_INF)
    {
        float32x4_t maxima[4] = {
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
            vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        uint32x4_t valid_masks[4] = {
            vdupq_n_u32(~0u), vdupq_n_u32(~0u),
            vdupq_n_u32(~0u), vdupq_n_u32(~0u)};
        for (std::size_t row = 0; row < rows; ++row)
        {
            const float* source = reinterpret_cast<const float*>(
                first.data + row * first_step);
            const float* other = difference
                ? reinterpret_cast<const float*>(
                      second->data + row * second_step)
                : nullptr;
            std::size_t x = 0;
            for (; x + 16 <= row_values; x += 16)
            {
                for (int stream = 0; stream < 4; ++stream)
                {
                    const std::size_t offset =
                        x + static_cast<std::size_t>(stream) * 4;
                    const float32x4_t first_values =
                        vld1q_f32(source + offset);
                    const float32x4_t values = difference
                        ? vsubq_f32(
                              first_values, vld1q_f32(other + offset))
                        : first_values;
                    valid_masks[stream] = vandq_u32(
                        valid_masks[stream], vceqq_f32(values, values));
                    maxima[stream] = vmaxnmq_f32(
                        maxima[stream], vabsq_f32(values));
                }
            }
            for (; x + 4 <= row_values; x += 4)
            {
                const float32x4_t first_values = vld1q_f32(source + x);
                const float32x4_t values = difference
                    ? vsubq_f32(first_values, vld1q_f32(other + x))
                    : first_values;
                valid_masks[0] = vandq_u32(
                    valid_masks[0], vceqq_f32(values, values));
                maxima[0] = vmaxnmq_f32(
                    maxima[0], vabsq_f32(values));
            }
            for (; x < row_values; ++x)
            {
                accumulate_norm_scalar(
                    source[x], difference ? other[x] : 0.0,
                    difference, norm_type, result);
            }
        }
        const float32x4_t maximum = vmaxnmq_f32(
            vmaxnmq_f32(maxima[0], maxima[1]),
            vmaxnmq_f32(maxima[2], maxima[3]));
        result.maximum = std::max(
            result.maximum, static_cast<double>(vmaxvq_f32(maximum)));
        const uint32x4_t valid = vandq_u32(
            vandq_u32(valid_masks[0], valid_masks[1]),
            vandq_u32(valid_masks[2], valid_masks[3]));
        result.has_nan = result.has_nan || vminvq_u32(valid) == 0;
        if (difference && std::isinf(result.maximum))
        {
            result = NormResult {};
            return norm_f32_f64(first, second, norm_type, result);
        }
        return true;
    }
    constexpr std::size_t values_per_chunk = 1024;
    bool requires_f64 = false;

    for (std::size_t row = 0; row < rows && !requires_f64; ++row)
    {
        const float* first_row = reinterpret_cast<const float*>(
            first.data + row * first_step);
        const float* second_row = difference
            ? reinterpret_cast<const float*>(second->data + row * second_step)
            : nullptr;
        std::size_t x = 0;
        while (x + 4 <= row_values && !requires_f64)
        {
            const std::size_t vector_end = std::min(
                row_values - (row_values - x) % 4,
                x + values_per_chunk);
            float32x4_t accumulators[4] = {
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f),
                vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
            int stream = 0;
            for (; x + 4 <= vector_end; x += 4)
            {
                const float32x4_t first_values = vld1q_f32(first_row + x);
                const float32x4_t values = difference
                    ? vsubq_f32(first_values, vld1q_f32(second_row + x))
                    : first_values;
                const float32x4_t magnitude = vabsq_f32(values);
                if (norm_type == NORM_L1)
                {
                    accumulators[stream] =
                        vaddq_f32(accumulators[stream], magnitude);
                }
                else
                {
                    accumulators[stream] = vfmaq_f32(
                        accumulators[stream], values, values);
                }
                stream = (stream + 1) & 3;
            }
            double block_sum = 0.0;
            for (int block = 0; block < 4; ++block)
            {
                const uint32x4_t finite = vandq_u32(
                    vceqq_f32(accumulators[block], accumulators[block]),
                    vcleq_f32(
                        vabsq_f32(accumulators[block]),
                        vdupq_n_f32(std::numeric_limits<float>::max())));
                if (vminvq_u32(finite) == 0)
                {
                    requires_f64 = true;
                    break;
                }
                block_sum +=
                    vaddvq_f64(convert_low_f64(accumulators[block]));
                block_sum +=
                    vaddvq_f64(convert_high_f64(accumulators[block]));
            }
            if (!requires_f64)
            {
                result.accumulator += static_cast<long double>(block_sum);
            }
        }
        for (; x < row_values && !requires_f64; ++x)
        {
            accumulate_norm_scalar(
                first_row[x],
                difference ? second_row[x] : 0.0,
                difference,
                norm_type,
                result);
        }
    }
    if (requires_f64)
    {
        result = NormResult {};
        return norm_f32_f64(first, second, norm_type, result);
    }
    return true;
}

inline long double sum_f32_row(const float* source,
                               std::size_t length,
                               bool square)
{
    float64x2_t low_accumulators[4] = {
        vdupq_n_f64(0.0), vdupq_n_f64(0.0),
        vdupq_n_f64(0.0), vdupq_n_f64(0.0)};
    float64x2_t high_accumulators[4] = {
        vdupq_n_f64(0.0), vdupq_n_f64(0.0),
        vdupq_n_f64(0.0), vdupq_n_f64(0.0)};
    std::size_t x = 0;
    for (; x + 16 <= length; x += 16)
    {
        for (int block = 0; block < 4; ++block)
        {
            const float32x4_t values =
                vld1q_f32(source + x + static_cast<std::size_t>(block) * 4);
            const float64x2_t low = convert_low_f64(values);
            const float64x2_t high = convert_high_f64(values);
            low_accumulators[block] = square
                ? vfmaq_f64(low_accumulators[block], low, low)
                : vaddq_f64(low_accumulators[block], low);
            high_accumulators[block] = square
                ? vfmaq_f64(high_accumulators[block], high, high)
                : vaddq_f64(high_accumulators[block], high);
        }
    }
    for (; x + 4 <= length; x += 4)
    {
        const float32x4_t values = vld1q_f32(source + x);
        const float64x2_t low = convert_low_f64(values);
        const float64x2_t high = convert_high_f64(values);
        low_accumulators[0] = square
            ? vfmaq_f64(low_accumulators[0], low, low)
            : vaddq_f64(low_accumulators[0], low);
        high_accumulators[0] = square
            ? vfmaq_f64(high_accumulators[0], high, high)
            : vaddq_f64(high_accumulators[0], high);
    }
    long double result = 0.0L;
    for (int block = 0; block < 4; ++block)
    {
        result += static_cast<long double>(
            vaddvq_f64(low_accumulators[block]));
        result += static_cast<long double>(
            vaddvq_f64(high_accumulators[block]));
    }
    for (; x < length; ++x)
    {
        const long double value = source[x];
        result += square ? value * value : value;
    }
    return result;
}

inline bool reduce_f32(const Mat& src,
                       Mat& dst,
                       int axis,
                       int rtype)
{
    if (!direct_neon_allowed() || src.depth() != CV_32F ||
        src.channels() != 1 || src.dims != 2 ||
        dst.depth() != CV_32F || dst.channels() != 1 ||
        (rtype != REDUCE_SUM && rtype != REDUCE_AVG &&
         rtype != REDUCE_SUM2))
    {
        return false;
    }
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    if ((axis == 0 && cols < 4) || (axis == 1 && cols < 4))
    {
        return false;
    }
    const bool square = rtype == REDUCE_SUM2;
    const double divisor = rtype == REDUCE_AVG
        ? static_cast<double>(axis == 0 ? rows : cols)
        : 1.0;
    if (axis == 1)
    {
        for (int row = 0; row < rows; ++row)
        {
            const float* source = reinterpret_cast<const float*>(
                src.data + static_cast<std::size_t>(row) * src.step(0));
            const long double value =
                sum_f32_row(source, static_cast<std::size_t>(cols), square) /
                static_cast<long double>(divisor);
            reinterpret_cast<float*>(
                dst.data + static_cast<std::size_t>(row) * dst.step(0))[0] =
                static_cast<float>(value);
        }
        return true;
    }

    std::vector<double> accumulators(static_cast<std::size_t>(cols), 0.0);
    std::vector<float> block_accumulators(static_cast<std::size_t>(cols));
    constexpr int rows_per_block = 256;
    bool requires_f64_fallback = false;
    for (int row_begin = 0;
         row_begin < rows && !requires_f64_fallback;
         row_begin += rows_per_block)
    {
        std::fill(block_accumulators.begin(), block_accumulators.end(), 0.0f);
        const int row_end = std::min(rows, row_begin + rows_per_block);
        int row = row_begin;
        for (; row + 4 <= row_end; row += 4)
        {
            const float* sources[4] = {
                reinterpret_cast<const float*>(
                    src.data + static_cast<std::size_t>(row) * src.step(0)),
                reinterpret_cast<const float*>(
                    src.data + static_cast<std::size_t>(row + 1) * src.step(0)),
                reinterpret_cast<const float*>(
                    src.data + static_cast<std::size_t>(row + 2) * src.step(0)),
                reinterpret_cast<const float*>(
                    src.data + static_cast<std::size_t>(row + 3) * src.step(0))};
            int col = 0;
            for (; col + 4 <= cols; col += 4)
            {
                float32x4_t accumulator =
                    vld1q_f32(block_accumulators.data() + col);
                for (int stream = 0; stream < 4; ++stream)
                {
                    const float32x4_t values =
                        vld1q_f32(sources[stream] + col);
                    accumulator = square
                        ? vfmaq_f32(accumulator, values, values)
                        : vaddq_f32(accumulator, values);
                }
                vst1q_f32(block_accumulators.data() + col, accumulator);
            }
            for (; col < cols; ++col)
            {
                float accumulator =
                    block_accumulators[static_cast<std::size_t>(col)];
                for (int stream = 0; stream < 4; ++stream)
                {
                    const float value = sources[stream][col];
                    accumulator += square ? value * value : value;
                }
                block_accumulators[static_cast<std::size_t>(col)] =
                    accumulator;
            }
        }
        for (; row < row_end; ++row)
        {
            const float* source = reinterpret_cast<const float*>(
                src.data + static_cast<std::size_t>(row) * src.step(0));
            int col = 0;
            for (; col + 4 <= cols; col += 4)
            {
                const float32x4_t values = vld1q_f32(source + col);
                float32x4_t accumulator =
                    vld1q_f32(block_accumulators.data() + col);
                accumulator = square
                    ? vfmaq_f32(accumulator, values, values)
                    : vaddq_f32(accumulator, values);
                vst1q_f32(block_accumulators.data() + col, accumulator);
            }
            for (; col < cols; ++col)
            {
                const float value = source[col];
                block_accumulators[static_cast<std::size_t>(col)] +=
                    square ? value * value : value;
            }
        }
        int col = 0;
        for (; col + 4 <= cols; col += 4)
        {
            const float32x4_t values =
                vld1q_f32(block_accumulators.data() + col);
            const uint32x4_t finite = vandq_u32(
                vceqq_f32(values, values),
                vcleq_f32(
                    vabsq_f32(values),
                    vdupq_n_f32(std::numeric_limits<float>::max())));
            if (vminvq_u32(finite) == 0)
            {
                requires_f64_fallback = true;
                break;
            }
            const float64x2_t low_values = convert_low_f64(values);
            const float64x2_t high_values = convert_high_f64(values);
            float64x2_t low = vld1q_f64(accumulators.data() + col);
            float64x2_t high = vld1q_f64(accumulators.data() + col + 2);
            low = vaddq_f64(low, low_values);
            high = vaddq_f64(high, high_values);
            vst1q_f64(accumulators.data() + col, low);
            vst1q_f64(accumulators.data() + col + 2, high);
        }
        for (; col < cols && !requires_f64_fallback; ++col)
        {
            const float value =
                block_accumulators[static_cast<std::size_t>(col)];
            if (!std::isfinite(value))
            {
                requires_f64_fallback = true;
                break;
            }
            accumulators[static_cast<std::size_t>(col)] += value;
        }
    }
    if (requires_f64_fallback)
    {
        std::fill(accumulators.begin(), accumulators.end(), 0.0);
        for (int row = 0; row < rows; ++row)
        {
            const float* source = reinterpret_cast<const float*>(
                src.data + static_cast<std::size_t>(row) * src.step(0));
            int col = 0;
            for (; col + 4 <= cols; col += 4)
            {
                const float32x4_t values = vld1q_f32(source + col);
                const float64x2_t low_values = convert_low_f64(values);
                const float64x2_t high_values = convert_high_f64(values);
                float64x2_t low = vld1q_f64(accumulators.data() + col);
                float64x2_t high = vld1q_f64(accumulators.data() + col + 2);
                low = square
                    ? vfmaq_f64(low, low_values, low_values)
                    : vaddq_f64(low, low_values);
                high = square
                    ? vfmaq_f64(high, high_values, high_values)
                    : vaddq_f64(high, high_values);
                vst1q_f64(accumulators.data() + col, low);
                vst1q_f64(accumulators.data() + col + 2, high);
            }
            for (; col < cols; ++col)
            {
                const double value = source[col];
                accumulators[static_cast<std::size_t>(col)] +=
                    square ? value * value : value;
            }
        }
    }
    float* destination = reinterpret_cast<float*>(dst.data);
    int col = 0;
    const float64x2_t scale = vdupq_n_f64(1.0 / divisor);
    for (; col + 4 <= cols; col += 4)
    {
        float64x2_t low = vld1q_f64(accumulators.data() + col);
        float64x2_t high = vld1q_f64(accumulators.data() + col + 2);
        if (rtype == REDUCE_AVG)
        {
            low = vmulq_f64(low, scale);
            high = vmulq_f64(high, scale);
        }
        vst1q_f32(
            destination + col,
            vcombine_f32(vcvt_f32_f64(low), vcvt_f32_f64(high)));
    }
    for (; col < cols; ++col)
    {
        destination[col] = static_cast<float>(
            accumulators[static_cast<std::size_t>(col)] / divisor);
    }
    return true;
}

inline bool apply_normalize_f32c1(const Mat& src,
                                  Mat& dst,
                                  double scale,
                                  double shift)
{
    if (!direct_neon_allowed() || src.dims > 2 ||
        src.type() != CV_32FC1 || dst.type() != CV_32FC1)
    {
        return false;
    }
    const bool flat = src.isContinuous() && dst.isContinuous();
    const std::size_t rows = flat
        ? 1
        : (src.dims > 1
               ? static_cast<std::size_t>(src.size.p[0])
               : 1);
    const std::size_t row_values = flat
        ? src.total()
        : (src.dims > 1 ? src.total(1, src.dims) : src.total());
    if (row_values < 4)
    {
        return false;
    }
    const std::size_t src_step = flat
        ? row_values * sizeof(float)
        : src.step(0);
    const std::size_t dst_step = flat
        ? row_values * sizeof(float)
        : dst.step(0);
    const float64x2_t v_scale = vdupq_n_f64(scale);
    const float64x2_t v_shift = vdupq_n_f64(shift);
    for (std::size_t row = 0; row < rows; ++row)
    {
        const float* source = reinterpret_cast<const float*>(
            src.data + row * src_step);
        float* destination = reinterpret_cast<float*>(
            dst.data + row * dst_step);
        std::size_t x = 0;
        for (; x + 16 <= row_values; x += 16)
        {
            for (int stream = 0; stream < 4; ++stream)
            {
                const std::size_t offset =
                    x + static_cast<std::size_t>(stream) * 4;
                const float32x4_t values = vld1q_f32(source + offset);
                const float64x2_t low = vfmaq_f64(
                    v_shift, convert_low_f64(values), v_scale);
                const float64x2_t high = vfmaq_f64(
                    v_shift, convert_high_f64(values), v_scale);
                vst1q_f32(
                    destination + offset,
                    vcombine_f32(
                        vcvt_f32_f64(low), vcvt_f32_f64(high)));
            }
        }
        for (; x + 4 <= row_values; x += 4)
        {
            const float32x4_t values = vld1q_f32(source + x);
            const float64x2_t low = vfmaq_f64(
                v_shift, convert_low_f64(values), v_scale);
            const float64x2_t high = vfmaq_f64(
                v_shift, convert_high_f64(values), v_scale);
            vst1q_f32(
                destination + x,
                vcombine_f32(vcvt_f32_f64(low), vcvt_f32_f64(high)));
        }
        for (; x < row_values; ++x)
        {
            destination[x] = static_cast<float>(
                static_cast<double>(source[x]) * scale + shift);
        }
    }
    return true;
}

inline void accumulate_statistics_channel(float32x4_t values,
                                          float64x2_t& low,
                                          float64x2_t& high)
{
    low = vaddq_f64(low, convert_low_f64(values));
    high = vaddq_f64(high, convert_high_f64(values));
}

inline void accumulate_statistics_m2(float32x4_t values,
                                     double mean,
                                     float64x2_t& low,
                                     float64x2_t& high)
{
    const float64x2_t mean_vector = vdupq_n_f64(mean);
    const float64x2_t low_delta =
        vsubq_f64(convert_low_f64(values), mean_vector);
    const float64x2_t high_delta =
        vsubq_f64(convert_high_f64(values), mean_vector);
    low = vfmaq_f64(low, low_delta, low_delta);
    high = vfmaq_f64(high, high_delta, high_delta);
}

inline bool stable_statistics_f32(const Mat& src,
                                  StableStatistics& result)
{
    if (!direct_neon_allowed() || src.depth() != CV_32F || src.empty() ||
        src.channels() != 3)
    {
        return false;
    }
    const bool flatten = src.isContinuous();
    const std::size_t rows = flatten
        ? 1
        : (src.dims > 1
               ? static_cast<std::size_t>(src.size.p[0])
               : 1);
    const std::size_t row_pixels = flatten
        ? src.total()
        : (src.dims > 1 ? src.total(1, src.dims) : src.total());
    if (row_pixels < 4)
    {
        return false;
    }
    const std::size_t step = flatten
        ? src.total() * src.elemSize()
        : (src.dims > 1 ? src.step(0) : row_pixels * src.elemSize());
    const int channels = src.channels();
    float64x2_t sum_low[3] = {
        vdupq_n_f64(0.0), vdupq_n_f64(0.0), vdupq_n_f64(0.0)};
    float64x2_t sum_high[3] = {
        vdupq_n_f64(0.0), vdupq_n_f64(0.0), vdupq_n_f64(0.0)};
    long double scalar_sums[3] = {0.0L, 0.0L, 0.0L};
    for (std::size_t row = 0; row < rows; ++row)
    {
        const float* source = reinterpret_cast<const float*>(
            src.data + row * step);
        std::size_t pixel = 0;
        if (channels == 1)
        {
            for (; pixel + 4 <= row_pixels; pixel += 4)
            {
                accumulate_statistics_channel(
                    vld1q_f32(source + pixel), sum_low[0], sum_high[0]);
            }
        }
        else
        {
            for (; pixel + 4 <= row_pixels; pixel += 4)
            {
                const float32x4x3_t values = vld3q_f32(source + pixel * 3);
                for (int channel = 0; channel < 3; ++channel)
                {
                    accumulate_statistics_channel(
                        values.val[channel],
                        sum_low[channel],
                        sum_high[channel]);
                }
            }
        }
        for (; pixel < row_pixels; ++pixel)
        {
            for (int channel = 0; channel < channels; ++channel)
            {
                scalar_sums[channel] +=
                    source[pixel * static_cast<std::size_t>(channels) +
                           static_cast<std::size_t>(channel)];
            }
        }
    }
    result.count = rows * row_pixels;
    double means[3] = {0.0, 0.0, 0.0};
    for (int channel = 0; channel < channels; ++channel)
    {
        const long double sum = scalar_sums[channel] +
            static_cast<long double>(vaddvq_f64(sum_low[channel])) +
            static_cast<long double>(vaddvq_f64(sum_high[channel]));
        result.means[channel] =
            sum / static_cast<long double>(result.count);
        means[channel] = static_cast<double>(result.means[channel]);
    }

    float64x2_t m2_low[3] = {
        vdupq_n_f64(0.0), vdupq_n_f64(0.0), vdupq_n_f64(0.0)};
    float64x2_t m2_high[3] = {
        vdupq_n_f64(0.0), vdupq_n_f64(0.0), vdupq_n_f64(0.0)};
    long double scalar_m2[3] = {0.0L, 0.0L, 0.0L};
    for (std::size_t row = 0; row < rows; ++row)
    {
        const float* source = reinterpret_cast<const float*>(
            src.data + row * step);
        std::size_t pixel = 0;
        if (channels == 1)
        {
            for (; pixel + 4 <= row_pixels; pixel += 4)
            {
                accumulate_statistics_m2(
                    vld1q_f32(source + pixel),
                    means[0],
                    m2_low[0],
                    m2_high[0]);
            }
        }
        else
        {
            for (; pixel + 4 <= row_pixels; pixel += 4)
            {
                const float32x4x3_t values = vld3q_f32(source + pixel * 3);
                for (int channel = 0; channel < 3; ++channel)
                {
                    accumulate_statistics_m2(
                        values.val[channel],
                        means[channel],
                        m2_low[channel],
                        m2_high[channel]);
                }
            }
        }
        for (; pixel < row_pixels; ++pixel)
        {
            for (int channel = 0; channel < channels; ++channel)
            {
                const long double delta =
                    static_cast<long double>(
                        source[pixel * static_cast<std::size_t>(channels) +
                               static_cast<std::size_t>(channel)]) -
                    result.means[channel];
                scalar_m2[channel] += delta * delta;
            }
        }
    }
    for (int channel = 0; channel < channels; ++channel)
    {
        result.m2[channel] = scalar_m2[channel] +
            static_cast<long double>(vaddvq_f64(m2_low[channel])) +
            static_cast<long double>(vaddvq_f64(m2_high[channel]));
    }
    return true;
}

#else

inline bool norm_f32(const Mat&, const Mat*, int, NormResult&)
{
    return false;
}

inline bool reduce_f32(const Mat&, Mat&, int, int)
{
    return false;
}

inline bool apply_normalize_f32c1(const Mat&, Mat&, double, double)
{
    return false;
}

inline bool stable_statistics_f32(const Mat&, StableStatistics&)
{
    return false;
}

#endif

}  // namespace reduction_neon
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_REDUCTION_NEON_HPP
