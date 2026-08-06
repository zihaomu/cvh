#ifndef CVH_IMGPROC_BILATERAL_FILTER_H
#define CVH_IMGPROC_BILATERAL_FILTER_H

#include "../core/detail/dispatch_control.h"
#include "detail/common.h"

#include <cmath>
#include <type_traits>
#include <vector>

namespace cvh
{
namespace bilateral_filter_detail
{

inline thread_local const char* g_last_bilateral_algorithm_path =
    "bilateral_generic";

inline const char* last_bilateral_algorithm_path()
{
    return g_last_bilateral_algorithm_path;
}

struct OffsetWeight
{
    int x;
    int y;
    float weight;
};

inline std::vector<OffsetWeight> spatial_weights(int radius,
                                                 double sigma_space)
{
    const double coefficient =
        -0.5 / (sigma_space * sigma_space);
    std::vector<OffsetWeight> weights;
    weights.reserve(static_cast<size_t>(radius * 2 + 1) *
                    static_cast<size_t>(radius * 2 + 1));
    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            const int distance_squared = x * x + y * y;
            if (distance_squared > radius * radius)
            {
                continue;
            }
            weights.push_back(
                {x,
                 y,
                 static_cast<float>(
                     std::exp(
                         static_cast<double>(distance_squared) *
                         coefficient))});
        }
    }
    return weights;
}

template<int Channels>
inline void run_u8_padded(
    const Mat& src,
    Mat& dst,
    const std::vector<OffsetWeight>& spatial,
    double sigma_color,
    int border_type)
{
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int count = static_cast<int>(spatial.size());
    const float color_coefficient =
        static_cast<float>(
            -0.5 / (sigma_color * sigma_color));
    std::vector<float> color_weights(
        static_cast<size_t>(Channels * 255 + 1));
    for (size_t difference = 0;
         difference < color_weights.size();
         ++difference)
    {
        const float value = static_cast<float>(difference);
        color_weights[difference] =
            std::exp(value * value * color_coefficient);
    }

    int radius = 0;
    for (const OffsetWeight& offset : spatial)
    {
        radius = std::max(
            radius,
            std::max(std::abs(offset.x), std::abs(offset.y)));
    }
    const int padded_cols = cols + radius * 2;
    const int padded_rows = rows + radius * 2;
    const int padded_stride = padded_cols * Channels;
    std::vector<int> source_x(static_cast<size_t>(padded_cols));
    for (int x = 0; x < padded_cols; ++x)
    {
        source_x[static_cast<size_t>(x)] =
            detail::border_interpolate(
                x - radius, cols, border_type);
    }
    std::vector<uchar> padded(
        static_cast<size_t>(padded_rows) *
        static_cast<size_t>(padded_stride));
    for (int y = 0; y < padded_rows; ++y)
    {
        const int source_y = detail::border_interpolate(
            y - radius, rows, border_type);
        uchar* padded_row =
            padded.data() + static_cast<size_t>(y) * padded_stride;
        if (source_y < 0)
        {
            std::fill(padded_row, padded_row + padded_stride, 0);
            continue;
        }
        const uchar* source_row =
            src.data + static_cast<size_t>(source_y) * src.step(0);
        for (int x = 0; x < padded_cols; ++x)
        {
            const int resolved_x = source_x[static_cast<size_t>(x)];
            uchar* destination_pixel =
                padded_row + static_cast<size_t>(x) * Channels;
            if (resolved_x < 0)
            {
                for (int channel = 0; channel < Channels; ++channel)
                {
                    destination_pixel[channel] = 0;
                }
                continue;
            }
            const uchar* source_pixel =
                source_row + static_cast<size_t>(resolved_x) * Channels;
            for (int channel = 0; channel < Channels; ++channel)
            {
                destination_pixel[channel] = source_pixel[channel];
            }
        }
    }

    std::vector<int> offsets(static_cast<size_t>(count));
    for (int index = 0; index < count; ++index)
    {
        const OffsetWeight& offset = spatial[static_cast<size_t>(index)];
        offsets[static_cast<size_t>(index)] =
            offset.y * padded_stride + offset.x * Channels;
    }

    for (int y = 0; y < rows; ++y)
    {
        const uchar* center_row =
            padded.data() +
            static_cast<size_t>(y + radius) * padded_stride +
            static_cast<size_t>(radius) * Channels;
        uchar* output =
            dst.data + static_cast<size_t>(y) * dst.step(0);
        for (int x = 0; x < cols; ++x)
        {
            const uchar* center =
                center_row + static_cast<size_t>(x) * Channels;
            float weighted_sum[3] = {0.0f, 0.0f, 0.0f};
            float weight_sum = 0.0f;
            for (int index = 0; index < count; ++index)
            {
                const uchar* sample =
                    center + offsets[static_cast<size_t>(index)];
                int difference = 0;
                for (int channel = 0;
                     channel < Channels;
                     ++channel)
                {
                    difference += std::abs(
                        static_cast<int>(sample[channel]) -
                        static_cast<int>(center[channel]));
                }
                const float weight =
                    spatial[static_cast<size_t>(index)].weight *
                    color_weights[static_cast<size_t>(difference)];
                weight_sum += weight;
                for (int channel = 0;
                     channel < Channels;
                     ++channel)
                {
                    weighted_sum[channel] +=
                        weight * sample[channel];
                }
            }
            const float inverse_weight =
                weight_sum > 0.0f ? 1.0f / weight_sum : 0.0f;
            for (int channel = 0; channel < Channels; ++channel)
            {
                output[
                    static_cast<size_t>(x) *
                        static_cast<size_t>(Channels) +
                    static_cast<size_t>(channel)] =
                    weight_sum > 0.0f
                        ? saturate_cast<uchar>(
                              weighted_sum[channel] *
                              inverse_weight)
                        : center[channel];
            }
        }
    }
}

inline void run_u8(
    const Mat& src,
    Mat& dst,
    const std::vector<OffsetWeight>& spatial,
    double sigma_color,
    int border_type)
{
    if (src.channels() == 1)
    {
        g_last_bilateral_algorithm_path = "bilateral_u8_c1_padded";
        run_u8_padded<1>(
            src, dst, spatial, sigma_color, border_type);
        return;
    }
    g_last_bilateral_algorithm_path = "bilateral_u8_c3_padded";
    run_u8_padded<3>(
        src, dst, spatial, sigma_color, border_type);
}

template<typename T>
inline void run(const Mat& src,
                Mat& dst,
                const std::vector<OffsetWeight>& spatial,
                double sigma_color,
                int border_type)
{
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int channels = src.channels();
    const double color_coefficient =
        -0.5 / (sigma_color * sigma_color);
    std::vector<double> color_weights;
    if constexpr (std::is_same<T, uchar>::value)
    {
        color_weights.resize(
            static_cast<size_t>(channels * 255 + 1));
        for (size_t difference = 0;
             difference < color_weights.size();
             ++difference)
        {
            color_weights[difference] =
                std::exp(
                    static_cast<double>(difference * difference) *
                    color_coefficient);
        }
    }

    for (int y = 0; y < rows; ++y)
    {
        const T* center_row = reinterpret_cast<const T*>(
            src.data + static_cast<size_t>(y) * src.step(0));
        T* output = reinterpret_cast<T*>(
            dst.data + static_cast<size_t>(y) * dst.step(0));
        for (int x = 0; x < cols; ++x)
        {
            const T* center =
                center_row + static_cast<size_t>(x) * channels;
            double weighted_sum[3] = {0.0, 0.0, 0.0};
            double weight_sum = 0.0;
            for (const OffsetWeight& offset : spatial)
            {
                const int source_y = detail::border_interpolate(
                    y + offset.y, rows, border_type);
                const int source_x = detail::border_interpolate(
                    x + offset.x, cols, border_type);
                if (source_y < 0 || source_x < 0)
                {
                    continue;
                }
                const T* sample_row = reinterpret_cast<const T*>(
                    src.data +
                    static_cast<size_t>(source_y) * src.step(0));
                const T* sample =
                    sample_row + static_cast<size_t>(source_x) * channels;
                double color_distance = 0.0;
                if constexpr (std::is_same<T, uchar>::value)
                {
                    int difference = 0;
                    for (int ch = 0; ch < channels; ++ch)
                    {
                        difference += std::abs(
                            static_cast<int>(sample[ch]) -
                            static_cast<int>(center[ch]));
                    }
                    color_distance =
                        color_weights[static_cast<size_t>(difference)];
                }
                else
                {
                    for (int ch = 0; ch < channels; ++ch)
                    {
                        color_distance += std::fabs(
                            static_cast<double>(sample[ch]) -
                            static_cast<double>(center[ch]));
                    }
                    color_distance =
                        std::exp(
                            color_distance * color_distance *
                            color_coefficient);
                }
                const double weight =
                    offset.weight * color_distance;
                weight_sum += weight;
                for (int ch = 0; ch < channels; ++ch)
                {
                    weighted_sum[ch] +=
                        weight * static_cast<double>(sample[ch]);
                }
            }
            for (int ch = 0; ch < channels; ++ch)
            {
                const double value =
                    weight_sum > 0.0
                        ? weighted_sum[ch] / weight_sum
                        : static_cast<double>(center[ch]);
                if constexpr (std::is_same<T, uchar>::value)
                {
                    output[static_cast<size_t>(x) * channels + ch] =
                        saturate_cast<uchar>(value);
                }
                else
                {
                    output[static_cast<size_t>(x) * channels + ch] =
                        static_cast<float>(value);
                }
            }
        }
    }
}

}  // namespace bilateral_filter_detail

inline void bilateralFilter(const Mat& src,
                            Mat& dst,
                            int d,
                            double sigmaColor,
                            double sigmaSpace,
                            int borderType = BORDER_DEFAULT)
{
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    bilateral_filter_detail::g_last_bilateral_algorithm_path =
        "bilateral_generic";
    if (src.empty() || src.dims != 2 ||
        (src.depth() != CV_8U && src.depth() != CV_32F) ||
        (src.channels() != 1 && src.channels() != 3))
    {
        CV_Error(Error::StsBadArg, "bilateralFilter unsupported source");
    }
    if (src.data == dst.data)
    {
        CV_Error(Error::StsBadArg, "bilateralFilter does not support in-place operation");
    }
    if (!std::isfinite(sigmaColor) || !std::isfinite(sigmaSpace))
    {
        CV_Error(Error::StsBadArg, "bilateralFilter sigma values must be finite");
    }
    const int border_type = detail::normalize_border_type(borderType);
    if (!detail::is_supported_filter_border(border_type))
    {
        CV_Error(Error::StsBadArg, "bilateralFilter unsupported border");
    }

    dst.create(src.shape(), src.type());
    if (sigmaColor <= 1e-6 || sigmaSpace <= 1e-6)
    {
        src.copyTo(dst);
        return;
    }
    int radius =
        d <= 0 ? static_cast<int>(std::lround(sigmaSpace * 1.5)) : d / 2;
    radius = std::max(radius, 1);
    const std::vector<bilateral_filter_detail::OffsetWeight> spatial =
        bilateral_filter_detail::spatial_weights(radius, sigmaSpace);
    if (src.depth() == CV_8U)
    {
        bilateral_filter_detail::run_u8(
            src, dst, spatial, sigmaColor, border_type);
    }
    else
    {
        bilateral_filter_detail::run<float>(
            src, dst, spatial, sigmaColor, border_type);
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_BILATERAL_FILTER_H
