#ifndef CVH_PIPELINE_DETAIL_SCALAR_STAGE_EXECUTOR_HPP
#define CVH_PIPELINE_DETAIL_SCALAR_STAGE_EXECUTOR_HPP

#include "ir.hpp"
#include "quantize.hpp"
#include "../../core/mat.h"
#include "../../core/saturate.h"

#include <cstddef>
#include <cstring>
#include <type_traits>

namespace cvh {
namespace detail {

inline void copyImage(const Mat& source, Mat& target)
{
    const std::size_t row_bytes =
        static_cast<std::size_t>(source.size[1]) * source.elemSize();
    for (int y = 0; y < source.size[0]; ++y)
    {
        std::memcpy(
            target.data + static_cast<std::size_t>(y) * target.step(0),
            source.data + static_cast<std::size_t>(y) * source.step(0),
            row_bytes);
    }
}

template <typename T>
inline PipelineStatus executeColorTyped(const PipelinePlannedStage& stage,
                                        const Mat& source,
                                        Mat& target)
{
    const Color source_color = stage.input.image.color;
    const Color target_color = stage.output.image.color;
    const int rows = source.size[0];
    const int cols = source.size[1];

    if (source_color == target_color)
    {
        copyImage(source, target);
        return PipelineStatus();
    }

    for (int y = 0; y < rows; ++y)
    {
        const T* source_row = reinterpret_cast<const T*>(
            source.data + static_cast<std::size_t>(y) * source.step(0));
        T* target_row = reinterpret_cast<T*>(
            target.data + static_cast<std::size_t>(y) * target.step(0));

        if ((source_color == Color::BGR && target_color == Color::RGB) ||
            (source_color == Color::RGB && target_color == Color::BGR))
        {
            for (int x = 0; x < cols; ++x)
            {
                const int offset = x * 3;
                target_row[offset] = source_row[offset + 2];
                target_row[offset + 1] = source_row[offset + 1];
                target_row[offset + 2] = source_row[offset];
            }
            continue;
        }

        if (source_color == Color::Gray &&
            (target_color == Color::BGR || target_color == Color::RGB))
        {
            for (int x = 0; x < cols; ++x)
            {
                const int offset = x * 3;
                target_row[offset] = source_row[x];
                target_row[offset + 1] = source_row[x];
                target_row[offset + 2] = source_row[x];
            }
            continue;
        }

        if ((source_color == Color::BGR || source_color == Color::RGB) &&
            target_color == Color::Gray)
        {
            const int blue_index = source_color == Color::RGB ? 2 : 0;
            const int red_index = source_color == Color::RGB ? 0 : 2;
            for (int x = 0; x < cols; ++x)
            {
                const int offset = x * 3;
                if constexpr (std::is_same<T, uchar>::value)
                {
                    const int blue = source_row[offset + blue_index];
                    const int green = source_row[offset + 1];
                    const int red = source_row[offset + red_index];
                    target_row[x] = static_cast<uchar>(
                        (7471 * blue +
                         38470 * green +
                         19595 * red +
                         (1 << 15)) >>
                        16);
                }
                else
                {
                    target_row[x] = static_cast<T>(
                        0.114f * source_row[offset + blue_index] +
                        0.587f * source_row[offset + 1] +
                        0.299f * source_row[offset + red_index]);
                }
            }
            continue;
        }

        return PipelineStatus::failure(
            PipelineStatusCode::Unsupported,
            -1,
            "scalar color conversion is unsupported");
    }
    return PipelineStatus();
}

inline PipelineStatus executeColor(const PipelinePlannedStage& stage,
                                   const Mat& source,
                                   Mat& target)
{
    if (stage.input.image.data_type == PipelineDataType::U8)
    {
        return executeColorTyped<uchar>(stage, source, target);
    }
    if (stage.input.image.data_type == PipelineDataType::F32)
    {
        return executeColorTyped<float>(stage, source, target);
    }
    return PipelineStatus::failure(
        PipelineStatusCode::Unsupported,
        -1,
        "scalar color data type is unsupported");
}

template <typename T>
inline PipelineStatus executeResizeTyped(const PipelinePlannedStage& stage,
                                         const Mat& source,
                                         Mat& target)
{
    const int channels = source.channels();
    const int target_rows = target.size[0];
    const int target_cols = target.size[1];
    const bool nearest =
        stage.operation.resize.interpolation == INTER_NEAREST;

    for (int y = 0; y < target_rows; ++y)
    {
        const int source_y0 = stage.y0[static_cast<std::size_t>(y)];
        const T* source_row0 = reinterpret_cast<const T*>(
            source.data +
            static_cast<std::size_t>(source_y0) * source.step(0));
        T* target_row = reinterpret_cast<T*>(
            target.data + static_cast<std::size_t>(y) * target.step(0));

        if (nearest)
        {
            for (int x = 0; x < target_cols; ++x)
            {
                const int source_x =
                    stage.x0[static_cast<std::size_t>(x)];
                for (int channel = 0; channel < channels; ++channel)
                {
                    target_row[x * channels + channel] =
                        source_row0[source_x * channels + channel];
                }
            }
            continue;
        }

        const int source_y1 = stage.y1[static_cast<std::size_t>(y)];
        const float weight_y = stage.wy[static_cast<std::size_t>(y)];
        const T* source_row1 = reinterpret_cast<const T*>(
            source.data +
            static_cast<std::size_t>(source_y1) * source.step(0));

        for (int x = 0; x < target_cols; ++x)
        {
            const int source_x0 =
                stage.x0[static_cast<std::size_t>(x)];
            const int source_x1 =
                stage.x1[static_cast<std::size_t>(x)];
            const float weight_x =
                stage.wx[static_cast<std::size_t>(x)];

            for (int channel = 0; channel < channels; ++channel)
            {
                const int offset00 = source_x0 * channels + channel;
                const int offset01 = source_x1 * channels + channel;
                const float top =
                    static_cast<float>(source_row0[offset00]) +
                    (static_cast<float>(source_row0[offset01]) -
                     static_cast<float>(source_row0[offset00])) *
                        weight_x;
                const float bottom =
                    static_cast<float>(source_row1[offset00]) +
                    (static_cast<float>(source_row1[offset01]) -
                     static_cast<float>(source_row1[offset00])) *
                        weight_x;
                const float value = top + (bottom - top) * weight_y;
                if constexpr (std::is_same<T, uchar>::value)
                {
                    target_row[x * channels + channel] =
                        saturate_cast<uchar>(value);
                }
                else
                {
                    target_row[x * channels + channel] =
                        static_cast<T>(value);
                }
            }
        }
    }
    return PipelineStatus();
}

inline PipelineStatus executeResize(const PipelinePlannedStage& stage,
                                    const Mat& source,
                                    Mat& target)
{
    if (stage.input.image.data_type == PipelineDataType::U8)
    {
        return executeResizeTyped<uchar>(stage, source, target);
    }
    if (stage.input.image.data_type == PipelineDataType::F32)
    {
        return executeResizeTyped<float>(stage, source, target);
    }
    return PipelineStatus::failure(
        PipelineStatusCode::Unsupported,
        -1,
        "scalar resize data type is unsupported");
}

template <typename T>
inline T letterboxPadValue(const PipelineLetterboxOperation& operation,
                           int channel)
{
    const int parameter = operation.pad_count == 1 ? 0 : channel;
    const float value =
        operation.pad_value[static_cast<std::size_t>(parameter)];
    if constexpr (std::is_same<T, uchar>::value)
    {
        return saturate_cast<uchar>(value);
    }
    return static_cast<T>(value);
}

template <typename T>
inline PipelineStatus executeLetterboxTyped(
    const PipelinePlannedStage& stage,
    const Mat& source,
    Mat& target)
{
    const PipelineTransform& transform = stage.transform;
    const PipelineLetterboxOperation& operation =
        stage.operation.letterbox;
    const int channels = source.channels();
    const bool nearest = operation.interpolation == INTER_NEAREST;

    for (int y = 0; y < target.size[0]; ++y)
    {
        T* target_row = reinterpret_cast<T*>(
            target.data + static_cast<std::size_t>(y) * target.step(0));
        const bool padded_y = y < transform.pad_top ||
            y >= transform.pad_top + transform.resized_height;
        const int resized_y = y - transform.pad_top;
        const int source_y0 = padded_y
            ? 0
            : stage.y0[static_cast<std::size_t>(resized_y)];
        const int source_y1 = padded_y || nearest
            ? source_y0
            : stage.y1[static_cast<std::size_t>(resized_y)];
        const float weight_y = padded_y || nearest
            ? 0.0f
            : stage.wy[static_cast<std::size_t>(resized_y)];
        const T* source_row0 = reinterpret_cast<const T*>(
            source.data +
            static_cast<std::size_t>(source_y0) * source.step(0));
        const T* source_row1 = reinterpret_cast<const T*>(
            source.data +
            static_cast<std::size_t>(source_y1) * source.step(0));

        for (int x = 0; x < target.size[1]; ++x)
        {
            const bool padded = padded_y || x < transform.pad_left ||
                x >= transform.pad_left + transform.resized_width;
            if (padded)
            {
                for (int channel = 0; channel < channels; ++channel)
                {
                    target_row[x * channels + channel] =
                        letterboxPadValue<T>(operation, channel);
                }
                continue;
            }

            const int resized_x = x - transform.pad_left;
            const int source_x0 =
                stage.x0[static_cast<std::size_t>(resized_x)];
            const int source_x1 = nearest
                ? source_x0
                : stage.x1[static_cast<std::size_t>(resized_x)];
            const float weight_x = nearest
                ? 0.0f
                : stage.wx[static_cast<std::size_t>(resized_x)];
            for (int channel = 0; channel < channels; ++channel)
            {
                const int offset00 = source_x0 * channels + channel;
                T value = source_row0[offset00];
                if (!nearest)
                {
                    const int offset01 =
                        source_x1 * channels + channel;
                    const float top =
                        static_cast<float>(source_row0[offset00]) +
                        (static_cast<float>(source_row0[offset01]) -
                         static_cast<float>(source_row0[offset00])) *
                            weight_x;
                    const float bottom =
                        static_cast<float>(source_row1[offset00]) +
                        (static_cast<float>(source_row1[offset01]) -
                         static_cast<float>(source_row1[offset00])) *
                            weight_x;
                    const float sampled =
                        top + (bottom - top) * weight_y;
                    if constexpr (std::is_same<T, uchar>::value)
                    {
                        value = saturate_cast<uchar>(sampled);
                    }
                    else
                    {
                        value = static_cast<T>(sampled);
                    }
                }
                target_row[x * channels + channel] = value;
            }
        }
    }
    return PipelineStatus();
}

inline PipelineStatus executeLetterbox(
    const PipelinePlannedStage& stage,
    const Mat& source,
    Mat& target)
{
    if (stage.input.image.data_type == PipelineDataType::U8)
    {
        return executeLetterboxTyped<uchar>(stage, source, target);
    }
    if (stage.input.image.data_type == PipelineDataType::F32)
    {
        return executeLetterboxTyped<float>(stage, source, target);
    }
    return PipelineStatus::failure(
        PipelineStatusCode::Unsupported,
        -1,
        "scalar letterbox data type is unsupported");
}

template <typename T>
inline PipelineStatus executeNormalizeTyped(
    const PipelinePlannedStage& stage,
    const Mat& source,
    Mat& target)
{
    const int channels = source.channels();
    const PipelineNormalizeOperation& normalize =
        stage.operation.normalize;
    for (int y = 0; y < source.size[0]; ++y)
    {
        const T* source_row = reinterpret_cast<const T*>(
            source.data + static_cast<std::size_t>(y) * source.step(0));
        float* target_row = reinterpret_cast<float*>(
            target.data + static_cast<std::size_t>(y) * target.step(0));
        for (int x = 0; x < source.size[1]; ++x)
        {
            for (int channel = 0; channel < channels; ++channel)
            {
                const int parameter =
                    normalize.count == 1 ? 0 : channel;
                const int offset = x * channels + channel;
                target_row[offset] =
                    (static_cast<float>(source_row[offset]) -
                     normalize.mean[static_cast<std::size_t>(parameter)]) /
                    normalize.stddev[
                        static_cast<std::size_t>(parameter)];
            }
        }
    }
    return PipelineStatus();
}

inline PipelineStatus executeNormalize(
    const PipelinePlannedStage& stage,
    const Mat& source,
    Mat& target)
{
    if (stage.input.image.data_type == PipelineDataType::U8)
    {
        return executeNormalizeTyped<uchar>(stage, source, target);
    }
    if (stage.input.image.data_type == PipelineDataType::F32)
    {
        return executeNormalizeTyped<float>(stage, source, target);
    }
    return PipelineStatus::failure(
        PipelineStatusCode::Unsupported,
        -1,
        "scalar normalize data type is unsupported");
}

template <typename T>
inline PipelineStatus executeQuantizeTyped(
    const PipelinePlannedStage& stage,
    const Mat& source,
    Mat& target)
{
    const PipelineQuantizeOperation& operation =
        stage.operation.quantize;
    for (int y = 0; y < source.size[0]; ++y)
    {
        const float* source_row = reinterpret_cast<const float*>(
            source.data + static_cast<std::size_t>(y) * source.step(0));
        T* target_row = reinterpret_cast<T*>(
            target.data + static_cast<std::size_t>(y) * target.step(0));
        const int value_count = source.size[1] * source.channels();
        for (int index = 0; index < value_count; ++index)
        {
            target_row[index] =
                quantizeValue<T>(source_row[index], operation);
        }
    }
    return PipelineStatus();
}

inline PipelineStatus executeQuantize(
    const PipelinePlannedStage& stage,
    const Mat& source,
    Mat& target)
{
    if (stage.output.image.data_type == PipelineDataType::U8)
    {
        return executeQuantizeTyped<uchar>(stage, source, target);
    }
    if (stage.output.image.data_type == PipelineDataType::S8)
    {
        return executeQuantizeTyped<schar>(stage, source, target);
    }
    return PipelineStatus::failure(
        PipelineStatusCode::Unsupported,
        -1,
        "scalar quantize target data type is unsupported");
}

template <typename T>
inline PipelineStatus executeLayoutTyped(const PipelinePlannedStage& stage,
                                         const Mat& source,
                                         Mat& target)
{
    const int rows = source.size[0];
    const int cols = source.size[1];
    const int channels = source.channels();
    T* target_data = reinterpret_cast<T*>(target.data);

    if (stage.operation.layout.target == Layout::NHWC)
    {
        const std::size_t row_bytes =
            static_cast<std::size_t>(cols * channels) * sizeof(T);
        for (int y = 0; y < rows; ++y)
        {
            std::memcpy(
                target_data +
                    static_cast<std::size_t>(y * cols * channels),
                source.data + static_cast<std::size_t>(y) * source.step(0),
                row_bytes);
        }
        return PipelineStatus();
    }

    for (int y = 0; y < rows; ++y)
    {
        const T* source_row = reinterpret_cast<const T*>(
            source.data + static_cast<std::size_t>(y) * source.step(0));
        for (int x = 0; x < cols; ++x)
        {
            for (int channel = 0; channel < channels; ++channel)
            {
                const std::size_t target_index =
                    (static_cast<std::size_t>(channel) *
                         static_cast<std::size_t>(rows) +
                     static_cast<std::size_t>(y)) *
                        static_cast<std::size_t>(cols) +
                    static_cast<std::size_t>(x);
                target_data[target_index] =
                    source_row[x * channels + channel];
            }
        }
    }
    return PipelineStatus();
}

inline PipelineStatus executeLayout(const PipelinePlannedStage& stage,
                                    const Mat& source,
                                    Mat& target)
{
    if (stage.input.image.data_type == PipelineDataType::U8)
    {
        return executeLayoutTyped<uchar>(stage, source, target);
    }
    if (stage.input.image.data_type == PipelineDataType::F32)
    {
        return executeLayoutTyped<float>(stage, source, target);
    }
    if (stage.input.image.data_type == PipelineDataType::S8)
    {
        return executeLayoutTyped<schar>(stage, source, target);
    }
    return PipelineStatus::failure(
        PipelineStatusCode::Unsupported,
        -1,
        "scalar layout data type is unsupported");
}

inline PipelineStatus executeStage(const PipelinePlannedStage& stage,
                                   const Mat& source,
                                   Mat& target)
{
    switch (stage.operation.kind)
    {
    case PipelineOperationKind::Color:
        return executeColor(stage, source, target);
    case PipelineOperationKind::Resize:
        return executeResize(stage, source, target);
    case PipelineOperationKind::Letterbox:
        return executeLetterbox(stage, source, target);
    case PipelineOperationKind::Normalize:
        return executeNormalize(stage, source, target);
    case PipelineOperationKind::Quantize:
        return executeQuantize(stage, source, target);
    case PipelineOperationKind::Layout:
        return executeLayout(stage, source, target);
    }
    return PipelineStatus::failure(
        PipelineStatusCode::InternalError,
        -1,
        "unknown pipeline operation");
}

}  // namespace detail
}  // namespace cvh

#endif  // CVH_PIPELINE_DETAIL_SCALAR_STAGE_EXECUTOR_HPP
