#ifndef CVH_PIPELINE_DETAIL_SCALAR_MODEL_INPUT_FUSED_HPP
#define CVH_PIPELINE_DETAIL_SCALAR_MODEL_INPUT_FUSED_HPP

#include "ir.hpp"
#include "../../core/mat.h"
#include "../../core/saturate.h"

#include <cstddef>

namespace cvh {
namespace detail {

inline PipelineStatus executeScalarModelInputFused(
    const PipelinePlanImpl& plan,
    const PipelineExecutionGroup& group,
    const Mat& source,
    Mat& target)
{
    std::size_t index = group.semantic_begin;
    const PipelinePlannedStage* color = nullptr;
    if (plan.stages[index].operation.kind ==
        PipelineOperationKind::Color)
    {
        color = &plan.stages[index++];
    }
    const PipelinePlannedStage& geometry = plan.stages[index++];
    const PipelinePlannedStage& normalize = plan.stages[index++];
    const PipelinePlannedStage& layout = plan.stages[index];
    if (layout.operation.kind != PipelineOperationKind::Layout ||
        group.semantic_end != index + 1)
    {
        return PipelineStatus::failure(
            PipelineStatusCode::InternalError,
            static_cast<int>(group.semantic_begin),
            "invalid scalar model-input fusion group");
    }

    const int rows = geometry.output.image.height;
    const int cols = geometry.output.image.width;
    const bool letterbox =
        geometry.operation.kind == PipelineOperationKind::Letterbox;
    const bool nearest =
        (letterbox
             ? geometry.operation.letterbox.interpolation
             : geometry.operation.resize.interpolation) == INTER_NEAREST;
    const Color source_color = plan.input.image.color;
    const Color output_color = color == nullptr
        ? source_color
        : color->operation.color.target;
    const bool swap_channels = source_color != output_color;
    const PipelineNormalizeOperation& parameters =
        normalize.operation.normalize;
    float* target_data = reinterpret_cast<float*>(target.data);
    float normalized_lut[3][256];
    for (int channel = 0; channel < 3; ++channel)
    {
        const int parameter = parameters.count == 1 ? 0 : channel;
        for (int value = 0; value < 256; ++value)
        {
            normalized_lut[channel][value] =
                (static_cast<float>(value) -
                 parameters.mean[static_cast<std::size_t>(parameter)]) /
                parameters.stddev[static_cast<std::size_t>(parameter)];
        }
    }

    if (!letterbox)
    {
        for (int y = 0; y < rows; ++y)
        {
            const int source_y0 =
                geometry.y0[static_cast<std::size_t>(y)];
            const uchar* source_row0 = source.data +
                static_cast<std::size_t>(source_y0) * source.step(0);
            const int source_y1 = nearest
                ? source_y0
                : geometry.y1[static_cast<std::size_t>(y)];
            const uchar* source_row1 = source.data +
                static_cast<std::size_t>(source_y1) * source.step(0);
            const float weight_y = nearest
                ? 0.0f
                : geometry.wy[static_cast<std::size_t>(y)];
            for (int x = 0; x < cols; ++x)
            {
                const int source_x0 =
                    geometry.x0[static_cast<std::size_t>(x)];
                const int source_x1 = nearest
                    ? source_x0
                    : geometry.x1[static_cast<std::size_t>(x)];
                const float weight_x = nearest
                    ? 0.0f
                    : geometry.wx[static_cast<std::size_t>(x)];
                for (int channel = 0; channel < 3; ++channel)
                {
                    const int source_channel = swap_channels
                        ? 2 - channel
                        : channel;
                    uchar resized_value = source_row0[
                        source_x0 * 3 + source_channel];
                    if (!nearest)
                    {
                        const int offset00 =
                            source_x0 * 3 + source_channel;
                        const int offset01 =
                            source_x1 * 3 + source_channel;
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
                        resized_value = saturate_cast<uchar>(
                            top + (bottom - top) * weight_y);
                    }
                    const float normalized =
                        normalized_lut[channel][resized_value];
                    const std::size_t output_index =
                        layout.operation.layout.target == Layout::NCHW
                        ? (static_cast<std::size_t>(channel) * rows + y) *
                                  cols +
                              x
                        : (static_cast<std::size_t>(y) * cols + x) * 3 +
                              channel;
                    target_data[output_index] = normalized;
                }
            }
        }
        return PipelineStatus();
    }

    for (int y = 0; y < rows; ++y)
    {
        const bool padded_y = letterbox &&
            (y < geometry.transform.pad_top ||
             y >= geometry.transform.pad_top +
                      geometry.transform.resized_height);
        const int geometry_y = letterbox
            ? y - geometry.transform.pad_top
            : y;
        const int source_y0 = padded_y
            ? 0
            : geometry.y0[static_cast<std::size_t>(geometry_y)];
        const uchar* source_row0 = source.data +
            static_cast<std::size_t>(source_y0) * source.step(0);
        const int source_y1 = nearest || padded_y
            ? source_y0
            : geometry.y1[static_cast<std::size_t>(geometry_y)];
        const uchar* source_row1 = source.data +
            static_cast<std::size_t>(source_y1) * source.step(0);
        const float weight_y = nearest || padded_y
            ? 0.0f
            : geometry.wy[static_cast<std::size_t>(geometry_y)];

        for (int x = 0; x < cols; ++x)
        {
            const bool padded = padded_y ||
                (letterbox &&
                 (x < geometry.transform.pad_left ||
                  x >= geometry.transform.pad_left +
                           geometry.transform.resized_width));
            const int geometry_x = letterbox
                ? x - geometry.transform.pad_left
                : x;
            const int source_x0 = padded
                ? 0
                : geometry.x0[static_cast<std::size_t>(geometry_x)];
            const int source_x1 = nearest || padded
                ? source_x0
                : geometry.x1[static_cast<std::size_t>(geometry_x)];
            const float weight_x = nearest || padded
                ? 0.0f
                : geometry.wx[static_cast<std::size_t>(geometry_x)];

            for (int channel = 0; channel < 3; ++channel)
            {
                const int source_channel = swap_channels
                    ? 2 - channel
                    : channel;
                uchar resized_value = 0;
                if (padded)
                {
                    const PipelineLetterboxOperation& operation =
                        geometry.operation.letterbox;
                    const int pad_parameter =
                        operation.pad_count == 1 ? 0 : channel;
                    resized_value = saturate_cast<uchar>(
                        operation.pad_value[
                            static_cast<std::size_t>(pad_parameter)]);
                }
                else
                {
                    resized_value = source_row0[
                        source_x0 * 3 + source_channel];
                    if (!nearest)
                    {
                        const int offset00 =
                            source_x0 * 3 + source_channel;
                        const int offset01 =
                            source_x1 * 3 + source_channel;
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
                        resized_value = saturate_cast<uchar>(
                            top + (bottom - top) * weight_y);
                    }
                }

                const float normalized =
                    normalized_lut[channel][resized_value];
                const std::size_t output_index =
                    layout.operation.layout.target == Layout::NCHW
                    ? (static_cast<std::size_t>(channel) * rows + y) *
                          cols + x
                    : (static_cast<std::size_t>(y) * cols + x) * 3 +
                          channel;
                target_data[output_index] = normalized;
            }
        }
    }
    return PipelineStatus();
}

}  // namespace detail
}  // namespace cvh

#endif  // CVH_PIPELINE_DETAIL_SCALAR_MODEL_INPUT_FUSED_HPP
