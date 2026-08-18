#ifndef CVH_PIPELINE_DETAIL_SCALAR_QUANTIZED_MODEL_INPUT_FUSED_HPP
#define CVH_PIPELINE_DETAIL_SCALAR_QUANTIZED_MODEL_INPUT_FUSED_HPP

#include "ir.hpp"
#include "quantize.hpp"
#include "../../core/mat.h"
#include "../../core/saturate.h"

#include <cstddef>

namespace cvh {
namespace detail {

template <typename T>
inline PipelineStatus executeScalarQuantizedModelInputFusedTyped(
    const PipelinePlanImpl& plan,
    const PipelineExecutionGroup& group,
    const Mat& source,
    Mat& target)
{
    std::size_t index = group.semantic_begin;
    const PipelinePlannedStage* color = nullptr;
    if (plan.stages[index].operation.kind == PipelineOperationKind::Color)
    {
        color = &plan.stages[index++];
    }
    const PipelinePlannedStage& geometry = plan.stages[index++];
    const PipelinePlannedStage& normalize = plan.stages[index++];
    const PipelinePlannedStage& quantize = plan.stages[index++];
    const PipelinePlannedStage& layout = plan.stages[index];
    if (layout.operation.kind != PipelineOperationKind::Layout ||
        group.semantic_end != index + 1)
    {
        return PipelineStatus::failure(
            PipelineStatusCode::InternalError,
            static_cast<int>(group.semantic_begin),
            "invalid scalar quantized model-input fusion group");
    }

    const int rows = geometry.output.image.height;
    const int cols = geometry.output.image.width;
    const bool letterbox =
        geometry.operation.kind == PipelineOperationKind::Letterbox;
    const bool nearest =
        (letterbox
             ? geometry.operation.letterbox.interpolation
             : geometry.operation.resize.interpolation) == INTER_NEAREST;
    const Color output_color = color == nullptr
        ? plan.input.image.color
        : color->operation.color.target;
    const bool swap_channels = plan.input.image.color != output_color;
    const PipelineNormalizeOperation& normalize_parameters =
        normalize.operation.normalize;
    const PipelineQuantizeOperation& quantize_parameters =
        quantize.operation.quantize;
    T* target_data = reinterpret_cast<T*>(target.data);

    if (!letterbox)
    {
        for (int y = 0; y < rows; ++y)
        {
            const int source_y0 =
                geometry.y0[static_cast<std::size_t>(y)];
            const int source_y1 = nearest
                ? source_y0
                : geometry.y1[static_cast<std::size_t>(y)];
            const float weight_y = nearest
                ? 0.0f
                : geometry.wy[static_cast<std::size_t>(y)];
            const uchar* source_row0 = source.data +
                static_cast<std::size_t>(source_y0) * source.step(0);
            const uchar* source_row1 = source.data +
                static_cast<std::size_t>(source_y1) * source.step(0);
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
                    const int offset00 =
                        source_x0 * 3 + source_channel;
                    uchar resized_value = source_row0[offset00];
                    if (!nearest)
                    {
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
                    const int parameter =
                        normalize_parameters.count == 1 ? 0 : channel;
                    const float real_value =
                        (static_cast<float>(resized_value) -
                         normalize_parameters.mean[
                             static_cast<std::size_t>(parameter)]) /
                        normalize_parameters.stddev[
                            static_cast<std::size_t>(parameter)];
                    const std::size_t output_index =
                        layout.operation.layout.target == Layout::NCHW
                        ? (static_cast<std::size_t>(channel) * rows + y) *
                                  cols +
                              x
                        : (static_cast<std::size_t>(y) * cols + x) * 3 +
                              channel;
                    target_data[output_index] = quantizeValue<T>(
                        real_value, quantize_parameters);
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
        const int source_y1 = nearest || padded_y
            ? source_y0
            : geometry.y1[static_cast<std::size_t>(geometry_y)];
        const float weight_y = nearest || padded_y
            ? 0.0f
            : geometry.wy[static_cast<std::size_t>(geometry_y)];
        const uchar* source_row0 = source.data +
            static_cast<std::size_t>(source_y0) * source.step(0);
        const uchar* source_row1 = source.data +
            static_cast<std::size_t>(source_y1) * source.step(0);

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
                uchar resized_value = 0;
                if (padded)
                {
                    const PipelineLetterboxOperation& operation =
                        geometry.operation.letterbox;
                    const int parameter =
                        operation.pad_count == 1 ? 0 : channel;
                    resized_value = saturate_cast<uchar>(
                        operation.pad_value[
                            static_cast<std::size_t>(parameter)]);
                }
                else
                {
                    const int source_channel = swap_channels
                        ? 2 - channel
                        : channel;
                    const int offset00 =
                        source_x0 * 3 + source_channel;
                    resized_value = source_row0[offset00];
                    if (!nearest)
                    {
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
                const int parameter =
                    normalize_parameters.count == 1 ? 0 : channel;
                const float real_value =
                    (static_cast<float>(resized_value) -
                     normalize_parameters.mean[
                         static_cast<std::size_t>(parameter)]) /
                    normalize_parameters.stddev[
                        static_cast<std::size_t>(parameter)];
                const std::size_t output_index =
                    layout.operation.layout.target == Layout::NCHW
                    ? (static_cast<std::size_t>(channel) * rows + y) * cols +
                          x
                    : (static_cast<std::size_t>(y) * cols + x) * 3 +
                          channel;
                target_data[output_index] =
                    quantizeValue<T>(real_value, quantize_parameters);
            }
        }
    }
    return PipelineStatus();
}

inline PipelineStatus executeScalarQuantizedModelInputFused(
    const PipelinePlanImpl& plan,
    const PipelineExecutionGroup& group,
    const Mat& source,
    Mat& target)
{
    if (plan.output.tensor.data_type == PipelineDataType::U8)
    {
        return executeScalarQuantizedModelInputFusedTyped<uchar>(
            plan, group, source, target);
    }
    if (plan.output.tensor.data_type == PipelineDataType::S8)
    {
        return executeScalarQuantizedModelInputFusedTyped<schar>(
            plan, group, source, target);
    }
    return PipelineStatus::failure(
        PipelineStatusCode::InternalError,
        static_cast<int>(group.semantic_begin),
        "quantized model-input output type is invalid");
}

}  // namespace detail
}  // namespace cvh

#endif  // CVH_PIPELINE_DETAIL_SCALAR_QUANTIZED_MODEL_INPUT_FUSED_HPP
