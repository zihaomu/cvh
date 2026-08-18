#ifndef CVH_PIPELINE_DETAIL_SCALAR_YUV_QUANTIZED_MODEL_INPUT_FUSED_HPP
#define CVH_PIPELINE_DETAIL_SCALAR_YUV_QUANTIZED_MODEL_INPUT_FUSED_HPP

#include "quantize.hpp"
#include "scalar_yuv_model_input_fused.hpp"

namespace cvh {
namespace detail {

template <typename T>
inline PipelineStatus executeScalarYuvQuantizedModelInputFusedTyped(
    const PipelinePlanImpl& plan,
    const PipelineExecutionGroup& group,
    ConstImageView source,
    Mat& target)
{
    if (group.semantic_begin != 0 || group.semantic_end != 5)
    {
        return PipelineStatus::failure(
            PipelineStatusCode::InternalError,
            static_cast<int>(group.semantic_begin),
            "invalid scalar YUV quantized model-input fusion group");
    }
    const PipelinePlannedStage& color = plan.stages[0];
    const PipelinePlannedStage& geometry = plan.stages[1];
    const PipelinePlannedStage& normalize = plan.stages[2];
    const PipelinePlannedStage& quantize = plan.stages[3];
    const PipelinePlannedStage& layout = plan.stages[4];
    const int rows = geometry.output.image.height;
    const int cols = geometry.output.image.width;
    const bool letterbox =
        geometry.operation.kind == PipelineOperationKind::Letterbox;
    const bool nearest =
        (letterbox
             ? geometry.operation.letterbox.interpolation
             : geometry.operation.resize.interpolation) == INTER_NEAREST;
    const Color output_color = color.operation.color.target;
    const PipelineNormalizeOperation& normalize_parameters =
        normalize.operation.normalize;
    const PipelineQuantizeOperation& quantize_parameters =
        quantize.operation.quantize;
    T* target_data = reinterpret_cast<T*>(target.data);

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
            const std::array<uchar, 3> pixel00 = padded
                ? std::array<uchar, 3>{}
                : convertYuvPixelToRgb(source, source_x0, source_y0);
            std::array<uchar, 3> pixel01{};
            std::array<uchar, 3> pixel10{};
            std::array<uchar, 3> pixel11{};
            if (!nearest && !padded)
            {
                pixel01 =
                    convertYuvPixelToRgb(source, source_x1, source_y0);
                pixel10 =
                    convertYuvPixelToRgb(source, source_x0, source_y1);
                pixel11 =
                    convertYuvPixelToRgb(source, source_x1, source_y1);
            }

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
                    resized_value = yuvOutputChannel(
                        pixel00, output_color, channel);
                    if (!nearest)
                    {
                        const float top =
                            static_cast<float>(yuvOutputChannel(
                                pixel00, output_color, channel)) +
                            (static_cast<float>(yuvOutputChannel(
                                 pixel01, output_color, channel)) -
                             static_cast<float>(yuvOutputChannel(
                                 pixel00, output_color, channel))) *
                                weight_x;
                        const float bottom =
                            static_cast<float>(yuvOutputChannel(
                                pixel10, output_color, channel)) +
                            (static_cast<float>(yuvOutputChannel(
                                 pixel11, output_color, channel)) -
                             static_cast<float>(yuvOutputChannel(
                                 pixel10, output_color, channel))) *
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

inline PipelineStatus executeScalarYuvQuantizedModelInputFused(
    const PipelinePlanImpl& plan,
    const PipelineExecutionGroup& group,
    ConstImageView source,
    Mat& target)
{
    if (plan.output.tensor.data_type == PipelineDataType::U8)
    {
        return executeScalarYuvQuantizedModelInputFusedTyped<uchar>(
            plan, group, source, target);
    }
    if (plan.output.tensor.data_type == PipelineDataType::S8)
    {
        return executeScalarYuvQuantizedModelInputFusedTyped<schar>(
            plan, group, source, target);
    }
    return PipelineStatus::failure(
        PipelineStatusCode::InternalError,
        static_cast<int>(group.semantic_begin),
        "YUV quantized model-input output type is invalid");
}

}  // namespace detail
}  // namespace cvh

#endif  // CVH_PIPELINE_DETAIL_SCALAR_YUV_QUANTIZED_MODEL_INPUT_FUSED_HPP
