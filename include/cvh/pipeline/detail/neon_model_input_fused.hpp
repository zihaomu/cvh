#ifndef CVH_PIPELINE_DETAIL_NEON_MODEL_INPUT_FUSED_HPP
#define CVH_PIPELINE_DETAIL_NEON_MODEL_INPUT_FUSED_HPP

#include "ir.hpp"
#include "../../core/detail/cpu_features.hpp"
#include "../../core/detail/dispatch_control.h"
#include "../../core/mat.h"
#include "../../core/saturate.h"

#include <cstddef>

namespace cvh {
namespace detail {

inline bool neonModelInputRuntimeAllowed()
{
    const cpu::DispatchMode mode = cpu::dispatch_mode();
    return cpu::neon_runtime_available() &&
           (mode == cpu::DispatchMode::Auto ||
            mode == cpu::DispatchMode::NeonOnly);
}

#if CVH_DETAIL_HAVE_NEON_KERNEL

inline float32x4_t normalizeModelInputU8x4(
    uint16x4_t values,
    float mean,
    float stddev)
{
    const float32x4_t converted =
        vcvtq_f32_u32(vmovl_u16(values));
    return vdivq_f32(
        vsubq_f32(converted, vdupq_n_f32(mean)),
        vdupq_n_f32(stddev));
}

inline PipelineStatus executeNeonModelInputFused(
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
    if (layout.operation.layout.target != Layout::NCHW)
    {
        return PipelineStatus::failure(
            PipelineStatusCode::InternalError,
            static_cast<int>(group.semantic_begin),
            "NEON model-input predicate requires NCHW output");
    }
    const bool letterbox =
        geometry.operation.kind == PipelineOperationKind::Letterbox;
    const int rows = geometry.output.image.height;
    const int cols = geometry.output.image.width;
    const int content_width = letterbox
        ? geometry.transform.resized_width
        : cols;
    const int content_height = letterbox
        ? geometry.transform.resized_height
        : rows;
    const int pad_left = letterbox ? geometry.transform.pad_left : 0;
    const int pad_top = letterbox ? geometry.transform.pad_top : 0;
    const Color output_color = color == nullptr
        ? plan.input.image.color
        : color->operation.color.target;
    const bool swap_channels = plan.input.image.color != output_color;
    const PipelineNormalizeOperation& parameters =
        normalize.operation.normalize;
    float* target_data = reinterpret_cast<float*>(target.data);

    const auto store_pad = [&](int y, int x) {
        const PipelineLetterboxOperation& operation =
            geometry.operation.letterbox;
        for (int channel = 0; channel < 3; ++channel)
        {
            const int pad_parameter =
                operation.pad_count == 1 ? 0 : channel;
            const uchar pad_value = saturate_cast<uchar>(
                operation.pad_value[
                    static_cast<std::size_t>(pad_parameter)]);
            const int normalize_parameter =
                parameters.count == 1 ? 0 : channel;
            target_data[
                (static_cast<std::size_t>(channel) * rows + y) * cols + x] =
                (static_cast<float>(pad_value) -
                 parameters.mean[
                     static_cast<std::size_t>(normalize_parameter)]) /
                parameters.stddev[
                    static_cast<std::size_t>(normalize_parameter)];
        }
    };

    for (int y = 0; y < rows; ++y)
    {
        const bool padded_y =
            y < pad_top || y >= pad_top + content_height;
        if (padded_y)
        {
            for (int x = 0; x < cols; ++x)
            {
                store_pad(y, x);
            }
            continue;
        }
        for (int x = 0; x < pad_left; ++x)
        {
            store_pad(y, x);
        }

        const int geometry_y = y - pad_top;
        const int source_y =
            geometry.y0[static_cast<std::size_t>(geometry_y)];
        const uchar* source_row = source.data +
            static_cast<std::size_t>(source_y) * source.step(0);
        int x = 0;
        for (; x + 8 <= content_width; x += 8)
        {
            const uint8x16x3_t packed = vld3q_u8(
                source_row + static_cast<std::size_t>(x) * 6);
            float32x4_t low[3];
            float32x4_t high[3];
            for (int channel = 0; channel < 3; ++channel)
            {
                const int source_channel = swap_channels
                    ? 2 - channel
                    : channel;
                const uint8x8_t selected = vget_low_u8(
                    vuzp1q_u8(
                        packed.val[source_channel],
                        packed.val[source_channel]));
                const uint16x8_t widened = vmovl_u8(selected);
                const int parameter =
                    parameters.count == 1 ? 0 : channel;
                low[channel] = normalizeModelInputU8x4(
                    vget_low_u16(widened),
                    parameters.mean[
                        static_cast<std::size_t>(parameter)],
                    parameters.stddev[
                        static_cast<std::size_t>(parameter)]);
                high[channel] = normalizeModelInputU8x4(
                    vget_high_u16(widened),
                    parameters.mean[
                        static_cast<std::size_t>(parameter)],
                    parameters.stddev[
                        static_cast<std::size_t>(parameter)]);
            }

            for (int channel = 0; channel < 3; ++channel)
            {
                float* output = target_data +
                    (static_cast<std::size_t>(channel) * rows + y) *
                        cols + pad_left + x;
                vst1q_f32(output, low[channel]);
                vst1q_f32(output + 4, high[channel]);
            }
        }

        for (; x < content_width; ++x)
        {
            const int source_x =
                geometry.x0[static_cast<std::size_t>(x)];
            for (int channel = 0; channel < 3; ++channel)
            {
                const int source_channel = swap_channels
                    ? 2 - channel
                    : channel;
                const int parameter =
                    parameters.count == 1 ? 0 : channel;
                const float value =
                    (static_cast<float>(
                         source_row[source_x * 3 + source_channel]) -
                     parameters.mean[
                         static_cast<std::size_t>(parameter)]) /
                    parameters.stddev[
                        static_cast<std::size_t>(parameter)];
                const std::size_t output_index =
                    (static_cast<std::size_t>(channel) * rows + y) * cols +
                    pad_left + x;
                target_data[output_index] = value;
            }
        }
        for (int output_x = pad_left + content_width;
             output_x < cols;
             ++output_x)
        {
            store_pad(y, output_x);
        }
    }
    return PipelineStatus();
}

#else

inline PipelineStatus executeNeonModelInputFused(
    const PipelinePlanImpl&,
    const PipelineExecutionGroup& group,
    const Mat&,
    Mat&)
{
    return PipelineStatus::failure(
        PipelineStatusCode::Unsupported,
        static_cast<int>(group.semantic_begin),
        "pipeline NEON model-input kernel was not compiled");
}

#endif

}  // namespace detail
}  // namespace cvh

#endif  // CVH_PIPELINE_DETAIL_NEON_MODEL_INPUT_FUSED_HPP
