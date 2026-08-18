#ifndef CVH_PIPELINE_DETAIL_SCALAR_YUV_MODEL_INPUT_FUSED_HPP
#define CVH_PIPELINE_DETAIL_SCALAR_YUV_MODEL_INPUT_FUSED_HPP

#include "ir.hpp"
#include "../../core/mat.h"
#include "../../core/saturate.h"
#include "../views.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace cvh {
namespace detail {

inline float sampleYuvChromaPlane(const ConstImageView& source,
                                  int source_x,
                                  int source_y,
                                  int component)
{
    const int chroma_width = source.descriptor.width / 2;
    const int chroma_height = source.descriptor.height / 2;
    const float chroma_x =
        source.descriptor.color_spec.chroma_location ==
                ChromaLocation::Left
        ? static_cast<float>(source_x) * 0.5f
        : (static_cast<float>(source_x) - 0.5f) * 0.5f;
    const float chroma_y =
        (static_cast<float>(source_y) - 0.5f) * 0.5f;
    const float clamped_x = std::clamp(
        chroma_x, 0.0f, static_cast<float>(chroma_width - 1));
    const float clamped_y = std::clamp(
        chroma_y, 0.0f, static_cast<float>(chroma_height - 1));
    const int x0 = static_cast<int>(std::floor(clamped_x));
    const int y0 = static_cast<int>(std::floor(clamped_y));
    const int x1 = std::min(x0 + 1, chroma_width - 1);
    const int y1 = std::min(y0 + 1, chroma_height - 1);
    const float wx = clamped_x - static_cast<float>(x0);
    const float wy = clamped_y - static_cast<float>(y0);
    const int uv_component =
        source.descriptor.pixel_format == PixelFormat::NV12
        ? component
        : 1 - component;
    const uchar* row0 = source.planes[1].data +
        static_cast<std::size_t>(y0) * source.planes[1].row_stride;
    const uchar* row1 = source.planes[1].data +
        static_cast<std::size_t>(y1) * source.planes[1].row_stride;
    const float top = static_cast<float>(row0[x0 * 2 + uv_component]) +
        (static_cast<float>(row0[x1 * 2 + uv_component]) -
         static_cast<float>(row0[x0 * 2 + uv_component])) * wx;
    const float bottom = static_cast<float>(row1[x0 * 2 + uv_component]) +
        (static_cast<float>(row1[x1 * 2 + uv_component]) -
         static_cast<float>(row1[x0 * 2 + uv_component])) * wx;
    return top + (bottom - top) * wy;
}

inline std::array<uchar, 3> convertYuvPixelToRgb(
    const ConstImageView& source,
    int source_x,
    int source_y)
{
    const uchar* y_row = source.planes[0].data +
        static_cast<std::size_t>(source_y) *
            source.planes[0].row_stride;
    float y = static_cast<float>(y_row[source_x]);
    float cb = sampleYuvChromaPlane(source, source_x, source_y, 0) -
        128.0f;
    float cr = sampleYuvChromaPlane(source, source_x, source_y, 1) -
        128.0f;
    if (source.descriptor.color_spec.range == ColorRange::Limited)
    {
        y = (y - 16.0f) * (255.0f / 219.0f);
        cb *= 255.0f / 224.0f;
        cr *= 255.0f / 224.0f;
    }

    float red_coefficient = 1.402f;
    float green_cb_coefficient = -0.344136f;
    float green_cr_coefficient = -0.714136f;
    float blue_coefficient = 1.772f;
    if (source.descriptor.color_spec.matrix == ColorMatrix::BT709)
    {
        red_coefficient = 1.5748f;
        green_cb_coefficient = -0.187324f;
        green_cr_coefficient = -0.468124f;
        blue_coefficient = 1.8556f;
    }
    else if (source.descriptor.color_spec.matrix == ColorMatrix::BT2020)
    {
        red_coefficient = 1.4746f;
        green_cb_coefficient = -0.164553f;
        green_cr_coefficient = -0.571353f;
        blue_coefficient = 1.8814f;
    }

    return {{
        saturate_cast<uchar>(y + red_coefficient * cr),
        saturate_cast<uchar>(
            y + green_cb_coefficient * cb +
            green_cr_coefficient * cr),
        saturate_cast<uchar>(y + blue_coefficient * cb)}};
}

inline uchar yuvOutputChannel(const std::array<uchar, 3>& rgb,
                              Color output_color,
                              int channel)
{
    return output_color == Color::RGB
        ? rgb[static_cast<std::size_t>(channel)]
        : rgb[static_cast<std::size_t>(2 - channel)];
}

inline PipelineStatus executeScalarYuvModelInputFused(
    const PipelinePlanImpl& plan,
    const PipelineExecutionGroup& group,
    ConstImageView source,
    Mat& target)
{
    if (group.semantic_begin != 0 || group.semantic_end != 4)
    {
        return PipelineStatus::failure(
            PipelineStatusCode::InternalError,
            static_cast<int>(group.semantic_begin),
            "invalid scalar YUV model-input fusion group");
    }
    const PipelinePlannedStage& color = plan.stages[0];
    const PipelinePlannedStage& geometry = plan.stages[1];
    const PipelinePlannedStage& normalize = plan.stages[2];
    const PipelinePlannedStage& layout = plan.stages[3];
    const int rows = geometry.output.image.height;
    const int cols = geometry.output.image.width;
    const bool letterbox =
        geometry.operation.kind == PipelineOperationKind::Letterbox;
    const bool nearest =
        (letterbox
             ? geometry.operation.letterbox.interpolation
             : geometry.operation.resize.interpolation) == INTER_NEAREST;
    const Color output_color = color.operation.color.target;
    const PipelineNormalizeOperation& parameters =
        normalize.operation.normalize;
    float* target_data = reinterpret_cast<float*>(target.data);

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
                uchar resized = 0;
                if (padded)
                {
                    const PipelineLetterboxOperation& operation =
                        geometry.operation.letterbox;
                    const int pad_parameter =
                        operation.pad_count == 1 ? 0 : channel;
                    resized = saturate_cast<uchar>(
                        operation.pad_value[
                            static_cast<std::size_t>(pad_parameter)]);
                }
                else
                {
                    resized = yuvOutputChannel(
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
                        resized = saturate_cast<uchar>(
                            top + (bottom - top) * weight_y);
                    }
                }
                const int parameter =
                    parameters.count == 1 ? 0 : channel;
                const float normalized =
                    (static_cast<float>(resized) -
                     parameters.mean[
                         static_cast<std::size_t>(parameter)]) /
                    parameters.stddev[
                        static_cast<std::size_t>(parameter)];
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

#endif  // CVH_PIPELINE_DETAIL_SCALAR_YUV_MODEL_INPUT_FUSED_HPP
