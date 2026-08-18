#ifndef CVH_PIPELINE_DETAIL_FUSION_RULES_HPP
#define CVH_PIPELINE_DETAIL_FUSION_RULES_HPP

#include "ir.hpp"
#include "../../core/detail/cpu_features.hpp"

namespace cvh {
namespace detail {

inline bool isModelInputGeometry(const PipelinePlannedStage& stage)
{
    const int interpolation =
        stage.operation.kind == PipelineOperationKind::Resize
        ? stage.operation.resize.interpolation
        : stage.operation.kind == PipelineOperationKind::Letterbox
            ? stage.operation.letterbox.interpolation
            : -1;
    return interpolation == INTER_NEAREST || interpolation == INTER_LINEAR;
}

inline bool isPackedF32ModelInputFusion(
    const PipelinePlanImpl& plan)
{
    const bool has_color = plan.stages.size() == 4;
    if ((!has_color && plan.stages.size() != 3) ||
        plan.input.kind != PipelineDataKind::Image ||
        plan.input.image.data_type != PipelineDataType::U8 ||
        (plan.input.image.pixel_format != PixelFormat::BGR8 &&
         plan.input.image.pixel_format != PixelFormat::RGB8) ||
        plan.input.image.plane_count != 1)
    {
        return false;
    }

    std::size_t index = 0;
    if (has_color)
    {
        const PipelinePlannedStage& color = plan.stages[index++];
        if (color.operation.kind != PipelineOperationKind::Color ||
            (color.operation.color.target != Color::BGR &&
             color.operation.color.target != Color::RGB))
        {
            return false;
        }
    }

    const PipelinePlannedStage& geometry = plan.stages[index++];
    const PipelinePlannedStage& normalize = plan.stages[index++];
    const PipelinePlannedStage& layout = plan.stages[index];
    if (!isModelInputGeometry(geometry) ||
        normalize.operation.kind != PipelineOperationKind::Normalize ||
        layout.operation.kind != PipelineOperationKind::Layout ||
        normalize.output.kind != PipelineDataKind::Image ||
        normalize.output.image.data_type != PipelineDataType::F32 ||
        plan.output.kind != PipelineDataKind::Tensor ||
        plan.output.tensor.data_type != PipelineDataType::F32 ||
        (plan.output.tensor.layout != Layout::NCHW &&
         plan.output.tensor.layout != Layout::NHWC))
    {
        return false;
    }
    return pipelineColorChannels(geometry.input.image.color) == 3 &&
           pipelineColorChannels(geometry.output.image.color) == 3;
}

inline bool isPackedF32NeonCandidate(const PipelinePlanImpl& plan)
{
#if CVH_DETAIL_HAVE_NEON_KERNEL
    if (!isPackedF32ModelInputFusion(plan))
    {
        return false;
    }
    std::size_t index = plan.stages.size() == 4 ? 1 : 0;
    const PipelinePlannedStage& geometry = plan.stages[index];
    const bool resize =
        geometry.operation.kind == PipelineOperationKind::Resize;
    const bool letterbox =
        geometry.operation.kind == PipelineOperationKind::Letterbox;
    const int interpolation = resize
        ? geometry.operation.resize.interpolation
        : letterbox
            ? geometry.operation.letterbox.interpolation
            : -1;
    const int content_width = letterbox
        ? geometry.transform.resized_width
        : geometry.output.image.width;
    const int content_height = letterbox
        ? geometry.transform.resized_height
        : geometry.output.image.height;
    return (resize || letterbox) &&
           interpolation == INTER_NEAREST &&
           plan.output.tensor.layout == Layout::NCHW &&
           static_cast<std::int64_t>(geometry.input.image.width) ==
               static_cast<std::int64_t>(content_width) * 2 &&
           content_width >= 8 &&
           static_cast<std::size_t>(content_width) *
                   static_cast<std::size_t>(content_height) >=
               256;
#else
    (void)plan;
    return false;
#endif
}

inline bool isYuvF32ModelInputFusion(const PipelinePlanImpl& plan)
{
    if (plan.stages.size() != 4 ||
        plan.input.kind != PipelineDataKind::Image ||
        plan.input.image.data_type != PipelineDataType::U8 ||
        plan.input.image.color != Color::YUV ||
        plan.input.image.plane_count != 2 ||
        (plan.input.image.pixel_format != PixelFormat::NV12 &&
         plan.input.image.pixel_format != PixelFormat::NV21))
    {
        return false;
    }
    const PipelinePlannedStage& color = plan.stages[0];
    const PipelinePlannedStage& geometry = plan.stages[1];
    const PipelinePlannedStage& normalize = plan.stages[2];
    const PipelinePlannedStage& layout = plan.stages[3];
    return color.operation.kind == PipelineOperationKind::Color &&
           (color.operation.color.target == Color::BGR ||
            color.operation.color.target == Color::RGB) &&
           isModelInputGeometry(geometry) &&
           normalize.operation.kind == PipelineOperationKind::Normalize &&
           layout.operation.kind == PipelineOperationKind::Layout &&
           normalize.output.kind == PipelineDataKind::Image &&
           normalize.output.image.data_type == PipelineDataType::F32 &&
           plan.output.kind == PipelineDataKind::Tensor &&
           plan.output.tensor.data_type == PipelineDataType::F32 &&
           (plan.output.tensor.layout == Layout::NCHW ||
            plan.output.tensor.layout == Layout::NHWC);
}

inline bool isPackedQuantizedModelInputFusion(
    const PipelinePlanImpl& plan)
{
    const bool has_color = plan.stages.size() == 5;
    if ((!has_color && plan.stages.size() != 4) ||
        plan.input.kind != PipelineDataKind::Image ||
        plan.input.image.data_type != PipelineDataType::U8 ||
        (plan.input.image.pixel_format != PixelFormat::BGR8 &&
         plan.input.image.pixel_format != PixelFormat::RGB8) ||
        plan.input.image.plane_count != 1)
    {
        return false;
    }
    std::size_t index = 0;
    if (has_color)
    {
        const PipelinePlannedStage& color = plan.stages[index++];
        if (color.operation.kind != PipelineOperationKind::Color ||
            (color.operation.color.target != Color::BGR &&
             color.operation.color.target != Color::RGB))
        {
            return false;
        }
    }
    const PipelinePlannedStage& geometry = plan.stages[index++];
    const PipelinePlannedStage& normalize = plan.stages[index++];
    const PipelinePlannedStage& quantize = plan.stages[index++];
    const PipelinePlannedStage& layout = plan.stages[index];
    return isModelInputGeometry(geometry) &&
           normalize.operation.kind == PipelineOperationKind::Normalize &&
           normalize.output.kind == PipelineDataKind::Image &&
           normalize.output.image.data_type == PipelineDataType::F32 &&
           quantize.operation.kind == PipelineOperationKind::Quantize &&
           (quantize.output.image.data_type == PipelineDataType::U8 ||
            quantize.output.image.data_type == PipelineDataType::S8) &&
           layout.operation.kind == PipelineOperationKind::Layout &&
           plan.output.kind == PipelineDataKind::Tensor &&
           plan.output.tensor.data_type == quantize.output.image.data_type &&
           (plan.output.tensor.layout == Layout::NCHW ||
            plan.output.tensor.layout == Layout::NHWC) &&
           pipelineColorChannels(geometry.input.image.color) == 3;
}

inline bool isYuvQuantizedModelInputFusion(
    const PipelinePlanImpl& plan)
{
    if (plan.stages.size() != 5 ||
        plan.input.kind != PipelineDataKind::Image ||
        plan.input.image.data_type != PipelineDataType::U8 ||
        plan.input.image.color != Color::YUV ||
        plan.input.image.plane_count != 2 ||
        (plan.input.image.pixel_format != PixelFormat::NV12 &&
         plan.input.image.pixel_format != PixelFormat::NV21))
    {
        return false;
    }
    const PipelinePlannedStage& color = plan.stages[0];
    const PipelinePlannedStage& geometry = plan.stages[1];
    const PipelinePlannedStage& normalize = plan.stages[2];
    const PipelinePlannedStage& quantize = plan.stages[3];
    const PipelinePlannedStage& layout = plan.stages[4];
    return color.operation.kind == PipelineOperationKind::Color &&
           (color.operation.color.target == Color::BGR ||
            color.operation.color.target == Color::RGB) &&
           isModelInputGeometry(geometry) &&
           normalize.operation.kind == PipelineOperationKind::Normalize &&
           normalize.output.image.data_type == PipelineDataType::F32 &&
           quantize.operation.kind == PipelineOperationKind::Quantize &&
           (quantize.output.image.data_type == PipelineDataType::U8 ||
            quantize.output.image.data_type == PipelineDataType::S8) &&
           layout.operation.kind == PipelineOperationKind::Layout &&
           plan.output.kind == PipelineDataKind::Tensor &&
           plan.output.tensor.data_type == quantize.output.image.data_type &&
           (plan.output.tensor.layout == Layout::NCHW ||
            plan.output.tensor.layout == Layout::NHWC);
}

}  // namespace detail
}  // namespace cvh

#endif  // CVH_PIPELINE_DETAIL_FUSION_RULES_HPP
