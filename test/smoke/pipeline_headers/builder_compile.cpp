#include "cvh/pipeline/builder.h"

int cvh_pipeline_builder_header_compile()
{
    const cvh::PipelinePlan plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::imageDesc(8, 8, cvh::PixelFormat::BGR8))
            .letterbox(8, 8, 114.0f, cvh::Interpolation::Nearest)
            .prepare();
    const cvh::PipelinePlan quantized =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::tensorDesc<uchar>({1, 3, 2, 2}, cvh::Layout::NCHW))
            .resize(2, 2)
            .normalize({128.0f}, {64.0f})
            .quantize(cvh::PipelineDataType::U8, 0.025f, 128)
            .layout(cvh::Layout::NCHW)
            .prepare();
    return cvh::PipelineStatus().ok() && plan.hasTransform() &&
                   quantized.info().execution_group_count == 1
        ? 0
        : 1;
}
