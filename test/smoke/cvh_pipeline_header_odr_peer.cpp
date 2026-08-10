#include "cvh/pipeline/pipeline.h"

#include <cstdint>
#include <string>

std::uint64_t cvh_pipeline_header_odr_peer()
{
    const cvh::PipelinePlan plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::imageDesc(2, 2, cvh::PixelFormat::BGR8))
            .resize(2, 2)
            .prepare();
    return static_cast<std::uint64_t>(plan.info().workspace_alignment) +
           static_cast<std::uint64_t>(plan.explain().size());
}
