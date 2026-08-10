#include "cvh/pipeline/pipeline.h"

int cvh_pipeline_pipeline_header_compile()
{
    const auto image =
        cvh::imageDesc(2, 2, cvh::PixelFormat::BGR8);
    return image.valid() ? 0 : 1;
}
