#include "cvh/pipeline/types.h"

int cvh_pipeline_types_header_compile()
{
    return cvh::pipelineColorChannels(cvh::Color::RGB) == 3 ? 0 : 1;
}
