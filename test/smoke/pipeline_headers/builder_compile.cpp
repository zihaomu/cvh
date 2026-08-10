#include "cvh/pipeline/builder.h"

int cvh_pipeline_builder_header_compile()
{
    return cvh::PipelineStatus().ok() ? 0 : 1;
}
