#include "cvh/pipeline/info.h"

int cvh_pipeline_info_header_compile()
{
    return cvh::PipelineInfo().allocations_per_run;
}
