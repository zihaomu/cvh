#include "cvh/pipeline/plan.h"

int cvh_pipeline_plan_header_compile()
{
    return cvh::PipelinePlan().valid() ? 1 : 0;
}
