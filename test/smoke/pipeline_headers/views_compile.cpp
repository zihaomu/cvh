#include "cvh/pipeline/views.h"

int cvh_pipeline_views_header_compile()
{
    return cvh::TensorView().size_bytes == 0 ? 0 : 1;
}
