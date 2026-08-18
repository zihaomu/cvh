#include "cvh/pipeline/operations.h"

int cvh_pipeline_operations_header_compile()
{
    cvh::PipelineOperation operation;
    operation.kind = cvh::PipelineOperationKind::Letterbox;
    operation.letterbox.pad_count = 1;
    return operation.letterbox.pad_count == 1 ? 0 : 1;
}
