#include "cvh/pipeline/workspace.h"

int cvh_pipeline_workspace_header_compile()
{
    return cvh::PipelineWorkspaceView().size() == 0 ? 0 : 1;
}
