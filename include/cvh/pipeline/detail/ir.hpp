#ifndef CVH_PIPELINE_DETAIL_IR_HPP
#define CVH_PIPELINE_DETAIL_IR_HPP

#include "../info.h"
#include "../operations.h"
#include "../types.h"

#include <cstddef>
#include <vector>

namespace cvh {
namespace detail {

struct PipelinePlannedStage
{
    PipelineOperation operation{};
    PipelineDataDescriptor input{};
    PipelineDataDescriptor output{};
    std::vector<int> x0;
    std::vector<int> x1;
    std::vector<int> y0;
    std::vector<int> y1;
    std::vector<float> wx;
    std::vector<float> wy;
    PipelineTransform transform{};
};

enum class PipelineExecutionGroupKind
{
    Copy,
    StagedStage,
    ModelInputFused,
    YuvModelInputFused,
    QuantizedModelInputFused,
    YuvQuantizedModelInputFused,
};

struct PipelineExecutionGroup
{
    PipelineExecutionGroupKind kind =
        PipelineExecutionGroupKind::StagedStage;
    std::size_t semantic_begin = 0;
    std::size_t semantic_end = 0;
    PipelineDataDescriptor input{};
    PipelineDataDescriptor output{};
    PipelineExecutionClass execution_class =
        PipelineExecutionClass::Direct;
    PipelineRoute candidate_route = PipelineRoute::Scalar;
    int workspace_slot = -1;
    std::size_t workspace_offset = 0;
    std::size_t workspace_bytes = 0;
};

struct PipelinePlanImpl
{
    PipelineDataDescriptor input{};
    PipelineDataDescriptor output{};
    std::vector<PipelinePlannedStage> stages;
    std::vector<PipelineExecutionGroup> execution_groups;
    PipelineInfo info{};
    PipelineTransform transform{};
    bool require_no_full_frame_intermediate = false;
    bool require_single_execution_group = false;
};

}  // namespace detail
}  // namespace cvh

#endif  // CVH_PIPELINE_DETAIL_IR_HPP
