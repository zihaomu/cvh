#ifndef CVH_PIPELINE_INFO_H
#define CVH_PIPELINE_INFO_H

#include <array>
#include <cstddef>
#include <cstdio>

namespace cvh {

enum class PipelineExecutionClass
{
    Direct,
    Staged,
    FusedTiled,
};

enum class PipelineRoute
{
    Unknown,
    Scalar,
    UniversalIntrinsics,
    Neon,
    Avx2,
};

enum class PipelineStatusCode
{
    Ok,
    InvalidDescriptor,
    InvalidOperation,
    ShapeMismatch,
    TypeMismatch,
    BufferTooSmall,
    WorkspaceMismatch,
    AliasingNotSupported,
    RequirementNotSatisfied,
    Unsupported,
    InternalError,
};

class PipelineStatus
{
public:
    PipelineStatus() = default;

    static PipelineStatus failure(PipelineStatusCode code,
                                  int stage,
                                  const char* message)
    {
        PipelineStatus status;
        status.code_ = code;
        status.stage_ = stage;
        std::snprintf(
            status.message_.data(), status.message_.size(), "%s",
            message != nullptr ? message : "");
        return status;
    }

    bool ok() const { return code_ == PipelineStatusCode::Ok; }
    explicit operator bool() const { return ok(); }
    PipelineStatusCode code() const { return code_; }
    int stage() const { return stage_; }
    const char* message() const { return message_.data(); }

private:
    PipelineStatusCode code_ = PipelineStatusCode::Ok;
    int stage_ = -1;
    std::array<char, 256> message_{};
};

struct PipelineInfo
{
    int semantic_stage_count = 0;
    int execution_group_count = 0;
    int full_frame_intermediates = 0;
    int allocations_per_run = 0;
    std::size_t workspace_bytes = 0;
    std::size_t workspace_alignment = 1;
    PipelineExecutionClass execution_class = PipelineExecutionClass::Direct;
    PipelineRoute candidate_route = PipelineRoute::Scalar;
};

struct PipelineRunInfo
{
    PipelineRoute actual_route = PipelineRoute::Unknown;
    PipelineRoute observed_isa = PipelineRoute::Unknown;
    int thread_count = 1;
    bool used_fallback = false;
    const char* fallback_reason = nullptr;
};

}  // namespace cvh

#endif  // CVH_PIPELINE_INFO_H
