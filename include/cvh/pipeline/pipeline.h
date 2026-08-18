#ifndef CVH_PIPELINE_PIPELINE_H
#define CVH_PIPELINE_PIPELINE_H

#include "info.h"
#include "operations.h"
#include "types.h"
#include "views.h"
#include "detail/ir.hpp"
#include "detail/neon_model_input_fused.hpp"
#include "detail/planner.hpp"
#include "detail/scalar_model_input_fused.hpp"
#include "detail/scalar_quantized_model_input_fused.hpp"
#include "detail/scalar_stage_executor.hpp"
#include "detail/scalar_yuv_model_input_fused.hpp"
#include "detail/scalar_yuv_quantized_model_input_fused.hpp"

#include "../core/mat.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cvh {

class PipelinePlan;
class PipelineWorkspace;
class ModelInputRecipeBuilder;

class PipelineWorkspaceView
{
public:
    PipelineWorkspaceView() = default;

    void* data() const { return data_; }
    std::size_t size() const { return size_; }

    const void* detailPlanToken() const { return plan_token_; }
    int detailIntermediateCount() const { return intermediate_count_; }

private:
    void* data_ = nullptr;
    std::size_t size_ = 0;
    const void* plan_token_ = nullptr;
    Mat* intermediates_ = nullptr;
    int intermediate_count_ = 0;
    Mat* borrowed_input_ = nullptr;
    Mat* borrowed_output_ = nullptr;

    friend class PipelinePlan;
    friend class PipelineWorkspace;
};

namespace detail {

inline int matTypeForDescriptor(const PipelineDataDescriptor& descriptor)
{
    const int depth =
        descriptor.kind == PipelineDataKind::Image
            ? pipelineDepth(descriptor.image.data_type)
            : pipelineDepth(descriptor.tensor.data_type);
    if (depth < 0)
    {
        CV_Error(Error::StsUnsupportedFormat, "unsupported pipeline data type");
    }
    const int channels =
        descriptor.kind == PipelineDataKind::Image
            ? pipelineColorChannels(descriptor.image.color)
            : 1;
    return CV_MAKETYPE(depth, channels);
}

inline void createMatForDescriptor(Mat& mat,
                                   const PipelineDataDescriptor& descriptor)
{
    if (descriptor.kind == PipelineDataKind::Image)
    {
        const int sizes[2] = {
            descriptor.image.height,
            descriptor.image.width};
        mat.create(2, sizes, matTypeForDescriptor(descriptor));
        return;
    }
    if (descriptor.kind == PipelineDataKind::Tensor)
    {
        mat.create(
            descriptor.tensor.dims,
            descriptor.tensor.shape.data(),
            matTypeForDescriptor(descriptor));
        return;
    }
    CV_Error(Error::StsBadArg, "cannot create Mat for invalid descriptor");
}

inline bool matMatchesDescriptor(const Mat& mat,
                                 const PipelineDataDescriptor& descriptor)
{
    if (mat.empty())
    {
        return false;
    }
    if (descriptor.kind == PipelineDataKind::Image)
    {
        return mat.dims == 2 &&
               mat.size[0] == descriptor.image.height &&
               mat.size[1] == descriptor.image.width &&
               mat.depth() == pipelineDepth(descriptor.image.data_type) &&
               mat.channels() ==
                   pipelineColorChannels(descriptor.image.color);
    }
    if (descriptor.kind == PipelineDataKind::Tensor)
    {
        if (mat.dims != descriptor.tensor.dims ||
            mat.depth() != pipelineDepth(descriptor.tensor.data_type) ||
            mat.channels() != 1 ||
            !mat.isContinuous())
        {
            return false;
        }
        for (int i = 0; i < mat.dims; ++i)
        {
            if (mat.size[i] !=
                descriptor.tensor.shape[static_cast<std::size_t>(i)])
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

inline bool matTypeMatchesDescriptor(
    const Mat& mat,
    const PipelineDataDescriptor& descriptor)
{
    if (mat.empty())
    {
        return false;
    }
    if (descriptor.kind == PipelineDataKind::Image)
    {
        return mat.depth() == pipelineDepth(descriptor.image.data_type) &&
               mat.channels() ==
                   pipelineColorChannels(descriptor.image.color);
    }
    if (descriptor.kind == PipelineDataKind::Tensor)
    {
        return mat.depth() == pipelineDepth(descriptor.tensor.data_type) &&
               mat.channels() == 1;
    }
    return false;
}

inline std::size_t matSpanBytes(const Mat& mat)
{
    if (mat.empty())
    {
        return 0;
    }
    if (mat.dims == 2)
    {
        return static_cast<std::size_t>(mat.size[0] - 1) * mat.step(0) +
               static_cast<std::size_t>(mat.size[1]) * mat.elemSize();
    }
    return mat.total() * mat.elemSize();
}

inline bool rangesOverlap(const void* lhs,
                          std::size_t lhs_bytes,
                          const void* rhs,
                          std::size_t rhs_bytes)
{
    if (lhs == nullptr || rhs == nullptr ||
        lhs_bytes == 0 || rhs_bytes == 0)
    {
        return false;
    }
    const std::uintptr_t lhs_begin =
        reinterpret_cast<std::uintptr_t>(lhs);
    const std::uintptr_t rhs_begin =
        reinterpret_cast<std::uintptr_t>(rhs);
    if (lhs_begin <= rhs_begin)
    {
        return rhs_begin - lhs_begin < lhs_bytes;
    }
    return lhs_begin - rhs_begin < rhs_bytes;
}

inline PipelineStatus validateWorkspace(
    const PipelinePlanImpl& plan,
    PipelineWorkspaceView workspace)
{
    if (workspace.detailPlanToken() != &plan ||
        workspace.size() < plan.info.workspace_bytes ||
        workspace.detailIntermediateCount() !=
            plan.info.full_frame_intermediates)
    {
        return PipelineStatus::failure(
            PipelineStatusCode::WorkspaceMismatch,
            -1,
            "PipelineWorkspace was not created for this PipelinePlan");
    }
    return PipelineStatus();
}

inline bool sameColorSpec(const ColorSpec& lhs,
                          const ColorSpec& rhs)
{
    return lhs.matrix == rhs.matrix &&
           lhs.range == rhs.range &&
           lhs.chroma_location == rhs.chroma_location;
}

inline bool checkedBufferSpan(std::size_t row_stride,
                              std::size_t row_bytes,
                              int height,
                              std::size_t& span)
{
    if (height <= 0)
    {
        return false;
    }
    const std::size_t preceding_rows =
        static_cast<std::size_t>(height - 1);
    if (row_stride != 0 &&
        preceding_rows >
            (std::numeric_limits<std::size_t>::max() - row_bytes) /
                row_stride)
    {
        return false;
    }
    span = preceding_rows * row_stride + row_bytes;
    return true;
}

inline bool pointerAlignedForType(const void* data,
                                  PipelineDataType type)
{
    const std::size_t alignment = pipelineDataTypeSize(type);
    return alignment != 0 &&
           reinterpret_cast<std::uintptr_t>(data) % alignment == 0;
}

inline PipelineStatus validateViewRun(
    const PipelinePlanImpl& plan,
    ConstImageView input,
    TensorView output,
    PipelineWorkspaceView workspace)
{
    const int output_stage = static_cast<int>(plan.stages.size());
    if (plan.input.kind != PipelineDataKind::Image ||
        plan.output.kind != PipelineDataKind::Tensor)
    {
        return PipelineStatus::failure(
            PipelineStatusCode::InvalidDescriptor,
            -1,
            "borrowed view run requires an Image input and Tensor output plan");
    }
    if (!input.descriptor.valid() ||
        input.plane_count != input.descriptor.plane_count)
    {
        return PipelineStatus::failure(
            PipelineStatusCode::InvalidDescriptor,
            -1,
            "borrowed input image view descriptor is invalid");
    }
    const ImageDescriptor& expected_input = plan.input.image;
    if (input.descriptor.data_type != expected_input.data_type ||
        input.descriptor.color != expected_input.color ||
        input.descriptor.pixel_format != expected_input.pixel_format ||
        input.descriptor.plane_count != expected_input.plane_count ||
        !sameColorSpec(
            input.descriptor.color_spec, expected_input.color_spec))
    {
        return PipelineStatus::failure(
            PipelineStatusCode::TypeMismatch,
            -1,
            "borrowed input image storage type does not match the prepared descriptor");
    }
    if (input.descriptor.width != expected_input.width ||
        input.descriptor.height != expected_input.height)
    {
        return PipelineStatus::failure(
            PipelineStatusCode::ShapeMismatch,
            -1,
            "borrowed input image shape does not match the prepared descriptor");
    }
    const std::size_t element_size =
        pipelineDataTypeSize(expected_input.data_type);
    std::array<std::size_t, 3> input_spans{};
    if (input.plane_count == 1)
    {
        const ConstPlaneView& input_plane = input.planes[0];
        const std::size_t channels = static_cast<std::size_t>(
            pipelineColorChannels(expected_input.color));
        if (element_size == 0 || channels == 0 ||
            static_cast<std::size_t>(expected_input.width) >
                std::numeric_limits<std::size_t>::max() /
                    channels / element_size)
        {
            return PipelineStatus::failure(
                PipelineStatusCode::InvalidDescriptor,
                -1,
                "borrowed input image row size overflows");
        }
        const std::size_t row_bytes =
            static_cast<std::size_t>(expected_input.width) *
            channels * element_size;
        if (input_plane.data == nullptr ||
            input_plane.row_stride < row_bytes ||
            !checkedBufferSpan(
                input_plane.row_stride,
                row_bytes,
                expected_input.height,
                input_spans[0]) ||
            input_plane.size_bytes < input_spans[0])
        {
            return PipelineStatus::failure(
                PipelineStatusCode::BufferTooSmall,
                -1,
                "borrowed input image buffer or row stride is too small");
        }
        if (!pointerAlignedForType(
                input_plane.data, expected_input.data_type) ||
            input_plane.row_stride % element_size != 0)
        {
            return PipelineStatus::failure(
                PipelineStatusCode::InvalidDescriptor,
                -1,
                "borrowed input image data or row stride is misaligned");
        }
    }
    else if (input.plane_count == 2 &&
             expected_input.data_type == PipelineDataType::U8 &&
             (expected_input.pixel_format == PixelFormat::NV12 ||
              expected_input.pixel_format == PixelFormat::NV21))
    {
        const std::size_t row_bytes =
            static_cast<std::size_t>(expected_input.width);
        const std::array<int, 2> plane_heights{{
            expected_input.height,
            expected_input.height / 2}};
        for (int plane_index = 0; plane_index < 2; ++plane_index)
        {
            const ConstPlaneView& plane =
                input.planes[static_cast<std::size_t>(plane_index)];
            if (plane.data == nullptr ||
                plane.row_stride < row_bytes ||
                !checkedBufferSpan(
                    plane.row_stride,
                    row_bytes,
                    plane_heights[static_cast<std::size_t>(plane_index)],
                    input_spans[static_cast<std::size_t>(plane_index)]) ||
                plane.size_bytes <
                    input_spans[static_cast<std::size_t>(plane_index)])
            {
                return PipelineStatus::failure(
                    PipelineStatusCode::BufferTooSmall,
                    -1,
                    "borrowed NV12/NV21 plane buffer or row stride is too small");
            }
        }
        if (rangesOverlap(
                input.planes[0].data,
                input.planes[0].size_bytes,
                input.planes[1].data,
                input.planes[1].size_bytes))
        {
            return PipelineStatus::failure(
                PipelineStatusCode::AliasingNotSupported,
                -1,
                "borrowed NV12/NV21 Y and UV planes must not overlap");
        }
    }
    else
    {
        return PipelineStatus::failure(
            PipelineStatusCode::Unsupported,
            -1,
            "borrowed image execution supports one packed plane or two-plane NV12/NV21");
    }

    if (!output.descriptor.valid())
    {
        return PipelineStatus::failure(
            PipelineStatusCode::InvalidDescriptor,
            output_stage,
            "borrowed output tensor view descriptor is invalid");
    }
    const TensorDescriptor& expected_output = plan.output.tensor;
    if (output.descriptor.data_type != expected_output.data_type ||
        output.descriptor.layout != expected_output.layout)
    {
        return PipelineStatus::failure(
            PipelineStatusCode::TypeMismatch,
            output_stage,
            "borrowed output tensor type or layout does not match the prepared descriptor");
    }
    if (output.descriptor.dims != expected_output.dims)
    {
        return PipelineStatus::failure(
            PipelineStatusCode::ShapeMismatch,
            output_stage,
            "borrowed output tensor rank does not match the prepared descriptor");
    }
    for (int dim = 0; dim < expected_output.dims; ++dim)
    {
        if (output.descriptor.shape[static_cast<std::size_t>(dim)] !=
            expected_output.shape[static_cast<std::size_t>(dim)])
        {
            return PipelineStatus::failure(
                PipelineStatusCode::ShapeMismatch,
                output_stage,
                "borrowed output tensor shape does not match the prepared descriptor");
        }
    }
    const std::size_t output_bytes = descriptorBytes(plan.output);
    if (output.data == nullptr || output.size_bytes < output_bytes)
    {
        return PipelineStatus::failure(
            PipelineStatusCode::BufferTooSmall,
            output_stage,
            "borrowed output tensor buffer is too small");
    }
    if (!pointerAlignedForType(
            output.data, expected_output.data_type))
    {
        return PipelineStatus::failure(
            PipelineStatusCode::InvalidDescriptor,
            output_stage,
            "borrowed output tensor data is misaligned");
    }

    const PipelineStatus workspace_status =
        validateWorkspace(plan, workspace);
    if (!workspace_status)
    {
        return workspace_status;
    }
    for (int plane_index = 0;
         plane_index < input.plane_count;
         ++plane_index)
    {
        const ConstPlaneView& plane =
            input.planes[static_cast<std::size_t>(plane_index)];
        if (rangesOverlap(
                plane.data,
                plane.size_bytes,
                output.data,
                output.size_bytes) ||
            rangesOverlap(
                plane.data,
                plane.size_bytes,
                workspace.data(),
                workspace.size()))
        {
            return PipelineStatus::failure(
                PipelineStatusCode::AliasingNotSupported,
                -1,
                "borrowed pipeline input, output, and workspace must not overlap");
        }
    }
    if (rangesOverlap(
            output.data,
            output.size_bytes,
            workspace.data(),
            workspace.size()))
    {
        return PipelineStatus::failure(
            PipelineStatusCode::AliasingNotSupported,
            -1,
            "borrowed pipeline input, output, and workspace must not overlap");
    }
    return PipelineStatus();
}

inline PipelineStatus validateRun(
    const PipelinePlanImpl& plan,
    const Mat& input,
    const Mat& output,
    PipelineWorkspaceView workspace)
{
    if (input.empty())
    {
        return PipelineStatus::failure(
            PipelineStatusCode::ShapeMismatch,
            -1,
            "pipeline input Mat is empty");
    }
    if (!matTypeMatchesDescriptor(input, plan.input))
    {
        return PipelineStatus::failure(
            PipelineStatusCode::TypeMismatch,
            -1,
            "pipeline input Mat type does not match the prepared descriptor");
    }
    if (!matMatchesDescriptor(input, plan.input))
    {
        return PipelineStatus::failure(
            PipelineStatusCode::ShapeMismatch,
            -1,
            "pipeline input Mat does not match the prepared descriptor");
    }
    if (output.empty())
    {
        return PipelineStatus::failure(
            PipelineStatusCode::ShapeMismatch,
            static_cast<int>(plan.stages.size()),
            "pipeline output Mat is empty");
    }
    if (!matTypeMatchesDescriptor(output, plan.output))
    {
        return PipelineStatus::failure(
            PipelineStatusCode::TypeMismatch,
            static_cast<int>(plan.stages.size()),
            "pipeline output Mat type does not match the inferred descriptor");
    }
    if (!matMatchesDescriptor(output, plan.output))
    {
        return PipelineStatus::failure(
            PipelineStatusCode::ShapeMismatch,
            static_cast<int>(plan.stages.size()),
            "pipeline output Mat does not match the inferred descriptor");
    }
    const PipelineStatus workspace_status =
        validateWorkspace(plan, workspace);
    if (!workspace_status)
    {
        return workspace_status;
    }

    const std::size_t input_bytes = matSpanBytes(input);
    const std::size_t output_bytes = matSpanBytes(output);
    if (rangesOverlap(input.data, input_bytes, output.data, output_bytes))
    {
        return PipelineStatus::failure(
            PipelineStatusCode::AliasingNotSupported,
            -1,
            "pipeline does not support overlapping input and output");
    }
    if (rangesOverlap(
            input.data, input_bytes, workspace.data(), workspace.size()) ||
        rangesOverlap(
            output.data, output_bytes, workspace.data(), workspace.size()))
    {
        return PipelineStatus::failure(
            PipelineStatusCode::AliasingNotSupported,
            -1,
            "pipeline workspace overlaps input or output");
    }
    return PipelineStatus();
}


inline int prepareErrorStage(const char* message)
{
    const char* marker = std::strstr(message, "pipeline stage ");
    if (marker == nullptr)
    {
        return -1;
    }
    marker += std::strlen("pipeline stage ");
    int stage = 0;
    bool has_digit = false;
    while (*marker >= '0' && *marker <= '9')
    {
        has_digit = true;
        stage = stage * 10 + (*marker - '0');
        ++marker;
    }
    return has_digit ? stage : -1;
}

inline PipelineStatus prepareExceptionStatus(const Exception& exception)
{
    const char* message = exception.err.c_str();
    PipelineStatusCode code = PipelineStatusCode::InvalidOperation;
    if (std::strstr(message, "requirement not satisfied") != nullptr)
    {
        code = PipelineStatusCode::RequirementNotSatisfied;
    }
    else if (std::strstr(message, "descriptor is invalid") != nullptr ||
             std::strstr(message, "pipeline output mismatch") != nullptr)
    {
        code = PipelineStatusCode::InvalidDescriptor;
    }
    else if (exception.code == Error::StsNotImplemented ||
             exception.code == Error::StsUnsupportedFormat)
    {
        code = PipelineStatusCode::Unsupported;
    }
    return PipelineStatus::failure(
        code, prepareErrorStage(message), message);
}

}  // namespace detail

class PipelinePlan
{
public:
    PipelinePlan() = default;

    bool valid() const { return static_cast<bool>(impl_); }

    const PipelineInfo& info() const
    {
        if (!impl_)
        {
            CV_Error(Error::StsBadArg, "PipelinePlan is empty");
        }
        return impl_->info;
    }

    const PipelineDataDescriptor& inputDescriptor() const
    {
        if (!impl_)
        {
            CV_Error(Error::StsBadArg, "PipelinePlan is empty");
        }
        return impl_->input;
    }

    const PipelineDataDescriptor& outputDescriptor() const
    {
        if (!impl_)
        {
            CV_Error(Error::StsBadArg, "PipelinePlan is empty");
        }
        return impl_->output;
    }

    bool hasTransform() const
    {
        return impl_ && impl_->transform.valid;
    }

    const PipelineTransform& transform() const
    {
        if (!impl_ || !impl_->transform.valid)
        {
            CV_Error(
                Error::StsBadArg,
                "PipelinePlan has no letterbox transform");
        }
        return impl_->transform;
    }

    std::string explain() const
    {
        if (!impl_)
        {
            return "invalid pipeline plan\n";
        }

        std::ostringstream out;
        if (impl_->info.recipe_id != nullptr)
        {
            out << "recipe: " << impl_->info.recipe_id
                << " v" << impl_->info.recipe_contract_version
                << ", fingerprint "
                << impl_->info.recipe_fingerprint << "\n\n";
        }
        out << "semantic stages: " << impl_->info.semantic_stage_count
            << "\n";
        for (std::size_t index = 0;
             index < impl_->stages.size();
             ++index)
        {
            const detail::PipelinePlannedStage& stage =
                impl_->stages[index];
            out << "  [" << index << "] "
                << detail::operationName(stage.operation.kind)
                << " " << detail::descriptorString(stage.input)
                << " -> " << detail::descriptorString(stage.output)
                << "\n";
        }

        out << "\nexecution groups: "
            << impl_->info.execution_group_count << "\n";
        for (std::size_t index = 0;
             index < impl_->execution_groups.size();
             ++index)
        {
            const detail::PipelineExecutionGroup& group =
                impl_->execution_groups[index];
            if (group.kind ==
                detail::PipelineExecutionGroupKind::Copy)
            {
                out << "  [" << index << "] scalar direct copy\n";
                continue;
            }
            if (group.kind ==
                detail::PipelineExecutionGroupKind::ModelInputFused)
            {
                out << "  [" << index << "] "
                    << detail::pipelineRouteName(group.candidate_route)
                    << " fused stages "
                    << group.semantic_begin << ".."
                    << group.semantic_end - 1
                    << ": model-input packed-f32\n";
                continue;
            }
            if (group.kind ==
                detail::PipelineExecutionGroupKind::YuvModelInputFused)
            {
                out << "  [" << index << "] scalar fused stages "
                    << group.semantic_begin << ".."
                    << group.semantic_end - 1
                    << ": yuv420-model-input\n";
                continue;
            }
            if (group.kind == detail::PipelineExecutionGroupKind::
                                  QuantizedModelInputFused)
            {
                out << "  [" << index << "] scalar fused stages "
                    << group.semantic_begin << ".."
                    << group.semantic_end - 1
                    << ": model-input quantized\n";
                continue;
            }
            if (group.kind == detail::PipelineExecutionGroupKind::
                                  YuvQuantizedModelInputFused)
            {
                out << "  [" << index << "] scalar fused stages "
                    << group.semantic_begin << ".."
                    << group.semantic_end - 1
                    << ": yuv420-model-input quantized\n";
                continue;
            }
            out << "  [" << index << "] scalar stages "
                << group.semantic_begin << ".." << group.semantic_end - 1
                << ": "
                << detail::operationName(
                       impl_->stages[group.semantic_begin].operation.kind)
                << "\n";
        }
        out << "\nfull-frame intermediates: "
            << impl_->info.full_frame_intermediates << "\n";
        out << "workspace: " << impl_->info.workspace_bytes
            << " bytes, alignment "
            << impl_->info.workspace_alignment << "\n";
        out << "candidate route: "
            << detail::pipelineRouteName(impl_->info.candidate_route)
            << "\n";
        return out.str();
    }

    PipelineStatus tryRun(const Mat& input,
                          Mat& output,
                          PipelineWorkspaceView workspace,
                          PipelineRunInfo* run_info = nullptr) const
    {
        if (!impl_)
        {
            return PipelineStatus::failure(
                PipelineStatusCode::InvalidDescriptor,
                -1,
                "PipelinePlan is empty");
        }

        if (impl_->input.kind == PipelineDataKind::Image &&
            impl_->input.image.plane_count != 1)
        {
            return PipelineStatus::failure(
                PipelineStatusCode::Unsupported,
                -1,
                "multi-plane pipeline input requires ConstImageView execution");
        }

        const PipelineStatus validation =
            detail::validateRun(*impl_, input, output, workspace);
        if (!validation)
        {
            return validation;
        }

        return executeValidated(input, output, workspace, run_info);
    }

    PipelineStatus tryRun(ConstImageView input,
                          TensorView output,
                          PipelineWorkspaceView workspace,
                          PipelineRunInfo* run_info = nullptr) const
    {
        if (!impl_)
        {
            return PipelineStatus::failure(
                PipelineStatusCode::InvalidDescriptor,
                -1,
                "PipelinePlan is empty");
        }

        const PipelineStatus validation =
            detail::validateViewRun(*impl_, input, output, workspace);
        if (!validation)
        {
            return validation;
        }
        if (workspace.borrowed_output_ == nullptr)
        {
            return PipelineStatus::failure(
                PipelineStatusCode::WorkspaceMismatch,
                -1,
                "PipelineWorkspace has no borrowed endpoint headers for this plan");
        }

        Mat& output_header = *workspace.borrowed_output_;
        output_header.data = static_cast<uchar*>(output.data);

        if (impl_->execution_groups.size() == 1 &&
            (impl_->execution_groups[0].kind ==
                 detail::PipelineExecutionGroupKind::YuvModelInputFused ||
             impl_->execution_groups[0].kind ==
                 detail::PipelineExecutionGroupKind::
                     YuvQuantizedModelInputFused))
        {
            cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
            const PipelineStatus status =
                impl_->execution_groups[0].kind ==
                        detail::PipelineExecutionGroupKind::
                            YuvModelInputFused
                ? detail::executeScalarYuvModelInputFused(
                      *impl_,
                      impl_->execution_groups[0],
                      input,
                      output_header)
                : detail::executeScalarYuvQuantizedModelInputFused(
                      *impl_,
                      impl_->execution_groups[0],
                      input,
                      output_header);
            if (status && run_info != nullptr)
            {
                run_info->actual_route = PipelineRoute::Scalar;
                run_info->observed_isa = PipelineRoute::Scalar;
                run_info->thread_count = 1;
                run_info->used_fallback = false;
                run_info->fallback_reason = nullptr;
            }
            return status;
        }

        if (workspace.borrowed_input_ == nullptr)
        {
            return PipelineStatus::failure(
                PipelineStatusCode::WorkspaceMismatch,
                -1,
                "PipelineWorkspace has no borrowed input header for this plan");
        }
        Mat& input_header = *workspace.borrowed_input_;
        input_header.data = const_cast<uchar*>(input.planes[0].data);
        input_header.stepBuf[0] = input.planes[0].row_stride;
        input_header.stepBuf[1] = input_header.elemSize();
        return executeValidated(
            input_header, output_header, workspace, run_info);
    }

    void run(ConstImageView input,
             TensorView output,
             PipelineWorkspaceView workspace,
             PipelineRunInfo* run_info = nullptr) const
    {
        const PipelineStatus status =
            tryRun(input, output, workspace, run_info);
        if (!status)
        {
            CV_Error_(
                Error::StsError,
                ("pipeline run failed at stage=%d: %s",
                 status.stage(),
                 status.message()));
        }
    }

    void run(const Mat& input,
             Mat& output,
             PipelineWorkspaceView workspace,
             PipelineRunInfo* run_info = nullptr) const
    {
        const PipelineStatus status =
            tryRun(input, output, workspace, run_info);
        if (!status)
        {
            CV_Error_(
                Error::StsError,
                ("pipeline run failed at stage=%d: %s",
                 status.stage(),
                 status.message()));
        }
    }

private:
    PipelineStatus executeValidated(
        const Mat& input,
        Mat& output,
        PipelineWorkspaceView workspace,
        PipelineRunInfo* run_info) const
    {
        PipelineRoute actual_route = PipelineRoute::Scalar;
        PipelineRoute observed_isa = PipelineRoute::Scalar;
        bool used_fallback = false;
        const char* fallback_reason = nullptr;
        cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);

        const Mat* current = &input;
        for (std::size_t group_index = 0;
             group_index < impl_->execution_groups.size();
             ++group_index)
        {
            const detail::PipelineExecutionGroup& group =
                impl_->execution_groups[group_index];
            Mat* target = nullptr;
            if (group_index + 1 == impl_->execution_groups.size())
            {
                target = &output;
            }
            else
            {
                target =
                    workspace.intermediates_ + group.workspace_slot;
            }

            PipelineStatus status;
            if (group.kind ==
                detail::PipelineExecutionGroupKind::Copy)
            {
                detail::copyImage(*current, *target);
            }
            else if (group.kind ==
                         detail::PipelineExecutionGroupKind::StagedStage &&
                     group.semantic_end == group.semantic_begin + 1 &&
                     group.semantic_end <= impl_->stages.size())
            {
                status = detail::executeStage(
                    impl_->stages[group.semantic_begin], *current, *target);
            }
            else if (group.kind ==
                     detail::PipelineExecutionGroupKind::ModelInputFused)
            {
                const cpu::DispatchMode mode = cpu::dispatch_mode();
                const bool use_neon =
                    group.candidate_route == PipelineRoute::Neon &&
                    detail::neonModelInputRuntimeAllowed();
                if (use_neon)
                {
                    status = detail::executeNeonModelInputFused(
                        *impl_, group, *current, *target);
                    actual_route = PipelineRoute::Neon;
                    observed_isa = PipelineRoute::Neon;
                    cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
                }
                else
                {
                    status = detail::executeScalarModelInputFused(
                        *impl_, group, *current, *target);
                    if (mode == cpu::DispatchMode::NeonOnly ||
                        (mode == cpu::DispatchMode::Auto &&
                         group.candidate_route == PipelineRoute::Neon))
                    {
                        used_fallback = true;
                        fallback_reason =
                            group.candidate_route == PipelineRoute::Neon
                            ? "NEON runtime capability is unavailable"
                            : "prepared plan has no NEON predicate";
                    }
                }
            }
            else if (group.kind ==
                     detail::PipelineExecutionGroupKind::YuvModelInputFused)
            {
                status = PipelineStatus::failure(
                    PipelineStatusCode::Unsupported,
                    static_cast<int>(group.semantic_begin),
                    "multi-plane YUV execution requires ConstImageView input");
            }
            else if (group.kind == detail::PipelineExecutionGroupKind::
                                       QuantizedModelInputFused)
            {
                status = detail::executeScalarQuantizedModelInputFused(
                    *impl_, group, *current, *target);
            }
            else if (group.kind == detail::PipelineExecutionGroupKind::
                                       YuvQuantizedModelInputFused)
            {
                status = PipelineStatus::failure(
                    PipelineStatusCode::Unsupported,
                    static_cast<int>(group.semantic_begin),
                    "multi-plane YUV execution requires ConstImageView input");
            }
            else
            {
                status = PipelineStatus::failure(
                    PipelineStatusCode::InternalError,
                    static_cast<int>(group.semantic_begin),
                    "pipeline execution group is invalid or unsupported");
            }
            if (!status)
            {
                return PipelineStatus::failure(
                    status.code(),
                    static_cast<int>(group.semantic_begin),
                    status.message());
            }
            current = target;
        }
        if (run_info != nullptr)
        {
            run_info->actual_route = actual_route;
            run_info->observed_isa = observed_isa;
            run_info->thread_count = 1;
            run_info->used_fallback = used_fallback;
            run_info->fallback_reason = fallback_reason;
        }
        return PipelineStatus();
    }

    explicit PipelinePlan(
        std::shared_ptr<const detail::PipelinePlanImpl> impl)
        : impl_(std::move(impl))
    {
    }

    std::shared_ptr<const detail::PipelinePlanImpl> impl_;

    friend class PipelineBuilder;
    friend class PipelineWorkspace;
};

class PipelineWorkspace
{
public:
    explicit PipelineWorkspace(const PipelinePlan& plan)
        : owner_(plan.impl_)
    {
        if (!owner_)
        {
            CV_Error(Error::StsBadArg, "PipelineWorkspace requires a valid plan");
        }

        if (owner_->input.kind == PipelineDataKind::Image &&
            owner_->input.image.plane_count == 1)
        {
            const int input_sizes[2] = {
                owner_->input.image.height,
                owner_->input.image.width};
            borrowed_input_header_ = Mat(
                2,
                input_sizes,
                detail::matTypeForDescriptor(owner_->input),
                borrowed_header_anchor_.data());
        }
        if (owner_->output.kind == PipelineDataKind::Tensor)
        {
            borrowed_output_header_ = Mat(
                owner_->output.tensor.dims,
                owner_->output.tensor.shape.data(),
                detail::matTypeForDescriptor(owner_->output),
                borrowed_header_anchor_.data());
        }

        const std::size_t bytes = owner_->info.workspace_bytes;
        const std::size_t alignment = owner_->info.workspace_alignment;
        if (bytes != 0)
        {
            storage_.resize(
                detail::checkedAdd(bytes, alignment - 1));
            const std::uintptr_t raw =
                reinterpret_cast<std::uintptr_t>(storage_.data());
            const std::uintptr_t aligned =
                (raw + alignment - 1) & ~(alignment - 1);
            aligned_data_ = reinterpret_cast<uchar*>(aligned);
        }

        intermediates_.reserve(
            static_cast<std::size_t>(
                owner_->info.full_frame_intermediates));
        for (const detail::PipelineExecutionGroup& group :
             owner_->execution_groups)
        {
            if (group.workspace_slot < 0)
            {
                continue;
            }
            uchar* data = aligned_data_ + group.workspace_offset;
            if (group.output.kind == PipelineDataKind::Image)
            {
                const int sizes[2] = {
                    group.output.image.height,
                    group.output.image.width};
                intermediates_.emplace_back(
                    2,
                    sizes,
                    detail::matTypeForDescriptor(group.output),
                    data);
            }
            else
            {
                intermediates_.emplace_back(
                    group.output.tensor.dims,
                    group.output.tensor.shape.data(),
                    detail::matTypeForDescriptor(group.output),
                    data);
            }
        }
    }

    PipelineWorkspace(const PipelineWorkspace&) = delete;
    PipelineWorkspace& operator=(const PipelineWorkspace&) = delete;
    PipelineWorkspace(PipelineWorkspace&&) = delete;
    PipelineWorkspace& operator=(PipelineWorkspace&&) = delete;

    PipelineWorkspaceView view()
    {
        PipelineWorkspaceView result;
        result.data_ = aligned_data_;
        result.size_ = owner_->info.workspace_bytes;
        result.plan_token_ = owner_.get();
        result.intermediates_ =
            intermediates_.empty() ? nullptr : intermediates_.data();
        result.intermediate_count_ =
            static_cast<int>(intermediates_.size());
        result.borrowed_input_ =
            owner_->input.kind == PipelineDataKind::Image &&
                    owner_->input.image.plane_count == 1
                ? &borrowed_input_header_
                : nullptr;
        result.borrowed_output_ =
            owner_->output.kind == PipelineDataKind::Tensor
                ? &borrowed_output_header_
                : nullptr;
        return result;
    }

    std::size_t size() const { return owner_->info.workspace_bytes; }

private:
    std::shared_ptr<const detail::PipelinePlanImpl> owner_;
    alignas(std::max_align_t)
        std::array<uchar, sizeof(std::max_align_t)>
            borrowed_header_anchor_{};
    Mat borrowed_input_header_;
    Mat borrowed_output_header_;
    std::vector<uchar> storage_;
    uchar* aligned_data_ = nullptr;
    std::vector<Mat> intermediates_;
};

class PipelineBuilder
{
public:
    PipelineBuilder(PipelineDataDescriptor input,
                    PipelineDataDescriptor output,
                    bool has_output_constraint)
        : input_(std::move(input)),
          output_(std::move(output)),
          has_output_constraint_(has_output_constraint)
    {
    }

    PipelineBuilder& color(Color target)
    {
        PipelineOperation operation;
        operation.kind = PipelineOperationKind::Color;
        operation.color.target = target;
        operations_.push_back(operation);
        return *this;
    }

    PipelineBuilder& resize(
        int width,
        int height,
        Interpolation interpolation = Interpolation::Linear)
    {
        return resize(width, height, static_cast<int>(interpolation));
    }

    PipelineBuilder& resize(int width,
                            int height,
                            int interpolation)
    {
        PipelineOperation operation;
        operation.kind = PipelineOperationKind::Resize;
        operation.resize.width = width;
        operation.resize.height = height;
        operation.resize.interpolation = interpolation;
        operations_.push_back(operation);
        return *this;
    }

    PipelineBuilder& letterbox(
        int width,
        int height,
        float pad_value = 114.0f,
        Interpolation interpolation = Interpolation::Linear)
    {
        PipelineOperation operation;
        operation.kind = PipelineOperationKind::Letterbox;
        operation.letterbox.width = width;
        operation.letterbox.height = height;
        operation.letterbox.interpolation =
            static_cast<int>(interpolation);
        operation.letterbox.pad_value[0] = pad_value;
        operation.letterbox.pad_count = 1;
        operations_.push_back(operation);
        return *this;
    }

    PipelineBuilder& letterbox(
        int width,
        int height,
        std::initializer_list<float> pad_value,
        Interpolation interpolation = Interpolation::Linear)
    {
        if (pad_value.size() == 0 || pad_value.size() > 4)
        {
            CV_Error(
                Error::StsBadArg,
                "letterbox pad value count must be in range 1..4");
        }
        PipelineOperation operation;
        operation.kind = PipelineOperationKind::Letterbox;
        operation.letterbox.width = width;
        operation.letterbox.height = height;
        operation.letterbox.interpolation =
            static_cast<int>(interpolation);
        operation.letterbox.pad_count =
            static_cast<int>(pad_value.size());
        std::copy(
            pad_value.begin(),
            pad_value.end(),
            operation.letterbox.pad_value.begin());
        operations_.push_back(operation);
        return *this;
    }

    PipelineBuilder& normalize(
        std::initializer_list<float> mean,
        std::initializer_list<float> stddev)
    {
        if (mean.size() == 0 ||
            mean.size() != stddev.size() ||
            mean.size() > 4)
        {
            CV_Error(
                Error::StsBadArg,
                "normalize expects equal mean/stddev counts in range 1..4");
        }
        return appendNormalize(
            mean.begin(), stddev.begin(), static_cast<int>(mean.size()));
    }

    PipelineBuilder& quantize(PipelineDataType target_type,
                              float scale,
                              int zero_point)
    {
        PipelineOperation operation;
        operation.kind = PipelineOperationKind::Quantize;
        operation.quantize.target_type = target_type;
        operation.quantize.scale = scale;
        operation.quantize.zero_point = zero_point;
        operations_.push_back(operation);
        return *this;
    }

    template <std::size_t Count>
    PipelineBuilder& normalize(
        const std::array<float, Count>& mean,
        const std::array<float, Count>& stddev)
    {
        static_assert(
            Count > 0 && Count <= 4,
            "normalize supports one to four channel parameters");
        return appendNormalize(
            mean.data(), stddev.data(), static_cast<int>(Count));
    }

    PipelineBuilder& normalize(const Scalar& mean,
                               const Scalar& stddev,
                               int count)
    {
        if (count <= 0 || count > 4)
        {
            CV_Error(
                Error::StsBadArg,
                "normalize Scalar count must be in range 1..4");
        }
        std::array<float, 4> mean_values{};
        std::array<float, 4> stddev_values{};
        for (int index = 0; index < count; ++index)
        {
            mean_values[static_cast<std::size_t>(index)] =
                static_cast<float>(mean[index]);
            stddev_values[static_cast<std::size_t>(index)] =
                static_cast<float>(stddev[index]);
        }
        return appendNormalize(
            mean_values.data(), stddev_values.data(), count);
    }

    PipelineBuilder& layout(Layout target)
    {
        PipelineOperation operation;
        operation.kind = PipelineOperationKind::Layout;
        operation.layout.target = target;
        operations_.push_back(operation);
        return *this;
    }

    PipelineBuilder& preferFusion()
    {
        return *this;
    }

    PipelineBuilder& requireNoFullFrameIntermediate()
    {
        require_no_full_frame_intermediate_ = true;
        return *this;
    }

    PipelineBuilder& requireSingleExecutionGroup()
    {
        require_single_execution_group_ = true;
        return *this;
    }

    PipelinePlan prepare() const
    {
        std::shared_ptr<detail::PipelinePlanImpl> plan =
            detail::buildPipelinePlan(
            input_,
            has_output_constraint_,
            output_,
            operations_,
            require_no_full_frame_intermediate_,
            require_single_execution_group_);
        plan->info.recipe_id = recipe_id_;
        plan->info.recipe_contract_version = recipe_contract_version_;
        plan->info.recipe_fingerprint = recipe_fingerprint_;
        return PipelinePlan(std::move(plan));
    }

    PipelineStatus tryPrepare(PipelinePlan& output) const
    {
        output = PipelinePlan();
        try
        {
            output = prepare();
            return PipelineStatus();
        }
        catch (const Exception& exception)
        {
            return detail::prepareExceptionStatus(exception);
        }
        catch (const std::exception& exception)
        {
            return PipelineStatus::failure(
                PipelineStatusCode::InternalError,
                -1,
                exception.what());
        }
    }

    void run() const
    {
        if (bound_input_ == nullptr || bound_output_ == nullptr)
        {
            CV_Error(
                Error::StsBadArg,
                "one-shot PipelineBuilder::run requires pipe(Mat, Mat)");
        }
        const PipelinePlan plan = prepare();
        if (bound_input_ == bound_output_)
        {
            CV_Error(
                Error::StsBadArg,
                "one-shot pipeline does not support identical input and output Mat objects");
        }
        detail::createMatForDescriptor(
            *bound_output_, plan.outputDescriptor());
        PipelineWorkspace workspace(plan);
        plan.run(*bound_input_, *bound_output_, workspace.view());
    }

private:
    PipelineBuilder& appendNormalize(const float* mean,
                                     const float* stddev,
                                     int count)
    {
        PipelineOperation operation;
        operation.kind = PipelineOperationKind::Normalize;
        operation.normalize.count = count;
        for (int index = 0; index < count; ++index)
        {
            operation.normalize.mean[static_cast<std::size_t>(index)] =
                mean[index];
            operation.normalize.stddev[static_cast<std::size_t>(index)] =
                stddev[index];
        }
        operations_.push_back(operation);
        return *this;
    }

    PipelineDataDescriptor input_{};
    PipelineDataDescriptor output_{};
    bool has_output_constraint_ = false;
    std::vector<PipelineOperation> operations_;
    bool require_no_full_frame_intermediate_ = false;
    bool require_single_execution_group_ = false;
    const char* recipe_id_ = nullptr;
    std::uint32_t recipe_contract_version_ = 0;
    std::uint64_t recipe_fingerprint_ = 0;
    const Mat* bound_input_ = nullptr;
    Mat* bound_output_ = nullptr;

    friend PipelineBuilder pipe(const Mat&, Mat&);
    friend class ModelInputRecipeBuilder;
};

inline PipelineBuilder pipe(const Mat& input, Mat& output)
{
    PipelineBuilder builder(
        detail::descriptorOf(input), PipelineDataDescriptor(), false);
    builder.bound_input_ = &input;
    builder.bound_output_ = &output;
    return builder;
}

inline PipelineBuilder pipe(PipelineDataDescriptor input,
                            PipelineDataDescriptor output)
{
    return PipelineBuilder(
        std::move(input), std::move(output), true);
}

}  // namespace cvh

#endif  // CVH_PIPELINE_PIPELINE_H
