#ifndef CVH_PIPELINE_PIPELINE_H
#define CVH_PIPELINE_PIPELINE_H

#include "info.h"
#include "operations.h"
#include "types.h"
#include "views.h"

#include "../core/mat.h"
#include "../core/saturate.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cvh {

class PipelinePlan;
class PipelineWorkspace;

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

    friend class PipelinePlan;
    friend class PipelineWorkspace;
};

namespace detail {

struct PipelinePlannedStage
{
    PipelineOperation operation{};
    PipelineDataDescriptor input{};
    PipelineDataDescriptor output{};
    int workspace_slot = -1;
    std::size_t workspace_offset = 0;
    std::size_t workspace_bytes = 0;
    std::vector<int> x0;
    std::vector<int> x1;
    std::vector<int> y0;
    std::vector<int> y1;
    std::vector<float> wx;
    std::vector<float> wy;
};

struct PipelinePlanImpl
{
    PipelineDataDescriptor input{};
    PipelineDataDescriptor output{};
    std::vector<PipelinePlannedStage> stages;
    PipelineInfo info{};
    bool require_no_full_frame_intermediate = false;
    bool require_single_execution_group = false;
};

inline int pipelineDepth(PipelineDataType type)
{
    switch (type)
    {
    case PipelineDataType::U8:
        return CV_8U;
    case PipelineDataType::S8:
        return CV_8S;
    case PipelineDataType::U16:
        return CV_16U;
    case PipelineDataType::S16:
        return CV_16S;
    case PipelineDataType::S32:
        return CV_32S;
    case PipelineDataType::F32:
        return CV_32F;
    case PipelineDataType::F64:
        return CV_64F;
    default:
        return -1;
    }
}

inline PipelineDataType pipelineDataTypeFromDepth(int depth)
{
    switch (depth)
    {
    case CV_8U:
        return PipelineDataType::U8;
    case CV_8S:
        return PipelineDataType::S8;
    case CV_16U:
        return PipelineDataType::U16;
    case CV_16S:
        return PipelineDataType::S16;
    case CV_32S:
        return PipelineDataType::S32;
    case CV_32F:
        return PipelineDataType::F32;
    case CV_64F:
        return PipelineDataType::F64;
    default:
        return PipelineDataType::Unknown;
    }
}

inline PixelFormat packedPixelFormat(PipelineDataType type, Color color)
{
    if (type != PipelineDataType::U8)
    {
        return PixelFormat::Unknown;
    }
    switch (color)
    {
    case Color::Gray:
        return PixelFormat::Gray8;
    case Color::RGB:
        return PixelFormat::RGB8;
    case Color::BGR:
        return PixelFormat::BGR8;
    case Color::RGBA:
        return PixelFormat::RGBA8;
    case Color::BGRA:
        return PixelFormat::BGRA8;
    default:
        return PixelFormat::Unknown;
    }
}

inline PipelineDataDescriptor descriptorOf(const Mat& mat)
{
    if (mat.empty())
    {
        CV_Error(Error::StsBadArg, "pipe input Mat must not be empty");
    }

    const PipelineDataType data_type =
        pipelineDataTypeFromDepth(mat.depth());
    if (data_type == PipelineDataType::Unknown)
    {
        CV_Error_(Error::StsUnsupportedFormat,
                  ("pipe input Mat depth=%d is unsupported", mat.depth()));
    }

    if (mat.dims != 2)
    {
        CV_Error(
            Error::StsNotImplemented,
            "P0 pipe(Mat, Mat) accepts only a two-dimensional image input");
    }

    ImageDescriptor image;
    image.width = mat.size[1];
    image.height = mat.size[0];
    image.data_type = data_type;
    image.plane_count = 1;

    switch (mat.channels())
    {
    case 1:
        image.color = Color::Gray;
        break;
    case 3:
        image.color = Color::BGR;
        break;
    case 4:
        image.color = Color::BGRA;
        break;
    default:
        CV_Error_(
            Error::StsUnsupportedFormat,
            ("P0 pipe input channels=%d are unsupported", mat.channels()));
    }
    image.pixel_format = packedPixelFormat(data_type, image.color);
    return PipelineDataDescriptor(image);
}

inline bool sameImageDescriptor(const ImageDescriptor& lhs,
                                const ImageDescriptor& rhs)
{
    return lhs.width == rhs.width &&
           lhs.height == rhs.height &&
           lhs.data_type == rhs.data_type &&
           lhs.color == rhs.color &&
           lhs.plane_count == rhs.plane_count;
}

inline bool sameTensorDescriptor(const TensorDescriptor& lhs,
                                 const TensorDescriptor& rhs)
{
    if (lhs.data_type != rhs.data_type ||
        lhs.layout != rhs.layout ||
        lhs.dims != rhs.dims)
    {
        return false;
    }
    for (int i = 0; i < lhs.dims; ++i)
    {
        if (lhs.shape[static_cast<std::size_t>(i)] !=
            rhs.shape[static_cast<std::size_t>(i)])
        {
            return false;
        }
    }
    return true;
}

inline bool sameDescriptor(const PipelineDataDescriptor& lhs,
                           const PipelineDataDescriptor& rhs)
{
    if (lhs.kind != rhs.kind)
    {
        return false;
    }
    if (lhs.kind == PipelineDataKind::Image)
    {
        return sameImageDescriptor(lhs.image, rhs.image);
    }
    if (lhs.kind == PipelineDataKind::Tensor)
    {
        return sameTensorDescriptor(lhs.tensor, rhs.tensor);
    }
    return false;
}

inline std::size_t checkedMultiply(std::size_t lhs, std::size_t rhs)
{
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs)
    {
        CV_Error(Error::StsOutOfMem, "pipeline descriptor byte size overflow");
    }
    return lhs * rhs;
}

inline std::size_t checkedAdd(std::size_t lhs, std::size_t rhs)
{
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs)
    {
        CV_Error(Error::StsOutOfMem, "pipeline descriptor byte size overflow");
    }
    return lhs + rhs;
}

inline std::size_t descriptorBytes(const PipelineDataDescriptor& descriptor)
{
    if (descriptor.kind == PipelineDataKind::Image)
    {
        const ImageDescriptor& image = descriptor.image;
        std::size_t total = static_cast<std::size_t>(image.width);
        total = checkedMultiply(total, static_cast<std::size_t>(image.height));
        total = checkedMultiply(
            total, static_cast<std::size_t>(pipelineColorChannels(image.color)));
        return checkedMultiply(total, pipelineDataTypeSize(image.data_type));
    }

    if (descriptor.kind == PipelineDataKind::Tensor)
    {
        const TensorDescriptor& tensor = descriptor.tensor;
        std::size_t total = 1;
        for (int i = 0; i < tensor.dims; ++i)
        {
            total = checkedMultiply(
                total,
                static_cast<std::size_t>(
                    tensor.shape[static_cast<std::size_t>(i)]));
        }
        return checkedMultiply(total, pipelineDataTypeSize(tensor.data_type));
    }
    return 0;
}

inline std::size_t alignUp(std::size_t value, std::size_t alignment)
{
    return checkedAdd(value, alignment - 1) & ~(alignment - 1);
}

inline const char* operationName(PipelineOperationKind kind)
{
    switch (kind)
    {
    case PipelineOperationKind::Color:
        return "color";
    case PipelineOperationKind::Resize:
        return "resize";
    case PipelineOperationKind::Normalize:
        return "normalize";
    case PipelineOperationKind::Layout:
        return "layout";
    }
    return "unknown";
}

inline const char* colorName(Color color)
{
    switch (color)
    {
    case Color::Gray:
        return "Gray";
    case Color::RGB:
        return "RGB";
    case Color::BGR:
        return "BGR";
    case Color::RGBA:
        return "RGBA";
    case Color::BGRA:
        return "BGRA";
    case Color::YUV:
        return "YUV";
    default:
        return "Unknown";
    }
}

inline const char* dataTypeName(PipelineDataType type)
{
    switch (type)
    {
    case PipelineDataType::U8:
        return "U8";
    case PipelineDataType::S8:
        return "S8";
    case PipelineDataType::U16:
        return "U16";
    case PipelineDataType::S16:
        return "S16";
    case PipelineDataType::S32:
        return "S32";
    case PipelineDataType::F32:
        return "F32";
    case PipelineDataType::F64:
        return "F64";
    default:
        return "Unknown";
    }
}

inline const char* layoutName(Layout layout)
{
    switch (layout)
    {
    case Layout::HWC:
        return "HWC";
    case Layout::CHW:
        return "CHW";
    case Layout::NHWC:
        return "NHWC";
    case Layout::NCHW:
        return "NCHW";
    default:
        return "Unknown";
    }
}

inline std::string descriptorString(const PipelineDataDescriptor& descriptor)
{
    std::ostringstream out;
    if (descriptor.kind == PipelineDataKind::Image)
    {
        out << "Image<" << dataTypeName(descriptor.image.data_type)
            << "," << colorName(descriptor.image.color) << ","
            << descriptor.image.width << "x" << descriptor.image.height
            << ">";
    }
    else if (descriptor.kind == PipelineDataKind::Tensor)
    {
        out << "Tensor<" << dataTypeName(descriptor.tensor.data_type)
            << "," << layoutName(descriptor.tensor.layout) << ",[";
        for (int i = 0; i < descriptor.tensor.dims; ++i)
        {
            if (i != 0)
            {
                out << ",";
            }
            out << descriptor.tensor.shape[static_cast<std::size_t>(i)];
        }
        out << "]>";
    }
    else
    {
        out << "Invalid";
    }
    return out.str();
}

inline void prepareResizeTables(PipelinePlannedStage& stage)
{
    const ImageDescriptor& source = stage.input.image;
    const ImageDescriptor& target = stage.output.image;
    const int interpolation = stage.operation.resize.interpolation;

    stage.x0.resize(static_cast<std::size_t>(target.width));
    stage.y0.resize(static_cast<std::size_t>(target.height));

    if (interpolation == INTER_NEAREST)
    {
        for (int x = 0; x < target.width; ++x)
        {
            stage.x0[static_cast<std::size_t>(x)] =
                std::min(
                    source.width - 1,
                    static_cast<int>(
                        (static_cast<std::int64_t>(x) * source.width) /
                        target.width));
        }
        for (int y = 0; y < target.height; ++y)
        {
            stage.y0[static_cast<std::size_t>(y)] =
                std::min(
                    source.height - 1,
                    static_cast<int>(
                        (static_cast<std::int64_t>(y) * source.height) /
                        target.height));
        }
        return;
    }

    stage.x1.resize(static_cast<std::size_t>(target.width));
    stage.y1.resize(static_cast<std::size_t>(target.height));
    stage.wx.resize(static_cast<std::size_t>(target.width));
    stage.wy.resize(static_cast<std::size_t>(target.height));

    const float scale_x =
        static_cast<float>(source.width) / static_cast<float>(target.width);
    const float scale_y =
        static_cast<float>(source.height) / static_cast<float>(target.height);

    for (int x = 0; x < target.width; ++x)
    {
        const float source_x =
            (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
        const int x0 =
            std::clamp(static_cast<int>(std::floor(source_x)),
                       0,
                       source.width - 1);
        stage.x0[static_cast<std::size_t>(x)] = x0;
        stage.x1[static_cast<std::size_t>(x)] =
            std::min(x0 + 1, source.width - 1);
        stage.wx[static_cast<std::size_t>(x)] =
            source_x - static_cast<float>(x0);
    }

    for (int y = 0; y < target.height; ++y)
    {
        const float source_y =
            (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 =
            std::clamp(static_cast<int>(std::floor(source_y)),
                       0,
                       source.height - 1);
        stage.y0[static_cast<std::size_t>(y)] = y0;
        stage.y1[static_cast<std::size_t>(y)] =
            std::min(y0 + 1, source.height - 1);
        stage.wy[static_cast<std::size_t>(y)] =
            source_y - static_cast<float>(y0);
    }
}

inline std::shared_ptr<const PipelinePlanImpl> buildPipelinePlan(
    const PipelineDataDescriptor& input,
    bool has_output_constraint,
    const PipelineDataDescriptor& output_constraint,
    const std::vector<PipelineOperation>& operations,
    bool require_no_full_frame_intermediate,
    bool require_single_execution_group)
{
    if (!input.valid())
    {
        CV_Error(Error::StsBadArg, "pipeline input descriptor is invalid");
    }
    if (input.kind != PipelineDataKind::Image)
    {
        CV_Error(
            Error::StsNotImplemented,
            "P0 pipeline accepts only Image input descriptors");
    }
    if (input.image.plane_count != 1 ||
        (input.image.color != Color::Gray &&
         input.image.color != Color::BGR &&
         input.image.color != Color::RGB) ||
        (input.image.data_type != PipelineDataType::U8 &&
         input.image.data_type != PipelineDataType::F32))
    {
        CV_Error(
            Error::StsNotImplemented,
            "P0 pipeline accepts packed Gray/BGR/RGB U8/F32 input");
    }

    std::shared_ptr<PipelinePlanImpl> plan =
        std::make_shared<PipelinePlanImpl>();
    plan->input = input;
    plan->require_no_full_frame_intermediate =
        require_no_full_frame_intermediate;
    plan->require_single_execution_group =
        require_single_execution_group;
    plan->stages.reserve(operations.size());

    PipelineDataDescriptor current = input;
    for (std::size_t index = 0; index < operations.size(); ++index)
    {
        const PipelineOperation& operation = operations[index];
        PipelinePlannedStage stage;
        stage.operation = operation;
        stage.input = current;

        switch (operation.kind)
        {
        case PipelineOperationKind::Color:
        {
            if (current.kind != PipelineDataKind::Image)
            {
                CV_Error_(
                    Error::StsBadArg,
                    ("pipeline stage %zu \"color\": expected Image, got %s",
                     index,
                     descriptorString(current).c_str()));
            }
            const Color target = operation.color.target;
            if (target != Color::Gray &&
                target != Color::BGR &&
                target != Color::RGB)
            {
                CV_Error_(
                    Error::StsNotImplemented,
                    ("pipeline stage %zu \"color\": P0 target is unsupported",
                     index));
            }
            ImageDescriptor image = current.image;
            image.color = target;
            image.pixel_format =
                packedPixelFormat(image.data_type, image.color);
            image.plane_count = 1;
            current = PipelineDataDescriptor(image);
            break;
        }
        case PipelineOperationKind::Resize:
        {
            if (current.kind != PipelineDataKind::Image)
            {
                CV_Error_(
                    Error::StsBadArg,
                    ("pipeline stage %zu \"resize\": expected Image, got %s",
                     index,
                     descriptorString(current).c_str()));
            }
            if (operation.resize.width <= 0 ||
                operation.resize.height <= 0)
            {
                CV_Error_(
                    Error::StsBadSize,
                    ("pipeline stage %zu \"resize\": dimensions must be positive",
                     index));
            }
            if (operation.resize.interpolation != INTER_NEAREST &&
                operation.resize.interpolation != INTER_LINEAR)
            {
                CV_Error_(
                    Error::StsNotImplemented,
                    ("pipeline stage %zu \"resize\": P0 interpolation=%d is unsupported",
                     index,
                     operation.resize.interpolation));
            }
            ImageDescriptor image = current.image;
            image.width = operation.resize.width;
            image.height = operation.resize.height;
            current = PipelineDataDescriptor(image);
            break;
        }
        case PipelineOperationKind::Normalize:
        {
            if (current.kind != PipelineDataKind::Image)
            {
                CV_Error_(
                    Error::StsBadArg,
                    ("pipeline stage %zu \"normalize\": expected Image, got %s",
                     index,
                     descriptorString(current).c_str()));
            }
            const int channels = pipelineColorChannels(current.image.color);
            if (operation.normalize.count != 1 &&
                operation.normalize.count != channels)
            {
                CV_Error_(
                    Error::StsBadArg,
                    ("pipeline stage %zu \"normalize\": parameter count=%d, channels=%d",
                     index,
                     operation.normalize.count,
                     channels));
            }
            for (int channel = 0;
                 channel < operation.normalize.count;
                 ++channel)
            {
                if (operation.normalize.stddev[
                        static_cast<std::size_t>(channel)] == 0.0f)
                {
                    CV_Error_(
                        Error::StsDivByZero,
                        ("pipeline stage %zu \"normalize\": stddev[%d] is zero",
                         index,
                         channel));
                }
            }
            ImageDescriptor image = current.image;
            image.data_type = PipelineDataType::F32;
            image.pixel_format = PixelFormat::Unknown;
            current = PipelineDataDescriptor(image);
            break;
        }
        case PipelineOperationKind::Layout:
        {
            if (current.kind != PipelineDataKind::Image)
            {
                CV_Error_(
                    Error::StsBadArg,
                    ("pipeline stage %zu \"layout\": expected Image, got %s",
                     index,
                     descriptorString(current).c_str()));
            }
            const Layout target = operation.layout.target;
            if (target != Layout::NCHW && target != Layout::NHWC)
            {
                CV_Error_(
                    Error::StsNotImplemented,
                    ("pipeline stage %zu \"layout\": P0 supports NCHW/NHWC",
                     index));
            }

            TensorDescriptor tensor;
            tensor.data_type = current.image.data_type;
            tensor.layout = target;
            tensor.dims = 4;
            const int channels = pipelineColorChannels(current.image.color);
            if (target == Layout::NCHW)
            {
                tensor.shape = {
                    1,
                    channels,
                    current.image.height,
                    current.image.width,
                    0,
                    0,
                    0,
                    0};
            }
            else
            {
                tensor.shape = {
                    1,
                    current.image.height,
                    current.image.width,
                    channels,
                    0,
                    0,
                    0,
                    0};
            }
            current = PipelineDataDescriptor(tensor);
            break;
        }
        }

        stage.output = current;
        if (operation.kind == PipelineOperationKind::Resize)
        {
            prepareResizeTables(stage);
        }
        plan->stages.push_back(std::move(stage));
    }

    if (has_output_constraint)
    {
        if (!output_constraint.valid())
        {
            CV_Error(Error::StsBadArg, "pipeline output descriptor is invalid");
        }
        if (!sameDescriptor(current, output_constraint))
        {
            CV_Error_(
                Error::StsUnmatchedFormats,
                ("pipeline output mismatch: inferred %s, requested %s",
                 descriptorString(current).c_str(),
                 descriptorString(output_constraint).c_str()));
        }
    }
    plan->output = current;

    constexpr std::size_t kWorkspaceAlignment = 64;
    std::size_t workspace_bytes = 0;
    int workspace_slot = 0;
    for (std::size_t index = 0; index + 1 < plan->stages.size(); ++index)
    {
        PipelinePlannedStage& stage = plan->stages[index];
        workspace_bytes = alignUp(workspace_bytes, kWorkspaceAlignment);
        stage.workspace_slot = workspace_slot++;
        stage.workspace_offset = workspace_bytes;
        stage.workspace_bytes = descriptorBytes(stage.output);
        workspace_bytes = checkedAdd(workspace_bytes, stage.workspace_bytes);
    }

    plan->info.semantic_stage_count =
        static_cast<int>(plan->stages.size());
    plan->info.execution_group_count =
        std::max(1, static_cast<int>(plan->stages.size()));
    plan->info.full_frame_intermediates = workspace_slot;
    plan->info.allocations_per_run = 0;
    plan->info.workspace_bytes = workspace_bytes;
    plan->info.workspace_alignment = kWorkspaceAlignment;
    plan->info.execution_class =
        plan->info.execution_group_count == 1
            ? PipelineExecutionClass::Direct
            : PipelineExecutionClass::Staged;
    plan->info.candidate_route = PipelineRoute::Scalar;

    if (require_no_full_frame_intermediate &&
        plan->info.full_frame_intermediates != 0)
    {
        const PipelinePlannedStage& blocker = plan->stages.front();
        CV_Error_(
            Error::StsNotImplemented,
            ("pipeline stage 0 \"%s\": requirement not satisfied; "
             "%d full-frame intermediates",
             operationName(blocker.operation.kind),
             plan->info.full_frame_intermediates));
    }
    if (require_single_execution_group &&
        plan->info.execution_group_count != 1)
    {
        const PipelinePlannedStage& blocker = plan->stages.front();
        CV_Error_(
            Error::StsNotImplemented,
            ("pipeline stage 0 \"%s\": requirement not satisfied; "
             "%d execution groups",
             operationName(blocker.operation.kind),
             plan->info.execution_group_count));
    }

    return plan;
}

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

    const std::size_t input_bytes = matSpanBytes(input);
    const std::size_t output_bytes = matSpanBytes(output);
    if (rangesOverlap(input.data, input_bytes, output.data, output_bytes))
    {
        return PipelineStatus::failure(
            PipelineStatusCode::AliasingNotSupported,
            -1,
            "P0 pipeline does not support overlapping input and output");
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

inline void copyImage(const Mat& source, Mat& target)
{
    const std::size_t row_bytes =
        static_cast<std::size_t>(source.size[1]) * source.elemSize();
    for (int y = 0; y < source.size[0]; ++y)
    {
        std::memcpy(
            target.data + static_cast<std::size_t>(y) * target.step(0),
            source.data + static_cast<std::size_t>(y) * source.step(0),
            row_bytes);
    }
}

template <typename T>
inline PipelineStatus executeColorTyped(const PipelinePlannedStage& stage,
                                        const Mat& source,
                                        Mat& target)
{
    const Color source_color = stage.input.image.color;
    const Color target_color = stage.output.image.color;
    const int rows = source.size[0];
    const int cols = source.size[1];

    if (source_color == target_color)
    {
        copyImage(source, target);
        return PipelineStatus();
    }

    for (int y = 0; y < rows; ++y)
    {
        const T* source_row = reinterpret_cast<const T*>(
            source.data + static_cast<std::size_t>(y) * source.step(0));
        T* target_row = reinterpret_cast<T*>(
            target.data + static_cast<std::size_t>(y) * target.step(0));

        if ((source_color == Color::BGR && target_color == Color::RGB) ||
            (source_color == Color::RGB && target_color == Color::BGR))
        {
            for (int x = 0; x < cols; ++x)
            {
                const int offset = x * 3;
                target_row[offset] = source_row[offset + 2];
                target_row[offset + 1] = source_row[offset + 1];
                target_row[offset + 2] = source_row[offset];
            }
            continue;
        }

        if (source_color == Color::Gray &&
            (target_color == Color::BGR || target_color == Color::RGB))
        {
            for (int x = 0; x < cols; ++x)
            {
                const int offset = x * 3;
                target_row[offset] = source_row[x];
                target_row[offset + 1] = source_row[x];
                target_row[offset + 2] = source_row[x];
            }
            continue;
        }

        if ((source_color == Color::BGR || source_color == Color::RGB) &&
            target_color == Color::Gray)
        {
            const int blue_index = source_color == Color::RGB ? 2 : 0;
            const int red_index = source_color == Color::RGB ? 0 : 2;
            for (int x = 0; x < cols; ++x)
            {
                const int offset = x * 3;
                if constexpr (std::is_same<T, uchar>::value)
                {
                    const int blue = source_row[offset + blue_index];
                    const int green = source_row[offset + 1];
                    const int red = source_row[offset + red_index];
                    target_row[x] = static_cast<uchar>(
                        (7471 * blue +
                         38470 * green +
                         19595 * red +
                         (1 << 15)) >>
                        16);
                }
                else
                {
                    target_row[x] = static_cast<T>(
                        0.114f * source_row[offset + blue_index] +
                        0.587f * source_row[offset + 1] +
                        0.299f * source_row[offset + red_index]);
                }
            }
            continue;
        }

        return PipelineStatus::failure(
            PipelineStatusCode::Unsupported,
            -1,
            "P0 scalar color conversion is unsupported");
    }
    return PipelineStatus();
}

inline PipelineStatus executeColor(const PipelinePlannedStage& stage,
                                   const Mat& source,
                                   Mat& target)
{
    if (stage.input.image.data_type == PipelineDataType::U8)
    {
        return executeColorTyped<uchar>(stage, source, target);
    }
    if (stage.input.image.data_type == PipelineDataType::F32)
    {
        return executeColorTyped<float>(stage, source, target);
    }
    return PipelineStatus::failure(
        PipelineStatusCode::Unsupported,
        -1,
        "P0 scalar color data type is unsupported");
}

template <typename T>
inline PipelineStatus executeResizeTyped(const PipelinePlannedStage& stage,
                                         const Mat& source,
                                         Mat& target)
{
    const int channels = source.channels();
    const int target_rows = target.size[0];
    const int target_cols = target.size[1];
    const bool nearest =
        stage.operation.resize.interpolation == INTER_NEAREST;

    for (int y = 0; y < target_rows; ++y)
    {
        const int source_y0 = stage.y0[static_cast<std::size_t>(y)];
        const T* source_row0 = reinterpret_cast<const T*>(
            source.data +
            static_cast<std::size_t>(source_y0) * source.step(0));
        T* target_row = reinterpret_cast<T*>(
            target.data + static_cast<std::size_t>(y) * target.step(0));

        if (nearest)
        {
            for (int x = 0; x < target_cols; ++x)
            {
                const int source_x =
                    stage.x0[static_cast<std::size_t>(x)];
                for (int channel = 0; channel < channels; ++channel)
                {
                    target_row[x * channels + channel] =
                        source_row0[source_x * channels + channel];
                }
            }
            continue;
        }

        const int source_y1 = stage.y1[static_cast<std::size_t>(y)];
        const float weight_y = stage.wy[static_cast<std::size_t>(y)];
        const T* source_row1 = reinterpret_cast<const T*>(
            source.data +
            static_cast<std::size_t>(source_y1) * source.step(0));

        for (int x = 0; x < target_cols; ++x)
        {
            const int source_x0 =
                stage.x0[static_cast<std::size_t>(x)];
            const int source_x1 =
                stage.x1[static_cast<std::size_t>(x)];
            const float weight_x =
                stage.wx[static_cast<std::size_t>(x)];

            for (int channel = 0; channel < channels; ++channel)
            {
                const int offset00 = source_x0 * channels + channel;
                const int offset01 = source_x1 * channels + channel;
                const float top =
                    static_cast<float>(source_row0[offset00]) +
                    (static_cast<float>(source_row0[offset01]) -
                     static_cast<float>(source_row0[offset00])) *
                        weight_x;
                const float bottom =
                    static_cast<float>(source_row1[offset00]) +
                    (static_cast<float>(source_row1[offset01]) -
                     static_cast<float>(source_row1[offset00])) *
                        weight_x;
                const float value = top + (bottom - top) * weight_y;
                if constexpr (std::is_same<T, uchar>::value)
                {
                    target_row[x * channels + channel] =
                        saturate_cast<uchar>(value);
                }
                else
                {
                    target_row[x * channels + channel] =
                        static_cast<T>(value);
                }
            }
        }
    }
    return PipelineStatus();
}

inline PipelineStatus executeResize(const PipelinePlannedStage& stage,
                                    const Mat& source,
                                    Mat& target)
{
    if (stage.input.image.data_type == PipelineDataType::U8)
    {
        return executeResizeTyped<uchar>(stage, source, target);
    }
    if (stage.input.image.data_type == PipelineDataType::F32)
    {
        return executeResizeTyped<float>(stage, source, target);
    }
    return PipelineStatus::failure(
        PipelineStatusCode::Unsupported,
        -1,
        "P0 scalar resize data type is unsupported");
}

template <typename T>
inline PipelineStatus executeNormalizeTyped(
    const PipelinePlannedStage& stage,
    const Mat& source,
    Mat& target)
{
    const int channels = source.channels();
    const PipelineNormalizeOperation& normalize =
        stage.operation.normalize;
    for (int y = 0; y < source.size[0]; ++y)
    {
        const T* source_row = reinterpret_cast<const T*>(
            source.data + static_cast<std::size_t>(y) * source.step(0));
        float* target_row = reinterpret_cast<float*>(
            target.data + static_cast<std::size_t>(y) * target.step(0));
        for (int x = 0; x < source.size[1]; ++x)
        {
            for (int channel = 0; channel < channels; ++channel)
            {
                const int parameter =
                    normalize.count == 1 ? 0 : channel;
                const int offset = x * channels + channel;
                target_row[offset] =
                    (static_cast<float>(source_row[offset]) -
                     normalize.mean[static_cast<std::size_t>(parameter)]) /
                    normalize.stddev[
                        static_cast<std::size_t>(parameter)];
            }
        }
    }
    return PipelineStatus();
}

inline PipelineStatus executeNormalize(
    const PipelinePlannedStage& stage,
    const Mat& source,
    Mat& target)
{
    if (stage.input.image.data_type == PipelineDataType::U8)
    {
        return executeNormalizeTyped<uchar>(stage, source, target);
    }
    if (stage.input.image.data_type == PipelineDataType::F32)
    {
        return executeNormalizeTyped<float>(stage, source, target);
    }
    return PipelineStatus::failure(
        PipelineStatusCode::Unsupported,
        -1,
        "P0 scalar normalize data type is unsupported");
}

template <typename T>
inline PipelineStatus executeLayoutTyped(const PipelinePlannedStage& stage,
                                         const Mat& source,
                                         Mat& target)
{
    const int rows = source.size[0];
    const int cols = source.size[1];
    const int channels = source.channels();
    T* target_data = reinterpret_cast<T*>(target.data);

    if (stage.operation.layout.target == Layout::NHWC)
    {
        const std::size_t row_bytes =
            static_cast<std::size_t>(cols * channels) * sizeof(T);
        for (int y = 0; y < rows; ++y)
        {
            std::memcpy(
                target_data +
                    static_cast<std::size_t>(y * cols * channels),
                source.data + static_cast<std::size_t>(y) * source.step(0),
                row_bytes);
        }
        return PipelineStatus();
    }

    for (int y = 0; y < rows; ++y)
    {
        const T* source_row = reinterpret_cast<const T*>(
            source.data + static_cast<std::size_t>(y) * source.step(0));
        for (int x = 0; x < cols; ++x)
        {
            for (int channel = 0; channel < channels; ++channel)
            {
                const std::size_t target_index =
                    (static_cast<std::size_t>(channel) *
                         static_cast<std::size_t>(rows) +
                     static_cast<std::size_t>(y)) *
                        static_cast<std::size_t>(cols) +
                    static_cast<std::size_t>(x);
                target_data[target_index] =
                    source_row[x * channels + channel];
            }
        }
    }
    return PipelineStatus();
}

inline PipelineStatus executeLayout(const PipelinePlannedStage& stage,
                                    const Mat& source,
                                    Mat& target)
{
    if (stage.input.image.data_type == PipelineDataType::U8)
    {
        return executeLayoutTyped<uchar>(stage, source, target);
    }
    if (stage.input.image.data_type == PipelineDataType::F32)
    {
        return executeLayoutTyped<float>(stage, source, target);
    }
    return PipelineStatus::failure(
        PipelineStatusCode::Unsupported,
        -1,
        "P0 scalar layout data type is unsupported");
}

inline PipelineStatus executeStage(const PipelinePlannedStage& stage,
                                   const Mat& source,
                                   Mat& target)
{
    switch (stage.operation.kind)
    {
    case PipelineOperationKind::Color:
        return executeColor(stage, source, target);
    case PipelineOperationKind::Resize:
        return executeResize(stage, source, target);
    case PipelineOperationKind::Normalize:
        return executeNormalize(stage, source, target);
    case PipelineOperationKind::Layout:
        return executeLayout(stage, source, target);
    }
    return PipelineStatus::failure(
        PipelineStatusCode::InternalError,
        -1,
        "unknown pipeline operation");
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

    std::string explain() const
    {
        if (!impl_)
        {
            return "invalid pipeline plan\n";
        }

        std::ostringstream out;
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
        if (impl_->stages.empty())
        {
            out << "  [0] scalar direct copy\n";
        }
        else
        {
            for (std::size_t index = 0;
                 index < impl_->stages.size();
                 ++index)
            {
                out << "  [" << index << "] scalar "
                    << detail::operationName(
                           impl_->stages[index].operation.kind)
                    << "\n";
            }
        }
        out << "\nfull-frame intermediates: "
            << impl_->info.full_frame_intermediates << "\n";
        out << "workspace: " << impl_->info.workspace_bytes
            << " bytes, alignment "
            << impl_->info.workspace_alignment << "\n";
        out << "candidate route: scalar\n";
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

        const PipelineStatus validation =
            detail::validateRun(*impl_, input, output, workspace);
        if (!validation)
        {
            return validation;
        }

        if (run_info != nullptr)
        {
            run_info->actual_route = PipelineRoute::Scalar;
            run_info->observed_isa = PipelineRoute::Scalar;
            run_info->thread_count = 1;
            run_info->used_fallback = false;
            run_info->fallback_reason = nullptr;
        }

        if (impl_->stages.empty())
        {
            detail::copyImage(input, output);
            return PipelineStatus();
        }

        const Mat* current = &input;
        for (std::size_t index = 0;
             index < impl_->stages.size();
             ++index)
        {
            const detail::PipelinePlannedStage& stage =
                impl_->stages[index];
            Mat* target = nullptr;
            if (index + 1 == impl_->stages.size())
            {
                target = &output;
            }
            else
            {
                target =
                    workspace.intermediates_ + stage.workspace_slot;
            }

            const PipelineStatus status =
                detail::executeStage(stage, *current, *target);
            if (!status)
            {
                return PipelineStatus::failure(
                    status.code(),
                    static_cast<int>(index),
                    status.message());
            }
            current = target;
        }
        return PipelineStatus();
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
        for (const detail::PipelinePlannedStage& stage : owner_->stages)
        {
            if (stage.workspace_slot < 0)
            {
                continue;
            }
            uchar* data = aligned_data_ + stage.workspace_offset;
            if (stage.output.kind == PipelineDataKind::Image)
            {
                const int sizes[2] = {
                    stage.output.image.height,
                    stage.output.image.width};
                intermediates_.emplace_back(
                    2,
                    sizes,
                    detail::matTypeForDescriptor(stage.output),
                    data);
            }
            else
            {
                intermediates_.emplace_back(
                    stage.output.tensor.dims,
                    stage.output.tensor.shape.data(),
                    detail::matTypeForDescriptor(stage.output),
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
        return result;
    }

    std::size_t size() const { return owner_->info.workspace_bytes; }

private:
    std::shared_ptr<const detail::PipelinePlanImpl> owner_;
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
        return PipelinePlan(detail::buildPipelinePlan(
            input_,
            has_output_constraint_,
            output_,
            operations_,
            require_no_full_frame_intermediate_,
            require_single_execution_group_));
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
                "P0 one-shot pipeline does not support identical input and output Mat objects");
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
    const Mat* bound_input_ = nullptr;
    Mat* bound_output_ = nullptr;

    friend PipelineBuilder pipe(const Mat&, Mat&);
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
