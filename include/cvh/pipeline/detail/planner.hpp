#ifndef CVH_PIPELINE_DETAIL_PLANNER_HPP
#define CVH_PIPELINE_DETAIL_PLANNER_HPP

#include "fusion_rules.hpp"
#include "ir.hpp"
#include "../../core/mat.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace cvh {
namespace detail {

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
            "pipe(Mat, Mat) accepts only a two-dimensional image input");
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
            ("pipe input channels=%d are unsupported", mat.channels()));
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
    case PipelineOperationKind::Letterbox:
        return "letterbox";
    case PipelineOperationKind::Normalize:
        return "normalize";
    case PipelineOperationKind::Quantize:
        return "quantize";
    case PipelineOperationKind::Layout:
        return "layout";
    }
    return "unknown";
}

inline const char* pipelineRouteName(PipelineRoute route)
{
    switch (route)
    {
    case PipelineRoute::Scalar:
        return "scalar";
    case PipelineRoute::UniversalIntrinsics:
        return "universal_intrinsics";
    case PipelineRoute::Neon:
        return "neon";
    case PipelineRoute::Avx2:
        return "avx2";
    default:
        return "unknown";
    }
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
    const int target_width =
        stage.operation.kind == PipelineOperationKind::Letterbox
        ? stage.transform.resized_width
        : stage.output.image.width;
    const int target_height =
        stage.operation.kind == PipelineOperationKind::Letterbox
        ? stage.transform.resized_height
        : stage.output.image.height;
    const int interpolation =
        stage.operation.kind == PipelineOperationKind::Letterbox
        ? stage.operation.letterbox.interpolation
        : stage.operation.resize.interpolation;

    stage.x0.resize(static_cast<std::size_t>(target_width));
    stage.y0.resize(static_cast<std::size_t>(target_height));

    if (interpolation == INTER_NEAREST)
    {
        for (int x = 0; x < target_width; ++x)
        {
            stage.x0[static_cast<std::size_t>(x)] =
                std::min(
                    source.width - 1,
                    static_cast<int>(
                        (static_cast<std::int64_t>(x) * source.width) /
                        target_width));
        }
        for (int y = 0; y < target_height; ++y)
        {
            stage.y0[static_cast<std::size_t>(y)] =
                std::min(
                    source.height - 1,
                    static_cast<int>(
                        (static_cast<std::int64_t>(y) * source.height) /
                        target_height));
        }
        return;
    }

    stage.x1.resize(static_cast<std::size_t>(target_width));
    stage.y1.resize(static_cast<std::size_t>(target_height));
    stage.wx.resize(static_cast<std::size_t>(target_width));
    stage.wy.resize(static_cast<std::size_t>(target_height));

    const float scale_x =
        static_cast<float>(source.width) / static_cast<float>(target_width);
    const float scale_y =
        static_cast<float>(source.height) / static_cast<float>(target_height);

    for (int x = 0; x < target_width; ++x)
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

    for (int y = 0; y < target_height; ++y)
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

inline std::shared_ptr<PipelinePlanImpl> buildPipelinePlan(
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
            "pipeline accepts only Image input descriptors");
    }
    const bool packed_input =
        input.image.plane_count == 1 &&
        (input.image.color == Color::Gray ||
         input.image.color == Color::BGR ||
         input.image.color == Color::RGB) &&
        (input.image.data_type == PipelineDataType::U8 ||
         input.image.data_type == PipelineDataType::F32);
    const bool yuv420_input =
        input.image.plane_count == 2 &&
        input.image.color == Color::YUV &&
        input.image.data_type == PipelineDataType::U8 &&
        (input.image.pixel_format == PixelFormat::NV12 ||
         input.image.pixel_format == PixelFormat::NV21) &&
        input.image.width % 2 == 0 &&
        input.image.height % 2 == 0;
    if (!packed_input && !yuv420_input)
    {
        CV_Error(
            Error::StsNotImplemented,
            "pipeline accepts packed Gray/BGR/RGB U8/F32 or even-sized NV12/NV21 U8 input");
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
                    ("pipeline stage %zu \"color\": target is unsupported",
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
                    ("pipeline stage %zu \"resize\": interpolation=%d is unsupported",
                     index,
                     operation.resize.interpolation));
            }
            ImageDescriptor image = current.image;
            image.width = operation.resize.width;
            image.height = operation.resize.height;
            current = PipelineDataDescriptor(image);
            break;
        }
        case PipelineOperationKind::Letterbox:
        {
            if (current.kind != PipelineDataKind::Image)
            {
                CV_Error_(
                    Error::StsBadArg,
                    ("pipeline stage %zu \"letterbox\": expected Image, got %s",
                     index,
                     descriptorString(current).c_str()));
            }
            if (plan->transform.valid)
            {
                CV_Error(
                    Error::StsNotImplemented,
                    "pipeline supports at most one letterbox operation per plan");
            }
            if (operation.letterbox.width <= 0 ||
                operation.letterbox.height <= 0)
            {
                CV_Error_(
                    Error::StsBadSize,
                    ("pipeline stage %zu \"letterbox\": dimensions must be positive",
                     index));
            }
            if (operation.letterbox.interpolation != INTER_NEAREST &&
                operation.letterbox.interpolation != INTER_LINEAR)
            {
                CV_Error_(
                    Error::StsNotImplemented,
                    ("pipeline stage %zu \"letterbox\": interpolation=%d is unsupported",
                     index,
                     operation.letterbox.interpolation));
            }
            const int channels = pipelineColorChannels(current.image.color);
            if (operation.letterbox.pad_count != 1 &&
                operation.letterbox.pad_count != channels)
            {
                CV_Error_(
                    Error::StsBadArg,
                    ("pipeline stage %zu \"letterbox\": pad count=%d, channels=%d",
                     index,
                     operation.letterbox.pad_count,
                     channels));
            }
            for (int channel = 0;
                 channel < operation.letterbox.pad_count;
                 ++channel)
            {
                if (!std::isfinite(operation.letterbox.pad_value[
                        static_cast<std::size_t>(channel)]))
                {
                    CV_Error_(
                        Error::StsBadArg,
                        ("pipeline stage %zu \"letterbox\": pad value must be finite",
                         index));
                }
            }

            PipelineTransform transform;
            transform.valid = true;
            transform.source_width = current.image.width;
            transform.source_height = current.image.height;
            transform.target_width = operation.letterbox.width;
            transform.target_height = operation.letterbox.height;
            transform.scale = std::min(
                static_cast<float>(transform.target_width) /
                    static_cast<float>(transform.source_width),
                static_cast<float>(transform.target_height) /
                    static_cast<float>(transform.source_height));
            transform.resized_width = std::clamp(
                static_cast<int>(std::floor(
                    static_cast<float>(transform.source_width) *
                        transform.scale +
                    0.5f)),
                1,
                transform.target_width);
            transform.resized_height = std::clamp(
                static_cast<int>(std::floor(
                    static_cast<float>(transform.source_height) *
                        transform.scale +
                    0.5f)),
                1,
                transform.target_height);
            const int horizontal_padding =
                transform.target_width - transform.resized_width;
            const int vertical_padding =
                transform.target_height - transform.resized_height;
            transform.pad_left = horizontal_padding / 2;
            transform.pad_right =
                horizontal_padding - transform.pad_left;
            transform.pad_top = vertical_padding / 2;
            transform.pad_bottom =
                vertical_padding - transform.pad_top;
            transform.scale_x =
                static_cast<float>(transform.resized_width) /
                static_cast<float>(transform.source_width);
            transform.scale_y =
                static_cast<float>(transform.resized_height) /
                static_cast<float>(transform.source_height);
            stage.transform = transform;
            plan->transform = transform;

            ImageDescriptor image = current.image;
            image.width = operation.letterbox.width;
            image.height = operation.letterbox.height;
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
        case PipelineOperationKind::Quantize:
        {
            if (current.kind != PipelineDataKind::Image)
            {
                CV_Error_(
                    Error::StsBadArg,
                    ("pipeline stage %zu \"quantize\": expected Image, got %s",
                     index,
                     descriptorString(current).c_str()));
            }
            if (current.image.data_type != PipelineDataType::F32)
            {
                CV_Error_(
                    Error::StsUnsupportedFormat,
                    ("pipeline stage %zu \"quantize\": input must be F32",
                     index));
            }
            if (operation.quantize.target_type != PipelineDataType::U8 &&
                operation.quantize.target_type != PipelineDataType::S8)
            {
                CV_Error_(
                    Error::StsNotImplemented,
                    ("pipeline stage %zu \"quantize\": target must be U8 or S8",
                     index));
            }
            if (!std::isfinite(operation.quantize.scale) ||
                operation.quantize.scale <= 0.0f)
            {
                CV_Error_(
                    Error::StsBadArg,
                    ("pipeline stage %zu \"quantize\": scale must be finite and positive",
                     index));
            }
            const int minimum =
                operation.quantize.target_type == PipelineDataType::U8
                ? 0
                : -128;
            const int maximum =
                operation.quantize.target_type == PipelineDataType::U8
                ? 255
                : 127;
            if (operation.quantize.zero_point < minimum ||
                operation.quantize.zero_point > maximum)
            {
                CV_Error_(
                    Error::StsOutOfRange,
                    ("pipeline stage %zu \"quantize\": zero point is outside target range",
                     index));
            }
            ImageDescriptor image = current.image;
            image.data_type = operation.quantize.target_type;
            image.pixel_format = packedPixelFormat(
                image.data_type, image.color);
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
                    ("pipeline stage %zu \"layout\": supports NCHW/NHWC",
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
        if (operation.kind == PipelineOperationKind::Resize ||
            operation.kind == PipelineOperationKind::Letterbox)
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

    if (isPackedF32ModelInputFusion(*plan))
    {
        PipelineExecutionGroup group;
        group.kind = PipelineExecutionGroupKind::ModelInputFused;
        group.semantic_begin = 0;
        group.semantic_end = plan->stages.size();
        group.input = plan->input;
        group.output = plan->output;
        group.execution_class = PipelineExecutionClass::FusedTiled;
        group.candidate_route =
            isPackedF32NeonCandidate(*plan) &&
                cpu::neon_runtime_available()
            ? PipelineRoute::Neon
            : PipelineRoute::Scalar;
        plan->execution_groups.push_back(std::move(group));
    }
    else if (isYuvF32ModelInputFusion(*plan))
    {
        PipelineExecutionGroup group;
        group.kind = PipelineExecutionGroupKind::YuvModelInputFused;
        group.semantic_begin = 0;
        group.semantic_end = plan->stages.size();
        group.input = plan->input;
        group.output = plan->output;
        group.execution_class = PipelineExecutionClass::FusedTiled;
        group.candidate_route = PipelineRoute::Scalar;
        plan->execution_groups.push_back(std::move(group));
    }
    else if (isPackedQuantizedModelInputFusion(*plan))
    {
        PipelineExecutionGroup group;
        group.kind = PipelineExecutionGroupKind::QuantizedModelInputFused;
        group.semantic_begin = 0;
        group.semantic_end = plan->stages.size();
        group.input = plan->input;
        group.output = plan->output;
        group.execution_class = PipelineExecutionClass::FusedTiled;
        group.candidate_route = PipelineRoute::Scalar;
        plan->execution_groups.push_back(std::move(group));
    }
    else if (isYuvQuantizedModelInputFusion(*plan))
    {
        PipelineExecutionGroup group;
        group.kind =
            PipelineExecutionGroupKind::YuvQuantizedModelInputFused;
        group.semantic_begin = 0;
        group.semantic_end = plan->stages.size();
        group.input = plan->input;
        group.output = plan->output;
        group.execution_class = PipelineExecutionClass::FusedTiled;
        group.candidate_route = PipelineRoute::Scalar;
        plan->execution_groups.push_back(std::move(group));
    }
    else if (yuv420_input)
    {
        CV_Error(
            Error::StsNotImplemented,
            "NV12/NV21 input requires color -> resize/letterbox -> normalize -> layout fused chain");
    }
    else if (plan->stages.empty())
    {
        PipelineExecutionGroup group;
        group.kind = PipelineExecutionGroupKind::Copy;
        group.input = plan->input;
        group.output = plan->output;
        group.execution_class = PipelineExecutionClass::Direct;
        group.candidate_route = PipelineRoute::Scalar;
        plan->execution_groups.push_back(std::move(group));
    }
    else
    {
        plan->execution_groups.reserve(plan->stages.size());
        for (std::size_t index = 0; index < plan->stages.size(); ++index)
        {
            const PipelinePlannedStage& stage = plan->stages[index];
            PipelineExecutionGroup group;
            group.kind = PipelineExecutionGroupKind::StagedStage;
            group.semantic_begin = index;
            group.semantic_end = index + 1;
            group.input = stage.input;
            group.output = stage.output;
            group.execution_class = PipelineExecutionClass::Direct;
            group.candidate_route = PipelineRoute::Scalar;
            plan->execution_groups.push_back(std::move(group));
        }
    }

    constexpr std::size_t kWorkspaceAlignment = 64;
    std::size_t workspace_bytes = 0;
    int workspace_slot = 0;
    for (std::size_t index = 0;
         index + 1 < plan->execution_groups.size();
         ++index)
    {
        PipelineExecutionGroup& group = plan->execution_groups[index];
        workspace_bytes = alignUp(workspace_bytes, kWorkspaceAlignment);
        group.workspace_slot = workspace_slot++;
        group.workspace_offset = workspace_bytes;
        group.workspace_bytes = descriptorBytes(group.output);
        workspace_bytes = checkedAdd(workspace_bytes, group.workspace_bytes);
    }

    plan->info.semantic_stage_count =
        static_cast<int>(plan->stages.size());
    plan->info.execution_group_count =
        static_cast<int>(plan->execution_groups.size());
    plan->info.full_frame_intermediates = workspace_slot;
    plan->info.allocations_per_run = 0;
    plan->info.workspace_bytes = workspace_bytes;
    plan->info.workspace_alignment = kWorkspaceAlignment;
    plan->info.execution_class =
        plan->info.execution_group_count == 1
            ? plan->execution_groups.front().execution_class
            : PipelineExecutionClass::Staged;
    plan->info.candidate_route =
        plan->execution_groups.front().candidate_route;

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


}  // namespace detail
}  // namespace cvh

#endif  // CVH_PIPELINE_DETAIL_PLANNER_HPP
