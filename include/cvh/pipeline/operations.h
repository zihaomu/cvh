#ifndef CVH_PIPELINE_OPERATIONS_H
#define CVH_PIPELINE_OPERATIONS_H

#include "types.h"
#include "../imgproc/detail/common.h"

#include <array>
#include <cstddef>

namespace cvh {

enum class Interpolation
{
    Nearest = INTER_NEAREST,
    Linear = INTER_LINEAR,
};

enum class PipelineOperationKind
{
    Color,
    Resize,
    Letterbox,
    Normalize,
    Quantize,
    Layout,
};

struct PipelineColorOperation
{
    Color target = Color::Unknown;
};

struct PipelineResizeOperation
{
    int width = 0;
    int height = 0;
    int interpolation = INTER_LINEAR;
};

struct PipelineLetterboxOperation
{
    int width = 0;
    int height = 0;
    int interpolation = INTER_LINEAR;
    std::array<float, 4> pad_value{};
    int pad_count = 0;
};

struct PipelineNormalizeOperation
{
    std::array<float, 4> mean{};
    std::array<float, 4> stddev{{1.0f, 1.0f, 1.0f, 1.0f}};
    int count = 0;
};

struct PipelineQuantizeOperation
{
    PipelineDataType target_type = PipelineDataType::Unknown;
    float scale = 0.0f;
    int zero_point = 0;
};

struct PipelineLayoutOperation
{
    Layout target = Layout::Unknown;
};

struct PipelineOperation
{
    PipelineOperationKind kind = PipelineOperationKind::Color;
    PipelineColorOperation color{};
    PipelineResizeOperation resize{};
    PipelineLetterboxOperation letterbox{};
    PipelineNormalizeOperation normalize{};
    PipelineQuantizeOperation quantize{};
    PipelineLayoutOperation layout{};
};

}  // namespace cvh

#endif  // CVH_PIPELINE_OPERATIONS_H
