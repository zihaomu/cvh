#ifndef CVH_PIPELINE_DETAIL_QUANTIZE_HPP
#define CVH_PIPELINE_DETAIL_QUANTIZE_HPP

#include "../operations.h"
#include "../../core/define.h"

#include <cmath>

namespace cvh {
namespace detail {

inline int quantizeToInt(float real_value,
                         const PipelineQuantizeOperation& operation)
{
    const int minimum =
        operation.target_type == PipelineDataType::U8 ? 0 : -128;
    const int maximum =
        operation.target_type == PipelineDataType::U8 ? 255 : 127;
    if (std::isnan(real_value))
    {
        return operation.zero_point;
    }
    if (std::isinf(real_value))
    {
        return real_value > 0.0f ? maximum : minimum;
    }
    const double rounded = std::round(
        static_cast<double>(real_value) /
        static_cast<double>(operation.scale));
    const double shifted =
        rounded + static_cast<double>(operation.zero_point);
    if (shifted <= static_cast<double>(minimum))
    {
        return minimum;
    }
    if (shifted >= static_cast<double>(maximum))
    {
        return maximum;
    }
    return static_cast<int>(shifted);
}

template <typename T>
inline T quantizeValue(float real_value,
                       const PipelineQuantizeOperation& operation)
{
    return static_cast<T>(quantizeToInt(real_value, operation));
}

}  // namespace detail
}  // namespace cvh

#endif  // CVH_PIPELINE_DETAIL_QUANTIZE_HPP
