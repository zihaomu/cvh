#ifndef CVH_CORE_TRANSFORM_H
#define CVH_CORE_TRANSFORM_H

#include "mat.h"

namespace cvh
{

void transform(const Mat& src, Mat& dst, const Mat& matrix);
void perspectiveTransform(const Mat& src, Mat& dst, const Mat& matrix);

}  // namespace cvh

#include "detail/transform_impl.hpp"

#endif  // CVH_CORE_TRANSFORM_H
