#ifndef CVH_CORE_RANDOM_H
#define CVH_CORE_RANDOM_H

#include "mat.h"

namespace cvh
{

void randu(Mat& dst, const Scalar& low, const Scalar& high);
void randn(Mat& dst, const Scalar& mean, const Scalar& stddev);

}  // namespace cvh

#include "detail/random_impl.hpp"

#endif  // CVH_CORE_RANDOM_H
