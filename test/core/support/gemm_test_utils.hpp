#pragma once

#include "cvh.h"
#include "test/core/support/floating_point_test_utils.hpp"

#include <cmath>

using namespace cvh;
using cvh::test::expect_mat_close;

namespace {

void fill_mat_deterministic(Mat& mat, float amp, float bias)
{
    float* data = reinterpret_cast<float*>(mat.data);
    for (size_t i = 0; i < mat.total(); ++i)
    {
        const float fi = static_cast<float>(i);
        data[i] = std::sin(fi * 0.113f) * amp + std::cos(fi * 0.071f) * (amp * 0.7f) + bias;
    }
}

}  // namespace
