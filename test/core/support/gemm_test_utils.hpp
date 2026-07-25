#pragma once

#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>

using namespace cvh;

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

void expect_mat_close(const Mat& out, const Mat& ref, float abs_tol, float rel_tol)
{
    ASSERT_EQ(out.shape(), ref.shape());
    ASSERT_EQ(out.type(), ref.type());

    const float* out_data = reinterpret_cast<const float*>(out.data);
    const float* ref_data = reinterpret_cast<const float*>(ref.data);
    const size_t count = out.total();

    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        const float abs_diff = std::abs(out_data[i] - ref_data[i]);
        const float denom = std::max(1.0f, std::abs(ref_data[i]));
        const float rel_diff = abs_diff / denom;
        max_abs = std::max(max_abs, abs_diff);
        max_rel = std::max(max_rel, rel_diff);
    }

    EXPECT_TRUE(max_abs <= abs_tol || max_rel <= rel_tol)
        << "max_abs=" << max_abs
        << ", max_rel=" << max_rel
        << ", abs_tol=" << abs_tol
        << ", rel_tol=" << rel_tol;
}

}  // namespace
