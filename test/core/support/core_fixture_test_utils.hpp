#pragma once

#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace cvh::test
{

inline std::string core_data_path(const std::string& filename)
{
    return std::string(M_ROOT_PATH) + "/test/core/data/npy/" + filename;
}

inline void expect_mat_close(
    const Mat& actual,
    const Mat& expected,
    float abs_tolerance,
    float rel_tolerance,
    const std::string& case_name)
{
    ASSERT_EQ(actual.shape(), expected.shape())
        << "shape mismatch, case=" << case_name;
    ASSERT_EQ(actual.type(), expected.type())
        << "type mismatch, case=" << case_name;

    const float* actual_data =
        reinterpret_cast<const float*>(actual.data);
    const float* expected_data =
        reinterpret_cast<const float*>(expected.data);

    double max_abs = 0.0;
    double max_rel = 0.0;
    size_t first_bad_index = actual.total();
    for (size_t index = 0; index < actual.total(); ++index)
    {
        const double difference =
            std::abs(
                static_cast<double>(actual_data[index]) -
                static_cast<double>(expected_data[index]));
        const double denominator =
            std::max(
                1.0,
                std::abs(static_cast<double>(expected_data[index])));
        const double relative = difference / denominator;
        max_abs = std::max(max_abs, difference);
        max_rel = std::max(max_rel, relative);
        if (first_bad_index == actual.total() &&
            difference > abs_tolerance &&
            relative > rel_tolerance)
        {
            first_bad_index = index;
        }
    }

    EXPECT_TRUE(
        max_abs <= abs_tolerance ||
        max_rel <= rel_tolerance)
        << "case=" << case_name
        << ", first_bad_index=" << first_bad_index
        << ", max_abs=" << max_abs
        << ", max_rel=" << max_rel
        << ", abs_tolerance=" << abs_tolerance
        << ", rel_tolerance=" << rel_tolerance;
}

}  // namespace cvh::test
