#pragma once

#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

namespace cvh::test
{

struct FloatComparison
{
    bool close;
    double absolute_error;
    double relative_error;
};

inline FloatComparison compare_float_values(
    float actual,
    float expected,
    float abs_tolerance,
    float rel_tolerance)
{
    if (std::isnan(actual) || std::isnan(expected))
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        return {false, nan, nan};
    }
    if (std::isinf(actual) || std::isinf(expected))
    {
        const double infinity =
            std::numeric_limits<double>::infinity();
        return {
            actual == expected,
            actual == expected ? 0.0 : infinity,
            actual == expected ? 0.0 : infinity,
        };
    }

    const double actual_value = static_cast<double>(actual);
    const double expected_value = static_cast<double>(expected);
    const double absolute_error =
        std::abs(actual_value - expected_value);
    const double expected_magnitude = std::abs(expected_value);
    const double relative_error =
        expected_magnitude == 0.0
            ? (absolute_error == 0.0
                   ? 0.0
                   : std::numeric_limits<double>::infinity())
            : absolute_error / expected_magnitude;
    return {
        absolute_error <= static_cast<double>(abs_tolerance) ||
            relative_error <= static_cast<double>(rel_tolerance),
        absolute_error,
        relative_error,
    };
}

inline float mat_float_at_scalar_index(
    const Mat& mat,
    size_t scalar_index)
{
    const size_t channels = static_cast<size_t>(mat.channels());
    size_t pixel_index = scalar_index / channels;
    const size_t channel = scalar_index % channels;
    size_t byte_offset = channel * sizeof(float);
    for (int dimension = mat.dims - 1; dimension >= 0; --dimension)
    {
        const size_t dimension_size =
            static_cast<size_t>(mat.size.p[dimension]);
        const size_t coordinate = pixel_index % dimension_size;
        pixel_index /= dimension_size;
        byte_offset += coordinate * mat.step(dimension);
    }
    return *reinterpret_cast<const float*>(mat.data + byte_offset);
}

inline testing::AssertionResult mat_close(
    const Mat& actual,
    const Mat& expected,
    float abs_tolerance,
    float rel_tolerance,
    const std::string& case_name = std::string())
{
    if (!std::isfinite(abs_tolerance) || abs_tolerance < 0.0f ||
        !std::isfinite(rel_tolerance) || rel_tolerance < 0.0f)
    {
        return testing::AssertionFailure()
            << "invalid tolerances, case=" << case_name
            << ", abs_tolerance=" << abs_tolerance
            << ", rel_tolerance=" << rel_tolerance;
    }
    if (actual.shape() != expected.shape())
    {
        return testing::AssertionFailure()
            << "shape mismatch, case=" << case_name;
    }
    if (actual.type() != expected.type())
    {
        return testing::AssertionFailure()
            << "type mismatch, case=" << case_name
            << ", actual_type=" << actual.type()
            << ", expected_type=" << expected.type();
    }
    if (actual.depth() != CV_32F)
    {
        return testing::AssertionFailure()
            << "floating-point comparison requires CV_32F, case="
            << case_name << ", actual_type=" << actual.type();
    }

    const size_t scalar_count =
        actual.total() * static_cast<size_t>(actual.channels());
    for (size_t index = 0; index < scalar_count; ++index)
    {
        const float actual_value =
            mat_float_at_scalar_index(actual, index);
        const float expected_value =
            mat_float_at_scalar_index(expected, index);
        const FloatComparison comparison = compare_float_values(
            actual_value,
            expected_value,
            abs_tolerance,
            rel_tolerance);
        if (!comparison.close)
        {
            return testing::AssertionFailure()
                << "case=" << case_name
                << ", first_bad_index=" << index
                << ", actual=" << actual_value
                << ", expected=" << expected_value
                << ", absolute_error=" << comparison.absolute_error
                << ", relative_error=" << comparison.relative_error
                << ", abs_tolerance=" << abs_tolerance
                << ", rel_tolerance=" << rel_tolerance;
        }
    }
    return testing::AssertionSuccess();
}

inline void expect_mat_close(
    const Mat& actual,
    const Mat& expected,
    float abs_tolerance,
    float rel_tolerance,
    const std::string& case_name = std::string())
{
    EXPECT_TRUE(mat_close(
        actual,
        expected,
        abs_tolerance,
        rel_tolerance,
        case_name));
}

}  // namespace cvh::test
