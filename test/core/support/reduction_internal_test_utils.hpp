#pragma once

#include "test/support/dispatch_mode_guard.hpp"
#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace cvh;

namespace {

using DispatchModeGuard = cvh::test::DispatchModeGuard;

void set_nonzero_at(Mat& mat, size_t index)
{
    switch (mat.depth())
    {
        case CV_8U:
            reinterpret_cast<uchar*>(mat.data)[index] = 1;
            return;
        case CV_8S:
            reinterpret_cast<schar*>(mat.data)[index] = -1;
            return;
        case CV_16U:
            reinterpret_cast<ushort*>(mat.data)[index] = 1;
            return;
        case CV_16S:
            reinterpret_cast<short*>(mat.data)[index] = -1;
            return;
        case CV_32S:
            reinterpret_cast<int*>(mat.data)[index] = -1;
            return;
        case CV_32U:
            reinterpret_cast<uint*>(mat.data)[index] = 1;
            return;
        case CV_16F:
            reinterpret_cast<hfloat*>(mat.data)[index] = hfloat(1.0f);
            return;
        case CV_32F:
            reinterpret_cast<float*>(mat.data)[index] = -1.0f;
            return;
        case CV_64F:
            reinterpret_cast<double*>(mat.data)[index] = -1.0;
            return;
        default:
            FAIL() << "unsupported test depth";
    }
}

void expect_nonzero_results(const Mat& src,
                            int expected_count,
                            cpu::DispatchTag expected_auto_tag)
{
    std::vector<Point> scalar_points;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        cpu::reset_last_dispatch_tag();
        EXPECT_EQ(countNonZero(src), expected_count);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
        cpu::reset_last_dispatch_tag();
        EXPECT_EQ(hasNonZero(src), expected_count != 0);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
        cpu::reset_last_dispatch_tag();
        findNonZero(src, scalar_points);
        EXPECT_EQ(
            scalar_points.size(),
            static_cast<size_t>(expected_count));
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    }

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    cpu::reset_last_dispatch_tag();
    EXPECT_EQ(countNonZero(src), expected_count);
    EXPECT_EQ(cpu::last_dispatch_tag(), expected_auto_tag);
    cpu::reset_last_dispatch_tag();
    EXPECT_EQ(hasNonZero(src), expected_count != 0);
    EXPECT_EQ(cpu::last_dispatch_tag(), expected_auto_tag);
    cpu::reset_last_dispatch_tag();
    std::vector<Point> auto_points;
    findNonZero(src, auto_points);
    EXPECT_EQ(auto_points, scalar_points);
    EXPECT_EQ(cpu::last_dispatch_tag(), expected_auto_tag);

    cpu::reset_last_dispatch_tag();
    Mat point_mat;
    findNonZero(src, point_mat);
    EXPECT_EQ(cpu::last_dispatch_tag(), expected_auto_tag);
    if (scalar_points.empty())
    {
        EXPECT_TRUE(point_mat.empty());
        return;
    }

    ASSERT_EQ(point_mat.type(), CV_32SC2);
    ASSERT_EQ(
        point_mat.shape(),
        MatShape({static_cast<int>(scalar_points.size()), 1}));
    const int* coordinates =
        reinterpret_cast<const int*>(point_mat.data);
    for (size_t index = 0; index < scalar_points.size(); ++index)
    {
        EXPECT_EQ(coordinates[index * 2], scalar_points[index].x);
        EXPECT_EQ(coordinates[index * 2 + 1], scalar_points[index].y);
    }
}

cpu::DispatchTag expected_nonzero_auto_tag(int depth, bool ui_enabled)
{
    if (!ui_enabled || depth == CV_16F)
    {
        return cpu::DispatchTag::Scalar;
    }
    if (depth == CV_64F)
    {
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        return cpu::DispatchTag::OpenCVUI;
#else
        return cpu::DispatchTag::Scalar;
#endif
    }
    return cpu::DispatchTag::OpenCVUI;
}

cpu::DispatchTag expected_statistics_auto_tag(int depth, bool ui_enabled)
{
    if (!ui_enabled || depth == CV_16F)
    {
        return cpu::DispatchTag::Scalar;
    }
    if (depth == CV_64F)
    {
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        return cpu::DispatchTag::OpenCVUI;
#else
        return cpu::DispatchTag::Scalar;
#endif
    }
    return cpu::DispatchTag::OpenCVUI;
}

cpu::DispatchTag expected_norm_auto_tag(int depth, bool ui_enabled)
{
    if (!ui_enabled)
    {
        return cpu::DispatchTag::Scalar;
    }
    if (depth == CV_8U)
    {
        return cpu::DispatchTag::OpenCVUI;
    }
    if (depth == CV_32F)
    {
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        return cpu::DispatchTag::OpenCVUI;
#else
        return cpu::DispatchTag::Scalar;
#endif
    }
    return cpu::DispatchTag::Scalar;
}

void set_test_value(Mat& mat,
                    int row,
                    int column,
                    int channel,
                    double value)
{
    uchar* row_data = mat.data + static_cast<size_t>(row) * mat.step(0);
    const size_t index =
        static_cast<size_t>(column * mat.channels() + channel);
    switch (mat.depth())
    {
        case CV_8U:
            reinterpret_cast<uchar*>(row_data)[index] =
                static_cast<uchar>(value);
            return;
        case CV_8S:
            reinterpret_cast<schar*>(row_data)[index] =
                static_cast<schar>(value);
            return;
        case CV_16U:
            reinterpret_cast<ushort*>(row_data)[index] =
                static_cast<ushort>(value);
            return;
        case CV_16S:
            reinterpret_cast<short*>(row_data)[index] =
                static_cast<short>(value);
            return;
        case CV_32S:
            reinterpret_cast<int*>(row_data)[index] =
                static_cast<int>(value);
            return;
        case CV_32U:
            reinterpret_cast<uint*>(row_data)[index] =
                static_cast<uint>(value);
            return;
        case CV_16F:
            reinterpret_cast<hfloat*>(row_data)[index] =
                hfloat(static_cast<float>(value));
            return;
        case CV_32F:
            reinterpret_cast<float*>(row_data)[index] =
                static_cast<float>(value);
            return;
        case CV_64F:
            reinterpret_cast<double*>(row_data)[index] = value;
            return;
        default:
            FAIL() << "unsupported test depth";
    }
}

void expect_scalar_close(const Scalar& actual,
                         const Scalar& expected,
                         int channels,
                         double absolute_tolerance,
                         double relative_tolerance)
{
    for (int channel = 0; channel < channels; ++channel)
    {
        if (std::isnan(expected[channel]))
        {
            EXPECT_TRUE(std::isnan(actual[channel]));
            continue;
        }
        if (std::isinf(expected[channel]))
        {
            EXPECT_EQ(actual[channel], expected[channel]);
            continue;
        }
        const double scale =
            std::max(std::fabs(actual[channel]), std::fabs(expected[channel]));
        EXPECT_NEAR(
            actual[channel],
            expected[channel],
            absolute_tolerance + relative_tolerance * scale);
    }
}

void expect_index_mat_equal(const Mat& actual, const Mat& expected)
{
    ASSERT_EQ(actual.shape(), expected.shape());
    ASSERT_EQ(actual.type(), CV_32SC1);
    ASSERT_EQ(expected.type(), CV_32SC1);
    for (size_t index = 0; index < actual.total(); ++index)
    {
        EXPECT_EQ(
            reinterpret_cast<const int*>(actual.data)[index],
            reinterpret_cast<const int*>(expected.data)[index]);
    }
}

void expect_f32_mat_close(const Mat& actual,
                          const Mat& expected,
                          float tolerance)
{
    ASSERT_EQ(actual.shape(), expected.shape());
    ASSERT_EQ(actual.type(), expected.type());
    ASSERT_EQ(actual.depth(), CV_32F);
    const size_t scalars =
        actual.total() * static_cast<size_t>(actual.channels());
    for (size_t index = 0; index < scalars; ++index)
    {
        EXPECT_NEAR(
            reinterpret_cast<const float*>(actual.data)[index],
            reinterpret_cast<const float*>(expected.data)[index],
            tolerance);
    }
}

void expect_reduce_mat_close(const Mat& actual,
                             const Mat& expected,
                             double absolute_tolerance,
                             double relative_tolerance)
{
    ASSERT_EQ(actual.shape(), expected.shape());
    ASSERT_EQ(actual.type(), expected.type());
    for (int row = 0; row < actual.size.p[0]; ++row)
    {
        const uchar* actual_row =
            actual.data + static_cast<size_t>(row) * actual.step(0);
        const uchar* expected_row =
            expected.data + static_cast<size_t>(row) * expected.step(0);
        const size_t row_scalars =
            static_cast<size_t>(actual.size.p[1]) *
            static_cast<size_t>(actual.channels());
        for (size_t scalar = 0; scalar < row_scalars; ++scalar)
        {
            const double actual_value =
                reduce_detail::read_scalar(
                    actual_row, scalar, actual.depth());
            const double expected_value =
                reduce_detail::read_scalar(
                    expected_row, scalar, expected.depth());
            if (std::isnan(expected_value))
            {
                EXPECT_TRUE(std::isnan(actual_value));
                continue;
            }
            if (std::isinf(expected_value))
            {
                EXPECT_EQ(actual_value, expected_value);
                continue;
            }
            const double scale = std::max(
                std::fabs(actual_value), std::fabs(expected_value));
            EXPECT_NEAR(
                actual_value,
                expected_value,
                absolute_tolerance + relative_tolerance * scale);
        }
    }
}

int reduce_test_lanes(int depth)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return depth == CV_8U
        ? cv::VTraits<cv::v_uint8>::vlanes()
        : cv::VTraits<cv::v_float32>::vlanes();
#else
    return depth == CV_8U ? 16 : 4;
#endif
}

cpu::DispatchTag expected_reduce_auto_tag(int depth,
                                          int axis,
                                          int rtype,
                                          bool ui_enabled)
{
    if (!ui_enabled)
    {
        return cpu::DispatchTag::Scalar;
    }
    if (depth == CV_32F && axis == 0 &&
        rtype != REDUCE_MAX && rtype != REDUCE_MIN)
    {
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        return cpu::DispatchTag::OpenCVUI;
#else
        return cpu::DispatchTag::Scalar;
#endif
    }
    return cpu::DispatchTag::OpenCVUI;
}

}  // namespace
