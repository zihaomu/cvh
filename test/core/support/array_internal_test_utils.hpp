#pragma once

#include "test/support/dispatch_mode_guard.hpp"
#include "cvh.h"
#include "cvh/core/detail/arithm_ui.hpp"
#include "cvh/core/detail/cpu_features.hpp"
#include "cvh/core/detail/dispatch_control.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

using namespace cvh;

namespace
{

using DispatchModeGuard = cvh::test::DispatchModeGuard;

void expect_mat_bytes_equal(const Mat& actual, const Mat& expected)
{
    ASSERT_EQ(actual.type(), expected.type());
    ASSERT_EQ(actual.shape(), expected.shape());
    const size_t row_bytes =
        static_cast<size_t>(actual.size[1]) * actual.elemSize();
    for (int row = 0; row < actual.size[0]; ++row)
    {
        const uchar* actual_row =
            actual.data + static_cast<size_t>(row) * actual.step(0);
        const uchar* expected_row =
            expected.data + static_cast<size_t>(row) * expected.step(0);
        for (size_t byte = 0; byte < row_bytes; ++byte)
        {
            ASSERT_EQ(actual_row[byte], expected_row[byte])
                << "row=" << row << ", byte=" << byte;
        }
    }
}

template<typename T>
void set_raw_bits(T& value, const void* bits)
{
    std::memcpy(&value, bits, sizeof(T));
}

template<typename UInt, typename T>
UInt raw_bits(const T& value)
{
    static_assert(sizeof(UInt) == sizeof(T), "raw bit type size mismatch");
    UInt bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template<typename T>
void expect_integer_ui_matches_scalar(int type)
{
    const int columns = cvh::test::accepted_fixed_width_test_length<T>();
    Mat a({3, columns}, type);
    Mat b({3, columns}, type);
    for (int y = 0; y < a.size[0]; ++y)
    {
        T* a_row = reinterpret_cast<T*>(a.data + static_cast<size_t>(y) * a.step(0));
        T* b_row = reinterpret_cast<T*>(b.data + static_cast<size_t>(y) * b.step(0));
        for (int x = 0; x < a.size[1]; ++x)
        {
            const int index = y * a.size[1] + x;
            if (index == 0)
            {
                a_row[x] = std::numeric_limits<T>::lowest();
                b_row[x] = std::numeric_limits<T>::max();
                continue;
            }
            if (index == 1)
            {
                a_row[x] = std::numeric_limits<T>::max();
                b_row[x] = std::numeric_limits<T>::lowest();
                continue;
            }
            if constexpr (std::is_signed<T>::value)
            {
                a_row[x] = static_cast<T>((index * 37) % 201 - 100);
                b_row[x] = static_cast<T>((index * 53 + 17) % 181 - 90);
            }
            else
            {
                a_row[x] = static_cast<T>((index * 37 + 11) % 211);
                b_row[x] = static_cast<T>((index * 53 + 17) % 197);
            }
        }
    }

    Mat scalar_abs;
    Mat scalar_min;
    Mat scalar_max;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        absdiff(a, b, scalar_abs);
        min(a, b, scalar_min);
        max(a, b, scalar_max);
    }

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    Mat actual;
    absdiff(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_abs);

    min(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_min);

    max(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_max);
}

template<typename T>
void expect_basic_integer_ui_matches_scalar(int type)
{
    const int columns = cvh::test::accepted_fixed_width_test_length<T>();
    Mat a({3, columns}, type);
    Mat b({3, columns}, type);
    for (int y = 0; y < a.size[0]; ++y)
    {
        T* a_row = reinterpret_cast<T*>(a.data + static_cast<size_t>(y) * a.step(0));
        T* b_row = reinterpret_cast<T*>(b.data + static_cast<size_t>(y) * b.step(0));
        for (int x = 0; x < a.size[1]; ++x)
        {
            const int index = y * a.size[1] + x;
            if constexpr (std::is_signed<T>::value)
            {
                a_row[x] = static_cast<T>((index * 13) % 41 - 20);
                b_row[x] = static_cast<T>((index * 7) % 17 - 8);
            }
            else
            {
                a_row[x] = static_cast<T>((index * 13) % 41);
                b_row[x] = static_cast<T>((index * 7) % 17);
            }
        }
    }

    if constexpr (sizeof(T) <= 2)
    {
        a.at<T>(0, 0) = std::numeric_limits<T>::max();
        b.at<T>(0, 0) = static_cast<T>(2);
        a.at<T>(0, 1) = std::numeric_limits<T>::lowest();
        b.at<T>(0, 1) = static_cast<T>(2);
    }

    Mat scalar_add;
    Mat scalar_sub;
    Mat scalar_mul;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        add(a, b, scalar_add);
        subtract(a, b, scalar_sub);
        multiply(a, b, scalar_mul);
    }

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    Mat actual;
    add(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_add);

    subtract(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_sub);

    multiply(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_mul);
}

template<typename T>
void expect_integer_scalar_ui_matches_scalar(int type)
{
    const int columns = cvh::test::accepted_fixed_width_test_length<T>();
    Mat src({3, columns}, type);
    const size_t row_scalars =
        static_cast<size_t>(src.size[1]) *
        static_cast<size_t>(src.channels());
    for (int y = 0; y < src.size[0]; ++y)
    {
        T* row = reinterpret_cast<T*>(
            src.data + static_cast<size_t>(y) * src.step(0));
        for (size_t x = 0; x < row_scalars; ++x)
        {
            const int index =
                y * static_cast<int>(row_scalars) + static_cast<int>(x);
            if constexpr (std::is_signed<T>::value)
            {
                row[x] = static_cast<T>((index * 29) % 181 - 90);
            }
            else
            {
                row[x] = static_cast<T>((index * 29 + 7) % 181);
            }
        }
    }

    const Scalar scalar(3.0, -5.0, 71.0);
    Mat scalar_add;
    Mat scalar_sub;
    Mat scalar_mul;
    Mat scalar_abs;
    Mat scalar_min;
    Mat scalar_max;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        add(src, scalar, scalar_add);
        subtract(scalar, src, scalar_sub);
        multiply(src, scalar, scalar_mul);
        absdiff(src, scalar, scalar_abs);
        min(src, scalar, scalar_min);
        max(src, scalar, scalar_max);
    }

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    Mat actual;
    add(src, scalar, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_add);

    subtract(scalar, src, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_sub);

    multiply(src, scalar, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_mul);

    absdiff(src, scalar, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_abs);

    min(src, scalar, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_min);

    max(src, scalar, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_max);
}

void expect_float_mat_near(const Mat& actual, const Mat& expected)
{
    ASSERT_EQ(actual.type(), CV_32FC1);
    ASSERT_EQ(actual.shape(), expected.shape());
    for (int y = 0; y < actual.size[0]; ++y)
    {
        for (int x = 0; x < actual.size[1]; ++x)
        {
            const float actual_value = actual.at<float>(y, x);
            const float expected_value = expected.at<float>(y, x);
            if (std::isnan(expected_value))
            {
                EXPECT_TRUE(std::isnan(actual_value)) << "y=" << y << ", x=" << x;
            }
            else if (std::isinf(expected_value))
            {
                EXPECT_TRUE(std::isinf(actual_value)) << "y=" << y << ", x=" << x;
                EXPECT_EQ(std::signbit(actual_value), std::signbit(expected_value));
            }
            else
            {
                const float tolerance =
                    1e-6f * std::max(1.0f, std::abs(expected_value));
                EXPECT_NEAR(actual_value, expected_value, tolerance)
                    << "y=" << y << ", x=" << x;
            }
        }
    }
}

}  // namespace
