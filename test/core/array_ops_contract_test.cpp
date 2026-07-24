#include "cvh.h"
#include "cvh/core/detail/arithm_ui.hpp"
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

class DispatchModeGuard
{
public:
    explicit DispatchModeGuard(cpu::DispatchMode mode)
        : previous_(cpu::dispatch_mode())
    {
        cpu::set_dispatch_mode(mode);
    }

    ~DispatchModeGuard()
    {
        cpu::set_dispatch_mode(previous_);
    }

private:
    cpu::DispatchMode previous_;
};

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
    Mat a({3, 19}, type);
    Mat b({3, 19}, type);
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
    Mat a({3, 19}, type);
    Mat b({3, 19}, type);
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
    Mat src({3, 19}, type);
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

TEST(ArrayOpsContract_TEST, ui_arithmetic_matches_scalar_for_all_integer_depths)
{
    if (!detail::arithm_ui::enabled())
        GTEST_SKIP() << "OpenCV UI arithmetic requires NEON or SSE/AVX";

    expect_integer_ui_matches_scalar<uchar>(CV_8UC1);
    expect_integer_ui_matches_scalar<schar>(CV_8SC1);
    expect_integer_ui_matches_scalar<ushort>(CV_16UC1);
    expect_integer_ui_matches_scalar<short>(CV_16SC1);
    expect_integer_ui_matches_scalar<int>(CV_32SC1);
    expect_integer_ui_matches_scalar<uint>(CV_32UC1);
}

TEST(ArrayOpsContract_TEST, ui_basic_arithmetic_matches_scalar_for_all_integer_depths)
{
    if (!detail::arithm_ui::enabled())
        GTEST_SKIP() << "OpenCV UI arithmetic requires NEON or SSE/AVX";

    expect_basic_integer_ui_matches_scalar<uchar>(CV_8UC1);
    expect_basic_integer_ui_matches_scalar<schar>(CV_8SC1);
    expect_basic_integer_ui_matches_scalar<ushort>(CV_16UC1);
    expect_basic_integer_ui_matches_scalar<short>(CV_16SC1);
    expect_basic_integer_ui_matches_scalar<int>(CV_32SC1);
    expect_basic_integer_ui_matches_scalar<uint>(CV_32UC1);
}

TEST(ArrayOpsContract_TEST, ui_mat_scalar_arithmetic_matches_scalar_for_broadcast_types)
{
    if (!detail::arithm_ui::enabled())
        GTEST_SKIP() << "OpenCV UI arithmetic requires NEON or SSE/AVX";

    expect_integer_scalar_ui_matches_scalar<uchar>(CV_8UC3);
    expect_integer_scalar_ui_matches_scalar<schar>(CV_8SC3);
    expect_integer_scalar_ui_matches_scalar<ushort>(CV_16UC3);
    expect_integer_scalar_ui_matches_scalar<short>(CV_16SC3);
    expect_integer_scalar_ui_matches_scalar<int>(CV_32SC3);
    expect_integer_scalar_ui_matches_scalar<uint>(CV_32UC3);

    Mat src({3, 19}, CV_32FC1);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            src.at<float>(y, x) =
                static_cast<float>(y * src.size[1] + x - 21) / 7.0f;
        }
    }
    src.at<float>(0, 0) = 0.0f;

    const Scalar scalar(2.5);
    Mat scalar_add;
    Mat scalar_sub;
    Mat scalar_mul;
    Mat scalar_div;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        add(src, scalar, scalar_add);
        subtract(scalar, src, scalar_sub);
        multiply(src, scalar, scalar_mul);
        divide(scalar, src, scalar_div);
    }

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    Mat actual;
    add(src, scalar, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_float_mat_near(actual, scalar_add);

    subtract(scalar, src, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_float_mat_near(actual, scalar_sub);

    multiply(src, scalar, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_float_mat_near(actual, scalar_mul);

    divide(scalar, src, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_float_mat_near(actual, scalar_div);

    Mat alias = scalar_add.clone();
    Mat expected_alias;
    {
        DispatchModeGuard scalar_guard(cpu::DispatchMode::ScalarOnly);
        subtract(Scalar(9.0), alias, expected_alias);
    }
    subtract(Scalar(9.0), alias, alias);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_float_mat_near(alias, expected_alias);
}

TEST(ArrayOpsContract_TEST, ui_basic_float_arithmetic_matches_scalar_including_divide_edges)
{
    if (!detail::arithm_ui::enabled())
        GTEST_SKIP() << "OpenCV UI arithmetic requires NEON or SSE/AVX";

    Mat a({3, 19}, CV_32FC1);
    Mat b({3, 19}, CV_32FC1);
    for (int y = 0; y < a.size[0]; ++y)
    {
        for (int x = 0; x < a.size[1]; ++x)
        {
            const int index = y * a.size[1] + x;
            a.at<float>(y, x) = static_cast<float>(index - 30) / 7.0f;
            b.at<float>(y, x) = static_cast<float>((index % 11) - 5) / 3.0f;
        }
    }
    a.at<float>(0, 0) = 0.0f;
    b.at<float>(0, 0) = 0.0f;
    a.at<float>(0, 1) = 1.0f;
    b.at<float>(0, 1) = 0.0f;
    a.at<float>(0, 2) = -1.0f;
    b.at<float>(0, 2) = 0.0f;

    Mat scalar_add;
    Mat scalar_sub;
    Mat scalar_mul;
    Mat scalar_div;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        add(a, b, scalar_add);
        subtract(a, b, scalar_sub);
        multiply(a, b, scalar_mul);
        divide(a, b, scalar_div);
    }

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    Mat actual;
    add(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_float_mat_near(actual, scalar_add);

    subtract(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_float_mat_near(actual, scalar_sub);

    multiply(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_float_mat_near(actual, scalar_mul);

    divide(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_float_mat_near(actual, scalar_div);
}

TEST(ArrayOpsContract_TEST, ui_arithmetic_matches_scalar_on_tail_roi_and_in_place_inputs)
{
    if (!detail::arithm_ui::enabled())
        GTEST_SKIP() << "OpenCV UI arithmetic requires NEON or SSE/AVX";

    Mat a_parent({4, 23}, CV_8UC3);
    Mat b_parent({4, 23}, CV_8UC3);
    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 23; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                a_parent.at<uchar>(y, x, ch) =
                    static_cast<uchar>((y * 71 + x * 13 + ch * 29) & 0xff);
                b_parent.at<uchar>(y, x, ch) =
                    static_cast<uchar>((y * 19 + x * 37 + ch * 11 + 5) & 0xff);
            }
        }
    }

    Mat a = a_parent.colRange(2, 21);
    Mat b = b_parent.colRange(2, 21);
    ASSERT_FALSE(a.isContinuous());
    ASSERT_FALSE(b.isContinuous());

    Mat scalar_abs;
    Mat scalar_add;
    Mat scalar_sub;
    Mat scalar_mul;
    Mat scalar_and;
    Mat scalar_or;
    Mat scalar_xor;
    Mat scalar_not;
    Mat scalar_min;
    Mat scalar_max;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        absdiff(a, b, scalar_abs);
        add(a, b, scalar_add);
        subtract(a, b, scalar_sub);
        multiply(a, b, scalar_mul);
        bitwise_and(a, b, scalar_and);
        bitwise_or(a, b, scalar_or);
        bitwise_xor(a, b, scalar_xor);
        bitwise_not(a, scalar_not);
        min(a, b, scalar_min);
        max(a, b, scalar_max);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    }

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    Mat actual;

    absdiff(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_abs);

    add(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_add);

    subtract(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_sub);

    multiply(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_mul);

    bitwise_and(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_and);

    bitwise_or(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_or);

    bitwise_xor(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_xor);

    bitwise_not(a, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_not);

    min(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_min);

    max(a, b, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_max);

    Mat in_place = a.clone();
    absdiff(in_place, b, in_place);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(in_place, scalar_abs);

    in_place = a.clone();
    add(in_place, b, in_place);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(in_place, scalar_add);
}

TEST(ArrayOpsContract_TEST, ui_arithmetic_keeps_short_rows_and_unsupported_types_on_scalar)
{
    Mat short_a({1, 5}, CV_8UC1);
    Mat short_b({1, 5}, CV_8UC1);
    short_a.setTo(Scalar::all(17));
    short_b.setTo(Scalar::all(9));

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    Mat out;
    absdiff(short_a, short_b, out);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    add(short_a, short_b, out);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

    Mat integer_dividend({2, 19}, CV_32SC1);
    Mat integer_divisor({2, 19}, CV_32SC1);
    integer_dividend.setTo(Scalar::all(21));
    integer_divisor.setTo(Scalar::all(4));
    out.release();
    divide(integer_dividend, integer_divisor, out);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    EXPECT_EQ(out.at<int>(0, 0), 5);

    Mat fp16_a({2, 19}, CV_16FC1);
    Mat fp16_b({2, 19}, CV_16FC1);
    fp16_a.setTo(Scalar::all(1.5));
    fp16_b.setTo(Scalar::all(2.0));
    out.release();
    add(fp16_a, fp16_b, out);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
}

TEST(ArrayOpsContract_TEST, public_numeric_apis_cover_mat_and_scalar_inputs)
{
    Mat a({1, 7}, CV_8UC1);
    Mat b({1, 7}, CV_8UC1);
    const uchar a_values[] = {0, 10, 20, 30, 200, 250, 255};
    const uchar b_values[] = {1, 5, 30, 20, 210, 240, 0};
    for (int x = 0; x < 7; ++x)
    {
        a.at<uchar>(0, x) = a_values[x];
        b.at<uchar>(0, x) = b_values[x];
    }

    Mat out;
    absdiff(a, b, out);
    const uchar abs_expected[] = {1, 5, 10, 10, 10, 10, 255};
    for (int x = 0; x < 7; ++x)
    {
        EXPECT_EQ(out.at<uchar>(0, x), abs_expected[x]);
    }

    absdiff(a, Scalar::all(20.0), out);
    EXPECT_EQ(out.at<uchar>(0, 0), 20);
    EXPECT_EQ(out.at<uchar>(0, 2), 0);
    EXPECT_EQ(out.at<uchar>(0, 6), 235);

    min(a, b, out);
    EXPECT_EQ(out.at<uchar>(0, 1), 5);
    EXPECT_EQ(out.at<uchar>(0, 4), 200);
    min(a, Scalar::all(25.0), out);
    EXPECT_EQ(out.at<uchar>(0, 0), 0);
    EXPECT_EQ(out.at<uchar>(0, 3), 25);
    max(a, Scalar::all(25.0), out);
    EXPECT_EQ(out.at<uchar>(0, 0), 25);
    EXPECT_EQ(out.at<uchar>(0, 3), 30);
}

TEST(ArrayOpsContract_TEST, bitwise_apis_use_raw_float_bits)
{
    Mat a({1, 3}, CV_32FC1);
    Mat b({1, 3}, CV_32FC1);
    const std::uint32_t a_bits[] = {0x3f800000u, 0x80000000u, 0x7fc12345u};
    const std::uint32_t b_bits[] = {0x00ff00ffu, 0xffffffffu, 0x0f0f0f0fu};
    for (int x = 0; x < 3; ++x)
    {
        set_raw_bits(a.at<float>(0, x), &a_bits[x]);
        set_raw_bits(b.at<float>(0, x), &b_bits[x]);
    }

    Mat out;
    bitwise_and(a, b, out);
    for (int x = 0; x < 3; ++x)
    {
        EXPECT_EQ(raw_bits<std::uint32_t>(out.at<float>(0, x)), a_bits[x] & b_bits[x]);
    }

    bitwise_or(a, b, out);
    for (int x = 0; x < 3; ++x)
    {
        EXPECT_EQ(raw_bits<std::uint32_t>(out.at<float>(0, x)), a_bits[x] | b_bits[x]);
    }

    bitwise_xor(a, b, out);
    for (int x = 0; x < 3; ++x)
    {
        EXPECT_EQ(raw_bits<std::uint32_t>(out.at<float>(0, x)), a_bits[x] ^ b_bits[x]);
    }

    bitwise_not(a, out);
    for (int x = 0; x < 3; ++x)
    {
        EXPECT_EQ(raw_bits<std::uint32_t>(out.at<float>(0, x)), ~a_bits[x]);
    }
}

TEST(ArrayOpsContract_TEST, bitwise_scalar_and_mask_preserve_unselected_pixels)
{
    Mat src({1, 5}, CV_8UC4);
    for (int x = 0; x < 5; ++x)
    {
        for (int ch = 0; ch < 4; ++ch)
        {
            src.at<uchar>(0, x, ch) = static_cast<uchar>(0x10 * (ch + 1) + x);
        }
    }
    Mat mask({1, 5}, CV_8UC1);
    mask.at<uchar>(0, 0) = 0;
    mask.at<uchar>(0, 1) = 255;
    mask.at<uchar>(0, 2) = 0;
    mask.at<uchar>(0, 3) = 1;
    mask.at<uchar>(0, 4) = 0;

    Mat dst({1, 5}, CV_8UC4);
    dst.setTo(Scalar::all(0xA5));
    bitwise_xor(src, Scalar(0x0F, 0xF0, 0x55, 0xAA), dst, mask);
    for (int x = 0; x < 5; ++x)
    {
        for (int ch = 0; ch < 4; ++ch)
        {
            const uchar expected =
                mask.at<uchar>(0, x) != 0
                    ? static_cast<uchar>(src.at<uchar>(0, x, ch) ^
                                         static_cast<uchar>(Scalar(0x0F, 0xF0, 0x55, 0xAA)[ch]))
                    : static_cast<uchar>(0xA5);
            EXPECT_EQ(dst.at<uchar>(0, x, ch), expected);
        }
    }

    Mat scalar_out;
    bitwise_and(src, Scalar::all(0x0F), scalar_out);
    EXPECT_EQ(scalar_out.at<uchar>(0, 2, 0),
              static_cast<uchar>(src.at<uchar>(0, 2, 0) & 0x0F));
    bitwise_or(Scalar::all(0x80), src, scalar_out);
    EXPECT_EQ(scalar_out.at<uchar>(0, 2, 1),
              static_cast<uchar>(0x80 | src.at<uchar>(0, 2, 1)));
    bitwise_xor(Scalar::all(0xFF), src, scalar_out);
    EXPECT_EQ(scalar_out.at<uchar>(0, 2, 2),
              static_cast<uchar>(0xFF ^ src.at<uchar>(0, 2, 2)));

    Mat allocated;
    bitwise_not(src, allocated, mask);
    for (int x = 0; x < 5; ++x)
    {
        for (int ch = 0; ch < 4; ++ch)
        {
            const uchar expected = mask.at<uchar>(0, x) != 0
                                       ? static_cast<uchar>(~src.at<uchar>(0, x, ch))
                                       : static_cast<uchar>(0);
            EXPECT_EQ(allocated.at<uchar>(0, x, ch), expected);
        }
    }
}

TEST(ArrayOpsContract_TEST, ui_masked_bitwise_matches_scalar_for_roi_alias_and_raw_bits)
{
    if (!detail::arithm_ui::enabled())
        GTEST_SKIP() << "OpenCV UI arithmetic requires NEON or SSE/AVX";

    Mat a_parent({4, 23}, CV_8UC3);
    Mat b_parent({4, 23}, CV_8UC3);
    Mat mask_parent({4, 23}, CV_8UC1);
    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 23; ++x)
        {
            mask_parent.at<uchar>(y, x) =
                (x + y) % 3 == 0 ? 0 : ((x + y) % 3 == 1 ? 1 : 255);
            for (int ch = 0; ch < 3; ++ch)
            {
                a_parent.at<uchar>(y, x, ch) =
                    static_cast<uchar>((y * 47 + x * 13 + ch * 31) & 0xff);
                b_parent.at<uchar>(y, x, ch) =
                    static_cast<uchar>((y * 17 + x * 29 + ch * 7) & 0xff);
            }
        }
    }

    Mat a = a_parent.colRange(2, 21);
    Mat b = b_parent.colRange(2, 21);
    Mat mask = mask_parent.colRange(2, 21);
    ASSERT_FALSE(a.isContinuous());
    ASSERT_FALSE(mask.isContinuous());

    Mat scalar_and(a.shape(), a.type());
    Mat scalar_xor(a.shape(), a.type());
    scalar_and.setTo(Scalar::all(0xA5));
    scalar_xor.setTo(Scalar::all(0x5A));
    Mat scalar_not;
    Mat scalar_alias = a.clone();
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        bitwise_and(a, b, scalar_and, mask);
        bitwise_xor(a, Scalar(0x0F, 0xF0, 0x55), scalar_xor, mask);
        bitwise_not(a, scalar_not, mask);
        bitwise_or(scalar_alias, b, scalar_alias, mask);
    }

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    Mat actual(a.shape(), a.type());
    actual.setTo(Scalar::all(0xA5));
    bitwise_and(a, b, actual, mask);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_and);

    actual.setTo(Scalar::all(0x5A));
    bitwise_xor(a, Scalar(0x0F, 0xF0, 0x55), actual, mask);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_xor);

    actual.release();
    bitwise_not(a, actual, mask);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_not);

    Mat alias = a.clone();
    bitwise_or(alias, b, alias, mask);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(alias, scalar_alias);

    Mat f32_a({2, 19}, CV_32FC1);
    Mat f32_b({2, 19}, CV_32FC1);
    Mat f32_mask({2, 19}, CV_8UC1);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 19; ++x)
        {
            const std::uint32_t a_bits =
                0x3F000000u + static_cast<std::uint32_t>(y * 19 + x) * 0x10101u;
            const std::uint32_t b_bits =
                0x7F00FF00u ^ static_cast<std::uint32_t>(y * 19 + x) * 0x010101u;
            set_raw_bits(f32_a.at<float>(y, x), &a_bits);
            set_raw_bits(f32_b.at<float>(y, x), &b_bits);
            f32_mask.at<uchar>(y, x) = x % 2 == 0 ? 255 : 0;
        }
    }
    Mat scalar_f32(f32_a.shape(), f32_a.type());
    scalar_f32.setTo(Scalar::all(1.0));
    {
        DispatchModeGuard scalar_guard(cpu::DispatchMode::ScalarOnly);
        bitwise_xor(f32_a, f32_b, scalar_f32, f32_mask);
    }
    Mat actual_f32(f32_a.shape(), f32_a.type());
    actual_f32.setTo(Scalar::all(1.0));
    bitwise_xor(f32_a, f32_b, actual_f32, f32_mask);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual_f32, scalar_f32);

    Mat wide_pixel_a({2, 19}, CV_64FC1);
    Mat wide_pixel_b({2, 19}, CV_64FC1);
    wide_pixel_a.setTo(Scalar::all(1.0));
    wide_pixel_b.setTo(Scalar::all(2.0));
    Mat wide_pixel_out;
    bitwise_and(wide_pixel_a, wide_pixel_b, wide_pixel_out, f32_mask);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

    Mat short_a({1, 5}, CV_8UC3);
    Mat short_b({1, 5}, CV_8UC3);
    Mat short_mask({1, 5}, CV_8UC1);
    short_a.setTo(Scalar::all(1.0));
    short_b.setTo(Scalar::all(2.0));
    short_mask.setTo(Scalar::all(255.0));
    Mat short_out;
    bitwise_and(short_a, short_b, short_out, short_mask);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
}

TEST(ArrayOpsContract_TEST, inrange_combines_channels_into_single_channel_mask)
{
    Mat src({1, 4}, CV_16SC3);
    const short values[4][3] = {
        {1, 10, 100},
        {2, 20, 200},
        {3, 30, 300},
        {4, 40, 400},
    };
    for (int x = 0; x < 4; ++x)
    {
        for (int ch = 0; ch < 3; ++ch)
        {
            src.at<short>(0, x, ch) = values[x][ch];
        }
    }

    Mat scalar_mask;
    inRange(src, Scalar(2.0, 15.0, 150.0), Scalar(4.0, 35.0, 350.0), scalar_mask);
    ASSERT_EQ(scalar_mask.type(), CV_8UC1);
    EXPECT_EQ(scalar_mask.at<uchar>(0, 0), 0);
    EXPECT_EQ(scalar_mask.at<uchar>(0, 1), 255);
    EXPECT_EQ(scalar_mask.at<uchar>(0, 2), 255);
    EXPECT_EQ(scalar_mask.at<uchar>(0, 3), 0);

    Mat lower(src.shape(), src.type());
    Mat upper(src.shape(), src.type());
    lower.setTo(Scalar(1.0, 10.0, 100.0));
    upper.setTo(Scalar(3.0, 30.0, 300.0));
    Mat mat_mask;
    inRange(src, lower, upper, mat_mask);
    EXPECT_EQ(mat_mask.at<uchar>(0, 0), 255);
    EXPECT_EQ(mat_mask.at<uchar>(0, 1), 255);
    EXPECT_EQ(mat_mask.at<uchar>(0, 2), 255);
    EXPECT_EQ(mat_mask.at<uchar>(0, 3), 0);
}

TEST(ArrayOpsContract_TEST, inrange_integer_scalar_uses_inclusive_fractional_bounds)
{
    Mat src({1, 5}, CV_32SC1);
    for (int x = 0; x < 5; ++x)
    {
        src.at<int>(0, x) = x + 1;
    }

    Mat mask;
    inRange(src, Scalar::all(2.5), Scalar::all(4.5), mask);
    const uchar expected[] = {0, 0, 255, 255, 0};
    for (int x = 0; x < 5; ++x)
    {
        EXPECT_EQ(mask.at<uchar>(0, x), expected[x]);
    }
}

TEST(ArrayOpsContract_TEST, ui_inrange_matches_scalar_for_bounds_roi_tail_and_edges)
{
    if (!detail::arithm_ui::enabled())
        GTEST_SKIP() << "OpenCV UI arithmetic requires NEON or SSE/AVX";

    Mat src_parent({4, 23}, CV_8UC3);
    Mat lower_parent({4, 23}, CV_8UC3);
    Mat upper_parent({4, 23}, CV_8UC3);
    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 23; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                src_parent.at<uchar>(y, x, ch) =
                    static_cast<uchar>((y * 61 + x * 17 + ch * 37) & 0xff);
                lower_parent.at<uchar>(y, x, ch) =
                    static_cast<uchar>(15 + ch * 20);
                upper_parent.at<uchar>(y, x, ch) =
                    static_cast<uchar>(190 + ch * 10);
            }
        }
    }

    Mat src = src_parent.colRange(2, 21);
    Mat lower = lower_parent.colRange(2, 21);
    Mat upper = upper_parent.colRange(2, 21);
    ASSERT_FALSE(src.isContinuous());

    Mat scalar_bounds;
    Mat mat_bounds;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        inRange(
            src,
            Scalar(-2.5, 20.25, 70.75),
            Scalar(127.5, 210.5, 300.0),
            scalar_bounds);
        inRange(src, lower, upper, mat_bounds);
    }

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    Mat actual;
    inRange(
        src,
        Scalar(-2.5, 20.25, 70.75),
        Scalar(127.5, 210.5, 300.0),
        actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_bounds);

    inRange(src, lower, upper, actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, mat_bounds);

    Mat f32({2, 19}, CV_32FC1);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 19; ++x)
        {
            f32.at<float>(y, x) =
                static_cast<float>(y * 19 + x - 12) / 3.0f;
        }
    }
    f32.at<float>(0, 0) = std::numeric_limits<float>::quiet_NaN();
    f32.at<float>(0, 1) = std::numeric_limits<float>::infinity();
    f32.at<float>(0, 2) = -std::numeric_limits<float>::infinity();
    actual.release();
    Mat scalar_f32;
    {
        DispatchModeGuard scalar_guard(cpu::DispatchMode::ScalarOnly);
        inRange(
            f32,
            Scalar::all(-std::numeric_limits<double>::infinity()),
            Scalar::all(std::numeric_limits<double>::infinity()),
            scalar_f32);
    }
    inRange(
        f32,
        Scalar::all(-std::numeric_limits<double>::infinity()),
        Scalar::all(std::numeric_limits<double>::infinity()),
        actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_f32);

    Mat u32({2, 19}, CV_32UC1);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 19; ++x)
        {
            u32.at<uint>(y, x) =
                static_cast<uint>(y * 100000 + x * 7919);
        }
    }
    Mat scalar_u32;
    {
        DispatchModeGuard scalar_guard(cpu::DispatchMode::ScalarOnly);
        inRange(u32, Scalar::all(1000.5), Scalar::all(110000.75), scalar_u32);
    }
    inRange(u32, Scalar::all(1000.5), Scalar::all(110000.75), actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_u32);

    Mat alias({2, 19}, CV_8UC1);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 19; ++x)
        {
            alias.at<uchar>(y, x) = static_cast<uchar>(x + y * 19);
        }
    }
    Mat alias_expected;
    {
        DispatchModeGuard scalar_guard(cpu::DispatchMode::ScalarOnly);
        inRange(alias, Scalar::all(5.5), Scalar::all(25.5), alias_expected);
    }
    inRange(alias, Scalar::all(5.5), Scalar::all(25.5), alias);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(alias, alias_expected);

    Mat f64({2, 19}, CV_64FC1);
    f64.setTo(Scalar::all(1.0));
    inRange(f64, Scalar::all(0.0), Scalar::all(2.0), actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

    Mat short_src({1, 5}, CV_8UC1);
    short_src.setTo(Scalar::all(1.0));
    actual.release();
    inRange(short_src, Scalar::all(0.0), Scalar::all(2.0), actual);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
}

TEST(ArrayOpsContract_TEST, public_ops_support_non_contiguous_roi_and_in_place_output)
{
    Mat a_base({3, 8}, CV_8UC3);
    Mat b_base({3, 8}, CV_8UC3);
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                a_base.at<uchar>(y, x, ch) = static_cast<uchar>(10 * y + x + ch);
                b_base.at<uchar>(y, x, ch) = static_cast<uchar>(2 * x + ch);
            }
        }
    }

    Mat a = a_base.colRange(1, 7);
    Mat b = b_base.colRange(1, 7);
    ASSERT_FALSE(a.isContinuous());
    ASSERT_FALSE(b.isContinuous());

    Mat expected;
    absdiff(a, b, expected);
    absdiff(a, b, a);
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 6; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                EXPECT_EQ(a.at<uchar>(y, x, ch), expected.at<uchar>(y, x, ch));
            }
        }
    }

    Mat range_mask;
    inRange(a, Scalar::all(0.0), Scalar::all(20.0), range_mask);
    ASSERT_EQ(range_mask.shape(), a.shape());
    ASSERT_EQ(range_mask.type(), CV_8UC1);
}

TEST(ArrayOpsContract_TEST, floating_numeric_edges_have_stable_operand_order_semantics)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    Mat a({1, 5}, CV_32FC1);
    Mat b({1, 5}, CV_32FC1);
    const float av[] = {nan, 1.0f, inf, -0.0f, 0.0f};
    const float bv[] = {2.0f, nan, inf, 0.0f, -0.0f};
    for (int x = 0; x < 5; ++x)
    {
        a.at<float>(0, x) = av[x];
        b.at<float>(0, x) = bv[x];
    }

    Mat out;
    min(a, b, out);
    EXPECT_TRUE(std::isnan(out.at<float>(0, 0)));
    EXPECT_TRUE(std::isnan(out.at<float>(0, 1)));
    EXPECT_TRUE(std::signbit(out.at<float>(0, 3)));
    EXPECT_FALSE(std::signbit(out.at<float>(0, 4)));

    max(a, b, out);
    EXPECT_TRUE(std::isnan(out.at<float>(0, 0)));
    EXPECT_TRUE(std::isnan(out.at<float>(0, 1)));
    EXPECT_FALSE(std::signbit(out.at<float>(0, 3)));
    EXPECT_FALSE(std::signbit(out.at<float>(0, 4)));

    absdiff(a, b, out);
    EXPECT_TRUE(std::isnan(out.at<float>(0, 0)));
    EXPECT_TRUE(std::isnan(out.at<float>(0, 1)));
    EXPECT_TRUE(std::isnan(out.at<float>(0, 2)));
    EXPECT_FALSE(std::signbit(out.at<float>(0, 3)));
    EXPECT_FALSE(std::signbit(out.at<float>(0, 4)));

    Mat a64({1, 2}, CV_64FC1);
    Mat b64({1, 2}, CV_64FC1);
    a64.at<double>(0, 0) = -std::numeric_limits<double>::infinity();
    a64.at<double>(0, 1) = -0.0;
    b64.at<double>(0, 0) = std::numeric_limits<double>::infinity();
    b64.at<double>(0, 1) = 0.0;
    out.release();
    absdiff(a64, b64, out);
    EXPECT_TRUE(std::isinf(out.at<double>(0, 0)));
    EXPECT_FALSE(std::signbit(out.at<double>(0, 1)));

    out.release();
    min(a64, b64, out);
    EXPECT_TRUE(std::isinf(out.at<double>(0, 0)));
    EXPECT_TRUE(std::signbit(out.at<double>(0, 0)));
    EXPECT_TRUE(std::signbit(out.at<double>(0, 1)));

    max(a64, b64, out);
    EXPECT_TRUE(std::isinf(out.at<double>(0, 0)));
    EXPECT_FALSE(std::signbit(out.at<double>(0, 0)));
    EXPECT_FALSE(std::signbit(out.at<double>(0, 1)));
}

TEST(ArrayOpsContract_TEST, invalid_shapes_types_and_masks_throw)
{
    Mat a({2, 3}, CV_8UC1);
    Mat wrong_shape({2, 4}, CV_8UC1);
    Mat wrong_type({2, 3}, CV_16UC1);
    Mat bad_mask({2, 3}, CV_8UC3);
    Mat out;

    EXPECT_THROW(absdiff(a, wrong_shape, out), Exception);
    EXPECT_THROW(min(a, wrong_type, out), Exception);
    EXPECT_THROW(bitwise_and(a, a, out, bad_mask), Exception);
    EXPECT_THROW(inRange(a, wrong_shape, wrong_shape, out), Exception);
}
