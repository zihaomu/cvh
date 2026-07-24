#include "cvh.h"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/detail/math_ui.hpp"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

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

std::uint16_t read_half_bits(const Mat& mat, int x)
{
    std::uint16_t bits = 0;
    std::memcpy(&bits, &mat.at<short>(0, x), sizeof(bits));
    return bits;
}

}  // namespace

TEST(MathOpsContract_TEST, scale_add_stays_scalar_after_ui_performance_gate)
{
    Mat a_parent({4, 23}, CV_32FC3);
    Mat b_parent({4, 23}, CV_32FC3);
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 23; ++col)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                const int index = (row * 23 + col) * 3 + ch;
                a_parent.at<float>(row, col, ch) =
                    static_cast<float>(index - 100) / 13.0f;
                b_parent.at<float>(row, col, ch) =
                    static_cast<float>((index * 7) % 83 - 41) / 9.0f;
            }
        }
    }

    Mat a = a_parent.colRange(2, 21);
    Mat b = b_parent.colRange(2, 21);
    ASSERT_FALSE(a.isContinuous());
    ASSERT_FALSE(b.isContinuous());

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    Mat expected;
    scaleAdd(a, 0.75, b, expected);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

    Mat src1_alias = a.clone();
    scaleAdd(src1_alias, 0.75, b, src1_alias);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    expect_mat_bytes_equal(src1_alias, expected);

    Mat src2_alias = b.clone();
    scaleAdd(a, 0.75, src2_alias, src2_alias);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    expect_mat_bytes_equal(src2_alias, expected);

    Mat a64({2, 19}, CV_64FC1);
    Mat b64({2, 19}, CV_64FC1);
    for (int row = 0; row < 2; ++row)
    {
        for (int col = 0; col < 19; ++col)
        {
            a64.at<double>(row, col) = (row * 19 + col - 15) / 7.0;
            b64.at<double>(row, col) = (row * 19 + col + 3) / 11.0;
        }
    }
    Mat actual64;
    scaleAdd(a64, 0.125, b64, actual64);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
}

TEST(MathOpsContract_TEST, ui_convert_scale_abs_matches_scalar_on_roi_tail_and_edges)
{
    if (!math_detail::ui::enabled())
        GTEST_SKIP() << "OpenCV UI math kernels require NEON or SSE/AVX";

    Mat parent({3, 23}, CV_32FC3);
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 23; ++col)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                const int index = (row * 23 + col) * 3 + ch;
                parent.at<float>(row, col, ch) =
                    static_cast<float>((index * 11) % 401 - 200) * 0.25f;
            }
        }
    }
    Mat src = parent.colRange(2, 21);
    ASSERT_FALSE(src.isContinuous());
    src.at<float>(0, 2, 0) = std::numeric_limits<float>::infinity();
    src.at<float>(0, 3, 1) = -std::numeric_limits<float>::infinity();
    src.at<float>(0, 4, 2) = std::numeric_limits<float>::quiet_NaN();

    Mat scalar_expected;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        convertScaleAbs(src, scalar_expected, -1.25, 2.0);
    }

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    Mat actual;
    convertScaleAbs(src, actual, -1.25, 2.0);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(actual, scalar_expected);

    Mat alias = src.clone();
    convertScaleAbs(alias, alias, -1.25, 2.0);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(alias, scalar_expected);
}

TEST(MathOpsContract_TEST, ui_convert_fp16_matches_scalar_in_both_directions)
{
    if (!math_detail::ui::enabled())
        GTEST_SKIP() << "OpenCV UI math kernels require NEON or SSE/AVX";

    static_assert(sizeof(cv::hfloat) == 2, "UI half layout must remain 16-bit");
    Mat parent({3, 23}, CV_32FC3);
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 23; ++col)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                const int index = (row * 23 + col) * 3 + ch;
                parent.at<float>(row, col, ch) =
                    static_cast<float>((index * 17) % 2001 - 1000) / 16.0f;
            }
        }
    }
    Mat src = parent.colRange(2, 21);
    ASSERT_FALSE(src.isContinuous());
    src.at<float>(0, 0, 0) = 0.0f;
    src.at<float>(0, 0, 1) = -0.0f;
    src.at<float>(0, 1, 0) = std::ldexp(1.0f, -24);
    src.at<float>(0, 1, 1) = 65504.0f;
    src.at<float>(0, 2, 0) = std::numeric_limits<float>::infinity();
    src.at<float>(0, 2, 1) = std::numeric_limits<float>::quiet_NaN();

    Mat scalar_half;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        convertFp16(src, scalar_half);
    }

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    Mat ui_half;
    convertFp16(src, ui_half);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(ui_half, scalar_half);

    Mat scalar_roundtrip;
    {
        DispatchModeGuard scalar_guard(cpu::DispatchMode::ScalarOnly);
        convertFp16(scalar_half, scalar_roundtrip);
    }
    Mat ui_roundtrip;
    convertFp16(ui_half, ui_roundtrip);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(ui_roundtrip, scalar_roundtrip);

    Mat alias = src.clone();
    convertFp16(alias, alias);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(alias, scalar_half);
}

TEST(MathOpsContract_TEST, ui_math_kernels_keep_uncovered_types_and_short_rows_scalar)
{
    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    Mat out;

    Mat short_f32({1, 3}, CV_32FC1);
    short_f32.setTo(Scalar::all(1.5));
    convertFp16(short_f32, out);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

    Mat short_f32_b({1, 3}, CV_32FC1);
    short_f32_b.setTo(Scalar::all(2.0));
    out.release();
    scaleAdd(short_f32, 0.75, short_f32_b, out);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

    Mat wide_u8({2, 37}, CV_8UC3);
    Mat wide_u8_b({2, 37}, CV_8UC3);
    wide_u8.setTo(Scalar(1, 2, 3));
    wide_u8_b.setTo(Scalar(4, 5, 6));
    out.release();
    scaleAdd(wide_u8, 2.0, wide_u8_b, out);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

    out.release();
    convertScaleAbs(wide_u8, out, 1.25, 3.0);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
}

TEST(MathOpsContract_TEST, scale_add_supports_multichannel_and_in_place)
{
    Mat a({1, 5}, CV_8UC3);
    Mat b({1, 5}, CV_8UC3);
    for (int x = 0; x < 5; ++x)
    {
        for (int ch = 0; ch < 3; ++ch)
        {
            a.at<uchar>(0, x, ch) = static_cast<uchar>(10 * x + ch);
            b.at<uchar>(0, x, ch) = static_cast<uchar>(100 + x + ch);
        }
    }

    Mat expected;
    scaleAdd(a, 2.0, b, expected);
    EXPECT_EQ(expected.at<uchar>(0, 0, 0), 100);
    EXPECT_EQ(expected.at<uchar>(0, 4, 2), 190);

    scaleAdd(a, 2.0, b, a);
    for (int x = 0; x < 5; ++x)
    {
        for (int ch = 0; ch < 3; ++ch)
        {
            EXPECT_EQ(a.at<uchar>(0, x, ch), expected.at<uchar>(0, x, ch));
        }
    }
}

TEST(MathOpsContract_TEST, convert_scale_abs_uses_even_rounding_and_saturation)
{
    Mat src({1, 9}, CV_32FC1);
    const float values[] = {
        -300.0f, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 254.5f, 300.0f,
    };
    for (int x = 0; x < 9; ++x)
    {
        src.at<float>(0, x) = values[x];
    }

    Mat dst;
    convertScaleAbs(src, dst);
    const uchar expected[] = {255, 2, 2, 0, 0, 2, 2, 254, 255};
    ASSERT_EQ(dst.type(), CV_8UC1);
    for (int x = 0; x < 9; ++x)
    {
        EXPECT_EQ(dst.at<uchar>(0, x), expected[x]);
    }

    convertScaleAbs(src, dst, -2.0, 1.0);
    EXPECT_EQ(dst.at<uchar>(0, 3), 2);
    EXPECT_EQ(dst.at<uchar>(0, 4), 0);

    Mat alias({1, 3}, CV_32FC1);
    alias.at<float>(0, 0) = -1.0f;
    alias.at<float>(0, 1) = 2.0f;
    alias.at<float>(0, 2) = 500.0f;
    convertScaleAbs(alias, alias);
    ASSERT_EQ(alias.type(), CV_8UC1);
    EXPECT_EQ(alias.at<uchar>(0, 0), 1);
    EXPECT_EQ(alias.at<uchar>(0, 1), 2);
    EXPECT_EQ(alias.at<uchar>(0, 2), 255);
}

TEST(MathOpsContract_TEST, convert_fp16_covers_special_and_subnormal_values)
{
    Mat src({1, 11}, CV_32FC1);
    const float denorm = std::ldexp(1.0f, -24);
    const float values[] = {
        0.0f,
        -0.0f,
        1.0f,
        -2.0f,
        65504.0f,
        std::ldexp(1.0f, -14),
        denorm,
        denorm * 0.25f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    for (int x = 0; x < 11; ++x)
    {
        src.at<float>(0, x) = values[x];
    }

    Mat fp16;
    convertFp16(src, fp16);
    ASSERT_EQ(fp16.type(), CV_16SC1);
    EXPECT_EQ(read_half_bits(fp16, 0), 0x0000u);
    EXPECT_EQ(read_half_bits(fp16, 1), 0x8000u);
    EXPECT_EQ(read_half_bits(fp16, 2), 0x3c00u);
    EXPECT_EQ(read_half_bits(fp16, 3), 0xc000u);
    EXPECT_EQ(read_half_bits(fp16, 4), 0x7bffu);
    EXPECT_EQ(read_half_bits(fp16, 5), 0x0400u);
    EXPECT_EQ(read_half_bits(fp16, 6), 0x0001u);
    EXPECT_EQ(read_half_bits(fp16, 7), 0x0000u);
    EXPECT_EQ(read_half_bits(fp16, 8), 0x7c00u);
    EXPECT_EQ(read_half_bits(fp16, 9), 0xfc00u);
    EXPECT_EQ(read_half_bits(fp16, 10) & 0x7c00u, 0x7c00u);
    EXPECT_NE(read_half_bits(fp16, 10) & 0x03ffu, 0u);

    Mat roundtrip;
    convertFp16(fp16, roundtrip);
    ASSERT_EQ(roundtrip.type(), CV_32FC1);
    EXPECT_FLOAT_EQ(roundtrip.at<float>(0, 2), 1.0f);
    EXPECT_FLOAT_EQ(roundtrip.at<float>(0, 3), -2.0f);
    EXPECT_FLOAT_EQ(roundtrip.at<float>(0, 4), 65504.0f);
    EXPECT_FLOAT_EQ(roundtrip.at<float>(0, 6), denorm);
    EXPECT_TRUE(std::isinf(roundtrip.at<float>(0, 8)));
    EXPECT_TRUE(std::isnan(roundtrip.at<float>(0, 10)));

    Mat native_half;
    src.convertTo(native_half, CV_16F);
    Mat native_roundtrip;
    convertFp16(native_half, native_roundtrip);
    EXPECT_FLOAT_EQ(native_roundtrip.at<float>(0, 2), 1.0f);
    EXPECT_TRUE(std::isnan(native_roundtrip.at<float>(0, 10)));
}

TEST(MathOpsContract_TEST, float_math_functions_cover_edges_and_power_sign_rule)
{
    Mat src({1, 7}, CV_32FC1);
    const float values[] = {
        0.0f,
        -0.0f,
        1.0f,
        4.0f,
        -4.0f,
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    for (int x = 0; x < 7; ++x)
    {
        src.at<float>(0, x) = values[x];
    }

    Mat out;
    sqrt(src, out);
    EXPECT_FLOAT_EQ(out.at<float>(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(out.at<float>(0, 3), 2.0f);
    EXPECT_TRUE(std::isnan(out.at<float>(0, 4)));
    EXPECT_TRUE(std::isinf(out.at<float>(0, 5)));

    pow(src, 3.0, out);
    EXPECT_FLOAT_EQ(out.at<float>(0, 4), -64.0f);
    pow(src, 0.5, out);
    EXPECT_TRUE(std::isnan(out.at<float>(0, 4)));

    exp(src, out);
    EXPECT_FLOAT_EQ(out.at<float>(0, 0), 1.0f);
    EXPECT_NEAR(out.at<float>(0, 2), std::exp(1.0f), 1e-6f);
    EXPECT_TRUE(std::isinf(out.at<float>(0, 5)));

    log(src, out);
    EXPECT_TRUE(std::isinf(out.at<float>(0, 0)));
    EXPECT_TRUE(std::signbit(out.at<float>(0, 0)));
    EXPECT_FLOAT_EQ(out.at<float>(0, 2), 0.0f);
    EXPECT_TRUE(std::isnan(out.at<float>(0, 4)));
}

TEST(MathOpsContract_TEST, double_math_functions_use_fixed_accuracy)
{
    Mat src({1, 4}, CV_64FC1);
    src.at<double>(0, 0) = 0.125;
    src.at<double>(0, 1) = 1.0;
    src.at<double>(0, 2) = 10.0;
    src.at<double>(0, 3) = 700.0;

    Mat out;
    sqrt(src, out);
    EXPECT_NEAR(out.at<double>(0, 0), std::sqrt(0.125), 1e-14);
    pow(src, 1.75, out);
    EXPECT_NEAR(out.at<double>(0, 2), std::pow(10.0, 1.75), 1e-12);
    exp(src, out);
    EXPECT_NEAR(out.at<double>(0, 1), std::exp(1.0), 1e-14);
    EXPECT_TRUE(std::isfinite(out.at<double>(0, 3)));
    log(src, out);
    EXPECT_NEAR(out.at<double>(0, 2), std::log(10.0), 1e-14);
}

TEST(MathOpsContract_TEST, unary_math_and_fp16_support_non_contiguous_roi_and_in_place)
{
    Mat parent({2, 7}, CV_32FC1);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 7; ++x)
        {
            parent.at<float>(y, x) = 0.25f + static_cast<float>(y * 7 + x) * 0.1f;
        }
    }
    Mat roi = parent.colRange(1, 6);
    ASSERT_FALSE(roi.isContinuous());

    Mat original = roi.clone();
    sqrt(roi, roi);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 5; ++x)
        {
            EXPECT_NEAR(
                roi.at<float>(y, x), std::sqrt(original.at<float>(y, x)), 1e-6f);
        }
    }

    original.copyTo(roi);
    pow(roi, 2.0, roi);
    EXPECT_NEAR(
        roi.at<float>(1, 4),
        original.at<float>(1, 4) * original.at<float>(1, 4),
        1e-6f);

    original.copyTo(roi);
    exp(roi, roi);
    EXPECT_NEAR(roi.at<float>(0, 2), std::exp(original.at<float>(0, 2)), 1e-6f);

    original.copyTo(roi);
    log(roi, roi);
    EXPECT_NEAR(roi.at<float>(1, 1), std::log(original.at<float>(1, 1)), 1e-6f);

    Mat fp16;
    convertFp16(original, fp16);
    ASSERT_EQ(fp16.type(), CV_16SC1);
    ASSERT_EQ(fp16.shape(), original.shape());
    Mat roundtrip;
    convertFp16(fp16, roundtrip);
    EXPECT_NEAR(roundtrip.at<float>(1, 3), original.at<float>(1, 3), 1e-3f);
}

TEST(MathOpsContract_TEST, check_range_reports_first_bad_pixel_and_upper_is_exclusive)
{
    Mat src({2, 4}, CV_32FC3);
    src.setTo(Scalar(1.0, 2.0, 3.0));
    src.at<float>(1, 2, 1) = std::numeric_limits<float>::quiet_NaN();
    src.at<float>(1, 3, 0) = 100.0f;

    Point position(-1, -1);
    EXPECT_FALSE(checkRange(src, true, &position, 0.0, 10.0));
    EXPECT_EQ(position, Point(2, 1));
    EXPECT_THROW(checkRange(src, false, nullptr, 0.0, 10.0), Exception);

    src.at<float>(1, 2, 1) = 2.0f;
    src.at<float>(1, 3, 0) = 10.0f;
    EXPECT_FALSE(checkRange(src, true, &position, 0.0, 10.0));
    EXPECT_EQ(position, Point(3, 1));

    src.at<float>(1, 3, 0) = 9.0f;
    EXPECT_TRUE(checkRange(src, true, &position, 0.0, 10.0));
}

TEST(MathOpsContract_TEST, check_range_and_patch_nans_cover_roi_and_type_errors)
{
    Mat parent({3, 7}, CV_32FC1);
    parent.setTo(Scalar::all(1.0));
    Mat roi = parent.colRange(1, 6);
    ASSERT_FALSE(roi.isContinuous());
    roi.at<float>(1, 3) = std::numeric_limits<float>::quiet_NaN();
    roi.at<float>(2, 4) = std::numeric_limits<float>::infinity();

    Point position;
    EXPECT_FALSE(checkRange(roi, true, &position));
    EXPECT_EQ(position, Point(3, 1));

    patchNaNs(roi, -7.0);
    EXPECT_FLOAT_EQ(roi.at<float>(1, 3), -7.0f);
    EXPECT_TRUE(std::isinf(roi.at<float>(2, 4)));
    EXPECT_FALSE(checkRange(roi));

    Mat f64({1, 2}, CV_64FC1);
    f64.at<double>(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(patchNaNs(f64), Exception);
}

TEST(MathOpsContract_TEST, invalid_math_inputs_throw)
{
    Mat u8({1, 3}, CV_8UC1);
    Mat f32({1, 3}, CV_32FC1);
    Mat out;

    EXPECT_THROW(scaleAdd(u8, 1.0, f32, out), Exception);
    EXPECT_THROW(sqrt(u8, out), Exception);
    EXPECT_THROW(exp(u8, out), Exception);
    EXPECT_THROW(log(u8, out), Exception);
    EXPECT_THROW(convertFp16(u8, out), Exception);
}
