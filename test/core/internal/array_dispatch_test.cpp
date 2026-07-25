#include "test/core/support/array_internal_test_utils.hpp"

TEST(ArrayDispatchInternalTest, ui_arithmetic_matches_scalar_for_all_integer_depths)
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

TEST(ArrayDispatchInternalTest, ui_basic_arithmetic_matches_scalar_for_all_integer_depths)
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

TEST(ArrayDispatchInternalTest, ui_mat_scalar_arithmetic_matches_scalar_for_broadcast_types)
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

TEST(ArrayDispatchInternalTest, ui_basic_float_arithmetic_matches_scalar_including_divide_edges)
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

TEST(ArrayDispatchInternalTest, ui_arithmetic_matches_scalar_on_tail_roi_and_in_place_inputs)
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

TEST(ArrayDispatchInternalTest, ui_arithmetic_keeps_short_rows_and_unsupported_types_on_scalar)
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

TEST(ArrayDispatchInternalTest, ui_masked_bitwise_matches_scalar_for_roi_alias_and_raw_bits)
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

TEST(ArrayDispatchInternalTest, ui_inrange_matches_scalar_for_bounds_roi_tail_and_edges)
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
