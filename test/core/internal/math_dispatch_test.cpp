#include "test/core/support/math_internal_test_utils.hpp"

TEST(MathDispatchInternalTest, scale_add_stays_scalar_after_ui_performance_gate)
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

TEST(MathDispatchInternalTest, ui_convert_scale_abs_matches_scalar_on_roi_tail_and_edges)
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

TEST(MathDispatchInternalTest, ui_convert_fp16_matches_scalar_in_both_directions)
{
    if (!math_detail::ui::enabled())
        GTEST_SKIP() << "OpenCV UI math kernels require NEON or SSE/AVX";

#if CVH_ENABLE_OPENCV_INTRIN
    static_assert(sizeof(cv::hfloat) == 2, "UI half layout must remain 16-bit");
#endif
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

TEST(MathDispatchInternalTest, ui_math_kernels_keep_uncovered_types_and_short_rows_scalar)
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

    exp(short_f32, out);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    log(short_f32, out);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    pow(short_f32, 1.75, out);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    patchNaNs(short_f32);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
}

TEST(MathDispatchInternalTest, ui_patch_nans_preserves_non_nan_bits_on_roi_and_tail)
{
    if (!math_detail::ui::enabled())
        GTEST_SKIP() << "OpenCV UI math kernels require NEON or SSE/AVX";

    Mat parent({3, 23}, CV_32FC3);
    for (int row = 0; row < parent.size[0]; ++row)
    {
        for (int col = 0; col < parent.size[1]; ++col)
        {
            for (int ch = 0; ch < parent.channels(); ++ch)
            {
                parent.at<float>(row, col, ch) =
                    static_cast<float>((row * 23 + col) * 3 + ch - 50) / 7.0f;
            }
        }
    }
    Mat src = parent.colRange(2, 21);
    ASSERT_FALSE(src.isContinuous());
    src.at<float>(0, 0, 0) = std::numeric_limits<float>::infinity();
    src.at<float>(0, 1, 1) = -std::numeric_limits<float>::infinity();
    src.at<float>(0, 2, 2) = -0.0f;
    const std::uint32_t nan_patterns[] = {
        0x7fc01234U,
        0xffc05678U,
        0x7f800001U,
    };
    for (int index = 0; index < 3; ++index)
    {
        std::memcpy(
            &src.at<float>(1, 3 + index, index),
            &nan_patterns[index],
            sizeof(nan_patterns[index]));
    }

    Mat scalar_expected = src.clone();
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        patchNaNs(scalar_expected, -7.25);
    }

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    patchNaNs(src, -7.25);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    expect_mat_bytes_equal(src, scalar_expected);
}

TEST(MathDispatchInternalTest, ui_exp_and_log_match_scalar_on_roi_alias_and_special_values)
{
    if (!math_detail::ui::enabled())
        GTEST_SKIP() << "OpenCV UI math kernels require NEON or SSE/AVX";

    Mat exp_parent({3, 23}, CV_32FC3);
    for (int row = 0; row < exp_parent.size[0]; ++row)
    {
        for (int col = 0; col < exp_parent.size[1]; ++col)
        {
            for (int ch = 0; ch < exp_parent.channels(); ++ch)
            {
                const int index = (row * 23 + col) * 3 + ch;
                exp_parent.at<float>(row, col, ch) =
                    static_cast<float>((index % 97) - 48) / 8.0f;
            }
        }
    }
    Mat exp_src = exp_parent.colRange(2, 21);
    ASSERT_FALSE(exp_src.isContinuous());
    exp_src.at<float>(0, 0, 0) = -std::numeric_limits<float>::infinity();
    exp_src.at<float>(0, 1, 1) = std::numeric_limits<float>::infinity();
    exp_src.at<float>(0, 2, 2) = std::numeric_limits<float>::quiet_NaN();
    exp_src.at<float>(0, 3, 0) = -104.0f;
    exp_src.at<float>(0, 4, 1) = 90.0f;

    Mat scalar_exp;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        exp(exp_src, scalar_exp);
    }
    Mat actual_exp;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        exp(exp_src, actual_exp);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    }
    expect_float_mat_near(actual_exp, scalar_exp, 2e-6f);

    Mat exp_alias = exp_src.clone();
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        exp(exp_alias, exp_alias);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    }
    expect_float_mat_near(exp_alias, scalar_exp, 2e-6f);

    Mat log_src = exp_src.clone();
    for (int row = 0; row < log_src.size[0]; ++row)
    {
        for (int col = 0; col < log_src.size[1]; ++col)
        {
            for (int ch = 0; ch < log_src.channels(); ++ch)
            {
                log_src.at<float>(row, col, ch) =
                    std::fabs(log_src.at<float>(row, col, ch)) + 0.125f;
            }
        }
    }
    log_src.at<float>(0, 0, 0) = 0.0f;
    log_src.at<float>(0, 1, 1) = -1.0f;
    log_src.at<float>(0, 2, 2) = std::numeric_limits<float>::infinity();
    log_src.at<float>(0, 3, 0) = std::numeric_limits<float>::quiet_NaN();
    log_src.at<float>(0, 4, 1) = std::numeric_limits<float>::denorm_min();

    Mat scalar_log;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        log(log_src, scalar_log);
    }
    Mat actual_log;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        log(log_src, actual_log);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    }
    expect_float_mat_near(actual_log, scalar_log, 2e-6f);

    Mat log_alias = log_src.clone();
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        log(log_alias, log_alias);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    }
    expect_float_mat_near(log_alias, scalar_log, 2e-6f);
}

TEST(MathDispatchInternalTest, ui_pow_splits_integer_and_generic_f32_paths)
{
    if (!math_detail::ui::enabled())
        GTEST_SKIP() << "OpenCV UI math kernels require NEON or SSE/AVX";

    Mat parent({3, 23}, CV_32FC3);
    for (int row = 0; row < parent.size[0]; ++row)
    {
        for (int col = 0; col < parent.size[1]; ++col)
        {
            for (int ch = 0; ch < parent.channels(); ++ch)
            {
                const int index = (row * 23 + col) * 3 + ch;
                parent.at<float>(row, col, ch) =
                    static_cast<float>((index % 41) - 20) / 8.0f;
            }
        }
    }
    Mat src = parent.colRange(2, 21);
    ASSERT_FALSE(src.isContinuous());
    src.at<float>(0, 0, 0) = 0.0f;
    src.at<float>(0, 1, 1) = -0.0f;
    src.at<float>(0, 2, 2) = std::numeric_limits<float>::infinity();
    src.at<float>(0, 3, 0) = -std::numeric_limits<float>::infinity();
    src.at<float>(0, 4, 1) = std::numeric_limits<float>::quiet_NaN();

    for (const double exponent : {3.0, -3.0})
    {
        Mat scalar_expected;
        {
            DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
            pow(src, exponent, scalar_expected);
        }
        Mat actual;
        {
            DispatchModeGuard guard(cpu::DispatchMode::Auto);
            pow(src, exponent, actual);
            EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
        }
        expect_float_mat_near(actual, scalar_expected, 2e-6f);
    }

    Mat positive = src.clone();
    for (int row = 0; row < positive.size[0]; ++row)
    {
        for (int col = 0; col < positive.size[1]; ++col)
        {
            for (int ch = 0; ch < positive.channels(); ++ch)
            {
                positive.at<float>(row, col, ch) =
                    std::fabs(positive.at<float>(row, col, ch)) + 0.25f;
            }
        }
    }
    positive.at<float>(0, 0, 0) = -1.0f;
    positive.at<float>(0, 1, 1) = 0.0f;
    positive.at<float>(0, 2, 2) = std::numeric_limits<float>::infinity();
    positive.at<float>(0, 3, 0) = std::numeric_limits<float>::quiet_NaN();
    positive.at<float>(0, 4, 1) = std::numeric_limits<float>::denorm_min();

    Mat scalar_generic;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        pow(positive, 1.75, scalar_generic);
    }
    Mat actual_generic;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        pow(positive, 1.75, actual_generic);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    }
    expect_float_mat_near(actual_generic, scalar_generic, 4e-6f);

    Mat alias = positive.clone();
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        pow(alias, 1.75, alias);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
    }
    expect_float_mat_near(alias, scalar_generic, 4e-6f);

    Mat f64({2, 19}, CV_64FC1);
    f64.setTo(Scalar::all(1.25));
    Mat f64_out;
    pow(f64, 1.75, f64_out);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
}
