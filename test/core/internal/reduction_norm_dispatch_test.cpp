#include "test/core/support/reduction_internal_test_utils.hpp"

TEST(ReductionNormDispatchInternalTest,
     norm_ui_matches_scalar_for_single_diff_mask_channels_roi_and_tail)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }
    const int depths[] = {CV_8U, CV_16S, CV_32F};
    const int norm_types[] = {NORM_INF, NORM_L1, NORM_L2};

    for (const int depth : depths)
    {
        for (int channels = 1; channels <= 4; ++channels)
        {
            Mat first_parent({3, 71}, CV_MAKETYPE(depth, channels));
            Mat second_parent({3, 71}, CV_MAKETYPE(depth, channels));
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 71; ++column)
                {
                    for (int channel = 0; channel < channels; ++channel)
                    {
                        const int seed =
                            row * 31 + column * 7 + channel * 5;
                        double first_value =
                            static_cast<double>(seed % 41 + 1);
                        double second_value =
                            static_cast<double>((seed * 3) % 37 + 2);
                        if (depth != CV_8U)
                        {
                            first_value =
                                static_cast<double>(seed % 41 - 20);
                            second_value =
                                static_cast<double>((seed * 3) % 37 - 18);
                        }
                        if (depth == CV_32F)
                        {
                            first_value *= 0.25;
                            second_value *= 0.125;
                        }
                        set_test_value(
                            first_parent,
                            row,
                            column,
                            channel,
                            first_value);
                        set_test_value(
                            second_parent,
                            row,
                            column,
                            channel,
                            second_value);
                    }
                }
            }
            Mat first = first_parent.colRange(2, 69);
            Mat second = second_parent.colRange(2, 69);
            ASSERT_FALSE(first.isContinuous());
            ASSERT_FALSE(second.isContinuous());

            Mat mask({3, 67}, CV_8UC1);
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 67; ++column)
                {
                    mask.at<uchar>(row, column) =
                        (column < 29 || column >= 36) ? 255 : 0;
                }
            }

            for (const int norm_type : norm_types)
            {
                for (int masked = 0; masked <= 1; ++masked)
                {
                    const Mat& active_mask = masked != 0 ? mask : Mat();
                    double scalar_single = 0.0;
                    double scalar_diff = 0.0;
                    {
                        DispatchModeGuard guard(
                            cpu::DispatchMode::ScalarOnly);
                        scalar_single =
                            norm(first, norm_type, active_mask);
                        scalar_diff =
                            norm(first, second, norm_type, active_mask);
                    }

                    DispatchModeGuard guard(cpu::DispatchMode::Auto);
                    cpu::reset_last_dispatch_tag();
                    const double auto_single =
                        norm(first, norm_type, active_mask);
                    EXPECT_EQ(
                        cpu::last_dispatch_tag(),
                        expected_norm_auto_tag(depth, ui_enabled));
                    cpu::reset_last_dispatch_tag();
                    const double auto_diff =
                        norm(first, second, norm_type, active_mask);
                    EXPECT_EQ(
                        cpu::last_dispatch_tag(),
                        expected_norm_auto_tag(depth, ui_enabled));

                    const double relative_tolerance =
                        depth == CV_32F ? 1e-6 : 0.0;
                    EXPECT_NEAR(
                        auto_single,
                        scalar_single,
                        relative_tolerance *
                            std::max(1.0, std::fabs(scalar_single)));
                    EXPECT_NEAR(
                        auto_diff,
                        scalar_diff,
                        relative_tolerance *
                            std::max(1.0, std::fabs(scalar_diff)));
                }
            }
        }
    }

    Mat short_row({1, 3}, CV_8UC1);
    short_row.setTo(Scalar::all(2.0));
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        EXPECT_DOUBLE_EQ(norm(short_row, NORM_L1), 6.0);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    }
}

TEST(ReductionNormDispatchInternalTest,
     norm_ui_preserves_nan_inf_and_wide_difference_semantics)
{
    Mat first({1, 33}, CV_32FC1);
    Mat second({1, 33}, CV_32FC1);
    first.setTo(Scalar::all(2.0));
    second.setTo(Scalar::all(-3.0));
    first.at<float>(0, 7) =
        std::numeric_limits<float>::infinity();
    second.at<float>(0, 7) =
        std::numeric_limits<float>::infinity();
    first.at<float>(0, 19) =
        std::numeric_limits<float>::quiet_NaN();

    EXPECT_TRUE(std::isnan(norm(first, NORM_INF)));
    EXPECT_TRUE(std::isnan(norm(first, NORM_L1)));
    EXPECT_TRUE(std::isnan(norm(first, NORM_L2)));
    EXPECT_TRUE(std::isnan(norm(first, second, NORM_INF)));
    EXPECT_TRUE(std::isnan(norm(first, second, NORM_L1)));
    EXPECT_TRUE(std::isnan(norm(first, second, NORM_L2)));

    Mat u8_low({1, 65}, CV_8UC1);
    Mat u8_high({1, 65}, CV_8UC1);
    u8_low.setTo(Scalar::all(0.0));
    u8_high.setTo(Scalar::all(255.0));
    EXPECT_DOUBLE_EQ(norm(u8_low, u8_high, NORM_INF), 255.0);
    EXPECT_DOUBLE_EQ(norm(u8_low, u8_high, NORM_L1), 65.0 * 255.0);
    EXPECT_DOUBLE_EQ(
        norm(u8_low, u8_high, NORM_L2),
        std::sqrt(65.0 * 255.0 * 255.0));

    Mat f32_high({1, 33}, CV_32FC1);
    Mat f32_low({1, 33}, CV_32FC1);
    f32_high.setTo(Scalar::all(
        std::numeric_limits<float>::max()));
    f32_low.setTo(Scalar::all(
        -std::numeric_limits<float>::max()));
    const double wide_difference =
        2.0 * static_cast<double>(
                  std::numeric_limits<float>::max());
    EXPECT_DOUBLE_EQ(
        norm(f32_high, f32_low, NORM_INF),
        wide_difference);
    EXPECT_NEAR(
        norm(f32_high, f32_low, NORM_L1),
        33.0 * wide_difference,
        1e-12 * 33.0 * wide_difference);
    EXPECT_NEAR(
        norm(f32_high, f32_low, NORM_L2),
        std::sqrt(33.0) * wide_difference,
        1e-12 * std::sqrt(33.0) * wide_difference);
}

TEST(ReductionNormDispatchInternalTest,
     normalize_ui_matches_scalar_for_mask_alias_dtype_and_constant_minmax)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }

    Mat parent({3, 71}, CV_32FC3);
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 71; ++column)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                set_test_value(
                    parent,
                    row,
                    column,
                    channel,
                    static_cast<double>(
                        row * 17 + column * 3 + channel - 40) *
                        0.125);
            }
        }
    }
    Mat src = parent.colRange(2, 69);
    ASSERT_FALSE(src.isContinuous());
    Mat mask({3, 67}, CV_8UC1);
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 67; ++column)
        {
            mask.at<uchar>(row, column) =
                (column < 31 || column >= 39) ? 1 : 0;
        }
    }

    for (const int norm_type : {NORM_INF, NORM_L1, NORM_L2})
    {
        Mat scalar_dst(src.shape(), src.type());
        Mat auto_dst(src.shape(), src.type());
        scalar_dst.setTo(Scalar::all(9.0));
        auto_dst.setTo(Scalar::all(9.0));
        {
            DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
            normalize(
                src,
                scalar_dst,
                3.0,
                0.0,
                norm_type,
                -1,
                mask);
        }
        {
            DispatchModeGuard guard(cpu::DispatchMode::Auto);
            cpu::reset_last_dispatch_tag();
            normalize(
                src,
                auto_dst,
                3.0,
                0.0,
                norm_type,
                -1,
                mask);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                expected_norm_auto_tag(CV_32F, ui_enabled));
        }
        expect_f32_mat_close(auto_dst, scalar_dst, 2e-6f);
    }

    Mat scalar_alias = src.clone();
    Mat auto_alias = src.clone();
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        normalize(
            scalar_alias,
            scalar_alias,
            -2.0,
            5.0,
            NORM_MINMAX);
    }
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        normalize(
            auto_alias,
            auto_alias,
            -2.0,
            5.0,
            NORM_MINMAX);
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            expected_norm_auto_tag(CV_32F, ui_enabled));
    }
    expect_f32_mat_close(auto_alias, scalar_alias, 2e-6f);

    Mat converted;
    normalize(src, converted, 2.0, 0.0, NORM_L2, CV_64F);
    EXPECT_EQ(converted.type(), CV_64FC3);

    Mat constant({2, 65}, CV_32FC1);
    constant.setTo(Scalar::all(7.0));
    Mat constant_dst;
    normalize(
        constant,
        constant_dst,
        -4.0,
        9.0,
        NORM_MINMAX);
    for (size_t index = 0; index < constant_dst.total(); ++index)
    {
        EXPECT_FLOAT_EQ(
            reinterpret_cast<const float*>(constant_dst.data)[index],
            -4.0f);
    }
}
