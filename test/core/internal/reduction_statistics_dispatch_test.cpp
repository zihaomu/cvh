#include "test/core/support/reduction_internal_test_utils.hpp"

TEST(ReductionStatisticsDispatchInternalTest,
     statistics_ui_matches_scalar_across_depths_channels_roi_and_tail)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }
    const int depths[] = {
        CV_8U,
        CV_8S,
        CV_16U,
        CV_16S,
        CV_32S,
        CV_32U,
        CV_16F,
        CV_32F,
        CV_64F,
    };

    for (const int depth : depths)
    {
        for (int channels = 1; channels <= 4; ++channels)
        {
            Mat parent({3, 43}, CV_MAKETYPE(depth, channels));
            for (int row = 0; row < parent.size.p[0]; ++row)
            {
                for (int column = 0; column < parent.size.p[1]; ++column)
                {
                    for (int channel = 0; channel < channels; ++channel)
                    {
                        const int seed =
                            row * 19 + column * 7 + channel * 3;
                        double value = static_cast<double>(seed % 23 + 1);
                        if (depth == CV_8S || depth == CV_16S ||
                            depth == CV_32S)
                        {
                            value = static_cast<double>(seed % 23 - 11);
                        }
                        if (depth == CV_16F || depth == CV_32F ||
                            depth == CV_64F)
                        {
                            value = static_cast<double>(seed % 23 - 11) *
                                    0.25;
                        }
                        set_test_value(
                            parent, row, column, channel, value);
                    }
                }
            }
            Mat src = parent.colRange(1, 42);
            ASSERT_FALSE(src.isContinuous());

            Scalar scalar_sum;
            Scalar scalar_mean;
            Scalar scalar_mean_value;
            Scalar scalar_stddev;
            {
                DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
                scalar_sum = sum(src);
                scalar_mean = mean(src);
                meanStdDev(src, scalar_mean_value, scalar_stddev);
                EXPECT_EQ(
                    cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
            }

            DispatchModeGuard guard(cpu::DispatchMode::Auto);
            cpu::reset_last_dispatch_tag();
            const Scalar auto_sum = sum(src);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                expected_statistics_auto_tag(depth, ui_enabled));
            cpu::reset_last_dispatch_tag();
            const Scalar auto_mean = mean(src);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                expected_statistics_auto_tag(depth, ui_enabled));
            Scalar auto_mean_value;
            Scalar auto_stddev;
            cpu::reset_last_dispatch_tag();
            meanStdDev(src, auto_mean_value, auto_stddev);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                expected_statistics_auto_tag(depth, ui_enabled));

            const bool floating =
                depth == CV_16F || depth == CV_32F || depth == CV_64F;
            const double absolute_tolerance =
                depth == CV_64F ? 1e-12 : (floating ? 1e-6 : 0.0);
            const double relative_tolerance =
                depth == CV_64F ? 1e-12 : (floating ? 1e-6 : 0.0);
            expect_scalar_close(
                auto_sum,
                scalar_sum,
                channels,
                absolute_tolerance,
                relative_tolerance);
            const double statistics_absolute_tolerance =
                floating ? absolute_tolerance : 1e-12;
            const double statistics_relative_tolerance =
                floating ? relative_tolerance : 1e-12;
            expect_scalar_close(
                auto_mean,
                scalar_mean,
                channels,
                statistics_absolute_tolerance,
                statistics_relative_tolerance);
            expect_scalar_close(
                auto_mean_value,
                scalar_mean_value,
                channels,
                statistics_absolute_tolerance,
                statistics_relative_tolerance);
            expect_scalar_close(
                auto_stddev,
                scalar_stddev,
                channels,
                statistics_absolute_tolerance,
                statistics_relative_tolerance);
        }
    }
}

TEST(ReductionStatisticsDispatchInternalTest,
     statistics_ui_masks_cover_empty_full_sparse_and_c1_to_c4)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }

    for (int channels = 1; channels <= 4; ++channels)
    {
        Mat src_parent({3, 47}, CV_MAKETYPE(CV_32F, channels));
        Mat mask_parent({3, 47}, CV_8UC1);
        for (int row = 0; row < src_parent.size.p[0]; ++row)
        {
            for (int column = 0; column < src_parent.size.p[1]; ++column)
            {
                for (int channel = 0; channel < channels; ++channel)
                {
                    set_test_value(
                        src_parent,
                        row,
                        column,
                        channel,
                        (row * 47 + column + channel * 0.25) * 0.5);
                }
            }
        }
        Mat src = src_parent.colRange(1, 46);
        Mat mask = mask_parent.colRange(1, 46);
        ASSERT_FALSE(src.isContinuous());
        ASSERT_FALSE(mask.isContinuous());

        for (int distribution = 0; distribution < 3; ++distribution)
        {
            mask.setTo(Scalar::all(distribution == 1 ? 255.0 : 0.0));
            if (distribution == 2)
            {
                for (int row = 0; row < mask.size.p[0]; ++row)
                {
                    for (int column = 0; column < mask.size.p[1]; ++column)
                    {
                        mask.at<uchar>(row, column) =
                            ((row * mask.size.p[1] + column) % 5) == 0
                            ? 255
                            : 0;
                    }
                }
            }

            Scalar scalar_mean;
            Scalar scalar_mean_value;
            Scalar scalar_stddev;
            {
                DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
                scalar_mean = mean(src, mask);
                meanStdDev(
                    src, scalar_mean_value, scalar_stddev, mask);
            }

            DispatchModeGuard guard(cpu::DispatchMode::Auto);
            cpu::reset_last_dispatch_tag();
            const Scalar auto_mean = mean(src, mask);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                expected_statistics_auto_tag(CV_32F, ui_enabled));
            Scalar auto_mean_value;
            Scalar auto_stddev;
            cpu::reset_last_dispatch_tag();
            meanStdDev(
                src, auto_mean_value, auto_stddev, mask);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                expected_statistics_auto_tag(CV_32F, ui_enabled));

            expect_scalar_close(
                auto_mean, scalar_mean, channels, 1e-6, 1e-6);
            expect_scalar_close(
                auto_mean_value,
                scalar_mean_value,
                channels,
                1e-6,
                1e-6);
            expect_scalar_close(
                auto_stddev,
                scalar_stddev,
                channels,
                1e-6,
                1e-6);
        }
    }
}

TEST(ReductionStatisticsDispatchInternalTest, statistics_ui_integer_sum_is_exact)
{
    Mat unsigned_values({1, 257}, CV_32UC1);
    Mat signed_values({1, 257}, CV_32SC1);
    for (int column = 0; column < 257; ++column)
    {
        unsigned_values.at<uint>(0, column) =
            std::numeric_limits<uint>::max() -
            static_cast<uint>(column);
        signed_values.at<int>(0, column) =
            (column % 2 == 0)
            ? std::numeric_limits<int>::max() - column
            : std::numeric_limits<int>::min() + column;
    }

    Scalar scalar_unsigned;
    Scalar scalar_signed;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        scalar_unsigned = sum(unsigned_values);
        scalar_signed = sum(signed_values);
    }
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        EXPECT_DOUBLE_EQ(sum(unsigned_values)[0], scalar_unsigned[0]);
        EXPECT_DOUBLE_EQ(sum(signed_values)[0], scalar_signed[0]);
    }
}

TEST(ReductionStatisticsDispatchInternalTest,
     statistics_ui_preserves_stability_special_values_and_fallback)
{
    Mat stable({1, 257}, CV_64FC1);
    for (int column = 0; column < stable.size.p[1]; ++column)
    {
        stable.at<double>(0, column) =
            1.0e12 + static_cast<double>(column % 5 - 2) * 0.001;
    }

    Scalar scalar_mean;
    Scalar scalar_stddev;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        meanStdDev(stable, scalar_mean, scalar_stddev);
    }
    Scalar auto_mean;
    Scalar auto_stddev;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        meanStdDev(stable, auto_mean, auto_stddev);
    }
    EXPECT_NEAR(auto_mean[0], scalar_mean[0], 2e-4);
    EXPECT_NEAR(auto_stddev[0], scalar_stddev[0], 2e-5);
    EXPECT_GT(auto_stddev[0], 0.0);

    Mat special({1, 33}, CV_32FC1);
    special.setTo(Scalar::all(1.0));
    special.at<float>(0, 7) = std::numeric_limits<float>::infinity();
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        const Scalar special_sum = sum(special);
        EXPECT_TRUE(std::isinf(special_sum[0]));
        meanStdDev(special, auto_mean, auto_stddev);
        EXPECT_FALSE(std::isfinite(auto_stddev[0]));
    }
    special.at<float>(0, 7) =
        std::numeric_limits<float>::quiet_NaN();
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        EXPECT_TRUE(std::isnan(sum(special)[0]));
        meanStdDev(special, auto_mean, auto_stddev);
        EXPECT_TRUE(std::isnan(auto_mean[0]));
        EXPECT_TRUE(std::isnan(auto_stddev[0]));
    }

    Mat short_row({1, 1}, CV_32FC4);
    short_row.setTo(Scalar(1.0, 2.0, 3.0, 4.0));
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        EXPECT_EQ(sum(short_row), Scalar(1.0, 2.0, 3.0, 4.0));
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    }
}
