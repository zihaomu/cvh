#include "test/core/support/reduction_internal_test_utils.hpp"

TEST(ReductionAxisDispatchInternalTest,
     reduce_ui_matches_scalar_across_axes_rtypes_channels_roi_and_tail)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }

    const int depths[] = {CV_8U, CV_32F};
    const int channel_counts[] = {1, 3};
    const int rtypes[] = {
        REDUCE_SUM,
        REDUCE_AVG,
        REDUCE_MAX,
        REDUCE_MIN,
        REDUCE_SUM2,
    };
    for (int depth : depths)
    {
        const int lanes = reduce_test_lanes(depth);
        for (int channels : channel_counts)
        {
            Mat parent(
                {7, 2 * lanes + 5},
                CV_MAKETYPE(depth, channels));
            for (int row = 0; row < parent.size.p[0]; ++row)
            {
                for (int column = 0;
                     column < parent.size.p[1];
                     ++column)
                {
                    for (int channel = 0;
                         channel < channels;
                         ++channel)
                    {
                        const int seed =
                            17 * row + 7 * column + 3 * channel;
                        const double value = depth == CV_8U
                            ? static_cast<double>(seed % 31)
                            : static_cast<double>(
                                  (seed % 37) - 18) /
                                  8.0;
                        set_test_value(
                            parent,
                            row,
                            column,
                            channel,
                            value);
                    }
                }
            }
            Mat src = parent.colRange(1, parent.size.p[1] - 1);
            ASSERT_FALSE(src.isContinuous());

            for (int axis = 0; axis <= 1; ++axis)
            {
                for (int rtype : rtypes)
                {
                    Mat expected;
                    {
                        DispatchModeGuard guard(
                            cpu::DispatchMode::ScalarOnly);
                        reduce(
                            src,
                            expected,
                            axis,
                            rtype,
                            CV_64F);
                        EXPECT_EQ(
                            cpu::last_dispatch_tag(),
                            cpu::DispatchTag::Scalar);
                    }

                    Mat actual;
                    {
                        DispatchModeGuard guard(
                            cpu::DispatchMode::Auto);
                        cpu::reset_last_dispatch_tag();
                        reduce(
                            src,
                            actual,
                            axis,
                            rtype,
                            CV_64F);
                        EXPECT_EQ(
                            cpu::last_dispatch_tag(),
                            expected_reduce_auto_tag(
                                depth,
                                axis,
                                rtype,
                                ui_enabled));
                    }
                    expect_reduce_mat_close(
                        actual, expected, 1e-12, 1e-12);
                }
            }
        }
    }
}

TEST(ReductionAxisDispatchInternalTest,
     reduce_ui_preserves_saturation_special_values_alias_and_fallbacks)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }

    const int u8_lanes = reduce_test_lanes(CV_8U);
    Mat saturated({5, u8_lanes + 3}, CV_8UC1);
    saturated.setTo(Scalar::all(255.0));
    Mat saturated_sum;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        reduce(
            saturated,
            saturated_sum,
            1,
            REDUCE_SUM,
            CV_8U);
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            ui_enabled
                ? cpu::DispatchTag::OpenCVUI
                : cpu::DispatchTag::Scalar);
    }
    for (int row = 0; row < saturated_sum.size.p[0]; ++row)
    {
        EXPECT_EQ(saturated_sum.at<uchar>(row, 0), 255);
    }

    const int f32_lanes = reduce_test_lanes(CV_32F);
    Mat special({3, f32_lanes + 3}, CV_32FC1);
    for (int row = 0; row < special.size.p[0]; ++row)
    {
        for (int column = 0; column < special.size.p[1]; ++column)
        {
            special.at<float>(row, column) =
                static_cast<float>(row * 10 + column + 1);
        }
    }
    special.at<float>(0, 0) =
        std::numeric_limits<float>::quiet_NaN();
    special.at<float>(1, 1) =
        std::numeric_limits<float>::quiet_NaN();
    special.at<float>(2, 2) =
        std::numeric_limits<float>::infinity();
    special.at<float>(2, 3) =
        -std::numeric_limits<float>::infinity();
    for (int axis = 0; axis <= 1; ++axis)
    {
        for (int rtype : {REDUCE_MAX, REDUCE_MIN, REDUCE_SUM})
        {
            Mat expected;
            {
                DispatchModeGuard guard(
                    cpu::DispatchMode::ScalarOnly);
                reduce(
                    special,
                    expected,
                    axis,
                    rtype,
                    CV_64F);
            }
            Mat actual;
            {
                DispatchModeGuard guard(cpu::DispatchMode::Auto);
                reduce(
                    special,
                    actual,
                    axis,
                    rtype,
                    CV_64F);
            }
            expect_reduce_mat_close(
                actual, expected, 0.0, 0.0);
        }
    }

    Mat alias_source({9, u8_lanes + 3}, CV_8UC3);
    for (int row = 0; row < alias_source.size.p[0]; ++row)
    {
        for (int column = 0;
             column < alias_source.size.p[1];
             ++column)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                alias_source.at<uchar>(row, column, channel) =
                    static_cast<uchar>(
                        (row + column + channel) % 23);
            }
        }
    }
    Mat expected_alias;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        reduce(
            alias_source,
            expected_alias,
            0,
            REDUCE_SUM2,
            CV_64F);
    }
    Mat alias = alias_source.clone();
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        reduce(alias, alias, 0, REDUCE_SUM2, CV_64F);
    }
    expect_reduce_mat_close(
        alias, expected_alias, 0.0, 0.0);

    Mat unsupported({3, 2 * u8_lanes + 1}, CV_16SC1);
    unsupported.setTo(Scalar::all(3.0));
    Mat unsupported_result;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        reduce(
            unsupported,
            unsupported_result,
            1,
            REDUCE_SUM,
            CV_64F);
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            cpu::DispatchTag::Scalar);
    }

    Mat short_row({3, std::max(1, u8_lanes - 1)}, CV_8UC1);
    short_row.setTo(Scalar::all(4.0));
    Mat short_result;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        reduce(
            short_row,
            short_result,
            1,
            REDUCE_SUM,
            CV_64F);
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            cpu::DispatchTag::Scalar);
    }
}
