#include "test/core/support/reduction_internal_test_utils.hpp"

TEST(ReductionArgDispatchInternalTest,
     reduce_arg_ui_matches_scalar_across_depths_axes_roi_and_tail)
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
        Mat parent({17, 263}, CV_MAKETYPE(depth, 1));
        for (int row = 0; row < 17; ++row)
        {
            for (int column = 0; column < 263; ++column)
            {
                const int seed = row * 13 + column * 7;
                double value = static_cast<double>(seed % 29 + 30);
                if (depth == CV_8S || depth == CV_16S ||
                    depth == CV_32S || depth == CV_16F ||
                    depth == CV_32F || depth == CV_64F)
                {
                    value = static_cast<double>(seed % 29 - 14);
                }
                set_test_value(parent, row, column, 0, value);
            }
        }
        Mat src = parent.colRange(2, 261);
        ASSERT_FALSE(src.isContinuous());

        for (int axis = 0; axis <= 1; ++axis)
        {
            for (int last = 0; last <= 1; ++last)
            {
                Mat scalar_min;
                Mat scalar_max;
                {
                    DispatchModeGuard guard(
                        cpu::DispatchMode::ScalarOnly);
                    reduceArgMin(src, scalar_min, axis, last != 0);
                    reduceArgMax(src, scalar_max, axis, last != 0);
                }

                Mat auto_min;
                Mat auto_max;
                {
                    DispatchModeGuard guard(cpu::DispatchMode::Auto);
                    cpu::reset_last_dispatch_tag();
                    reduceArgMin(src, auto_min, axis, last != 0);
                    EXPECT_EQ(
                        cpu::last_dispatch_tag(),
                        expected_statistics_auto_tag(
                            depth, ui_enabled));
                    cpu::reset_last_dispatch_tag();
                    reduceArgMax(src, auto_max, axis, last != 0);
                    EXPECT_EQ(
                        cpu::last_dispatch_tag(),
                        expected_statistics_auto_tag(
                            depth, ui_enabled));
                }
                expect_index_mat_equal(auto_min, scalar_min);
                expect_index_mat_equal(auto_max, scalar_max);
            }
        }
    }
}

TEST(ReductionArgDispatchInternalTest,
     reduce_arg_ui_preserves_nan_signed_zero_constant_and_alias_semantics)
{
    Mat rows({3, 33}, CV_32FC1);
    rows.setTo(Scalar::all(5.0));
    rows.at<float>(0, 0) =
        std::numeric_limits<float>::quiet_NaN();
    rows.at<float>(1, 5) = 0.0f;
    rows.at<float>(1, 9) = -0.0f;
    rows.at<float>(1, 17) =
        std::numeric_limits<float>::quiet_NaN();

    Mat indices;
    reduceArgMin(rows, indices, 1, false);
    EXPECT_EQ(indices.at<int>(0, 0), 0);
    EXPECT_EQ(indices.at<int>(1, 0), 5);
    EXPECT_EQ(indices.at<int>(2, 0), 0);
    reduceArgMin(rows, indices, 1, true);
    EXPECT_EQ(indices.at<int>(0, 0), 0);
    EXPECT_EQ(indices.at<int>(1, 0), 9);
    EXPECT_EQ(indices.at<int>(2, 0), 32);
    reduceArgMax(rows, indices, 1, false);
    EXPECT_EQ(indices.at<int>(0, 0), 0);
    EXPECT_EQ(indices.at<int>(2, 0), 0);
    reduceArgMax(rows, indices, 1, true);
    EXPECT_EQ(indices.at<int>(0, 0), 0);
    EXPECT_EQ(indices.at<int>(2, 0), 32);

    Mat columns({17, 33}, CV_32FC1);
    columns.setTo(Scalar::all(2.0));
    reduceArgMin(columns, indices, 0, false);
    EXPECT_EQ(indices.at<int>(0, 0), 0);
    EXPECT_EQ(indices.at<int>(0, 32), 0);
    reduceArgMin(columns, indices, 0, true);
    EXPECT_EQ(indices.at<int>(0, 0), 16);
    EXPECT_EQ(indices.at<int>(0, 32), 16);

    Mat expected;
    reduceArgMax(columns, expected, 1, true);
    Mat alias = columns.clone();
    reduceArgMax(alias, alias, 1, true);
    expect_index_mat_equal(alias, expected);

    Mat short_row({1, 1}, CV_32FC1);
    short_row.at<float>(0, 0) = 3.0f;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        reduceArgMin(short_row, indices, 1, false);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    }
    EXPECT_EQ(indices.at<int>(0, 0), 0);
}
