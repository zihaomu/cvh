#include "test/core/support/reduction_internal_test_utils.hpp"

TEST(ReductionMinMaxDispatchInternalTest,
     minmax_ui_matches_scalar_across_depths_roi_tail_and_ties)
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
        Mat parent({3, 263}, CV_MAKETYPE(depth, 1));
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 263; ++column)
            {
                const int seed = row * 263 + column;
                double value = static_cast<double>(seed % 101 + 10);
                if (depth == CV_8S || depth == CV_16S ||
                    depth == CV_32S || depth == CV_16F ||
                    depth == CV_32F || depth == CV_64F)
                {
                    value = static_cast<double>(seed % 101 - 50);
                }
                set_test_value(parent, row, column, 0, value);
            }
        }
        Mat src = parent.colRange(2, 261);
        ASSERT_FALSE(src.isContinuous());
        set_test_value(src, 0, 17, 0, depth == CV_8U || depth == CV_16U ||
                                          depth == CV_32U
                                      ? 0.0
                                      : -100.0);
        set_test_value(src, 1, 19, 0, depth == CV_8U || depth == CV_16U ||
                                          depth == CV_32U
                                      ? 0.0
                                      : -100.0);
        set_test_value(src, 0, 31, 0, depth == CV_8U ? 255.0 : 120.0);
        set_test_value(src, 2, 258, 0, depth == CV_8U ? 255.0 : 120.0);

        double scalar_min = 0.0;
        double scalar_max = 0.0;
        Point scalar_min_location;
        Point scalar_max_location;
        {
            DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
            minMaxLoc(
                src,
                &scalar_min,
                &scalar_max,
                &scalar_min_location,
                &scalar_max_location);
        }

        double auto_min = 0.0;
        double auto_max = 0.0;
        Point auto_min_location;
        Point auto_max_location;
        {
            DispatchModeGuard guard(cpu::DispatchMode::Auto);
            cpu::reset_last_dispatch_tag();
            minMaxLoc(
                src,
                &auto_min,
                &auto_max,
                &auto_min_location,
                &auto_max_location);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                expected_statistics_auto_tag(depth, ui_enabled));
        }
        EXPECT_DOUBLE_EQ(auto_min, scalar_min);
        EXPECT_DOUBLE_EQ(auto_max, scalar_max);
        EXPECT_EQ(auto_min_location, scalar_min_location);
        EXPECT_EQ(auto_max_location, scalar_max_location);
    }
}

TEST(ReductionMinMaxDispatchInternalTest,
     minmax_ui_preserves_mask_nan_inf_and_signed_zero_semantics)
{
    Mat parent({2, 67}, CV_32FC1);
    parent.setTo(Scalar::all(10.0));
    Mat src = parent.colRange(1, 66);
    Mat mask_parent({2, 67}, CV_8UC1);
    Mat mask = mask_parent.colRange(1, 66);
    ASSERT_FALSE(src.isContinuous());
    ASSERT_FALSE(mask.isContinuous());

    src.at<float>(0, 2) = std::numeric_limits<float>::quiet_NaN();
    src.at<float>(0, 5) = 0.0f;
    src.at<float>(0, 9) = -0.0f;
    src.at<float>(1, 7) = std::numeric_limits<float>::infinity();
    src.at<float>(1, 11) = -std::numeric_limits<float>::infinity();

    mask.setTo(Scalar::all(0.0));
    mask.at<uchar>(0, 2) = 255;
    mask.at<uchar>(0, 5) = 255;
    mask.at<uchar>(0, 9) = 255;
    mask.at<uchar>(1, 7) = 255;
    mask.at<uchar>(1, 11) = 255;

    double scalar_min = 0.0;
    double scalar_max = 0.0;
    Point scalar_min_location;
    Point scalar_max_location;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        minMaxLoc(
            src,
            &scalar_min,
            &scalar_max,
            &scalar_min_location,
            &scalar_max_location,
            mask);
    }
    double auto_min = 0.0;
    double auto_max = 0.0;
    Point auto_min_location;
    Point auto_max_location;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        minMaxLoc(
            src,
            &auto_min,
            &auto_max,
            &auto_min_location,
            &auto_max_location,
            mask);
    }
    EXPECT_EQ(auto_min, scalar_min);
    EXPECT_EQ(auto_max, scalar_max);
    EXPECT_EQ(auto_min_location, scalar_min_location);
    EXPECT_EQ(auto_max_location, scalar_max_location);
    EXPECT_TRUE(std::isinf(auto_min) && auto_min < 0.0);
    EXPECT_TRUE(std::isinf(auto_max) && auto_max > 0.0);

    mask.setTo(Scalar::all(0.0));
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        minMaxLoc(
            src,
            &auto_min,
            &auto_max,
            &auto_min_location,
            &auto_max_location,
            mask);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    }
    EXPECT_DOUBLE_EQ(auto_min, 0.0);
    EXPECT_DOUBLE_EQ(auto_max, 0.0);
    EXPECT_EQ(auto_min_location, Point(-1, -1));
    EXPECT_EQ(auto_max_location, Point(-1, -1));

    Mat zeros({1, 33}, CV_32FC1);
    zeros.setTo(Scalar::all(1.0));
    zeros.at<float>(0, 5) = 0.0f;
    zeros.at<float>(0, 9) = -0.0f;
    minMaxLoc(
        zeros,
        &auto_min,
        &auto_max,
        &auto_min_location,
        &auto_max_location);
    EXPECT_FALSE(std::signbit(auto_min));
    EXPECT_EQ(auto_min_location, Point(5, 0));

    Mat all_nan({1, 33}, CV_32FC1);
    all_nan.setTo(Scalar::all(
        std::numeric_limits<float>::quiet_NaN()));
    minMaxLoc(
        all_nan,
        &auto_min,
        &auto_max,
        &auto_min_location,
        &auto_max_location);
    EXPECT_DOUBLE_EQ(auto_min, 0.0);
    EXPECT_DOUBLE_EQ(auto_max, 0.0);
    EXPECT_EQ(auto_min_location, Point(-1, -1));
    EXPECT_EQ(auto_max_location, Point(-1, -1));
}
