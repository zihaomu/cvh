#include "test/imgproc/support/kernel_family_test_utils.hpp"

TEST(SqrBoxFilterTest, squared_box_filter_uses_wide_accumulation)
{
    Mat large({35, 35}, CV_8UC1);
    large.setTo(Scalar::all(255));
    Mat normalized;
    sqrBoxFilter(
        large,
        normalized,
        CV_64F,
        Size(31, 31),
        Point(-1, -1),
        true,
        BORDER_REPLICATE);
    EXPECT_DOUBLE_EQ(normalized.at<double>(17, 17), 65025.0);

    Mat unnormalized;
    sqrBoxFilter(
        large,
        unnormalized,
        CV_64F,
        Size(31, 31),
        Point(-1, -1),
        false,
        BORDER_REPLICATE);
    EXPECT_DOUBLE_EQ(
        unnormalized.at<double>(17, 17), 65025.0 * 31.0 * 31.0);

    Mat color_parent({5, 7}, CV_32FC3);
    color_parent.setTo(Scalar(1.0, 2.0, 3.0));
    Mat color_roi = color_parent(Range(1, 5), Range(1, 6));
    Mat color_result;
    sqrBoxFilter(
        color_roi,
        color_result,
        CV_32F,
        Size(3, 3),
        Point(-1, -1),
        true,
        BORDER_REPLICATE | BORDER_ISOLATED);
    EXPECT_FLOAT_EQ(color_result.at<float>(2, 2, 0), 1.0f);
    EXPECT_FLOAT_EQ(color_result.at<float>(2, 2, 2), 9.0f);
}

TEST(SqrBoxFilterTest, rejects_unsupported_output_depth)
{
    Mat source({3, 3}, CV_32FC1);
    Mat output;

    EXPECT_THROW(
        sqrBoxFilter(source, output, CV_16S, Size(3, 3)),
        Exception);
}
