#include "test/imgproc/support/pyramid_color_test_utils.hpp"

TEST(BlendLinearTest, blend_linear_handles_zero_and_non_normalized_weights)
{
    Mat first({2, 3}, CV_8UC3);
    Mat second({2, 3}, CV_8UC3);
    first.setTo(Scalar(20, 40, 60));
    second.setTo(Scalar(100, 120, 140));
    Mat weight1({2, 3}, CV_32FC1);
    Mat weight2({2, 3}, CV_32FC1);
    weight1.setTo(Scalar::all(0.0));
    weight2.setTo(Scalar::all(0.0));
    Mat result;
    blendLinear(first, second, weight1, weight2, result);
    EXPECT_EQ(result.at<uchar>(0, 0, 0), 0);

    weight1.setTo(Scalar::all(2.0));
    weight2.setTo(Scalar::all(1.0));
    blendLinear(first, second, weight1, weight2, result);
    EXPECT_NEAR(result.at<uchar>(1, 2, 0), 47, 1);

    weight1.setTo(Scalar::all(0.25));
    weight2.setTo(Scalar::all(0.75));
    blendLinear(first, second, weight1, weight2, first);
    EXPECT_NEAR(first.at<uchar>(0, 0, 2), 120, 1);
}
