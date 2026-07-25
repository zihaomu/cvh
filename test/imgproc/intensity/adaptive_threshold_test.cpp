#include "test/imgproc/support/intensity_filter_test_utils.hpp"

TEST(AdaptiveThresholdTest, adaptive_threshold_covers_mean_gaussian_and_in_place)
{
    Mat ramp({7, 9}, CV_8UC1);
    for (int y = 0; y < ramp.size.p[0]; ++y)
    {
        for (int x = 0; x < ramp.size.p[1]; ++x)
        {
            ramp.at<uchar>(y, x) =
                static_cast<uchar>(10 * x + y);
        }
    }
    Mat mean_binary;
    adaptiveThreshold(
        ramp,
        mean_binary,
        200,
        ADAPTIVE_THRESH_MEAN_C,
        THRESH_BINARY,
        3,
        2.0);
    EXPECT_EQ(mean_binary.type(), CV_8UC1);
    EXPECT_TRUE(
        mean_binary.at<uchar>(3, 4) == 0 ||
        mean_binary.at<uchar>(3, 4) == 200);

    Mat gaussian_inverse = ramp.clone();
    adaptiveThreshold(
        gaussian_inverse,
        gaussian_inverse,
        255,
        ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY_INV,
        5,
        -1.25);
    EXPECT_EQ(gaussian_inverse.shape(), ramp.shape());
    EXPECT_THROW(
        adaptiveThreshold(
            ramp,
            mean_binary,
            255,
            ADAPTIVE_THRESH_MEAN_C,
            THRESH_BINARY,
            4,
            1.0),
        Exception);
    EXPECT_THROW(
        adaptiveThreshold(
            ramp,
            mean_binary,
            255,
            ADAPTIVE_THRESH_MEAN_C,
            THRESH_BINARY,
            3,
            std::numeric_limits<double>::infinity()),
        Exception);
    EXPECT_THROW(
        adaptiveThreshold(
            ramp,
            mean_binary,
            255,
            99,
            THRESH_BINARY,
            3,
            1.0),
        Exception);
}
