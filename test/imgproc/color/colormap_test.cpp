#include "test/imgproc/support/intensity_filter_test_utils.hpp"

TEST(ColormapTest, apply_color_map_uses_bgr_and_user_lut)
{
    Mat values({1, 3}, CV_8UC1);
    values.at<uchar>(0, 0) = 0;
    values.at<uchar>(0, 1) = 128;
    values.at<uchar>(0, 2) = 255;
    Mat colored;
    applyColorMap(values, colored, COLORMAP_AUTUMN);
    EXPECT_EQ(colored.type(), CV_8UC3);
    EXPECT_EQ(colored.at<uchar>(0, 0, 0), 0);
    EXPECT_EQ(colored.at<uchar>(0, 0, 2), 255);
    EXPECT_EQ(colored.at<uchar>(0, 2, 1), 255);

    Mat lookup_parent({258, 3}, CV_8UC3);
    Mat lookup = lookup_parent(Range(1, 257), Range(1, 2));
    for (int i = 0; i < 256; ++i)
    {
        lookup.at<uchar>(i, 0, 0) = static_cast<uchar>(i);
        lookup.at<uchar>(i, 0, 1) = static_cast<uchar>(255 - i);
        lookup.at<uchar>(i, 0, 2) = 17;
    }
    applyColorMap(values, colored, lookup);
    EXPECT_EQ(colored.at<uchar>(0, 1, 0), 128);
    EXPECT_EQ(colored.at<uchar>(0, 1, 1), 127);
    EXPECT_EQ(colored.at<uchar>(0, 1, 2), 17);
    EXPECT_THROW(
        applyColorMap(values, colored, COLORMAP_VIRIDIS),
        Exception);
}
