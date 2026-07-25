#include "test/imgproc/support/geometric_sampling_test_utils.hpp"

TEST(RectSubPixTest, get_rect_sub_pix_covers_depth_and_edge)
{
    Mat source({4, 5}, CV_8UC1);
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 5; ++col)
        {
            source.at<uchar>(row, col) =
                static_cast<uchar>(row * 20 + col * 4);
        }
    }
    Mat center_patch;
    getRectSubPix(
        source,
        Size(1, 1),
        Point2f(1.5f, 2.5f),
        center_patch,
        CV_32F);
    EXPECT_EQ(center_patch.type(), CV_32FC1);
    EXPECT_FLOAT_EQ(center_patch.at<float>(0, 0), 56.0f);

    Mat edge_patch;
    getRectSubPix(
        source,
        Size(3, 3),
        Point2f(0.0f, 0.0f),
        edge_patch);
    EXPECT_EQ(edge_patch.type(), CV_8UC1);
    EXPECT_EQ(edge_patch.at<uchar>(0, 0), source.at<uchar>(0, 0));
    EXPECT_EQ(edge_patch.at<uchar>(2, 2), source.at<uchar>(1, 1));
}

TEST(RectSubPixTest, rejects_negative_center)
{
    Mat source({4, 5}, CV_8UC1);
    Mat output;

    EXPECT_THROW(
        getRectSubPix(
            source,
            Size(3, 3),
            Point2f(-0.1f, 0.0f),
            output),
        Exception);
}
