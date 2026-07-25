#include "test/imgproc/support/intensity_filter_test_utils.hpp"

TEST(BilateralFilterTest, bilateral_filter_preserves_constants_and_rejects_alias)
{
    Mat constant({1, 7}, CV_8UC3);
    constant.setTo(Scalar(11, 37, 201));
    Mat filtered;
    bilateralFilter(
        constant, filtered, 5, 30.0, 2.0, BORDER_REFLECT_101);
    for (int x = 0; x < 7; ++x)
    {
        EXPECT_EQ(filtered.at<uchar>(0, x, 0), 11);
        EXPECT_EQ(filtered.at<uchar>(0, x, 1), 37);
        EXPECT_EQ(filtered.at<uchar>(0, x, 2), 201);
    }

    Mat edge({5, 5}, CV_32FC1);
    for (int y = 0; y < 5; ++y)
    {
        for (int x = 0; x < 5; ++x)
        {
            edge.at<float>(y, x) = x < 2 ? 0.0f : 100.0f;
        }
    }
    bilateralFilter(edge, filtered, 3, 1.0, 2.0);
    EXPECT_LT(filtered.at<float>(2, 1), 1.0f);
    EXPECT_GT(filtered.at<float>(2, 2), 99.0f);
    EXPECT_THROW(
        bilateralFilter(edge, edge, 3, 1.0, 2.0),
        Exception);
    EXPECT_THROW(
        bilateralFilter(edge, filtered, 3, 1.0, 2.0, BORDER_WRAP),
        Exception);
    EXPECT_THROW(
        bilateralFilter(
            edge,
            filtered,
            3,
            std::numeric_limits<double>::quiet_NaN(),
            2.0),
        Exception);
}
