#include "test/imgproc/support/pyramid_color_test_utils.hpp"

TEST(AccumulateTest, accumulate_family_covers_mask_and_repeated_updates)
{
    Mat src1({3, 4}, CV_8UC3);
    Mat src2({3, 4}, CV_8UC3);
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                src1.at<uchar>(y, x, ch) =
                    static_cast<uchar>(1 + x + y + ch);
                src2.at<uchar>(y, x, ch) =
                    static_cast<uchar>(2 + 2 * x + ch);
            }
        }
    }
    Mat mask({3, 4}, CV_8UC1);
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            mask.at<uchar>(y, x) = ((x + y) & 1) ? 255 : 0;
        }
    }

    Mat dst({3, 4}, CV_32FC3);
    dst.setTo(Scalar::all(1.0));
    accumulate(src1, dst, mask);
    accumulate(src1, dst, mask);
    EXPECT_FLOAT_EQ(dst.at<float>(0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(
        dst.at<float>(0, 1, 2),
        1.0f + 2.0f * src1.at<uchar>(0, 1, 2));

    dst.setTo(Scalar::all(0.0));
    accumulateSquare(src1, dst);
    EXPECT_FLOAT_EQ(
        dst.at<float>(2, 3, 1),
        static_cast<float>(
            src1.at<uchar>(2, 3, 1) *
            src1.at<uchar>(2, 3, 1)));

    dst.setTo(Scalar::all(0.0));
    accumulateProduct(src1, src2, dst, mask);
    EXPECT_FLOAT_EQ(dst.at<float>(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(
        dst.at<float>(1, 0, 2),
        static_cast<float>(
            src1.at<uchar>(1, 0, 2) *
            src2.at<uchar>(1, 0, 2)));

    Mat wrong;
    EXPECT_THROW(accumulate(src1, wrong), Exception);
}

TEST(AccumulateTest, accumulate_weighted_handles_alpha_extremes)
{
    Mat src({2, 3}, CV_32FC1);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            src.at<float>(y, x) = static_cast<float>(10 + y * 3 + x);
        }
    }
    Mat dst({2, 3}, CV_32FC1);
    dst.setTo(Scalar::all(3.0));
    accumulateWeighted(src, dst, 0.0);
    EXPECT_FLOAT_EQ(dst.at<float>(1, 2), 3.0f);
    accumulateWeighted(src, dst, 1.0);
    EXPECT_FLOAT_EQ(dst.at<float>(1, 2), src.at<float>(1, 2));
    dst.setTo(Scalar::all(2.0));
    accumulateWeighted(src, dst, 0.25);
    EXPECT_FLOAT_EQ(
        dst.at<float>(0, 0),
        0.75f * 2.0f + 0.25f * src.at<float>(0, 0));
}
