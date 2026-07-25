#include "test/imgproc/support/filter_test_utils.hpp"

TEST(FilterUpstreamTest, Imgproc_Blur_borderTypes)
{
    // Upstream reference:
    // modules/imgproc/test/test_filter.cpp :: TEST(Imgproc_Blur, borderTypes)
    Mat parent({9, 11}, CV_8UC3);
    for (int y = 0; y < parent.size[0]; ++y)
    {
        for (int x = 0; x < parent.size[1]; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                parent.at<uchar>(y, x, c) = static_cast<uchar>((y * 19 + x * 13 + c * 23) % 256);
            }
        }
    }

    Mat src_roi = parent(Range(2, 8), Range(3, 10));
    ASSERT_FALSE(src_roi.isContinuous());

    Mat dst;
    blur(src_roi, dst, Size(3, 3), Point(-1, -1), BORDER_REPLICATE);

    Mat dst_isolated;
    blur(src_roi, dst_isolated, Size(3, 3), Point(-1, -1), BORDER_REPLICATE | BORDER_ISOLATED);

    EXPECT_EQ(max_abs_diff_u8(dst, dst_isolated), 0);

    const Mat ref = box_filter_reference_u8(src_roi, Size(3, 3), Point(-1, -1), true, BORDER_REPLICATE);
    EXPECT_EQ(max_abs_diff_u8(dst, ref), 0);
}

TEST(FilterUpstreamTest, Imgproc_GaussianBlur_borderTypes)
{
    // Upstream reference:
    // modules/imgproc/test/test_filter.cpp :: TEST(Imgproc_GaussianBlur, borderTypes)
    Mat parent({10, 12}, CV_8UC1);
    for (int y = 0; y < parent.size[0]; ++y)
    {
        for (int x = 0; x < parent.size[1]; ++x)
        {
            parent.at<uchar>(y, x) = static_cast<uchar>((y * 37 + x * 11) % 256);
        }
    }

    Mat src_roi = parent(Range(2, 9), Range(1, 10));
    ASSERT_FALSE(src_roi.isContinuous());

    Mat dst;
    GaussianBlur(src_roi, dst, Size(5, 5), 0.0, 0.0, BORDER_REPLICATE);

    Mat dst_isolated;
    GaussianBlur(src_roi, dst_isolated, Size(5, 5), 0.0, 0.0, BORDER_REPLICATE | BORDER_ISOLATED);
    EXPECT_EQ(max_abs_diff_u8(dst, dst_isolated), 0);

    Mat dst_default;
    GaussianBlur(src_roi, dst_default, Size(5, 5), 0.0, 0.0, BORDER_DEFAULT);
    const Mat ref_default = gaussian_blur_reference_u8(src_roi, Size(5, 5), 0.0, 0.0, BORDER_DEFAULT);
    EXPECT_LE(max_abs_diff_u8(dst_default, ref_default), 1);
}

TEST(FilterUpstreamTest, GaussianBlur_Bitexact_regression_15015)
{
    // Upstream reference:
    // modules/imgproc/test/test_smooth_bitexact.cpp :: TEST(GaussianBlur_Bitexact, regression_15015)
    Mat src({100, 100}, CV_8UC3);
    src = Scalar(255.0, 255.0, 255.0, 0.0);

    Mat dst;
    GaussianBlur(src, dst, Size(5, 5), 0.0);

    ASSERT_EQ(dst.type(), src.type());
    ASSERT_EQ(dst.size[0], src.size[0]);
    ASSERT_EQ(dst.size[1], src.size[1]);
    EXPECT_EQ(max_abs_diff_u8(dst, src), 0);
}

TEST(FilterUpstreamTest, Imgproc_GaussianBlur_regression_11303)
{
    // Upstream reference:
    // modules/imgproc/test/test_filter.cpp :: TEST(Imgproc_GaussianBlur, regression_11303)
    const int width = 2115;
    const int height = 211;
    const double sigma = 8.64421;

    Mat src({height, width}, CV_32FC1);
    src = 1.0f;

    Mat dst;
    GaussianBlur(src, dst, Size(), sigma, sigma);

    ASSERT_EQ(dst.type(), src.type());
    ASSERT_EQ(dst.size[0], src.size[0]);
    ASSERT_EQ(dst.size[1], src.size[1]);
    EXPECT_LE(l2_norm_diff_f32(src, dst), 1e-3);
}
