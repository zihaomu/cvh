#include "test/imgproc/support/filter_test_utils.hpp"

TEST(GaussianBlurTest, gaussian_blur_roi_and_inplace_match_reference)
{
    Mat parent({11, 14}, CV_8UC4);
    for (int y = 0; y < parent.size[0]; ++y)
    {
        for (int x = 0; x < parent.size[1]; ++x)
        {
            parent.at<uchar>(y, x, 0) = static_cast<uchar>((y * 7 + x * 37 + 5) % 256);
            parent.at<uchar>(y, x, 1) = static_cast<uchar>((y * 19 + x * 13 + 17) % 256);
            parent.at<uchar>(y, x, 2) = static_cast<uchar>((y * 29 + x * 3 + 23) % 256);
            parent.at<uchar>(y, x, 3) = static_cast<uchar>((y * 11 + x * 41 + 31) % 256);
        }
    }

    Mat roi = parent(Range(2, 10), Range(1, 13));
    ASSERT_FALSE(roi.isContinuous());

    const Size roi_ksize(7, 5);
    const Mat roi_expected = gaussian_blur_reference_u8(roi, roi_ksize, 1.4, 1.2, BORDER_REFLECT);
    Mat roi_actual;
    GaussianBlur(roi, roi_actual, roi_ksize, 1.4, 1.2, BORDER_REFLECT);
    EXPECT_LE(max_abs_diff_u8(roi_actual, roi_expected), 1);

    Mat in_place_src({13, 15}, CV_8UC1);
    for (int y = 0; y < in_place_src.size[0]; ++y)
    {
        for (int x = 0; x < in_place_src.size[1]; ++x)
        {
            in_place_src.at<uchar>(y, x) = static_cast<uchar>((y * 43 + x * 17 + 29) % 256);
        }
    }

    const Size in_place_ksize(5, 5);
    const Mat in_place_expected = gaussian_blur_reference_u8(in_place_src, in_place_ksize, 0.0, 0.0, BORDER_CONSTANT);
    Mat in_place_actual = in_place_src.clone();
    GaussianBlur(in_place_actual, in_place_actual, in_place_ksize, 0.0, 0.0, BORDER_CONSTANT);
    EXPECT_LE(max_abs_diff_u8(in_place_actual, in_place_expected), 1);
}

TEST(GaussianBlurTest, supports_cv32f_gaussian_roi_and_inplace)
{
    Mat base({10, 13}, CV_32FC3);
    for (int y = 0; y < base.size[0]; ++y)
    {
        for (int x = 0; x < base.size[1]; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                base.at<float>(y, x, c) = static_cast<float>(y * 1.1 + x * 0.4 - c * 0.9);
            }
        }
    }
    Mat roi = base(Range(2, 9), Range(1, 12));
    ASSERT_FALSE(roi.isContinuous());

    Mat actual;
    GaussianBlur(roi, actual, Size(5, 7), 1.2, 1.6, BORDER_REPLICATE);
    const Mat expected = gaussian_blur_reference_f32(roi, Size(5, 7), 1.2, 1.6, BORDER_REPLICATE);
    EXPECT_LE(max_abs_diff_f32(actual, expected), 2e-4f);

    Mat in_place = roi.clone();
    const Mat in_place_ref = gaussian_blur_reference_f32(in_place, Size(3, 3), 0.0, 0.0, BORDER_CONSTANT);
    GaussianBlur(in_place, in_place, Size(3, 3), 0.0, 0.0, BORDER_CONSTANT);
    EXPECT_LE(max_abs_diff_f32(in_place, in_place_ref), 2e-4f);
}
