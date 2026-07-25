#include "test/imgproc/support/filter_test_utils.hpp"

TEST(BoxFilterTest, boxfilter_non_contiguous_roi_custom_anchor_and_normalize_off_matches_reference)
{
    Mat parent({9, 13}, CV_8UC4);
    for (int y = 0; y < parent.size[0]; ++y)
    {
        for (int x = 0; x < parent.size[1]; ++x)
        {
            parent.at<uchar>(y, x, 0) = static_cast<uchar>((y * 17 + x * 3 + 11) % 256);
            parent.at<uchar>(y, x, 1) = static_cast<uchar>((y * 5 + x * 29 + 7) % 256);
            parent.at<uchar>(y, x, 2) = static_cast<uchar>((y * 13 + x * 19 + 3) % 256);
            parent.at<uchar>(y, x, 3) = static_cast<uchar>((y * 23 + x * 11 + 1) % 256);
        }
    }

    Mat roi = parent(Range(1, 8), Range(2, 12));
    ASSERT_FALSE(roi.isContinuous());

    const Size ksize(5, 3);
    const Point anchor(1, 0);

    Mat actual;
    boxFilter(roi, actual, -1, ksize, anchor, false, BORDER_REFLECT_101);

    const Mat expected = box_filter_reference_u8(roi, ksize, anchor, false, BORDER_REFLECT_101);
    EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
}

TEST(BoxFilterTest, boxfilter_inplace_matches_reference_with_constant_border)
{
    Mat src({17, 19}, CV_8UC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            src.at<uchar>(y, x, 0) = static_cast<uchar>((y * 31 + x * 7 + 3) % 256);
            src.at<uchar>(y, x, 1) = static_cast<uchar>((y * 11 + x * 5 + 9) % 256);
            src.at<uchar>(y, x, 2) = static_cast<uchar>((y * 13 + x * 17 + 15) % 256);
        }
    }

    const Size ksize(7, 5);
    const Point anchor(-1, -1);
    const Mat expected = box_filter_reference_u8(src, ksize, anchor, true, BORDER_CONSTANT);

    Mat in_place = src.clone();
    boxFilter(in_place, in_place, -1, ksize, anchor, true, BORDER_CONSTANT);

    EXPECT_EQ(max_abs_diff_u8(in_place, expected), 0);
}

TEST(BoxFilterTest, supports_cv32f_boxfilter_roi_and_inplace)
{
    Mat base({9, 12}, CV_32FC4);
    for (int y = 0; y < base.size[0]; ++y)
    {
        for (int x = 0; x < base.size[1]; ++x)
        {
            for (int c = 0; c < 4; ++c)
            {
                base.at<float>(y, x, c) = static_cast<float>(y * 0.8 - x * 0.35 + c * 1.2);
            }
        }
    }
    Mat roi = base(Range(1, 8), Range(2, 11));
    ASSERT_FALSE(roi.isContinuous());

    Mat actual;
    boxFilter(roi, actual, -1, Size(5, 3), Point(1, 0), true, BORDER_REFLECT_101);
    const Mat expected = box_filter_reference_f32(roi, Size(5, 3), Point(1, 0), true, BORDER_REFLECT_101);
    EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-5f);

    Mat in_place = roi.clone();
    const Mat in_place_ref = box_filter_reference_f32(in_place, Size(3, 3), Point(-1, -1), false, BORDER_CONSTANT);
    boxFilter(in_place, in_place, -1, Size(3, 3), Point(-1, -1), false, BORDER_CONSTANT);
    EXPECT_LE(max_abs_diff_f32(in_place, in_place_ref), 1e-5f);
}
