#include "test/imgproc/support/cvtcolor_test_utils.hpp"

TEST(CvtColorYuv422Test, bgr_rgb_to_nv16_nv61_yuv422sp_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 6;

    Mat bgr({kRows, kCols}, CV_8UC3);
    Mat rgb({kRows, kCols}, CV_8UC3);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(19 + (y * 17 + x * 9) % 180);
            const uchar g = static_cast<uchar>(35 + (y * 13 + x * 7) % 170);
            const uchar r = static_cast<uchar>(51 + (y * 11 + x * 5) % 160);
            bgr.at<uchar>(y, x, 0) = b;
            bgr.at<uchar>(y, x, 1) = g;
            bgr.at<uchar>(y, x, 2) = r;
            rgb.at<uchar>(y, x, 0) = r;
            rgb.at<uchar>(y, x, 1) = g;
            rgb.at<uchar>(y, x, 2) = b;
        }
    }

    Mat nv16_from_bgr_expected = color3_to_yuv422sp_reference_u8(bgr, false, false);
    Mat nv16_from_bgr_actual;
    cvtColor(bgr, nv16_from_bgr_actual, COLOR_BGR2YUV_NV16);
    ASSERT_EQ(nv16_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(nv16_from_bgr_actual.size[0], kRows * 2);
    EXPECT_EQ(nv16_from_bgr_actual.size[1], kCols);
    EXPECT_EQ(max_abs_diff_u8(nv16_from_bgr_expected, nv16_from_bgr_actual), 0);

    Mat nv61_from_bgr_expected = color3_to_yuv422sp_reference_u8(bgr, false, true);
    Mat nv61_from_bgr_actual;
    cvtColor(bgr, nv61_from_bgr_actual, COLOR_BGR2YUV_NV61);
    ASSERT_EQ(nv61_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv61_from_bgr_expected, nv61_from_bgr_actual), 0);

    Mat nv16_from_rgb_expected = color3_to_yuv422sp_reference_u8(rgb, true, false);
    Mat nv16_from_rgb_actual;
    cvtColor(rgb, nv16_from_rgb_actual, COLOR_RGB2YUV_NV16);
    ASSERT_EQ(nv16_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv16_from_rgb_expected, nv16_from_rgb_actual), 0);

    Mat nv61_from_rgb_expected = color3_to_yuv422sp_reference_u8(rgb, true, true);
    Mat nv61_from_rgb_actual;
    cvtColor(rgb, nv61_from_rgb_actual, COLOR_RGB2YUV_NV61);
    ASSERT_EQ(nv61_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv61_from_rgb_expected, nv61_from_rgb_actual), 0);
}

TEST(CvtColorYuv422Test, bgr_rgb_to_yuy2_uyvy_yuv422packed_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 6;

    Mat bgr({kRows, kCols}, CV_8UC3);
    Mat rgb({kRows, kCols}, CV_8UC3);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(23 + (y * 17 + x * 9) % 180);
            const uchar g = static_cast<uchar>(41 + (y * 13 + x * 7) % 170);
            const uchar r = static_cast<uchar>(59 + (y * 11 + x * 5) % 160);
            bgr.at<uchar>(y, x, 0) = b;
            bgr.at<uchar>(y, x, 1) = g;
            bgr.at<uchar>(y, x, 2) = r;
            rgb.at<uchar>(y, x, 0) = r;
            rgb.at<uchar>(y, x, 1) = g;
            rgb.at<uchar>(y, x, 2) = b;
        }
    }

    Mat yuy2_from_bgr_expected = color3_to_yuv422packed_reference_u8(bgr, false, false);
    Mat yuy2_from_bgr_actual;
    cvtColor(bgr, yuy2_from_bgr_actual, COLOR_BGR2YUV_YUY2);
    ASSERT_EQ(yuy2_from_bgr_actual.type(), CV_8UC2);
    EXPECT_EQ(max_abs_diff_u8(yuy2_from_bgr_expected, yuy2_from_bgr_actual), 0);

    Mat uyvy_from_bgr_expected = color3_to_yuv422packed_reference_u8(bgr, false, true);
    Mat uyvy_from_bgr_actual;
    cvtColor(bgr, uyvy_from_bgr_actual, COLOR_BGR2YUV_UYVY);
    ASSERT_EQ(uyvy_from_bgr_actual.type(), CV_8UC2);
    EXPECT_EQ(max_abs_diff_u8(uyvy_from_bgr_expected, uyvy_from_bgr_actual), 0);

    Mat yuy2_from_rgb_expected = color3_to_yuv422packed_reference_u8(rgb, true, false);
    Mat yuy2_from_rgb_actual;
    cvtColor(rgb, yuy2_from_rgb_actual, COLOR_RGB2YUV_YUY2);
    ASSERT_EQ(yuy2_from_rgb_actual.type(), CV_8UC2);
    EXPECT_EQ(max_abs_diff_u8(yuy2_from_rgb_expected, yuy2_from_rgb_actual), 0);

    Mat uyvy_from_rgb_expected = color3_to_yuv422packed_reference_u8(rgb, true, true);
    Mat uyvy_from_rgb_actual;
    cvtColor(rgb, uyvy_from_rgb_actual, COLOR_RGB2YUV_UYVY);
    ASSERT_EQ(uyvy_from_rgb_actual.type(), CV_8UC2);
    EXPECT_EQ(max_abs_diff_u8(uyvy_from_rgb_expected, uyvy_from_rgb_actual), 0);
}

TEST(CvtColorYuv422Test, nv16_nv61_yuv422sp_u8_matches_reference)
{
    constexpr int kRows = 5;
    constexpr int kCols = 6;

    Mat nv16({kRows * 2, kCols}, CV_8UC1);
    Mat nv61({kRows * 2, kCols}, CV_8UC1);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(20 + (y * 19 + x * 11) % 200);
            nv16.at<uchar>(y, x) = yy;
            nv61.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; x += 2)
        {
            const uchar uu = static_cast<uchar>(46 + (y * 13 + x * 5) % 150);
            const uchar vv = static_cast<uchar>(60 + (y * 7 + x * 9) % 140);
            nv16.at<uchar>(kRows + y, x + 0) = uu;
            nv16.at<uchar>(kRows + y, x + 1) = vv;
            nv61.at<uchar>(kRows + y, x + 0) = vv;
            nv61.at<uchar>(kRows + y, x + 1) = uu;
        }
    }

    Mat bgr_nv16_expected = yuv422sp_to_color3_reference_u8(nv16, false, false);
    Mat bgr_nv16_actual;
    cvtColor(nv16, bgr_nv16_actual, COLOR_YUV2BGR_NV16);
    ASSERT_EQ(bgr_nv16_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv16_expected, bgr_nv16_actual), 0);

    Mat rgb_nv16_expected = yuv422sp_to_color3_reference_u8(nv16, false, true);
    Mat rgb_nv16_actual;
    cvtColor(nv16, rgb_nv16_actual, COLOR_YUV2RGB_NV16);
    ASSERT_EQ(rgb_nv16_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv16_expected, rgb_nv16_actual), 0);

    Mat bgr_nv61_expected = yuv422sp_to_color3_reference_u8(nv61, true, false);
    Mat bgr_nv61_actual;
    cvtColor(nv61, bgr_nv61_actual, COLOR_YUV2BGR_NV61);
    ASSERT_EQ(bgr_nv61_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv61_expected, bgr_nv61_actual), 0);

    Mat rgb_nv61_expected = yuv422sp_to_color3_reference_u8(nv61, true, true);
    Mat rgb_nv61_actual;
    cvtColor(nv61, rgb_nv61_actual, COLOR_YUV2RGB_NV61);
    ASSERT_EQ(rgb_nv61_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv61_expected, rgb_nv61_actual), 0);
}

TEST(CvtColorYuv422Test, yuy2_uyvy_yuv422packed_u8_matches_reference)
{
    constexpr int kRows = 5;
    constexpr int kCols = 6;

    Mat yuy2({kRows, kCols}, CV_8UC2);
    Mat uyvy({kRows, kCols}, CV_8UC2);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; x += 2)
        {
            const uchar yy0 = static_cast<uchar>(22 + (y * 17 + x * 9) % 190);
            const uchar yy1 = static_cast<uchar>(35 + (y * 11 + x * 7) % 180);
            const uchar uu = static_cast<uchar>(48 + (y * 13 + x * 5) % 150);
            const uchar vv = static_cast<uchar>(62 + (y * 7 + x * 3) % 140);
            set_yuv422_packed_pair_u8(yuy2, y, x, false, yy0, yy1, uu, vv);
            set_yuv422_packed_pair_u8(uyvy, y, x, true, yy0, yy1, uu, vv);
        }
    }

    Mat bgr_yuy2_expected = yuv422packed_to_color3_reference_u8(yuy2, false, false);
    Mat bgr_yuy2_actual;
    cvtColor(yuy2, bgr_yuy2_actual, COLOR_YUV2BGR_YUY2);
    ASSERT_EQ(bgr_yuy2_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_yuy2_expected, bgr_yuy2_actual), 0);

    Mat rgb_yuy2_expected = yuv422packed_to_color3_reference_u8(yuy2, false, true);
    Mat rgb_yuy2_actual;
    cvtColor(yuy2, rgb_yuy2_actual, COLOR_YUV2RGB_YUY2);
    ASSERT_EQ(rgb_yuy2_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_yuy2_expected, rgb_yuy2_actual), 0);

    Mat bgr_uyvy_expected = yuv422packed_to_color3_reference_u8(uyvy, true, false);
    Mat bgr_uyvy_actual;
    cvtColor(uyvy, bgr_uyvy_actual, COLOR_YUV2BGR_UYVY);
    ASSERT_EQ(bgr_uyvy_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_uyvy_expected, bgr_uyvy_actual), 0);

    Mat rgb_uyvy_expected = yuv422packed_to_color3_reference_u8(uyvy, true, true);
    Mat rgb_uyvy_actual;
    cvtColor(uyvy, rgb_uyvy_actual, COLOR_YUV2RGB_UYVY);
    ASSERT_EQ(rgb_uyvy_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_uyvy_expected, rgb_uyvy_actual), 0);
}

TEST(CvtColorYuv422Test, non_contiguous_roi_for_nv16_nv61_encode_matches_reference)
{
    constexpr int kRows = 5;
    constexpr int kCols = 8;

    Mat base_bgr({kRows, kCols + 4}, CV_8UC3);
    Mat bgr_roi = base_bgr.colRange(2, 2 + kCols);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat base_rgb({kRows, kCols + 4}, CV_8UC3);
    Mat rgb_roi = base_rgb.colRange(2, 2 + kCols);
    ASSERT_FALSE(rgb_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(27 + (y * 19 + x * 9) % 170);
            const uchar g = static_cast<uchar>(45 + (y * 11 + x * 7) % 160);
            const uchar r = static_cast<uchar>(63 + (y * 13 + x * 5) % 150);
            bgr_roi.at<uchar>(y, x, 0) = b;
            bgr_roi.at<uchar>(y, x, 1) = g;
            bgr_roi.at<uchar>(y, x, 2) = r;
            rgb_roi.at<uchar>(y, x, 0) = r;
            rgb_roi.at<uchar>(y, x, 1) = g;
            rgb_roi.at<uchar>(y, x, 2) = b;
        }
    }

    Mat nv16_expected = color3_to_yuv422sp_reference_u8(bgr_roi, false, false);
    Mat nv16_actual;
    cvtColor(bgr_roi, nv16_actual, COLOR_BGR2YUV_NV16);
    EXPECT_EQ(max_abs_diff_u8(nv16_expected, nv16_actual), 0);

    Mat nv61_expected = color3_to_yuv422sp_reference_u8(rgb_roi, true, true);
    Mat nv61_actual;
    cvtColor(rgb_roi, nv61_actual, COLOR_RGB2YUV_NV61);
    EXPECT_EQ(max_abs_diff_u8(nv61_expected, nv61_actual), 0);
}

TEST(CvtColorYuv422Test, non_contiguous_roi_for_yuy2_uyvy_encode_matches_reference)
{
    constexpr int kRows = 6;
    constexpr int kCols = 8;

    Mat base_bgr({kRows, kCols + 3}, CV_8UC3);
    Mat bgr_roi = base_bgr.colRange(1, 1 + kCols);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat base_rgb({kRows, kCols + 3}, CV_8UC3);
    Mat rgb_roi = base_rgb.colRange(1, 1 + kCols);
    ASSERT_FALSE(rgb_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(31 + (y * 19 + x * 9) % 170);
            const uchar g = static_cast<uchar>(49 + (y * 11 + x * 7) % 160);
            const uchar r = static_cast<uchar>(67 + (y * 13 + x * 5) % 150);
            bgr_roi.at<uchar>(y, x, 0) = b;
            bgr_roi.at<uchar>(y, x, 1) = g;
            bgr_roi.at<uchar>(y, x, 2) = r;
            rgb_roi.at<uchar>(y, x, 0) = r;
            rgb_roi.at<uchar>(y, x, 1) = g;
            rgb_roi.at<uchar>(y, x, 2) = b;
        }
    }

    Mat yuy2_expected = color3_to_yuv422packed_reference_u8(bgr_roi, false, false);
    Mat yuy2_actual;
    cvtColor(bgr_roi, yuy2_actual, COLOR_BGR2YUV_YUY2);
    EXPECT_EQ(max_abs_diff_u8(yuy2_expected, yuy2_actual), 0);

    Mat uyvy_expected = color3_to_yuv422packed_reference_u8(rgb_roi, true, true);
    Mat uyvy_actual;
    cvtColor(rgb_roi, uyvy_actual, COLOR_RGB2YUV_UYVY);
    EXPECT_EQ(max_abs_diff_u8(uyvy_expected, uyvy_actual), 0);
}

TEST(CvtColorYuv422Test, non_contiguous_step_for_nv16_nv61_matches_reference)
{
    constexpr int kRows = 6;
    constexpr int kCols = 8;

    Mat base_nv16({kRows * 2, kCols + 4}, CV_8UC1);
    Mat nv16_roi = base_nv16.colRange(2, 2 + kCols);
    ASSERT_FALSE(nv16_roi.isContinuous());

    Mat base_nv61({kRows * 2, kCols + 4}, CV_8UC1);
    Mat nv61_roi = base_nv61.colRange(2, 2 + kCols);
    ASSERT_FALSE(nv61_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(24 + (y * 17 + x * 7) % 200);
            nv16_roi.at<uchar>(y, x) = yy;
            nv61_roi.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; x += 2)
        {
            const uchar uu = static_cast<uchar>(50 + (y * 9 + x * 5) % 150);
            const uchar vv = static_cast<uchar>(66 + (y * 11 + x * 3) % 140);
            nv16_roi.at<uchar>(kRows + y, x + 0) = uu;
            nv16_roi.at<uchar>(kRows + y, x + 1) = vv;
            nv61_roi.at<uchar>(kRows + y, x + 0) = vv;
            nv61_roi.at<uchar>(kRows + y, x + 1) = uu;
        }
    }

    Mat bgr_nv16_expected = yuv422sp_to_color3_reference_u8(nv16_roi, false, false);
    Mat bgr_nv16_actual;
    cvtColor(nv16_roi, bgr_nv16_actual, COLOR_YUV2BGR_NV16);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv16_expected, bgr_nv16_actual), 0);

    Mat rgb_nv61_expected = yuv422sp_to_color3_reference_u8(nv61_roi, true, true);
    Mat rgb_nv61_actual;
    cvtColor(nv61_roi, rgb_nv61_actual, COLOR_YUV2RGB_NV61);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv61_expected, rgb_nv61_actual), 0);
}

TEST(CvtColorYuv422Test, non_contiguous_step_for_yuy2_uyvy_matches_reference)
{
    constexpr int kRows = 6;
    constexpr int kCols = 8;

    Mat base_yuy2({kRows, kCols + 3}, CV_8UC2);
    Mat yuy2_roi = base_yuy2.colRange(1, 1 + kCols);
    ASSERT_FALSE(yuy2_roi.isContinuous());

    Mat base_uyvy({kRows, kCols + 3}, CV_8UC2);
    Mat uyvy_roi = base_uyvy.colRange(1, 1 + kCols);
    ASSERT_FALSE(uyvy_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; x += 2)
        {
            const uchar yy0 = static_cast<uchar>(26 + (y * 15 + x * 7) % 190);
            const uchar yy1 = static_cast<uchar>(41 + (y * 9 + x * 11) % 180);
            const uchar uu = static_cast<uchar>(54 + (y * 5 + x * 3) % 150);
            const uchar vv = static_cast<uchar>(68 + (y * 7 + x * 5) % 140);
            set_yuv422_packed_pair_u8(yuy2_roi, y, x, false, yy0, yy1, uu, vv);
            set_yuv422_packed_pair_u8(uyvy_roi, y, x, true, yy0, yy1, uu, vv);
        }
    }

    Mat bgr_yuy2_expected = yuv422packed_to_color3_reference_u8(yuy2_roi, false, false);
    Mat bgr_yuy2_actual;
    cvtColor(yuy2_roi, bgr_yuy2_actual, COLOR_YUV2BGR_YUY2);
    EXPECT_EQ(max_abs_diff_u8(bgr_yuy2_expected, bgr_yuy2_actual), 0);

    Mat rgb_uyvy_expected = yuv422packed_to_color3_reference_u8(uyvy_roi, true, true);
    Mat rgb_uyvy_actual;
    cvtColor(uyvy_roi, rgb_uyvy_actual, COLOR_YUV2RGB_UYVY);
    EXPECT_EQ(max_abs_diff_u8(rgb_uyvy_expected, rgb_uyvy_actual), 0);
}

TEST(CvtColorYuv422Test, throws_on_invalid_nv16_nv61_layouts)
{
    Mat dst;

    Mat odd_width({8, 5}, CV_8UC1);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_YUV2BGR_NV16), Exception);

    Mat bad_rows({7, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(bad_rows, dst, COLOR_YUV2RGB_NV61), Exception);

    Mat three_channel({8, 6}, CV_8UC3);
    EXPECT_THROW(cvtColor(three_channel, dst, COLOR_YUV2BGR_NV61), Exception);

    Mat f32_src({8, 6}, CV_32FC1);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_YUV2RGB_NV16), Exception);
}

TEST(CvtColorYuv422Test, throws_on_invalid_bgr_rgb_to_nv16_nv61_inputs)
{
    Mat dst;

    Mat gray_src({5, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_BGR2YUV_NV16), Exception);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_RGB2YUV_NV61), Exception);

    Mat bgra_src({5, 6}, CV_8UC4);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_RGB2YUV_NV16), Exception);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_BGR2YUV_NV61), Exception);

    Mat odd_width({5, 5}, CV_8UC3);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_BGR2YUV_NV16), Exception);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_RGB2YUV_NV61), Exception);

    Mat f32_src({5, 6}, CV_32FC3);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_BGR2YUV_NV16), Exception);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_RGB2YUV_NV61), Exception);
}

TEST(CvtColorYuv422Test, throws_on_invalid_bgr_rgb_to_yuy2_uyvy_inputs)
{
    Mat dst;

    Mat gray_src({5, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_BGR2YUV_YUY2), Exception);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_RGB2YUV_UYVY), Exception);

    Mat bgra_src({5, 6}, CV_8UC4);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_RGB2YUV_YUY2), Exception);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_BGR2YUV_UYVY), Exception);

    Mat odd_width({5, 5}, CV_8UC3);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_BGR2YUV_YUY2), Exception);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_RGB2YUV_UYVY), Exception);

    Mat f32_src({5, 6}, CV_32FC3);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_BGR2YUV_YUY2), Exception);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_RGB2YUV_UYVY), Exception);
}

TEST(CvtColorYuv422Test, throws_on_invalid_yuy2_uyvy_layouts)
{
    Mat dst;

    Mat odd_width({6, 5}, CV_8UC2);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_YUV2BGR_YUY2), Exception);

    Mat one_channel({6, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(one_channel, dst, COLOR_YUV2RGB_YUY2), Exception);

    Mat three_channel({6, 6}, CV_8UC3);
    EXPECT_THROW(cvtColor(three_channel, dst, COLOR_YUV2BGR_UYVY), Exception);

    Mat f32_src({6, 6}, CV_32FC2);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_YUV2RGB_UYVY), Exception);
}
