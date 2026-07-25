#include "test/imgproc/support/cvtcolor_test_utils.hpp"

TEST(CvtColorYuv444Test, bgr_rgb_yuv_family_u8_matches_reference)
{
    Mat bgr({2, 5}, CV_8UC3);
    for (int y = 0; y < bgr.size[0]; ++y)
    {
        for (int x = 0; x < bgr.size[1]; ++x)
        {
            bgr.at<uchar>(y, x, 0) = static_cast<uchar>(7 + y * 29 + x * 3);
            bgr.at<uchar>(y, x, 1) = static_cast<uchar>(13 + y * 17 + x * 5);
            bgr.at<uchar>(y, x, 2) = static_cast<uchar>(19 + y * 11 + x * 7);
        }
    }

    Mat yuv_expected = color3_to_yuv_reference<uchar>(bgr, false);
    Mat yuv_actual;
    cvtColor(bgr, yuv_actual, COLOR_BGR2YUV);
    ASSERT_EQ(yuv_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(yuv_expected, yuv_actual), 0);

    Mat bgr_expected = yuv_to_color3_reference<uchar>(yuv_actual, false);
    Mat bgr_actual;
    cvtColor(yuv_actual, bgr_actual, COLOR_YUV2BGR);
    ASSERT_EQ(bgr_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_expected, bgr_actual), 0);

    Mat rgb = bgr2rgb_reference<uchar>(bgr);
    Mat yuv_from_rgb_expected = color3_to_yuv_reference<uchar>(rgb, true);
    Mat yuv_from_rgb_actual;
    cvtColor(rgb, yuv_from_rgb_actual, COLOR_RGB2YUV);
    ASSERT_EQ(yuv_from_rgb_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(yuv_from_rgb_expected, yuv_from_rgb_actual), 0);

    Mat rgb_expected = yuv_to_color3_reference<uchar>(yuv_from_rgb_actual, true);
    Mat rgb_actual;
    cvtColor(yuv_from_rgb_actual, rgb_actual, COLOR_YUV2RGB);
    ASSERT_EQ(rgb_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_expected, rgb_actual), 0);
}

TEST(CvtColorYuv444Test, bgr_rgb_to_nv24_nv42_yuv444sp_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 5;

    Mat bgr({kRows, kCols}, CV_8UC3);
    Mat rgb({kRows, kCols}, CV_8UC3);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(17 + (y * 19 + x * 11) % 200);
            const uchar g = static_cast<uchar>(33 + (y * 13 + x * 7) % 180);
            const uchar r = static_cast<uchar>(49 + (y * 9 + x * 5) % 160);
            bgr.at<uchar>(y, x, 0) = b;
            bgr.at<uchar>(y, x, 1) = g;
            bgr.at<uchar>(y, x, 2) = r;
            rgb.at<uchar>(y, x, 0) = r;
            rgb.at<uchar>(y, x, 1) = g;
            rgb.at<uchar>(y, x, 2) = b;
        }
    }

    Mat nv24_from_bgr_expected = color3_to_yuv444sp_reference_u8(bgr, false, false);
    Mat nv24_from_bgr_actual;
    cvtColor(bgr, nv24_from_bgr_actual, COLOR_BGR2YUV_NV24);
    ASSERT_EQ(nv24_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(nv24_from_bgr_actual.size[0], kRows * 3);
    EXPECT_EQ(nv24_from_bgr_actual.size[1], kCols);
    EXPECT_EQ(max_abs_diff_u8(nv24_from_bgr_expected, nv24_from_bgr_actual), 0);

    Mat nv42_from_bgr_expected = color3_to_yuv444sp_reference_u8(bgr, false, true);
    Mat nv42_from_bgr_actual;
    cvtColor(bgr, nv42_from_bgr_actual, COLOR_BGR2YUV_NV42);
    ASSERT_EQ(nv42_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv42_from_bgr_expected, nv42_from_bgr_actual), 0);

    Mat nv24_from_rgb_expected = color3_to_yuv444sp_reference_u8(rgb, true, false);
    Mat nv24_from_rgb_actual;
    cvtColor(rgb, nv24_from_rgb_actual, COLOR_RGB2YUV_NV24);
    ASSERT_EQ(nv24_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv24_from_rgb_expected, nv24_from_rgb_actual), 0);

    Mat nv42_from_rgb_expected = color3_to_yuv444sp_reference_u8(rgb, true, true);
    Mat nv42_from_rgb_actual;
    cvtColor(rgb, nv42_from_rgb_actual, COLOR_RGB2YUV_NV42);
    ASSERT_EQ(nv42_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv42_from_rgb_expected, nv42_from_rgb_actual), 0);
}

TEST(CvtColorYuv444Test, bgr_rgb_to_i444_yv24_yuv444p_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 5;

    Mat bgr({kRows, kCols}, CV_8UC3);
    Mat rgb({kRows, kCols}, CV_8UC3);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(21 + (y * 17 + x * 9) % 190);
            const uchar g = static_cast<uchar>(37 + (y * 11 + x * 7) % 170);
            const uchar r = static_cast<uchar>(53 + (y * 13 + x * 5) % 150);
            bgr.at<uchar>(y, x, 0) = b;
            bgr.at<uchar>(y, x, 1) = g;
            bgr.at<uchar>(y, x, 2) = r;
            rgb.at<uchar>(y, x, 0) = r;
            rgb.at<uchar>(y, x, 1) = g;
            rgb.at<uchar>(y, x, 2) = b;
        }
    }

    Mat i444_from_bgr_expected = color3_to_yuv444p_reference_u8(bgr, false, false);
    Mat i444_from_bgr_actual;
    cvtColor(bgr, i444_from_bgr_actual, COLOR_BGR2YUV_I444);
    ASSERT_EQ(i444_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(i444_from_bgr_actual.size[0], kRows * 3);
    EXPECT_EQ(i444_from_bgr_actual.size[1], kCols);
    EXPECT_EQ(max_abs_diff_u8(i444_from_bgr_expected, i444_from_bgr_actual), 0);

    Mat yv24_from_bgr_expected = color3_to_yuv444p_reference_u8(bgr, false, true);
    Mat yv24_from_bgr_actual;
    cvtColor(bgr, yv24_from_bgr_actual, COLOR_BGR2YUV_YV24);
    ASSERT_EQ(yv24_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(yv24_from_bgr_expected, yv24_from_bgr_actual), 0);

    Mat i444_from_rgb_expected = color3_to_yuv444p_reference_u8(rgb, true, false);
    Mat i444_from_rgb_actual;
    cvtColor(rgb, i444_from_rgb_actual, COLOR_RGB2YUV_I444);
    ASSERT_EQ(i444_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(i444_from_rgb_expected, i444_from_rgb_actual), 0);

    Mat yv24_from_rgb_expected = color3_to_yuv444p_reference_u8(rgb, true, true);
    Mat yv24_from_rgb_actual;
    cvtColor(rgb, yv24_from_rgb_actual, COLOR_RGB2YUV_YV24);
    ASSERT_EQ(yv24_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(yv24_from_rgb_expected, yv24_from_rgb_actual), 0);
}

TEST(CvtColorYuv444Test, i444_yv24_yuv444p_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 5;
    constexpr int kPlaneSize = kRows * kCols;

    Mat i444({kRows * 3, kCols}, CV_8UC1);
    Mat yv24({kRows * 3, kCols}, CV_8UC1);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(20 + (y * 19 + x * 13) % 200);
            i444.at<uchar>(y, x) = yy;
            yv24.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const int chroma_index = y * kCols + x;
            const uchar uu = static_cast<uchar>(48 + (y * 11 + x * 7) % 150);
            const uchar vv = static_cast<uchar>(62 + (y * 17 + x * 5) % 140);
            set_yuv444p_plane_byte_u8(i444, kRows, kCols, 0, chroma_index, uu);
            set_yuv444p_plane_byte_u8(i444, kRows, kCols, kPlaneSize, chroma_index, vv);
            set_yuv444p_plane_byte_u8(yv24, kRows, kCols, 0, chroma_index, vv);
            set_yuv444p_plane_byte_u8(yv24, kRows, kCols, kPlaneSize, chroma_index, uu);
        }
    }

    Mat bgr_i444_expected = yuv444p_to_color3_reference_u8(i444, false, false);
    Mat bgr_i444_actual;
    cvtColor(i444, bgr_i444_actual, COLOR_YUV2BGR_I444);
    ASSERT_EQ(bgr_i444_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_i444_expected, bgr_i444_actual), 0);

    Mat rgb_i444_expected = yuv444p_to_color3_reference_u8(i444, false, true);
    Mat rgb_i444_actual;
    cvtColor(i444, rgb_i444_actual, COLOR_YUV2RGB_I444);
    ASSERT_EQ(rgb_i444_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_i444_expected, rgb_i444_actual), 0);

    Mat bgr_yv24_expected = yuv444p_to_color3_reference_u8(yv24, true, false);
    Mat bgr_yv24_actual;
    cvtColor(yv24, bgr_yv24_actual, COLOR_YUV2BGR_YV24);
    ASSERT_EQ(bgr_yv24_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_yv24_expected, bgr_yv24_actual), 0);

    Mat rgb_yv24_expected = yuv444p_to_color3_reference_u8(yv24, true, true);
    Mat rgb_yv24_actual;
    cvtColor(yv24, rgb_yv24_actual, COLOR_YUV2RGB_YV24);
    ASSERT_EQ(rgb_yv24_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_yv24_expected, rgb_yv24_actual), 0);
}

TEST(CvtColorYuv444Test, nv24_nv42_yuv444sp_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 5;

    Mat nv24({kRows * 3, kCols}, CV_8UC1);
    Mat nv42({kRows * 3, kCols}, CV_8UC1);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(18 + (y * 23 + x * 13) % 210);
            nv24.at<uchar>(y, x) = yy;
            nv42.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar uu = static_cast<uchar>(44 + (y * 17 + x * 7) % 160);
            const uchar vv = static_cast<uchar>(58 + (y * 11 + x * 9) % 150);
            const int base = y * (kCols * 2) + x * 2;
            set_yuv444sp_plane_byte_u8(nv24, kRows, kCols, base + 0, uu);
            set_yuv444sp_plane_byte_u8(nv24, kRows, kCols, base + 1, vv);
            set_yuv444sp_plane_byte_u8(nv42, kRows, kCols, base + 0, vv);
            set_yuv444sp_plane_byte_u8(nv42, kRows, kCols, base + 1, uu);
        }
    }

    Mat bgr_nv24_expected = yuv444sp_to_color3_reference_u8(nv24, false, false);
    Mat bgr_nv24_actual;
    cvtColor(nv24, bgr_nv24_actual, COLOR_YUV2BGR_NV24);
    ASSERT_EQ(bgr_nv24_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv24_expected, bgr_nv24_actual), 0);

    Mat rgb_nv24_expected = yuv444sp_to_color3_reference_u8(nv24, false, true);
    Mat rgb_nv24_actual;
    cvtColor(nv24, rgb_nv24_actual, COLOR_YUV2RGB_NV24);
    ASSERT_EQ(rgb_nv24_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv24_expected, rgb_nv24_actual), 0);

    Mat bgr_nv42_expected = yuv444sp_to_color3_reference_u8(nv42, true, false);
    Mat bgr_nv42_actual;
    cvtColor(nv42, bgr_nv42_actual, COLOR_YUV2BGR_NV42);
    ASSERT_EQ(bgr_nv42_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv42_expected, bgr_nv42_actual), 0);

    Mat rgb_nv42_expected = yuv444sp_to_color3_reference_u8(nv42, true, true);
    Mat rgb_nv42_actual;
    cvtColor(nv42, rgb_nv42_actual, COLOR_YUV2RGB_NV42);
    ASSERT_EQ(rgb_nv42_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv42_expected, rgb_nv42_actual), 0);
}

TEST(CvtColorYuv444Test, non_contiguous_roi_for_yuv_family_matches_reference)
{
    Mat base_bgr({6, 10}, CV_8UC3);
    for (int y = 0; y < base_bgr.size[0]; ++y)
    {
        for (int x = 0; x < base_bgr.size[1]; ++x)
        {
            base_bgr.at<uchar>(y, x, 0) = static_cast<uchar>((y * 23 + x * 3 + 1) % 256);
            base_bgr.at<uchar>(y, x, 1) = static_cast<uchar>((y * 17 + x * 5 + 2) % 256);
            base_bgr.at<uchar>(y, x, 2) = static_cast<uchar>((y * 11 + x * 7 + 3) % 256);
        }
    }
    Mat bgr_roi = base_bgr.colRange(1, 9);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat yuv_expected = color3_to_yuv_reference<uchar>(bgr_roi, false);
    Mat yuv_actual;
    cvtColor(bgr_roi, yuv_actual, COLOR_BGR2YUV);
    EXPECT_EQ(max_abs_diff_u8(yuv_expected, yuv_actual), 0);

    Mat base_yuv_f32({7, 11}, CV_32FC3);
    for (int y = 0; y < base_yuv_f32.size[0]; ++y)
    {
        for (int x = 0; x < base_yuv_f32.size[1]; ++x)
        {
            base_yuv_f32.at<float>(y, x, 0) = static_cast<float>(0.15 + y * 0.04 + x * 0.03);
            base_yuv_f32.at<float>(y, x, 1) = static_cast<float>(0.50 - y * 0.02 + x * 0.01);
            base_yuv_f32.at<float>(y, x, 2) = static_cast<float>(0.45 + y * 0.03 - x * 0.02);
        }
    }
    Mat yuv_roi = base_yuv_f32.colRange(2, 10);
    ASSERT_FALSE(yuv_roi.isContinuous());

    Mat rgb_expected = yuv_to_color3_reference<float>(yuv_roi, true);
    Mat rgb_actual;
    cvtColor(yuv_roi, rgb_actual, COLOR_YUV2RGB);
    EXPECT_LE(max_abs_diff_f32(rgb_expected, rgb_actual), 1e-6f);
}

TEST(CvtColorYuv444Test, non_contiguous_roi_for_nv24_nv42_encode_matches_reference)
{
    constexpr int kRows = 5;
    constexpr int kCols = 6;

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
            const uchar b = static_cast<uchar>(23 + (y * 17 + x * 7) % 190);
            const uchar g = static_cast<uchar>(41 + (y * 11 + x * 5) % 170);
            const uchar r = static_cast<uchar>(59 + (y * 13 + x * 9) % 150);
            bgr_roi.at<uchar>(y, x, 0) = b;
            bgr_roi.at<uchar>(y, x, 1) = g;
            bgr_roi.at<uchar>(y, x, 2) = r;
            rgb_roi.at<uchar>(y, x, 0) = r;
            rgb_roi.at<uchar>(y, x, 1) = g;
            rgb_roi.at<uchar>(y, x, 2) = b;
        }
    }

    Mat nv24_expected = color3_to_yuv444sp_reference_u8(bgr_roi, false, false);
    Mat nv24_actual;
    cvtColor(bgr_roi, nv24_actual, COLOR_BGR2YUV_NV24);
    EXPECT_EQ(max_abs_diff_u8(nv24_expected, nv24_actual), 0);

    Mat nv42_expected = color3_to_yuv444sp_reference_u8(rgb_roi, true, true);
    Mat nv42_actual;
    cvtColor(rgb_roi, nv42_actual, COLOR_RGB2YUV_NV42);
    EXPECT_EQ(max_abs_diff_u8(nv42_expected, nv42_actual), 0);
}

TEST(CvtColorYuv444Test, non_contiguous_roi_for_i444_yv24_encode_matches_reference)
{
    constexpr int kRows = 5;
    constexpr int kCols = 6;

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
            const uchar b = static_cast<uchar>(25 + (y * 19 + x * 7) % 180);
            const uchar g = static_cast<uchar>(43 + (y * 13 + x * 5) % 170);
            const uchar r = static_cast<uchar>(61 + (y * 11 + x * 9) % 150);
            bgr_roi.at<uchar>(y, x, 0) = b;
            bgr_roi.at<uchar>(y, x, 1) = g;
            bgr_roi.at<uchar>(y, x, 2) = r;
            rgb_roi.at<uchar>(y, x, 0) = r;
            rgb_roi.at<uchar>(y, x, 1) = g;
            rgb_roi.at<uchar>(y, x, 2) = b;
        }
    }

    Mat i444_expected = color3_to_yuv444p_reference_u8(bgr_roi, false, false);
    Mat i444_actual;
    cvtColor(bgr_roi, i444_actual, COLOR_BGR2YUV_I444);
    EXPECT_EQ(max_abs_diff_u8(i444_expected, i444_actual), 0);

    Mat yv24_expected = color3_to_yuv444p_reference_u8(rgb_roi, true, true);
    Mat yv24_actual;
    cvtColor(rgb_roi, yv24_actual, COLOR_RGB2YUV_YV24);
    EXPECT_EQ(max_abs_diff_u8(yv24_expected, yv24_actual), 0);
}

TEST(CvtColorYuv444Test, non_contiguous_step_for_i444_yv24_matches_reference)
{
    constexpr int kRows = 5;
    constexpr int kCols = 6;
    constexpr int kPlaneSize = kRows * kCols;

    Mat base_i444({kRows * 3, kCols + 3}, CV_8UC1);
    Mat i444_roi = base_i444.colRange(1, 1 + kCols);
    ASSERT_FALSE(i444_roi.isContinuous());

    Mat base_yv24({kRows * 3, kCols + 3}, CV_8UC1);
    Mat yv24_roi = base_yv24.colRange(1, 1 + kCols);
    ASSERT_FALSE(yv24_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(24 + (y * 17 + x * 9) % 190);
            i444_roi.at<uchar>(y, x) = yy;
            yv24_roi.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const int chroma_index = y * kCols + x;
            const uchar uu = static_cast<uchar>(50 + (y * 13 + x * 5) % 150);
            const uchar vv = static_cast<uchar>(68 + (y * 7 + x * 11) % 140);
            set_yuv444p_plane_byte_u8(i444_roi, kRows, kCols, 0, chroma_index, uu);
            set_yuv444p_plane_byte_u8(i444_roi, kRows, kCols, kPlaneSize, chroma_index, vv);
            set_yuv444p_plane_byte_u8(yv24_roi, kRows, kCols, 0, chroma_index, vv);
            set_yuv444p_plane_byte_u8(yv24_roi, kRows, kCols, kPlaneSize, chroma_index, uu);
        }
    }

    Mat bgr_i444_expected = yuv444p_to_color3_reference_u8(i444_roi, false, false);
    Mat bgr_i444_actual;
    cvtColor(i444_roi, bgr_i444_actual, COLOR_YUV2BGR_I444);
    EXPECT_EQ(max_abs_diff_u8(bgr_i444_expected, bgr_i444_actual), 0);

    Mat rgb_yv24_expected = yuv444p_to_color3_reference_u8(yv24_roi, true, true);
    Mat rgb_yv24_actual;
    cvtColor(yv24_roi, rgb_yv24_actual, COLOR_YUV2RGB_YV24);
    EXPECT_EQ(max_abs_diff_u8(rgb_yv24_expected, rgb_yv24_actual), 0);
}

TEST(CvtColorYuv444Test, non_contiguous_step_for_nv24_nv42_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 5;

    Mat base_nv24({kRows * 3, kCols + 3}, CV_8UC1);
    Mat nv24_roi = base_nv24.colRange(1, 1 + kCols);
    ASSERT_FALSE(nv24_roi.isContinuous());

    Mat base_nv42({kRows * 3, kCols + 3}, CV_8UC1);
    Mat nv42_roi = base_nv42.colRange(1, 1 + kCols);
    ASSERT_FALSE(nv42_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(26 + (y * 19 + x * 11) % 200);
            nv24_roi.at<uchar>(y, x) = yy;
            nv42_roi.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar uu = static_cast<uchar>(52 + (y * 13 + x * 5) % 150);
            const uchar vv = static_cast<uchar>(70 + (y * 7 + x * 9) % 140);
            const int base = y * (kCols * 2) + x * 2;
            set_yuv444sp_plane_byte_u8(nv24_roi, kRows, kCols, base + 0, uu);
            set_yuv444sp_plane_byte_u8(nv24_roi, kRows, kCols, base + 1, vv);
            set_yuv444sp_plane_byte_u8(nv42_roi, kRows, kCols, base + 0, vv);
            set_yuv444sp_plane_byte_u8(nv42_roi, kRows, kCols, base + 1, uu);
        }
    }

    Mat bgr_nv24_expected = yuv444sp_to_color3_reference_u8(nv24_roi, false, false);
    Mat bgr_nv24_actual;
    cvtColor(nv24_roi, bgr_nv24_actual, COLOR_YUV2BGR_NV24);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv24_expected, bgr_nv24_actual), 0);

    Mat rgb_nv42_expected = yuv444sp_to_color3_reference_u8(nv42_roi, true, true);
    Mat rgb_nv42_actual;
    cvtColor(nv42_roi, rgb_nv42_actual, COLOR_YUV2RGB_NV42);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv42_expected, rgb_nv42_actual), 0);
}

TEST(CvtColorYuv444Test, throws_on_invalid_i444_yv24_layouts)
{
    Mat dst;

    Mat bad_rows({11, 5}, CV_8UC1);
    EXPECT_THROW(cvtColor(bad_rows, dst, COLOR_YUV2BGR_I444), Exception);

    Mat three_channel({12, 5}, CV_8UC3);
    EXPECT_THROW(cvtColor(three_channel, dst, COLOR_YUV2RGB_YV24), Exception);

    Mat f32_src({12, 5}, CV_32FC1);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_YUV2RGB_I444), Exception);
}

TEST(CvtColorYuv444Test, throws_on_invalid_nv24_nv42_layouts)
{
    Mat dst;

    Mat bad_rows({11, 5}, CV_8UC1);
    EXPECT_THROW(cvtColor(bad_rows, dst, COLOR_YUV2BGR_NV24), Exception);

    Mat three_channel({12, 5}, CV_8UC3);
    EXPECT_THROW(cvtColor(three_channel, dst, COLOR_YUV2RGB_NV42), Exception);

    Mat f32_src({12, 5}, CV_32FC1);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_YUV2BGR_NV42), Exception);
}

TEST(CvtColorYuv444Test, throws_on_invalid_bgr_rgb_to_nv24_nv42_inputs)
{
    Mat dst;

    Mat gray_src({5, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_BGR2YUV_NV24), Exception);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_RGB2YUV_NV42), Exception);

    Mat bgra_src({5, 6}, CV_8UC4);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_RGB2YUV_NV24), Exception);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_BGR2YUV_NV42), Exception);

    Mat f32_src({5, 6}, CV_32FC3);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_BGR2YUV_NV24), Exception);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_RGB2YUV_NV42), Exception);
}

TEST(CvtColorYuv444Test, throws_on_invalid_bgr_rgb_to_i444_yv24_inputs)
{
    Mat dst;

    Mat gray_src({5, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_BGR2YUV_I444), Exception);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_RGB2YUV_YV24), Exception);

    Mat bgra_src({5, 6}, CV_8UC4);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_RGB2YUV_I444), Exception);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_BGR2YUV_YV24), Exception);

    Mat f32_src({5, 6}, CV_32FC3);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_BGR2YUV_I444), Exception);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_RGB2YUV_YV24), Exception);
}

TEST(CvtColorYuv444Test, supports_cv32f_yuv_family_conversions)
{
    Mat bgr({3, 4}, CV_32FC3);
    for (int y = 0; y < bgr.size[0]; ++y)
    {
        for (int x = 0; x < bgr.size[1]; ++x)
        {
            bgr.at<float>(y, x, 0) = static_cast<float>(0.10 + y * 0.07 + x * 0.03);
            bgr.at<float>(y, x, 1) = static_cast<float>(0.20 + y * 0.05 + x * 0.04);
            bgr.at<float>(y, x, 2) = static_cast<float>(0.30 + y * 0.06 + x * 0.02);
        }
    }

    Mat yuv_expected = color3_to_yuv_reference<float>(bgr, false);
    Mat yuv_actual;
    cvtColor(bgr, yuv_actual, COLOR_BGR2YUV);
    ASSERT_EQ(yuv_actual.type(), CV_32FC3);
    EXPECT_LE(max_abs_diff_f32(yuv_expected, yuv_actual), 1e-6f);

    Mat bgr_expected = yuv_to_color3_reference<float>(yuv_actual, false);
    Mat bgr_actual;
    cvtColor(yuv_actual, bgr_actual, COLOR_YUV2BGR);
    ASSERT_EQ(bgr_actual.type(), CV_32FC3);
    EXPECT_LE(max_abs_diff_f32(bgr_expected, bgr_actual), 1e-6f);

    Mat rgb = bgr2rgb_reference<float>(bgr);
    Mat yuv_from_rgb_expected = color3_to_yuv_reference<float>(rgb, true);
    Mat yuv_from_rgb_actual;
    cvtColor(rgb, yuv_from_rgb_actual, COLOR_RGB2YUV);
    ASSERT_EQ(yuv_from_rgb_actual.type(), CV_32FC3);
    EXPECT_LE(max_abs_diff_f32(yuv_from_rgb_expected, yuv_from_rgb_actual), 1e-6f);

    Mat rgb_expected = yuv_to_color3_reference<float>(yuv_from_rgb_actual, true);
    Mat rgb_actual;
    cvtColor(yuv_from_rgb_actual, rgb_actual, COLOR_YUV2RGB);
    ASSERT_EQ(rgb_actual.type(), CV_32FC3);
    EXPECT_LE(max_abs_diff_f32(rgb_expected, rgb_actual), 1e-6f);
}
