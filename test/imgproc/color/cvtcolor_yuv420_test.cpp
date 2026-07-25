#include "test/imgproc/support/cvtcolor_test_utils.hpp"

TEST(CvtColorYuv420Test, bgr_rgb_to_nv12_nv21_yuv420sp_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 6;

    Mat bgr({kRows, kCols}, CV_8UC3);
    Mat rgb({kRows, kCols}, CV_8UC3);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(17 + (y * 19 + x * 9) % 180);
            const uchar g = static_cast<uchar>(33 + (y * 13 + x * 7) % 170);
            const uchar r = static_cast<uchar>(49 + (y * 11 + x * 5) % 160);
            bgr.at<uchar>(y, x, 0) = b;
            bgr.at<uchar>(y, x, 1) = g;
            bgr.at<uchar>(y, x, 2) = r;
            rgb.at<uchar>(y, x, 0) = r;
            rgb.at<uchar>(y, x, 1) = g;
            rgb.at<uchar>(y, x, 2) = b;
        }
    }

    Mat nv12_from_bgr_expected = color3_to_yuv420sp_reference_u8(bgr, false, false);
    Mat nv12_from_bgr_actual;
    cvtColor(bgr, nv12_from_bgr_actual, COLOR_BGR2YUV_NV12);
    ASSERT_EQ(nv12_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(nv12_from_bgr_actual.size[0], kRows * 3 / 2);
    EXPECT_EQ(nv12_from_bgr_actual.size[1], kCols);
    EXPECT_EQ(max_abs_diff_u8(nv12_from_bgr_expected, nv12_from_bgr_actual), 0);

    Mat nv21_from_bgr_expected = color3_to_yuv420sp_reference_u8(bgr, false, true);
    Mat nv21_from_bgr_actual;
    cvtColor(bgr, nv21_from_bgr_actual, COLOR_BGR2YUV_NV21);
    ASSERT_EQ(nv21_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv21_from_bgr_expected, nv21_from_bgr_actual), 0);

    Mat nv12_from_rgb_expected = color3_to_yuv420sp_reference_u8(rgb, true, false);
    Mat nv12_from_rgb_actual;
    cvtColor(rgb, nv12_from_rgb_actual, COLOR_RGB2YUV_NV12);
    ASSERT_EQ(nv12_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv12_from_rgb_expected, nv12_from_rgb_actual), 0);

    Mat nv21_from_rgb_expected = color3_to_yuv420sp_reference_u8(rgb, true, true);
    Mat nv21_from_rgb_actual;
    cvtColor(rgb, nv21_from_rgb_actual, COLOR_RGB2YUV_NV21);
    ASSERT_EQ(nv21_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(nv21_from_rgb_expected, nv21_from_rgb_actual), 0);
}

TEST(CvtColorYuv420Test, bgr_rgb_to_i420_yv12_yuv420p_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 6;

    Mat bgr({kRows, kCols}, CV_8UC3);
    Mat rgb({kRows, kCols}, CV_8UC3);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(21 + (y * 17 + x * 9) % 180);
            const uchar g = static_cast<uchar>(37 + (y * 11 + x * 7) % 170);
            const uchar r = static_cast<uchar>(53 + (y * 13 + x * 5) % 160);
            bgr.at<uchar>(y, x, 0) = b;
            bgr.at<uchar>(y, x, 1) = g;
            bgr.at<uchar>(y, x, 2) = r;
            rgb.at<uchar>(y, x, 0) = r;
            rgb.at<uchar>(y, x, 1) = g;
            rgb.at<uchar>(y, x, 2) = b;
        }
    }

    Mat i420_from_bgr_expected = color3_to_yuv420p_reference_u8(bgr, false, false);
    Mat i420_from_bgr_actual;
    cvtColor(bgr, i420_from_bgr_actual, COLOR_BGR2YUV_I420);
    ASSERT_EQ(i420_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(i420_from_bgr_actual.size[0], kRows * 3 / 2);
    EXPECT_EQ(i420_from_bgr_actual.size[1], kCols);
    EXPECT_EQ(max_abs_diff_u8(i420_from_bgr_expected, i420_from_bgr_actual), 0);

    Mat yv12_from_bgr_expected = color3_to_yuv420p_reference_u8(bgr, false, true);
    Mat yv12_from_bgr_actual;
    cvtColor(bgr, yv12_from_bgr_actual, COLOR_BGR2YUV_YV12);
    ASSERT_EQ(yv12_from_bgr_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(yv12_from_bgr_expected, yv12_from_bgr_actual), 0);

    Mat i420_from_rgb_expected = color3_to_yuv420p_reference_u8(rgb, true, false);
    Mat i420_from_rgb_actual;
    cvtColor(rgb, i420_from_rgb_actual, COLOR_RGB2YUV_I420);
    ASSERT_EQ(i420_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(i420_from_rgb_expected, i420_from_rgb_actual), 0);

    Mat yv12_from_rgb_expected = color3_to_yuv420p_reference_u8(rgb, true, true);
    Mat yv12_from_rgb_actual;
    cvtColor(rgb, yv12_from_rgb_actual, COLOR_RGB2YUV_YV12);
    ASSERT_EQ(yv12_from_rgb_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(yv12_from_rgb_expected, yv12_from_rgb_actual), 0);
}

TEST(CvtColorYuv420Test, nv12_nv21_yuv420sp_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 6;

    Mat nv12({kRows * 3 / 2, kCols}, CV_8UC1);
    Mat nv21({kRows * 3 / 2, kCols}, CV_8UC1);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(16 + (y * 23 + x * 11) % 220);
            nv12.at<uchar>(y, x) = yy;
            nv21.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows / 2; ++y)
    {
        for (int x = 0; x < kCols; x += 2)
        {
            const uchar uu = static_cast<uchar>(40 + (y * 17 + x * 9) % 160);
            const uchar vv = static_cast<uchar>(60 + (y * 13 + x * 7) % 150);

            nv12.at<uchar>(kRows + y, x + 0) = uu;
            nv12.at<uchar>(kRows + y, x + 1) = vv;
            nv21.at<uchar>(kRows + y, x + 0) = vv;
            nv21.at<uchar>(kRows + y, x + 1) = uu;
        }
    }

    Mat bgr_nv12_expected = yuv420sp_to_color3_reference_u8(nv12, false, false);
    Mat bgr_nv12_actual;
    cvtColor(nv12, bgr_nv12_actual, COLOR_YUV2BGR_NV12);
    ASSERT_EQ(bgr_nv12_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv12_expected, bgr_nv12_actual), 0);

    Mat rgb_nv12_expected = yuv420sp_to_color3_reference_u8(nv12, false, true);
    Mat rgb_nv12_actual;
    cvtColor(nv12, rgb_nv12_actual, COLOR_YUV2RGB_NV12);
    ASSERT_EQ(rgb_nv12_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv12_expected, rgb_nv12_actual), 0);

    Mat bgr_nv21_expected = yuv420sp_to_color3_reference_u8(nv21, true, false);
    Mat bgr_nv21_actual;
    cvtColor(nv21, bgr_nv21_actual, COLOR_YUV2BGR_NV21);
    ASSERT_EQ(bgr_nv21_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv21_expected, bgr_nv21_actual), 0);

    Mat rgb_nv21_expected = yuv420sp_to_color3_reference_u8(nv21, true, true);
    Mat rgb_nv21_actual;
    cvtColor(nv21, rgb_nv21_actual, COLOR_YUV2RGB_NV21);
    ASSERT_EQ(rgb_nv21_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv21_expected, rgb_nv21_actual), 0);
}

TEST(CvtColorYuv420Test, i420_yv12_yuv420p_u8_matches_reference)
{
    constexpr int kRows = 4;
    constexpr int kCols = 6;
    constexpr int kUvSize = kRows * kCols / 4;

    Mat i420({kRows * 3 / 2, kCols}, CV_8UC1);
    Mat yv12({kRows * 3 / 2, kCols}, CV_8UC1);
    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(16 + (y * 19 + x * 13) % 220);
            i420.at<uchar>(y, x) = yy;
            yv12.at<uchar>(y, x) = yy;
        }
    }

    for (int i = 0; i < kUvSize; ++i)
    {
        const uchar uu = static_cast<uchar>(44 + (i * 9) % 160);
        const uchar vv = static_cast<uchar>(58 + (i * 11) % 150);
        set_yuv420p_plane_byte_u8(i420, kRows, kCols, 0, i, uu);
        set_yuv420p_plane_byte_u8(i420, kRows, kCols, kUvSize, i, vv);
        set_yuv420p_plane_byte_u8(yv12, kRows, kCols, 0, i, vv);
        set_yuv420p_plane_byte_u8(yv12, kRows, kCols, kUvSize, i, uu);
    }

    Mat bgr_i420_expected = yuv420p_to_color3_reference_u8(i420, false, false);
    Mat bgr_i420_actual;
    cvtColor(i420, bgr_i420_actual, COLOR_YUV2BGR_I420);
    ASSERT_EQ(bgr_i420_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_i420_expected, bgr_i420_actual), 0);

    Mat rgb_i420_expected = yuv420p_to_color3_reference_u8(i420, false, true);
    Mat rgb_i420_actual;
    cvtColor(i420, rgb_i420_actual, COLOR_YUV2RGB_I420);
    ASSERT_EQ(rgb_i420_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_i420_expected, rgb_i420_actual), 0);

    Mat bgr_yv12_expected = yuv420p_to_color3_reference_u8(yv12, true, false);
    Mat bgr_yv12_actual;
    cvtColor(yv12, bgr_yv12_actual, COLOR_YUV2BGR_YV12);
    ASSERT_EQ(bgr_yv12_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_yv12_expected, bgr_yv12_actual), 0);

    Mat rgb_yv12_expected = yuv420p_to_color3_reference_u8(yv12, true, true);
    Mat rgb_yv12_actual;
    cvtColor(yv12, rgb_yv12_actual, COLOR_YUV2RGB_YV12);
    ASSERT_EQ(rgb_yv12_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(rgb_yv12_expected, rgb_yv12_actual), 0);
}

TEST(CvtColorYuv420Test, non_contiguous_roi_for_nv12_nv21_encode_matches_reference)
{
    constexpr int kRows = 6;
    constexpr int kCols = 8;

    Mat base_bgr({kRows + 2, kCols + 3}, CV_8UC3);
    Mat bgr_roi = base_bgr.rowRange(1, 1 + kRows).colRange(1, 1 + kCols);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat base_rgb({kRows + 2, kCols + 3}, CV_8UC3);
    Mat rgb_roi = base_rgb.rowRange(1, 1 + kRows).colRange(1, 1 + kCols);
    ASSERT_FALSE(rgb_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(29 + (y * 17 + x * 9) % 170);
            const uchar g = static_cast<uchar>(47 + (y * 11 + x * 7) % 160);
            const uchar r = static_cast<uchar>(65 + (y * 13 + x * 5) % 150);
            bgr_roi.at<uchar>(y, x, 0) = b;
            bgr_roi.at<uchar>(y, x, 1) = g;
            bgr_roi.at<uchar>(y, x, 2) = r;
            rgb_roi.at<uchar>(y, x, 0) = r;
            rgb_roi.at<uchar>(y, x, 1) = g;
            rgb_roi.at<uchar>(y, x, 2) = b;
        }
    }

    Mat nv12_expected = color3_to_yuv420sp_reference_u8(bgr_roi, false, false);
    Mat nv12_actual;
    cvtColor(bgr_roi, nv12_actual, COLOR_BGR2YUV_NV12);
    EXPECT_EQ(max_abs_diff_u8(nv12_expected, nv12_actual), 0);

    Mat nv21_expected = color3_to_yuv420sp_reference_u8(rgb_roi, true, true);
    Mat nv21_actual;
    cvtColor(rgb_roi, nv21_actual, COLOR_RGB2YUV_NV21);
    EXPECT_EQ(max_abs_diff_u8(nv21_expected, nv21_actual), 0);
}

TEST(CvtColorYuv420Test, non_contiguous_roi_for_i420_yv12_encode_matches_reference)
{
    constexpr int kRows = 6;
    constexpr int kCols = 8;

    Mat base_bgr({kRows + 2, kCols + 3}, CV_8UC3);
    Mat bgr_roi = base_bgr.rowRange(1, 1 + kRows).colRange(1, 1 + kCols);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat base_rgb({kRows + 2, kCols + 3}, CV_8UC3);
    Mat rgb_roi = base_rgb.rowRange(1, 1 + kRows).colRange(1, 1 + kCols);
    ASSERT_FALSE(rgb_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar b = static_cast<uchar>(33 + (y * 17 + x * 9) % 170);
            const uchar g = static_cast<uchar>(51 + (y * 11 + x * 7) % 160);
            const uchar r = static_cast<uchar>(69 + (y * 13 + x * 5) % 150);
            bgr_roi.at<uchar>(y, x, 0) = b;
            bgr_roi.at<uchar>(y, x, 1) = g;
            bgr_roi.at<uchar>(y, x, 2) = r;
            rgb_roi.at<uchar>(y, x, 0) = r;
            rgb_roi.at<uchar>(y, x, 1) = g;
            rgb_roi.at<uchar>(y, x, 2) = b;
        }
    }

    Mat i420_expected = color3_to_yuv420p_reference_u8(bgr_roi, false, false);
    Mat i420_actual;
    cvtColor(bgr_roi, i420_actual, COLOR_BGR2YUV_I420);
    EXPECT_EQ(max_abs_diff_u8(i420_expected, i420_actual), 0);

    Mat yv12_expected = color3_to_yuv420p_reference_u8(rgb_roi, true, true);
    Mat yv12_actual;
    cvtColor(rgb_roi, yv12_actual, COLOR_RGB2YUV_YV12);
    EXPECT_EQ(max_abs_diff_u8(yv12_expected, yv12_actual), 0);
}

TEST(CvtColorYuv420Test, non_contiguous_step_for_nv12_nv21_matches_reference)
{
    constexpr int kRows = 6;
    constexpr int kCols = 8;

    Mat base_nv12({kRows * 3 / 2, kCols + 4}, CV_8UC1);
    Mat nv12_roi = base_nv12.colRange(2, 2 + kCols);
    ASSERT_FALSE(nv12_roi.isContinuous());

    Mat base_nv21({kRows * 3 / 2, kCols + 4}, CV_8UC1);
    Mat nv21_roi = base_nv21.colRange(2, 2 + kCols);
    ASSERT_FALSE(nv21_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(32 + (y * 19 + x * 7) % 180);
            nv12_roi.at<uchar>(y, x) = yy;
            nv21_roi.at<uchar>(y, x) = yy;
        }
    }

    for (int y = 0; y < kRows / 2; ++y)
    {
        for (int x = 0; x < kCols; x += 2)
        {
            const uchar uu = static_cast<uchar>(48 + (y * 9 + x * 5) % 150);
            const uchar vv = static_cast<uchar>(70 + (y * 11 + x * 3) % 140);

            nv12_roi.at<uchar>(kRows + y, x + 0) = uu;
            nv12_roi.at<uchar>(kRows + y, x + 1) = vv;
            nv21_roi.at<uchar>(kRows + y, x + 0) = vv;
            nv21_roi.at<uchar>(kRows + y, x + 1) = uu;
        }
    }

    Mat bgr_nv12_expected = yuv420sp_to_color3_reference_u8(nv12_roi, false, false);
    Mat bgr_nv12_actual;
    cvtColor(nv12_roi, bgr_nv12_actual, COLOR_YUV2BGR_NV12);
    EXPECT_EQ(max_abs_diff_u8(bgr_nv12_expected, bgr_nv12_actual), 0);

    Mat rgb_nv21_expected = yuv420sp_to_color3_reference_u8(nv21_roi, true, true);
    Mat rgb_nv21_actual;
    cvtColor(nv21_roi, rgb_nv21_actual, COLOR_YUV2RGB_NV21);
    EXPECT_EQ(max_abs_diff_u8(rgb_nv21_expected, rgb_nv21_actual), 0);
}

TEST(CvtColorYuv420Test, non_contiguous_step_for_i420_yv12_matches_reference)
{
    constexpr int kRows = 6;
    constexpr int kCols = 8;
    constexpr int kUvSize = kRows * kCols / 4;

    Mat base_i420({kRows * 3 / 2, kCols + 4}, CV_8UC1);
    Mat i420_roi = base_i420.colRange(2, 2 + kCols);
    ASSERT_FALSE(i420_roi.isContinuous());

    Mat base_yv12({kRows * 3 / 2, kCols + 4}, CV_8UC1);
    Mat yv12_roi = base_yv12.colRange(2, 2 + kCols);
    ASSERT_FALSE(yv12_roi.isContinuous());

    for (int y = 0; y < kRows; ++y)
    {
        for (int x = 0; x < kCols; ++x)
        {
            const uchar yy = static_cast<uchar>(28 + (y * 17 + x * 5) % 190);
            i420_roi.at<uchar>(y, x) = yy;
            yv12_roi.at<uchar>(y, x) = yy;
        }
    }

    for (int i = 0; i < kUvSize; ++i)
    {
        const uchar uu = static_cast<uchar>(52 + (i * 7) % 150);
        const uchar vv = static_cast<uchar>(66 + (i * 9) % 140);
        set_yuv420p_plane_byte_u8(i420_roi, kRows, kCols, 0, i, uu);
        set_yuv420p_plane_byte_u8(i420_roi, kRows, kCols, kUvSize, i, vv);
        set_yuv420p_plane_byte_u8(yv12_roi, kRows, kCols, 0, i, vv);
        set_yuv420p_plane_byte_u8(yv12_roi, kRows, kCols, kUvSize, i, uu);
    }

    Mat bgr_i420_expected = yuv420p_to_color3_reference_u8(i420_roi, false, false);
    Mat bgr_i420_actual;
    cvtColor(i420_roi, bgr_i420_actual, COLOR_YUV2BGR_I420);
    EXPECT_EQ(max_abs_diff_u8(bgr_i420_expected, bgr_i420_actual), 0);

    Mat rgb_yv12_expected = yuv420p_to_color3_reference_u8(yv12_roi, true, true);
    Mat rgb_yv12_actual;
    cvtColor(yv12_roi, rgb_yv12_actual, COLOR_YUV2RGB_YV12);
    EXPECT_EQ(max_abs_diff_u8(rgb_yv12_expected, rgb_yv12_actual), 0);
}

TEST(CvtColorYuv420Test, throws_on_invalid_nv12_nv21_layouts)
{
    Mat dst;

    Mat odd_width({6, 5}, CV_8UC1);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_YUV2BGR_NV12), Exception);

    Mat bad_rows({5, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(bad_rows, dst, COLOR_YUV2RGB_NV12), Exception);

    Mat three_channel({6, 6}, CV_8UC3);
    EXPECT_THROW(cvtColor(three_channel, dst, COLOR_YUV2BGR_NV21), Exception);

    Mat f32_src({6, 6}, CV_32FC1);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_YUV2RGB_NV21), Exception);
}

TEST(CvtColorYuv420Test, throws_on_invalid_i420_yv12_layouts)
{
    Mat dst;

    Mat odd_width({6, 5}, CV_8UC1);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_YUV2BGR_I420), Exception);

    Mat bad_rows({5, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(bad_rows, dst, COLOR_YUV2RGB_YV12), Exception);

    Mat three_channel({6, 6}, CV_8UC3);
    EXPECT_THROW(cvtColor(three_channel, dst, COLOR_YUV2BGR_YV12), Exception);

    Mat f32_src({6, 6}, CV_32FC1);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_YUV2RGB_I420), Exception);
}

TEST(CvtColorYuv420Test, throws_on_invalid_bgr_rgb_to_nv12_nv21_inputs)
{
    Mat dst;

    Mat gray_src({6, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_BGR2YUV_NV12), Exception);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_RGB2YUV_NV21), Exception);

    Mat bgra_src({6, 6}, CV_8UC4);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_RGB2YUV_NV12), Exception);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_BGR2YUV_NV21), Exception);

    Mat odd_width({6, 5}, CV_8UC3);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_BGR2YUV_NV12), Exception);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_RGB2YUV_NV21), Exception);

    Mat odd_height({5, 6}, CV_8UC3);
    EXPECT_THROW(cvtColor(odd_height, dst, COLOR_BGR2YUV_NV12), Exception);
    EXPECT_THROW(cvtColor(odd_height, dst, COLOR_RGB2YUV_NV21), Exception);

    Mat f32_src({6, 6}, CV_32FC3);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_BGR2YUV_NV12), Exception);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_RGB2YUV_NV21), Exception);
}

TEST(CvtColorYuv420Test, throws_on_invalid_bgr_rgb_to_i420_yv12_inputs)
{
    Mat dst;

    Mat gray_src({6, 6}, CV_8UC1);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_BGR2YUV_I420), Exception);
    EXPECT_THROW(cvtColor(gray_src, dst, COLOR_RGB2YUV_YV12), Exception);

    Mat bgra_src({6, 6}, CV_8UC4);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_RGB2YUV_I420), Exception);
    EXPECT_THROW(cvtColor(bgra_src, dst, COLOR_BGR2YUV_YV12), Exception);

    Mat odd_width({6, 5}, CV_8UC3);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_BGR2YUV_I420), Exception);
    EXPECT_THROW(cvtColor(odd_width, dst, COLOR_RGB2YUV_YV12), Exception);

    Mat odd_height({5, 6}, CV_8UC3);
    EXPECT_THROW(cvtColor(odd_height, dst, COLOR_BGR2YUV_I420), Exception);
    EXPECT_THROW(cvtColor(odd_height, dst, COLOR_RGB2YUV_YV12), Exception);

    Mat f32_src({6, 6}, CV_32FC3);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_BGR2YUV_I420), Exception);
    EXPECT_THROW(cvtColor(f32_src, dst, COLOR_RGB2YUV_YV12), Exception);
}
