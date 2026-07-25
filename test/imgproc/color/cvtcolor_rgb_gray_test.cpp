#include "test/imgproc/support/cvtcolor_test_utils.hpp"

TEST(CvtColorRgbGrayTest, bgr2gray_matches_known_values)
{
    Mat src({2, 2}, CV_8UC3);
    src.at<uchar>(0, 0, 0) = 10;  src.at<uchar>(0, 0, 1) = 20;  src.at<uchar>(0, 0, 2) = 30;
    src.at<uchar>(0, 1, 0) = 100; src.at<uchar>(0, 1, 1) = 110; src.at<uchar>(0, 1, 2) = 120;
    src.at<uchar>(1, 0, 0) = 0;   src.at<uchar>(1, 0, 1) = 0;   src.at<uchar>(1, 0, 2) = 255;
    src.at<uchar>(1, 1, 0) = 255; src.at<uchar>(1, 1, 1) = 0;   src.at<uchar>(1, 1, 2) = 0;

    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    ASSERT_EQ(gray.type(), CV_8UC1);
    ASSERT_EQ(gray.size[0], 2);
    ASSERT_EQ(gray.size[1], 2);

    const uchar expected[4] = {22, 112, 76, 29};
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            EXPECT_EQ(gray.at<uchar>(y, x), expected[y * 2 + x]);
        }
    }
}

TEST(CvtColorRgbGrayTest, rgb2gray_matches_reference)
{
    Mat src({3, 5}, CV_8UC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            src.at<uchar>(y, x, 0) = static_cast<uchar>((y * 31 + x * 17 + 3) & 0xff);
            src.at<uchar>(y, x, 1) = static_cast<uchar>((y * 13 + x * 29 + 7) & 0xff);
            src.at<uchar>(y, x, 2) = static_cast<uchar>((y * 19 + x * 11 + 23) & 0xff);
        }
    }

    Mat expected = rgb2gray_reference_u8(src);
    Mat actual;
    cvtColor(src, actual, COLOR_RGB2GRAY);
    ASSERT_EQ(actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(expected, actual), 0);
}

TEST(CvtColorRgbGrayTest, gray2bgr_replicates_channels)
{
    Mat gray({2, 3}, CV_8UC1);
    gray.at<uchar>(0, 0) = 10;
    gray.at<uchar>(0, 1) = 20;
    gray.at<uchar>(0, 2) = 30;
    gray.at<uchar>(1, 0) = 40;
    gray.at<uchar>(1, 1) = 50;
    gray.at<uchar>(1, 2) = 60;

    Mat bgr;
    cvtColor(gray, bgr, COLOR_GRAY2BGR);
    ASSERT_EQ(bgr.type(), CV_8UC3);
    ASSERT_EQ(bgr.size[0], gray.size[0]);
    ASSERT_EQ(bgr.size[1], gray.size[1]);

    for (int y = 0; y < bgr.size[0]; ++y)
    {
        for (int x = 0; x < bgr.size[1]; ++x)
        {
            const uchar v = gray.at<uchar>(y, x);
            EXPECT_EQ(bgr.at<uchar>(y, x, 0), v);
            EXPECT_EQ(bgr.at<uchar>(y, x, 1), v);
            EXPECT_EQ(bgr.at<uchar>(y, x, 2), v);
        }
    }
}

TEST(CvtColorRgbGrayTest, gray2bgr_then_bgr2gray_roundtrip_is_identity)
{
    Mat gray({4, 5}, CV_8UC1);
    for (int y = 0; y < gray.size[0]; ++y)
    {
        for (int x = 0; x < gray.size[1]; ++x)
        {
            gray.at<uchar>(y, x) = static_cast<uchar>((y * 31 + x * 17) % 256);
        }
    }

    Mat bgr;
    cvtColor(gray, bgr, COLOR_GRAY2BGR);

    Mat back;
    cvtColor(bgr, back, COLOR_BGR2GRAY);
    ASSERT_EQ(back.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(gray, back), 0);
}

TEST(CvtColorRgbGrayTest, bgr2rgb_swaps_blue_and_red_channels)
{
    Mat src({2, 3}, CV_8UC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            src.at<uchar>(y, x, 0) = static_cast<uchar>(10 + y * 20 + x * 3);
            src.at<uchar>(y, x, 1) = static_cast<uchar>(30 + y * 20 + x * 5);
            src.at<uchar>(y, x, 2) = static_cast<uchar>(50 + y * 20 + x * 7);
        }
    }

    Mat expected = bgr2rgb_reference<uchar>(src);
    Mat actual;
    cvtColor(src, actual, COLOR_BGR2RGB);
    ASSERT_EQ(actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(expected, actual), 0);

    Mat roundtrip;
    cvtColor(actual, roundtrip, COLOR_RGB2BGR);
    EXPECT_EQ(max_abs_diff_u8(src, roundtrip), 0);
}

TEST(CvtColorRgbGrayTest, bgr2bgra_and_bgra2bgr_match_reference)
{
    Mat src({2, 4}, CV_8UC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            src.at<uchar>(y, x, 0) = static_cast<uchar>(1 + y * 17 + x * 2);
            src.at<uchar>(y, x, 1) = static_cast<uchar>(2 + y * 19 + x * 3);
            src.at<uchar>(y, x, 2) = static_cast<uchar>(3 + y * 23 + x * 5);
        }
    }

    Mat bgra_expected = bgr2bgra_reference<uchar>(src);
    Mat bgra_actual;
    cvtColor(src, bgra_actual, COLOR_BGR2BGRA);
    ASSERT_EQ(bgra_actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(bgra_expected, bgra_actual), 0);

    Mat bgr_expected = bgra2bgr_reference<uchar>(bgra_actual);
    Mat bgr_actual;
    cvtColor(bgra_actual, bgr_actual, COLOR_BGRA2BGR);
    ASSERT_EQ(bgr_actual.type(), CV_8UC3);
    EXPECT_EQ(max_abs_diff_u8(bgr_expected, bgr_actual), 0);
}

TEST(CvtColorRgbGrayTest, rgb_rgba_bgr_bgra_family_u8_matches_reference)
{
    Mat rgb({2, 4}, CV_8UC3);
    for (int y = 0; y < rgb.size[0]; ++y)
    {
        for (int x = 0; x < rgb.size[1]; ++x)
        {
            rgb.at<uchar>(y, x, 0) = static_cast<uchar>(11 + y * 13 + x * 2);
            rgb.at<uchar>(y, x, 1) = static_cast<uchar>(21 + y * 17 + x * 3);
            rgb.at<uchar>(y, x, 2) = static_cast<uchar>(31 + y * 19 + x * 5);
        }
    }

    Mat rgba_expected = rgb2rgba_reference<uchar>(rgb);
    Mat rgba_actual;
    cvtColor(rgb, rgba_actual, COLOR_RGB2RGBA);
    ASSERT_EQ(rgba_actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(rgba_expected, rgba_actual), 0);

    Mat rgb_roundtrip;
    cvtColor(rgba_actual, rgb_roundtrip, COLOR_RGBA2RGB);
    EXPECT_EQ(max_abs_diff_u8(rgb, rgb_roundtrip), 0);

    Mat bgra_expected = rgb2bgra_reference<uchar>(rgb);
    Mat bgra_actual;
    cvtColor(rgb, bgra_actual, COLOR_RGB2BGRA);
    ASSERT_EQ(bgra_actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(bgra_expected, bgra_actual), 0);

    Mat rgb_from_bgra;
    cvtColor(bgra_actual, rgb_from_bgra, COLOR_BGRA2RGB);
    EXPECT_EQ(max_abs_diff_u8(rgb, rgb_from_bgra), 0);

    Mat bgr({2, 4}, CV_8UC3);
    for (int y = 0; y < bgr.size[0]; ++y)
    {
        for (int x = 0; x < bgr.size[1]; ++x)
        {
            bgr.at<uchar>(y, x, 0) = static_cast<uchar>(7 + y * 23 + x * 2);
            bgr.at<uchar>(y, x, 1) = static_cast<uchar>(9 + y * 11 + x * 7);
            bgr.at<uchar>(y, x, 2) = static_cast<uchar>(13 + y * 5 + x * 9);
        }
    }

    Mat rgba_from_bgr_expected = bgr2rgba_reference<uchar>(bgr);
    Mat rgba_from_bgr_actual;
    cvtColor(bgr, rgba_from_bgr_actual, COLOR_BGR2RGBA);
    ASSERT_EQ(rgba_from_bgr_actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(rgba_from_bgr_expected, rgba_from_bgr_actual), 0);

    Mat bgr_roundtrip;
    cvtColor(rgba_from_bgr_actual, bgr_roundtrip, COLOR_RGBA2BGR);
    EXPECT_EQ(max_abs_diff_u8(bgr, bgr_roundtrip), 0);

    Mat rgba_swapped_expected = swap_rb_4ch_reference<uchar>(bgra_actual);
    Mat rgba_swapped_actual;
    cvtColor(bgra_actual, rgba_swapped_actual, COLOR_BGRA2RGBA);
    ASSERT_EQ(rgba_swapped_actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(rgba_swapped_expected, rgba_swapped_actual), 0);

    Mat bgra_roundtrip;
    cvtColor(rgba_swapped_actual, bgra_roundtrip, COLOR_RGBA2BGRA);
    EXPECT_EQ(max_abs_diff_u8(bgra_actual, bgra_roundtrip), 0);
}

TEST(CvtColorRgbGrayTest, gray_rgba_bgra_family_u8_matches_reference)
{
    Mat gray({2, 5}, CV_8UC1);
    for (int y = 0; y < gray.size[0]; ++y)
    {
        for (int x = 0; x < gray.size[1]; ++x)
        {
            gray.at<uchar>(y, x) = static_cast<uchar>(10 + y * 31 + x * 7);
        }
    }

    Mat bgra_expected = gray2bgra_reference<uchar>(gray);
    Mat bgra_actual;
    cvtColor(gray, bgra_actual, COLOR_GRAY2BGRA);
    ASSERT_EQ(bgra_actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(bgra_expected, bgra_actual), 0);

    Mat rgba_actual;
    cvtColor(gray, rgba_actual, COLOR_GRAY2RGBA);
    ASSERT_EQ(rgba_actual.type(), CV_8UC4);
    EXPECT_EQ(max_abs_diff_u8(bgra_expected, rgba_actual), 0);

    Mat gray_from_bgra_expected = color4_to_gray_reference<uchar>(bgra_actual, false);
    Mat gray_from_bgra_actual;
    cvtColor(bgra_actual, gray_from_bgra_actual, COLOR_BGRA2GRAY);
    ASSERT_EQ(gray_from_bgra_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(gray_from_bgra_expected, gray_from_bgra_actual), 0);

    Mat gray_from_rgba_expected = color4_to_gray_reference<uchar>(rgba_actual, true);
    Mat gray_from_rgba_actual;
    cvtColor(rgba_actual, gray_from_rgba_actual, COLOR_RGBA2GRAY);
    ASSERT_EQ(gray_from_rgba_actual.type(), CV_8UC1);
    EXPECT_EQ(max_abs_diff_u8(gray_from_rgba_expected, gray_from_rgba_actual), 0);
}

TEST(CvtColorRgbGrayTest, throws_on_invalid_input_channels_or_code)
{
    // Ported idea from OpenCV:
    // modules/imgproc/test/test_color.cpp
    // TEST(ImgProc_cvtColor_InvalidNumOfChannels, regression_25971)
    Mat src_gray({8, 8}, CV_8UC1);
    Mat src_bgr({8, 8}, CV_8UC3);
    Mat dst;

    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGR2GRAY), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_RGB2GRAY), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_GRAY2BGR), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGR2RGB), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGR2BGRA), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_RGB2RGBA), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGR2RGBA), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_RGBA2RGB), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_BGRA2RGBA), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_GRAY2BGRA), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_GRAY2RGBA), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGRA2GRAY), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_RGBA2GRAY), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGR2YUV), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_RGB2YUV), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGR2YUV_NV24), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_RGB2YUV_NV24), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_BGR2YUV_NV42), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_RGB2YUV_NV42), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2BGR), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2RGB), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2BGR_NV12), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2RGB_NV12), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2BGR_NV21), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2RGB_NV21), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2BGR_I420), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2RGB_I420), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2BGR_YV12), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2RGB_YV12), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2BGR_NV24), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2RGB_NV24), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2BGR_NV42), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2RGB_NV42), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2BGR_NV16), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2RGB_NV16), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2BGR_NV61), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, COLOR_YUV2RGB_NV61), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2BGR_YUY2), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2RGB_YUY2), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2BGR_UYVY), Exception);
    EXPECT_THROW(cvtColor(src_gray, dst, COLOR_YUV2RGB_UYVY), Exception);
    EXPECT_THROW(cvtColor(src_bgr, dst, -999), Exception);

    Mat src_u16({8, 8}, CV_16UC3);
    EXPECT_THROW(cvtColor(src_u16, dst, COLOR_BGR2GRAY), Exception);
    EXPECT_THROW(cvtColor(src_u16, dst, COLOR_RGB2GRAY), Exception);
}

TEST(CvtColorRgbGrayTest, non_contiguous_roi_matches_reference)
{
    Mat base_bgr({7, 11}, CV_8UC3);
    for (int y = 0; y < base_bgr.size[0]; ++y)
    {
        for (int x = 0; x < base_bgr.size[1]; ++x)
        {
            base_bgr.at<uchar>(y, x, 0) = static_cast<uchar>((y * 13 + x * 3 + 1) % 256);
            base_bgr.at<uchar>(y, x, 1) = static_cast<uchar>((y * 5 + x * 17 + 2) % 256);
            base_bgr.at<uchar>(y, x, 2) = static_cast<uchar>((y * 19 + x * 7 + 3) % 256);
        }
    }
    Mat bgr_roi = base_bgr.colRange(2, 10);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat gray_expected = bgr2gray_reference_u8(bgr_roi);
    Mat gray_actual;
    cvtColor(bgr_roi, gray_actual, COLOR_BGR2GRAY);
    EXPECT_EQ(max_abs_diff_u8(gray_expected, gray_actual), 0);

    Mat rgb_gray_expected = rgb2gray_reference_u8(bgr_roi);
    Mat rgb_gray_actual;
    cvtColor(bgr_roi, rgb_gray_actual, COLOR_RGB2GRAY);
    EXPECT_EQ(max_abs_diff_u8(rgb_gray_expected, rgb_gray_actual), 0);

    Mat base_gray({6, 12}, CV_8UC1);
    for (int y = 0; y < base_gray.size[0]; ++y)
    {
        for (int x = 0; x < base_gray.size[1]; ++x)
        {
            base_gray.at<uchar>(y, x) = static_cast<uchar>((y * 29 + x * 11 + 7) % 256);
        }
    }
    Mat gray_roi = base_gray.colRange(1, 10);
    ASSERT_FALSE(gray_roi.isContinuous());

    Mat bgr_expected = gray2bgr_reference_u8(gray_roi);
    Mat bgr_actual;
    cvtColor(gray_roi, bgr_actual, COLOR_GRAY2BGR);
    EXPECT_EQ(max_abs_diff_u8(bgr_expected, bgr_actual), 0);
}

TEST(CvtColorRgbGrayTest, non_contiguous_roi_for_rgb_and_bgra_paths_matches_reference)
{
    Mat base_bgr({6, 10}, CV_8UC3);
    for (int y = 0; y < base_bgr.size[0]; ++y)
    {
        for (int x = 0; x < base_bgr.size[1]; ++x)
        {
            base_bgr.at<uchar>(y, x, 0) = static_cast<uchar>((y * 11 + x * 3 + 1) % 256);
            base_bgr.at<uchar>(y, x, 1) = static_cast<uchar>((y * 7 + x * 5 + 2) % 256);
            base_bgr.at<uchar>(y, x, 2) = static_cast<uchar>((y * 13 + x * 9 + 3) % 256);
        }
    }
    Mat bgr_roi = base_bgr.colRange(1, 9);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat rgb_expected = bgr2rgb_reference<uchar>(bgr_roi);
    Mat rgb_actual;
    cvtColor(bgr_roi, rgb_actual, COLOR_BGR2RGB);
    EXPECT_EQ(max_abs_diff_u8(rgb_expected, rgb_actual), 0);

    Mat base_bgr_f32({7, 11}, CV_32FC3);
    for (int y = 0; y < base_bgr_f32.size[0]; ++y)
    {
        for (int x = 0; x < base_bgr_f32.size[1]; ++x)
        {
            base_bgr_f32.at<float>(y, x, 0) = static_cast<float>(y * 0.30 - x * 0.12 + 0.75);
            base_bgr_f32.at<float>(y, x, 1) = static_cast<float>(y * 0.55 + x * 0.18 - 1.25);
            base_bgr_f32.at<float>(y, x, 2) = static_cast<float>(y * 0.08 + x * 0.72 + 2.50);
        }
    }
    Mat bgr_f32_roi = base_bgr_f32.colRange(2, 10);
    ASSERT_FALSE(bgr_f32_roi.isContinuous());

    Mat bgra_expected = bgr2bgra_reference<float>(bgr_f32_roi);
    Mat bgra_actual;
    cvtColor(bgr_f32_roi, bgra_actual, COLOR_BGR2BGRA);
    EXPECT_LE(max_abs_diff_f32(bgra_expected, bgra_actual), 1e-6f);

    Mat bgr_expected = bgra2bgr_reference<float>(bgra_actual);
    Mat bgr_actual;
    cvtColor(bgra_actual, bgr_actual, COLOR_BGRA2BGR);
    EXPECT_LE(max_abs_diff_f32(bgr_expected, bgr_actual), 1e-6f);
}

TEST(CvtColorRgbGrayTest, non_contiguous_roi_for_rgba_family_matches_reference)
{
    Mat base_rgb({6, 10}, CV_8UC3);
    for (int y = 0; y < base_rgb.size[0]; ++y)
    {
        for (int x = 0; x < base_rgb.size[1]; ++x)
        {
            base_rgb.at<uchar>(y, x, 0) = static_cast<uchar>((y * 13 + x * 3 + 5) % 256);
            base_rgb.at<uchar>(y, x, 1) = static_cast<uchar>((y * 7 + x * 11 + 9) % 256);
            base_rgb.at<uchar>(y, x, 2) = static_cast<uchar>((y * 5 + x * 17 + 1) % 256);
        }
    }
    Mat rgb_roi = base_rgb.colRange(1, 9);
    ASSERT_FALSE(rgb_roi.isContinuous());

    Mat rgba_expected = rgb2rgba_reference<uchar>(rgb_roi);
    Mat rgba_actual;
    cvtColor(rgb_roi, rgba_actual, COLOR_RGB2RGBA);
    EXPECT_EQ(max_abs_diff_u8(rgba_expected, rgba_actual), 0);

    Mat bgra_expected = rgb2bgra_reference<uchar>(rgb_roi);
    Mat bgra_actual;
    cvtColor(rgb_roi, bgra_actual, COLOR_RGB2BGRA);
    EXPECT_EQ(max_abs_diff_u8(bgra_expected, bgra_actual), 0);

    Mat base_rgba_f32({7, 11}, CV_32FC4);
    for (int y = 0; y < base_rgba_f32.size[0]; ++y)
    {
        for (int x = 0; x < base_rgba_f32.size[1]; ++x)
        {
            base_rgba_f32.at<float>(y, x, 0) = static_cast<float>(y * 0.35 + x * 0.07 - 1.00);
            base_rgba_f32.at<float>(y, x, 1) = static_cast<float>(y * 0.12 - x * 0.28 + 2.25);
            base_rgba_f32.at<float>(y, x, 2) = static_cast<float>(y * 0.50 + x * 0.21 - 0.75);
            base_rgba_f32.at<float>(y, x, 3) = static_cast<float>(0.25 + y * 0.03 + x * 0.04);
        }
    }
    Mat rgba_roi = base_rgba_f32.colRange(2, 10);
    ASSERT_FALSE(rgba_roi.isContinuous());

    Mat rgb_expected = rgba2rgb_reference<float>(rgba_roi);
    Mat rgb_actual;
    cvtColor(rgba_roi, rgb_actual, COLOR_RGBA2RGB);
    EXPECT_LE(max_abs_diff_f32(rgb_expected, rgb_actual), 1e-6f);

    Mat bgr_expected = rgba2bgr_reference<float>(rgba_roi);
    Mat bgr_actual;
    cvtColor(rgba_roi, bgr_actual, COLOR_RGBA2BGR);
    EXPECT_LE(max_abs_diff_f32(bgr_expected, bgr_actual), 1e-6f);

    Mat bgra_expected_f32 = swap_rb_4ch_reference<float>(rgba_roi);
    Mat bgra_actual_f32;
    cvtColor(rgba_roi, bgra_actual_f32, COLOR_RGBA2BGRA);
    EXPECT_LE(max_abs_diff_f32(bgra_expected_f32, bgra_actual_f32), 1e-6f);
}

TEST(CvtColorRgbGrayTest, non_contiguous_roi_for_gray_rgba_family_matches_reference)
{
    Mat base_gray({6, 10}, CV_8UC1);
    for (int y = 0; y < base_gray.size[0]; ++y)
    {
        for (int x = 0; x < base_gray.size[1]; ++x)
        {
            base_gray.at<uchar>(y, x) = static_cast<uchar>((y * 19 + x * 13 + 4) % 256);
        }
    }
    Mat gray_roi = base_gray.colRange(1, 9);
    ASSERT_FALSE(gray_roi.isContinuous());

    Mat bgra_expected = gray2bgra_reference<uchar>(gray_roi);
    Mat bgra_actual;
    cvtColor(gray_roi, bgra_actual, COLOR_GRAY2BGRA);
    EXPECT_EQ(max_abs_diff_u8(bgra_expected, bgra_actual), 0);

    Mat rgba_actual;
    cvtColor(gray_roi, rgba_actual, COLOR_GRAY2RGBA);
    EXPECT_EQ(max_abs_diff_u8(bgra_expected, rgba_actual), 0);

    Mat base_bgra_f32({7, 11}, CV_32FC4);
    for (int y = 0; y < base_bgra_f32.size[0]; ++y)
    {
        for (int x = 0; x < base_bgra_f32.size[1]; ++x)
        {
            base_bgra_f32.at<float>(y, x, 0) = static_cast<float>(y * 0.25 + x * 0.05 - 1.5);
            base_bgra_f32.at<float>(y, x, 1) = static_cast<float>(y * 0.18 - x * 0.12 + 0.5);
            base_bgra_f32.at<float>(y, x, 2) = static_cast<float>(y * 0.07 + x * 0.31 + 2.0);
            base_bgra_f32.at<float>(y, x, 3) = static_cast<float>(0.2 + y * 0.01 + x * 0.03);
        }
    }
    Mat bgra_roi = base_bgra_f32.colRange(2, 10);
    ASSERT_FALSE(bgra_roi.isContinuous());

    Mat gray_from_bgra_expected = color4_to_gray_reference<float>(bgra_roi, false);
    Mat gray_from_bgra_actual;
    cvtColor(bgra_roi, gray_from_bgra_actual, COLOR_BGRA2GRAY);
    EXPECT_LE(max_abs_diff_f32(gray_from_bgra_expected, gray_from_bgra_actual), 1e-6f);

    Mat rgba_roi = swap_rb_4ch_reference<float>(bgra_roi);
    Mat gray_from_rgba_expected = color4_to_gray_reference<float>(rgba_roi, true);
    Mat gray_from_rgba_actual;
    cvtColor(rgba_roi, gray_from_rgba_actual, COLOR_RGBA2GRAY);
    EXPECT_LE(max_abs_diff_f32(gray_from_rgba_expected, gray_from_rgba_actual), 1e-6f);
}

TEST(CvtColorRgbGrayTest, supports_single_row_and_single_col_images)
{
    Mat row_bgr({1, 9}, CV_8UC3);
    for (int x = 0; x < row_bgr.size[1]; ++x)
    {
        row_bgr.at<uchar>(0, x, 0) = static_cast<uchar>((x * 3 + 1) % 256);
        row_bgr.at<uchar>(0, x, 1) = static_cast<uchar>((x * 5 + 2) % 256);
        row_bgr.at<uchar>(0, x, 2) = static_cast<uchar>((x * 7 + 3) % 256);
    }

    Mat row_gray;
    cvtColor(row_bgr, row_gray, COLOR_BGR2GRAY);
    Mat row_expected = bgr2gray_reference_u8(row_bgr);
    EXPECT_EQ(max_abs_diff_u8(row_gray, row_expected), 0);

    Mat col_gray({9, 1}, CV_8UC1);
    for (int y = 0; y < col_gray.size[0]; ++y)
    {
        col_gray.at<uchar>(y, 0) = static_cast<uchar>((y * 17 + 9) % 256);
    }

    Mat col_bgr;
    cvtColor(col_gray, col_bgr, COLOR_GRAY2BGR);
    Mat col_expected = gray2bgr_reference_u8(col_gray);
    EXPECT_EQ(max_abs_diff_u8(col_bgr, col_expected), 0);
}

TEST(CvtColorRgbGrayTest, supports_cv32f_bgr2gray_and_gray2bgr)
{
    Mat src({3, 4}, CV_32FC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            src.at<float>(y, x, 0) = static_cast<float>(-1.5 + y * 0.25 + x * 0.10);
            src.at<float>(y, x, 1) = static_cast<float>(2.0 + y * 0.75 - x * 0.20);
            src.at<float>(y, x, 2) = static_cast<float>(0.5 - y * 0.10 + x * 1.30);
        }
    }

    Mat gray_expected = bgr2gray_reference_f32(src);
    Mat gray_actual;
    cvtColor(src, gray_actual, COLOR_BGR2GRAY);
    ASSERT_EQ(gray_actual.type(), CV_32FC1);
    EXPECT_LE(max_abs_diff_f32(gray_expected, gray_actual), 1e-6f);

    Mat rgb_gray_expected = rgb2gray_reference_f32(src);
    Mat rgb_gray_actual;
    cvtColor(src, rgb_gray_actual, COLOR_RGB2GRAY);
    ASSERT_EQ(rgb_gray_actual.type(), CV_32FC1);
    EXPECT_LE(max_abs_diff_f32(rgb_gray_expected, rgb_gray_actual), 1e-6f);

    Mat bgr_expected = gray2bgr_reference_f32(gray_actual);
    Mat bgr_actual;
    cvtColor(gray_actual, bgr_actual, COLOR_GRAY2BGR);
    ASSERT_EQ(bgr_actual.type(), CV_32FC3);
    EXPECT_LE(max_abs_diff_f32(bgr_expected, bgr_actual), 1e-6f);
}

TEST(CvtColorRgbGrayTest, cv32f_non_contiguous_roi_matches_reference)
{
    Mat base_bgr({6, 10}, CV_32FC3);
    for (int y = 0; y < base_bgr.size[0]; ++y)
    {
        for (int x = 0; x < base_bgr.size[1]; ++x)
        {
            base_bgr.at<float>(y, x, 0) = static_cast<float>(y * 0.45 - x * 0.15 + 0.25);
            base_bgr.at<float>(y, x, 1) = static_cast<float>(y * 1.20 + x * 0.35 - 0.50);
            base_bgr.at<float>(y, x, 2) = static_cast<float>(-y * 0.30 + x * 0.90 + 1.75);
        }
    }
    Mat bgr_roi = base_bgr.colRange(1, 9);
    ASSERT_FALSE(bgr_roi.isContinuous());

    Mat gray_expected = bgr2gray_reference_f32(bgr_roi);
    Mat gray_actual;
    cvtColor(bgr_roi, gray_actual, COLOR_BGR2GRAY);
    EXPECT_LE(max_abs_diff_f32(gray_expected, gray_actual), 1e-6f);

    Mat rgb_gray_expected = rgb2gray_reference_f32(bgr_roi);
    Mat rgb_gray_actual;
    cvtColor(bgr_roi, rgb_gray_actual, COLOR_RGB2GRAY);
    EXPECT_LE(max_abs_diff_f32(rgb_gray_expected, rgb_gray_actual), 1e-6f);

    Mat base_gray({7, 11}, CV_32FC1);
    for (int y = 0; y < base_gray.size[0]; ++y)
    {
        for (int x = 0; x < base_gray.size[1]; ++x)
        {
            base_gray.at<float>(y, x) = static_cast<float>(y * 0.60 - x * 0.22 + 3.0);
        }
    }
    Mat gray_roi = base_gray.colRange(2, 10);
    ASSERT_FALSE(gray_roi.isContinuous());

    Mat bgr_expected = gray2bgr_reference_f32(gray_roi);
    Mat bgr_actual;
    cvtColor(gray_roi, bgr_actual, COLOR_GRAY2BGR);
    EXPECT_LE(max_abs_diff_f32(bgr_expected, bgr_actual), 1e-6f);
}

TEST(CvtColorRgbGrayTest, supports_cv32f_bgr2rgb_and_bgr2bgra)
{
    Mat src({3, 5}, CV_32FC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            src.at<float>(y, x, 0) = static_cast<float>(-0.5 + y * 0.20 + x * 0.10);
            src.at<float>(y, x, 1) = static_cast<float>(1.5 + y * 0.35 - x * 0.40);
            src.at<float>(y, x, 2) = static_cast<float>(2.5 - y * 0.15 + x * 0.60);
        }
    }

    Mat rgb_expected = bgr2rgb_reference<float>(src);
    Mat rgb_actual;
    cvtColor(src, rgb_actual, COLOR_BGR2RGB);
    ASSERT_EQ(rgb_actual.type(), CV_32FC3);
    EXPECT_LE(max_abs_diff_f32(rgb_expected, rgb_actual), 1e-6f);

    Mat bgr_roundtrip;
    cvtColor(rgb_actual, bgr_roundtrip, COLOR_RGB2BGR);
    EXPECT_LE(max_abs_diff_f32(src, bgr_roundtrip), 1e-6f);

    Mat bgra_expected = bgr2bgra_reference<float>(src);
    Mat bgra_actual;
    cvtColor(src, bgra_actual, COLOR_BGR2BGRA);
    ASSERT_EQ(bgra_actual.type(), CV_32FC4);
    EXPECT_LE(max_abs_diff_f32(bgra_expected, bgra_actual), 1e-6f);

    Mat bgr_expected = bgra2bgr_reference<float>(bgra_actual);
    Mat bgr_actual;
    cvtColor(bgra_actual, bgr_actual, COLOR_BGRA2BGR);
    ASSERT_EQ(bgr_actual.type(), CV_32FC3);
    EXPECT_LE(max_abs_diff_f32(bgr_expected, bgr_actual), 1e-6f);
}

TEST(CvtColorRgbGrayTest, supports_cv32f_rgba_family_conversions)
{
    Mat rgb({3, 5}, CV_32FC3);
    for (int y = 0; y < rgb.size[0]; ++y)
    {
        for (int x = 0; x < rgb.size[1]; ++x)
        {
            rgb.at<float>(y, x, 0) = static_cast<float>(-1.0 + y * 0.15 + x * 0.40);
            rgb.at<float>(y, x, 1) = static_cast<float>(0.5 + y * 0.22 - x * 0.17);
            rgb.at<float>(y, x, 2) = static_cast<float>(2.0 - y * 0.31 + x * 0.09);
        }
    }

    Mat rgba_expected = rgb2rgba_reference<float>(rgb);
    Mat rgba_actual;
    cvtColor(rgb, rgba_actual, COLOR_RGB2RGBA);
    ASSERT_EQ(rgba_actual.type(), CV_32FC4);
    EXPECT_LE(max_abs_diff_f32(rgba_expected, rgba_actual), 1e-6f);

    Mat bgra_expected = rgb2bgra_reference<float>(rgb);
    Mat bgra_actual;
    cvtColor(rgb, bgra_actual, COLOR_RGB2BGRA);
    ASSERT_EQ(bgra_actual.type(), CV_32FC4);
    EXPECT_LE(max_abs_diff_f32(bgra_expected, bgra_actual), 1e-6f);

    Mat rgb_roundtrip;
    cvtColor(rgba_actual, rgb_roundtrip, COLOR_RGBA2RGB);
    EXPECT_LE(max_abs_diff_f32(rgb, rgb_roundtrip), 1e-6f);

    Mat rgb_from_bgra;
    cvtColor(bgra_actual, rgb_from_bgra, COLOR_BGRA2RGB);
    EXPECT_LE(max_abs_diff_f32(rgb, rgb_from_bgra), 1e-6f);

    Mat bgr({3, 5}, CV_32FC3);
    for (int y = 0; y < bgr.size[0]; ++y)
    {
        for (int x = 0; x < bgr.size[1]; ++x)
        {
            bgr.at<float>(y, x, 0) = static_cast<float>(1.5 + y * 0.11 + x * 0.27);
            bgr.at<float>(y, x, 1) = static_cast<float>(-0.5 + y * 0.45 - x * 0.12);
            bgr.at<float>(y, x, 2) = static_cast<float>(0.25 - y * 0.08 + x * 0.51);
        }
    }

    Mat rgba_from_bgr_expected = bgr2rgba_reference<float>(bgr);
    Mat rgba_from_bgr_actual;
    cvtColor(bgr, rgba_from_bgr_actual, COLOR_BGR2RGBA);
    ASSERT_EQ(rgba_from_bgr_actual.type(), CV_32FC4);
    EXPECT_LE(max_abs_diff_f32(rgba_from_bgr_expected, rgba_from_bgr_actual), 1e-6f);

    Mat bgr_roundtrip;
    cvtColor(rgba_from_bgr_actual, bgr_roundtrip, COLOR_RGBA2BGR);
    EXPECT_LE(max_abs_diff_f32(bgr, bgr_roundtrip), 1e-6f);

    Mat rgba_swapped_expected = swap_rb_4ch_reference<float>(bgra_actual);
    Mat rgba_swapped_actual;
    cvtColor(bgra_actual, rgba_swapped_actual, COLOR_BGRA2RGBA);
    ASSERT_EQ(rgba_swapped_actual.type(), CV_32FC4);
    EXPECT_LE(max_abs_diff_f32(rgba_swapped_expected, rgba_swapped_actual), 1e-6f);

    Mat bgra_roundtrip;
    cvtColor(rgba_swapped_actual, bgra_roundtrip, COLOR_RGBA2BGRA);
    EXPECT_LE(max_abs_diff_f32(bgra_actual, bgra_roundtrip), 1e-6f);
}

TEST(CvtColorRgbGrayTest, supports_cv32f_gray_rgba_family_conversions)
{
    Mat gray({3, 4}, CV_32FC1);
    for (int y = 0; y < gray.size[0]; ++y)
    {
        for (int x = 0; x < gray.size[1]; ++x)
        {
            gray.at<float>(y, x) = static_cast<float>(-0.75 + y * 0.40 + x * 0.15);
        }
    }

    Mat bgra_expected = gray2bgra_reference<float>(gray);
    Mat bgra_actual;
    cvtColor(gray, bgra_actual, COLOR_GRAY2BGRA);
    ASSERT_EQ(bgra_actual.type(), CV_32FC4);
    EXPECT_LE(max_abs_diff_f32(bgra_expected, bgra_actual), 1e-6f);

    Mat rgba_actual;
    cvtColor(gray, rgba_actual, COLOR_GRAY2RGBA);
    ASSERT_EQ(rgba_actual.type(), CV_32FC4);
    EXPECT_LE(max_abs_diff_f32(bgra_expected, rgba_actual), 1e-6f);

    Mat gray_from_bgra_expected = color4_to_gray_reference<float>(bgra_actual, false);
    Mat gray_from_bgra_actual;
    cvtColor(bgra_actual, gray_from_bgra_actual, COLOR_BGRA2GRAY);
    ASSERT_EQ(gray_from_bgra_actual.type(), CV_32FC1);
    EXPECT_LE(max_abs_diff_f32(gray_from_bgra_expected, gray_from_bgra_actual), 1e-6f);

    Mat gray_from_rgba_expected = color4_to_gray_reference<float>(rgba_actual, true);
    Mat gray_from_rgba_actual;
    cvtColor(rgba_actual, gray_from_rgba_actual, COLOR_RGBA2GRAY);
    ASSERT_EQ(gray_from_rgba_actual.type(), CV_32FC1);
    EXPECT_LE(max_abs_diff_f32(gray_from_rgba_expected, gray_from_rgba_actual), 1e-6f);
}
