#include "test/imgproc/support/morphology_derivatives_test_utils.hpp"

TEST(MorphologyTest, erode_dilate_u8_c1_matches_reference)
{
    Mat src({7, 9}, CV_8UC1);
    fill_u8_pattern(src);

    Mat erode_actual;
    Mat dilate_actual;
    erode(src, erode_actual, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    dilate(src, dilate_actual, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);

    const Mat erode_expected = morph_reference_u8(src, true, BORDER_REPLICATE, Scalar::all(255.0));
    const Mat dilate_expected = morph_reference_u8(src, false, BORDER_REPLICATE, Scalar::all(0.0));

    EXPECT_EQ(max_abs_diff_u8(erode_actual, erode_expected), 0);
    EXPECT_EQ(max_abs_diff_u8(dilate_actual, dilate_expected), 0);
}

TEST(MorphologyTest, erode_dilate_u8_c3_roi_matches_reference)
{
    Mat src_full({8, 10}, CV_8UC3);
    fill_u8_pattern(src_full);
    Mat roi = src_full(Range(1, 7), Range(2, 9));

    Mat erode_actual;
    Mat dilate_actual;
    erode(roi, erode_actual, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    dilate(roi, dilate_actual, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);

    const Mat erode_expected = morph_reference_u8(roi, true, BORDER_REPLICATE, Scalar::all(255.0));
    const Mat dilate_expected = morph_reference_u8(roi, false, BORDER_REPLICATE, Scalar::all(0.0));

    EXPECT_EQ(max_abs_diff_u8(erode_actual, erode_expected), 0);
    EXPECT_EQ(max_abs_diff_u8(dilate_actual, dilate_expected), 0);
}

TEST(MorphologyTest, morphologyEx_open_close_gradient_match_reference)
{
    Mat src({9, 11}, CV_8UC3);
    fill_u8_pattern(src);

    Mat expected_erode = morph_reference_u8(src, true, BORDER_REPLICATE, Scalar::all(255.0));
    Mat expected_dilate = morph_reference_u8(src, false, BORDER_REPLICATE, Scalar::all(0.0));

    Mat expected_open;
    dilate(expected_erode, expected_open, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    Mat expected_close;
    erode(expected_dilate, expected_close, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);

    Mat expected_gradient({src.size[0], src.size[1]}, src.type());
    const size_t count = src.total() * static_cast<size_t>(src.channels());
    for (size_t i = 0; i < count; ++i)
    {
        expected_gradient.data[i] =
            static_cast<uchar>(static_cast<int>(expected_dilate.data[i]) - static_cast<int>(expected_erode.data[i]));
    }

    Mat actual_open;
    Mat actual_close;
    Mat actual_gradient;
    morphologyEx(src, actual_open, MORPH_OPEN, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    morphologyEx(src, actual_close, MORPH_CLOSE, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    morphologyEx(src, actual_gradient, MORPH_GRADIENT, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);

    EXPECT_EQ(max_abs_diff_u8(actual_open, expected_open), 0);
    EXPECT_EQ(max_abs_diff_u8(actual_close, expected_close), 0);
    EXPECT_EQ(max_abs_diff_u8(actual_gradient, expected_gradient), 0);
}

TEST(MorphologyTest, morphologyEx_tophat_blackhat_match_reference)
{
    Mat src({9, 12}, CV_8UC4);
    fill_u8_pattern(src);

    const Mat expected_erode = morph_reference_u8(src, true, BORDER_REPLICATE, Scalar::all(255.0));
    const Mat expected_dilate = morph_reference_u8(src, false, BORDER_REPLICATE, Scalar::all(0.0));
    const Mat expected_open = morph_reference_u8(expected_erode, false, BORDER_REPLICATE, Scalar::all(0.0));
    const Mat expected_close = morph_reference_u8(expected_dilate, true, BORDER_REPLICATE, Scalar::all(255.0));

    Mat expected_tophat({src.size[0], src.size[1]}, src.type());
    Mat expected_blackhat({src.size[0], src.size[1]}, src.type());
    const size_t count = src.total() * static_cast<size_t>(src.channels());
    for (size_t i = 0; i < count; ++i)
    {
        const int tophat = static_cast<int>(src.data[i]) - static_cast<int>(expected_open.data[i]);
        const int blackhat = static_cast<int>(expected_close.data[i]) - static_cast<int>(src.data[i]);
        expected_tophat.data[i] = static_cast<uchar>(tophat < 0 ? 0 : tophat);
        expected_blackhat.data[i] = static_cast<uchar>(blackhat < 0 ? 0 : blackhat);
    }

    Mat actual_tophat;
    Mat actual_blackhat;
    morphologyEx(src, actual_tophat, MORPH_TOPHAT, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
    morphologyEx(src, actual_blackhat, MORPH_BLACKHAT, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);

    EXPECT_EQ(max_abs_diff_u8(actual_tophat, expected_tophat), 0);
    EXPECT_EQ(max_abs_diff_u8(actual_blackhat, expected_blackhat), 0);
}

TEST(MorphologyTest, morphologyEx_hitmiss_signed_kernel_semantics)
{
    Mat src({3, 3}, CV_8UC1);
    src = 0;
    src.at<uchar>(1, 1) = 255;

    Mat kernel({3, 3}, CV_8SC1);
    kernel = 0;
    kernel.at<schar>(0, 1) = -1;
    kernel.at<schar>(1, 0) = -1;
    kernel.at<schar>(1, 1) = 1;
    kernel.at<schar>(1, 2) = -1;
    kernel.at<schar>(2, 1) = -1;

    Mat dst;
    morphologyEx(src, dst, MORPH_HITMISS, kernel);

    Mat expected({3, 3}, CV_8UC1);
    expected = 0;
    expected.at<uchar>(1, 1) = 255;
    EXPECT_EQ(max_abs_diff_u8(dst, expected), 0);

    src.at<uchar>(0, 1) = 255;
    morphologyEx(src, dst, MORPH_HITMISS, kernel);

    expected = 0;
    EXPECT_EQ(max_abs_diff_u8(dst, expected), 0);
}

TEST(MorphologyTest, rejects_unsupported_depth_and_iteration_count)
{
    Mat src_u8({5, 6}, CV_8UC1);
    fill_u8_pattern(src_u8);
    Mat src_u16({5, 6}, CV_16UC1);
    src_u16 = 7;
    Mat dst;

    EXPECT_THROW(erode(src_u16, dst), Exception);
    EXPECT_THROW(dilate(src_u16, dst), Exception);
    EXPECT_THROW(erode(src_u8, dst, Mat(), Point(-1, -1), 0, BORDER_REPLICATE), Exception);
}
