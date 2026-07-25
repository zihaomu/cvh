#include "test/imgproc/support/morphology_derivatives_test_utils.hpp"

// Ported from OpenCV:
// modules/imgproc/test/test_filter.cpp
// TEST(Imgproc_Morphology, iterated)
TEST(MorphologyDerivativesUpstreamTest, Imgproc_Morphology_iterated)
{
    std::uint32_t state = 0xC0FFEEu;
    for (int iter = 0; iter < 20; ++iter)
    {
        state = lcg_next(state);
        const int width = 5 + static_cast<int>(state % 28u);
        state = lcg_next(state);
        const int height = 5 + static_cast<int>(state % 28u);
        state = lcg_next(state);
        const int cn = 1 + static_cast<int>(state % 4u);
        state = lcg_next(state);
        const int iterations = 1 + static_cast<int>(state % 10u);
        state = lcg_next(state);
        const bool do_dilate = (state & 1u) == 0u;

        Mat src({height, width}, CV_MAKETYPE(CV_8U, cn));
        fill_u8_lcg(src, state ^ 0x91u);

        Mat dst0;
        Mat dst1;
        Mat dst2;

        if (do_dilate)
        {
            dilate(src, dst0, Mat(), Point(-1, -1), iterations);
        }
        else
        {
            erode(src, dst0, Mat(), Point(-1, -1), iterations);
        }

        for (int i = 0; i < iterations; ++i)
        {
            if (do_dilate)
            {
                dilate(i == 0 ? src : dst1, dst1, Mat(), Point(-1, -1), 1);
            }
            else
            {
                erode(i == 0 ? src : dst1, dst1, Mat(), Point(-1, -1), 1);
            }
        }

        Mat kern({3, 3}, CV_8UC1);
        kern = 1;
        if (do_dilate)
        {
            dilate(src, dst2, kern, Point(-1, -1), iterations);
        }
        else
        {
            erode(src, dst2, kern, Point(-1, -1), iterations);
        }

        EXPECT_EQ(0, max_abs_diff_u8(dst0, dst1));
        EXPECT_EQ(0, max_abs_diff_u8(dst0, dst2));
    }
}

// Ported from OpenCV:
// modules/imgproc/test/test_filter.cpp
// TEST(Imgproc, morphologyEx_small_input_22893)
TEST(MorphologyDerivativesUpstreamTest, Imgproc_morphologyEx_small_input_22893)
{
    Mat img({1, 4}, CV_8UC1);
    img.at<uchar>(0, 0) = 1;
    img.at<uchar>(0, 1) = 2;
    img.at<uchar>(0, 2) = 3;
    img.at<uchar>(0, 3) = 4;

    Mat gold({1, 4}, CV_8UC1);
    gold.at<uchar>(0, 0) = 2;
    gold.at<uchar>(0, 1) = 3;
    gold.at<uchar>(0, 2) = 4;
    gold.at<uchar>(0, 3) = 4;

    Mat kernel({4, 4}, CV_8UC1);
    kernel = 1;

    Mat result;
    morphologyEx(img, result, MORPH_DILATE, kernel);

    ASSERT_EQ(result.type(), gold.type());
    ASSERT_EQ(result.size[0], gold.size[0]);
    ASSERT_EQ(result.size[1], gold.size[1]);
    EXPECT_EQ(max_abs_diff_u8(result, gold), 0);
}

// Ported from OpenCV:
// modules/imgproc/test/test_filter.cpp
// TEST(Imgproc_MorphEx, hitmiss_regression_8957)
TEST(MorphologyDerivativesUpstreamTest, Imgproc_MorphEx_hitmiss_regression_8957)
{
    Mat src({3, 3}, CV_8UC1);
    src.at<uchar>(0, 0) = 0;   src.at<uchar>(0, 1) = 255; src.at<uchar>(0, 2) = 0;
    src.at<uchar>(1, 0) = 0;   src.at<uchar>(1, 1) = 0;   src.at<uchar>(1, 2) = 0;
    src.at<uchar>(2, 0) = 0;   src.at<uchar>(2, 1) = 255; src.at<uchar>(2, 2) = 0;

    Mat kernel({3, 3}, CV_8UC1);
    kernel.at<uchar>(0, 0) = 0; kernel.at<uchar>(0, 1) = 1; kernel.at<uchar>(0, 2) = 0;
    kernel.at<uchar>(1, 0) = 0; kernel.at<uchar>(1, 1) = 0; kernel.at<uchar>(1, 2) = 0;
    kernel.at<uchar>(2, 0) = 0; kernel.at<uchar>(2, 1) = 1; kernel.at<uchar>(2, 2) = 0;

    Mat dst;
    morphologyEx(src, dst, MORPH_HITMISS, kernel);

    Mat ref({3, 3}, CV_8UC1);
    ref = 0;
    ref.at<uchar>(1, 1) = 255;
    EXPECT_EQ(max_abs_diff_u8(dst, ref), 0);

    src.at<uchar>(1, 1) = 255;
    ref.at<uchar>(0, 1) = 255;
    ref.at<uchar>(2, 1) = 255;
    morphologyEx(src, dst, MORPH_HITMISS, kernel);
    EXPECT_EQ(max_abs_diff_u8(dst, ref), 0);
}

// Ported from OpenCV:
// modules/imgproc/test/test_filter.cpp
// TEST(Imgproc_MorphEx, hitmiss_zero_kernel)
TEST(MorphologyDerivativesUpstreamTest, Imgproc_MorphEx_hitmiss_zero_kernel)
{
    Mat src({3, 3}, CV_8UC1);
    src.at<uchar>(0, 0) = 0;   src.at<uchar>(0, 1) = 255; src.at<uchar>(0, 2) = 0;
    src.at<uchar>(1, 0) = 0;   src.at<uchar>(1, 1) = 0;   src.at<uchar>(1, 2) = 0;
    src.at<uchar>(2, 0) = 0;   src.at<uchar>(2, 1) = 255; src.at<uchar>(2, 2) = 0;

    Mat kernel({3, 3}, CV_8UC1);
    kernel = 0;

    Mat dst;
    morphologyEx(src, dst, MORPH_HITMISS, kernel);
    EXPECT_EQ(max_abs_diff_u8(dst, src), 0);
}

// Ported from OpenCV (implemented-ops coverage):
// modules/imgproc/test/test_filter.cpp
// TEST(Imgproc, filter_empty_src_16857)
TEST(MorphologyDerivativesUpstreamTest, Imgproc_filter_empty_src_16857)
{
    Mat src, dst, dst2;

    EXPECT_THROW(blur(src, dst, Size(3, 3)), Exception);
    EXPECT_THROW(boxFilter(src, dst, CV_8U, Size(3, 3)), Exception);
    EXPECT_THROW(GaussianBlur(src, dst, Size(3, 3), 0.0), Exception);
    EXPECT_THROW(Sobel(src, dst, CV_32F, 1, 0, 3), Exception);
    EXPECT_THROW(dilate(src, dst, Mat()), Exception);
    EXPECT_THROW(erode(src, dst, Mat()), Exception);
    EXPECT_THROW(morphologyEx(src, dst, MORPH_OPEN, Mat()), Exception);

    EXPECT_TRUE(src.empty());
    EXPECT_TRUE(dst.empty());
    EXPECT_TRUE(dst2.empty());
}

// Ported from OpenCV:
// modules/imgproc/test/test_filter.cpp
// TEST(Imgproc_Sobel, borderTypes)
TEST(MorphologyDerivativesUpstreamTest, Imgproc_Sobel_borderTypes)
{
    const int kernelSize = 3;
    Mat dst;

    Mat src({3, 3}, CV_8UC1);
    src.at<uchar>(0, 0) = 1; src.at<uchar>(0, 1) = 2; src.at<uchar>(0, 2) = 3;
    src.at<uchar>(1, 0) = 4; src.at<uchar>(1, 1) = 5; src.at<uchar>(1, 2) = 6;
    src.at<uchar>(2, 0) = 7; src.at<uchar>(2, 1) = 8; src.at<uchar>(2, 2) = 9;

    Mat src_roi = src(Range(1, 2), Range(1, 2));
    src_roi.setTo(0);

    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REPLICATE);
    EXPECT_FLOAT_EQ(8.0f, dst.at<float>(0, 0));
    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REFLECT);
    EXPECT_FLOAT_EQ(8.0f, dst.at<float>(0, 0));

    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REPLICATE | BORDER_ISOLATED);
    EXPECT_FLOAT_EQ(0.0f, dst.at<float>(0, 0));
    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REFLECT | BORDER_ISOLATED);
    EXPECT_FLOAT_EQ(0.0f, dst.at<float>(0, 0));

    src = Mat({5, 5}, CV_8UC1);
    src = 5;
    src_roi = src(Range(1, 4), Range(1, 4));
    src_roi.setTo(0);

    Mat expected({3, 3}, CV_32FC1);
    expected.at<float>(0, 0) = -15.0f; expected.at<float>(0, 1) = 0.0f; expected.at<float>(0, 2) = 15.0f;
    expected.at<float>(1, 0) = -20.0f; expected.at<float>(1, 1) = 0.0f; expected.at<float>(1, 2) = 20.0f;
    expected.at<float>(2, 0) = -15.0f; expected.at<float>(2, 1) = 0.0f; expected.at<float>(2, 2) = 15.0f;

    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REPLICATE);
    EXPECT_LE(max_abs_diff_f32(expected, dst), 1e-6f);
    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REFLECT);
    EXPECT_LE(max_abs_diff_f32(expected, dst), 1e-6f);

    Mat expected_zero({3, 3}, CV_32FC1);
    expected_zero = 0.0f;
    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REPLICATE | BORDER_ISOLATED);
    EXPECT_LE(max_abs_diff_f32(expected_zero, dst), 1e-6f);
    Sobel(src_roi, dst, CV_32F, 1, 0, kernelSize, 1.0, 0.0, BORDER_REFLECT | BORDER_ISOLATED);
    EXPECT_LE(max_abs_diff_f32(expected_zero, dst), 1e-6f);
}

// Ported from OpenCV:
// modules/imgproc/test/test_filter.cpp
// TEST(Imgproc_Sobel, s16_regression_13506)
TEST(MorphologyDerivativesUpstreamTest, Imgproc_Sobel_s16_regression_13506)
{
    static const short src_values[8 * 16] = {
        127, 138, 130, 102, 118, 97, 76, 84, 124, 90, 146, 63, 130, 87, 212, 85,
        164, 3, 51, 124, 151, 89, 154, 117, 36, 88, 116, 117, 180, 112, 147, 124,
        63, 50, 115, 103, 83, 148, 106, 79, 213, 106, 135, 53, 79, 106, 122, 112,
        218, 107, 81, 126, 78, 138, 85, 142, 151, 108, 104, 158, 155, 81, 112, 178,
        184, 96, 187, 148, 150, 112, 138, 162, 222, 146, 128, 49, 124, 46, 165, 104,
        119, 164, 77, 144, 186, 98, 106, 148, 155, 157, 160, 151, 156, 149, 43, 122,
        106, 155, 120, 132, 159, 115, 126, 188, 44, 79, 164, 201, 153, 97, 139, 133,
        133, 98, 111, 165, 66, 106, 131, 85, 176, 156, 67, 108, 142, 91, 74, 137,
    };

    static const short ref_values[8 * 16] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1020, -796, -489, -469, -247, 317, 760, 1429, 1983, 1384, 254, -459, -899, -1197, -1172, -1058,
        2552, 2340, 1617, 591, 9, 96, 722, 1985, 2746, 1916, 676, 9, -635, -1115, -779, -380,
        3546, 3349, 2838, 2206, 1388, 669, 938, 1880, 2252, 1785, 1083, 606, 180, -298, -464, -418,
        816, 966, 1255, 1652, 1619, 924, 535, 288, 5, 601, 1581, 1870, 1520, 625, -627, -1260,
        -782, -610, -395, -267, -122, -42, -317, -1378, -2293, -1451, 596, 1870, 1679, 763, -69, -394,
        -882, -681, -463, -818, -1167, -732, -463, -1042, -1604, -1592, -1047, -334, -104, -117, 229, 512,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    Mat src({8, 16}, CV_16SC1);
    Mat ref({8, 16}, CV_16SC1);
    short* src_ptr = reinterpret_cast<short*>(src.data);
    short* ref_ptr = reinterpret_cast<short*>(ref.data);
    for (size_t i = 0; i < 8u * 16u; ++i)
    {
        src_ptr[i] = src_values[i];
        ref_ptr[i] = ref_values[i];
    }

    Mat dst;
    Sobel(src, dst, CV_16S, 0, 1, 5);

    ASSERT_EQ(dst.type(), ref.type());
    ASSERT_EQ(dst.size[0], ref.size[0]);
    ASSERT_EQ(dst.size[1], ref.size[1]);
    EXPECT_EQ(max_abs_diff_s16(dst, ref), 0);
}
