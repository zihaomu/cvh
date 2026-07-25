#include "test/imgproc/support/morphology_derivatives_test_utils.hpp"

TEST(DerivativesTest, sobel_u8_to_f32_c1_matches_reference)
{
    Mat src({6, 7}, CV_8UC1);
    fill_u8_pattern(src);

    Mat actual;
    Sobel(src, actual, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
    const Mat expected = sobel_reference_u8_to_f32(src, 1, 0, BORDER_REPLICATE);

    EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-6f);
}

TEST(DerivativesTest, sobel_u8_to_f32_c4_roi_isolated_matches_reference)
{
    Mat src_full({9, 11}, CV_8UC4);
    fill_u8_pattern(src_full);
    Mat roi = src_full(Range(2, 8), Range(3, 10));

    Mat actual;
    // Keep this contract as ROI-local sampling; non-isolated ROI behavior is
    // covered by MorphologyDerivativesUpstreamTest.Imgproc_Sobel_borderTypes.
    Sobel(roi, actual, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE | BORDER_ISOLATED);
    const Mat expected = sobel_reference_u8_to_f32(roi, 1, 0, BORDER_REPLICATE);

    EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-6f);
}

#include "test/imgproc/support/kernel_family_test_utils.hpp"

TEST(DerivativesTest, scharr_laplacian_and_spatial_gradient_share_semantics)
{
    Mat ramp({7, 9}, CV_8UC1);
    for (int y = 0; y < 7; ++y)
    {
        for (int x = 0; x < 9; ++x)
        {
            ramp.at<uchar>(y, x) = static_cast<uchar>(x + 2 * y);
        }
    }
    Mat scharr_x;
    Scharr(ramp, scharr_x, CV_16S, 1, 0);
    EXPECT_EQ(scharr_x.at<short>(3, 4), 32);

    Mat constant({7, 9}, CV_32FC3);
    constant.setTo(Scalar::all(5.0));
    Mat laplacian;
    Laplacian(constant, laplacian, CV_32F, 3);
    EXPECT_NEAR(norm(laplacian, NORM_INF), 0.0, 1e-6);

    Mat impulse({5, 5}, CV_32FC1);
    impulse.setTo(Scalar::all(0.0));
    impulse.at<float>(2, 2) = 1.0f;
    Laplacian(impulse, laplacian, CV_32F, 1, 1.0, 0.0, BORDER_CONSTANT);
    EXPECT_FLOAT_EQ(laplacian.at<float>(2, 2), -4.0f);

    Mat dx;
    Mat dy;
    Mat expected_dx;
    Mat expected_dy;
    spatialGradient(ramp, dx, dy);
    Sobel(ramp, expected_dx, CV_16S, 1, 0, 3);
    Sobel(ramp, expected_dy, CV_16S, 0, 1, 3);
    EXPECT_EQ(
        std::memcmp(
            dx.data,
            expected_dx.data,
            dx.total() * dx.elemSize()),
        0);
    EXPECT_EQ(
        std::memcmp(
            dy.data,
            expected_dy.data,
            dy.total() * dy.elemSize()),
        0);
}

TEST(DerivativesTest, rejects_unsupported_derivative_parameters)
{
    Mat source({3, 3}, CV_32FC1);
    Mat output_x;
    Mat output_y;

    EXPECT_THROW(
        Scharr(source, output_x, CV_32F, 1, 1),
        Exception);
    EXPECT_THROW(
        Laplacian(source, output_x, CV_32F, 7),
        Exception);
    EXPECT_THROW(
        spatialGradient(source, output_x, output_y, 5),
        Exception);
}

TEST(DerivativesTest, sobel_rejects_empty_unsupported_depth_and_order)
{
    Mat empty;
    Mat source_u8({5, 6}, CV_8UC1);
    Mat source_u16({5, 6}, CV_16UC1);
    Mat output;

    EXPECT_THROW(
        Sobel(
            empty,
            output,
            CV_32F,
            1,
            0,
            3,
            1.0,
            0.0,
            BORDER_REPLICATE),
        Exception);
    EXPECT_THROW(
        Sobel(
            source_u16,
            output,
            CV_32F,
            1,
            0,
            3,
            1.0,
            0.0,
            BORDER_REPLICATE),
        Exception);
    EXPECT_THROW(
        Sobel(
            source_u8,
            output,
            CV_8U,
            1,
            0,
            3,
            1.0,
            0.0,
            BORDER_REPLICATE),
        Exception);
    EXPECT_THROW(
        Sobel(
            source_u8,
            output,
            CV_32F,
            2,
            0,
            3,
            1.0,
            0.0,
            BORDER_REPLICATE),
        Exception);
}
