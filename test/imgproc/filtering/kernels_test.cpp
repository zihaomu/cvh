#include "test/imgproc/support/kernel_family_test_utils.hpp"

TEST(KernelsTest, structuring_elements_cover_shapes_anchor_and_errors)
{
    Mat rectangle = getStructuringElement(MORPH_RECT, Size(5, 3));
    EXPECT_EQ(countNonZero(rectangle), 15);

    Mat cross = getStructuringElement(MORPH_CROSS, Size(5, 3), Point(1, 0));
    EXPECT_EQ(countNonZero(cross), 7);
    EXPECT_EQ(cross.at<uchar>(0, 4), 1);
    EXPECT_EQ(cross.at<uchar>(2, 1), 1);
    EXPECT_EQ(cross.at<uchar>(2, 2), 0);

    Mat ellipse = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    EXPECT_EQ(ellipse.at<uchar>(0, 2), 1);
    EXPECT_EQ(ellipse.at<uchar>(2, 0), 1);
    EXPECT_EQ(ellipse.at<uchar>(0, 0), 0);

    EXPECT_THROW(
        getStructuringElement(MORPH_RECT, Size(0, 3)),
        Exception);
    EXPECT_THROW(
        getStructuringElement(MORPH_CROSS, Size(3, 3), Point(3, 1)),
        Exception);
}

TEST(KernelsTest, gaussian_and_hanning_have_fixed_numeric_contracts)
{
    for (const int type : {CV_32F, CV_64F})
    {
        Mat gaussian = getGaussianKernel(7, 0.0, type);
        EXPECT_EQ(gaussian.shape(), MatShape({7, 1}));
        EXPECT_NEAR(sum(gaussian)[0], 1.0, type == CV_32F ? 1e-7 : 1e-15);
        for (int i = 0; i < 7; ++i)
        {
            const double left = type == CV_32F
                                    ? gaussian.at<float>(i, 0)
                                    : gaussian.at<double>(i, 0);
            const double right = type == CV_32F
                                     ? gaussian.at<float>(6 - i, 0)
                                     : gaussian.at<double>(6 - i, 0);
            EXPECT_DOUBLE_EQ(left, right);
        }

        Mat hanning;
        createHanningWindow(hanning, Size(5, 5), type);
        EXPECT_DOUBLE_EQ(
            type == CV_32F ? hanning.at<float>(0, 2) : hanning.at<double>(0, 2),
            0.0);
        EXPECT_NEAR(
            type == CV_32F ? hanning.at<float>(2, 2) : hanning.at<double>(2, 2),
            1.0,
            type == CV_32F ? 1e-7 : 1e-15);
    }
}

TEST(KernelsTest, derivative_and_gabor_generators_are_stable)
{
    Mat kx;
    Mat ky;
    getDerivKernels(kx, ky, 1, 0, 3, false, CV_64F);
    EXPECT_DOUBLE_EQ(kx.at<double>(0, 0), -1.0);
    EXPECT_DOUBLE_EQ(kx.at<double>(2, 0), 1.0);
    EXPECT_DOUBLE_EQ(ky.at<double>(1, 0), 2.0);

    getDerivKernels(kx, ky, 1, 0, -1, true, CV_32F);
    EXPECT_FLOAT_EQ(kx.at<float>(0, 0), -1.0f);
    EXPECT_FLOAT_EQ(ky.at<float>(0, 0), 3.0f / 32.0f);
    EXPECT_NEAR(sum(ky)[0], 0.5, 1e-7);

    Mat gabor = getGaborKernel(
        Size(7, 5), 2.0, 0.3, 4.0, 0.8, 0.0, CV_64F);
    EXPECT_EQ(gabor.shape(), MatShape({5, 7}));
    EXPECT_TRUE(std::isfinite(gabor.at<double>(2, 3)));
    EXPECT_THROW(
        getGaborKernel(Size(3, 3), 0.0, 0.0, 2.0, 1.0),
        Exception);
}
