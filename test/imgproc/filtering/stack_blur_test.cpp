#include "test/imgproc/support/intensity_filter_test_utils.hpp"

TEST(StackBlurTest, stack_blur_has_triangular_kernel_and_alias_contract)
{
    Mat impulse({1, 5}, CV_32FC1);
    impulse.setTo(Scalar::all(0));
    impulse.at<float>(0, 2) = 9.0f;
    Mat filtered;
    stackBlur(impulse, filtered, Size(3, 1));
    EXPECT_FLOAT_EQ(filtered.at<float>(0, 1), 2.25f);
    EXPECT_FLOAT_EQ(filtered.at<float>(0, 2), 4.5f);
    EXPECT_FLOAT_EQ(filtered.at<float>(0, 3), 2.25f);

    Mat color({5, 4}, CV_8UC4);
    color.setTo(Scalar(2, 20, 90, 255));
    stackBlur(color, color, Size(5, 3));
    EXPECT_EQ(color.at<uchar>(2, 2, 0), 2);
    EXPECT_EQ(color.at<uchar>(2, 2, 3), 255);
    EXPECT_THROW(stackBlur(color, filtered, Size(4, 3)), Exception);
}

TEST(StackBlurTest, stack_blur_sliding_u8_matches_naive_for_roi_and_channels)
{
    for (const int type : {CV_8UC1, CV_8UC3, CV_8UC4})
    {
        Mat parent({9, 13}, type);
        fill_pattern(parent);
        const Mat roi =
            parent(Range(1, 8), Range(2, 12));
        for (const Size ksize : {Size(3, 5), Size(5, 3)})
        {
            const Mat expected =
                naive_stack_blur_u8(roi, ksize);
            Mat actual;
            stackBlur(roi, actual, ksize);
            ASSERT_EQ(expected.shape(), actual.shape());
            EXPECT_EQ(
                std::memcmp(
                    expected.data,
                    actual.data,
                    expected.total() * expected.elemSize()),
                0);
        }
    }
}
