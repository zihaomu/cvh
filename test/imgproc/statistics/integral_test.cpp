#include "test/imgproc/support/kernel_family_test_utils.hpp"

TEST(IntegralTest, integral_has_zero_border_and_multichannel_values)
{
    Mat src({2, 3}, CV_8UC3);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                src.at<uchar>(y, x, ch) =
                    static_cast<uchar>(1 + y * 3 + x + ch);
            }
        }
    }
    Mat sum32;
    integral(src, sum32);
    ASSERT_EQ(sum32.shape(), MatShape({3, 4}));
    ASSERT_EQ(sum32.type(), CV_32SC3);
    EXPECT_EQ(sum32.at<int>(0, 3, 2), 0);
    EXPECT_EQ(sum32.at<int>(2, 3, 0), 21);
    EXPECT_EQ(sum32.at<int>(2, 3, 2), 33);

    Mat sum64;
    integral(src, sum64, CV_64F);
    EXPECT_DOUBLE_EQ(sum64.at<double>(2, 3, 0), 21.0);

    Mat roi_parent({4, 5}, CV_8UC1);
    roi_parent.setTo(Scalar::all(2));
    Mat roi = roi_parent(Range(1, 4), Range(1, 5));
    integral(roi, sum32);
    EXPECT_EQ(sum32.at<int>(3, 4), 24);
}

TEST(IntegralTest, rejects_unsupported_float_input)
{
    Mat source({3, 3}, CV_32FC1);
    Mat output;

    EXPECT_THROW(integral(source, output), Exception);
}
