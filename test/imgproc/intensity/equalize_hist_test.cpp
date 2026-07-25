#include "test/imgproc/support/intensity_filter_test_utils.hpp"

TEST(EqualizeHistTest, equalize_hist_handles_constant_bimodal_ramp_and_roi)
{
    Mat constant({4, 5}, CV_8UC1);
    constant.setTo(Scalar::all(73));
    Mat output;
    equalizeHist(constant, output);
    EXPECT_EQ(output.at<uchar>(2, 3), 73);

    Mat bimodal({1, 6}, CV_8UC1);
    for (int x = 0; x < 6; ++x)
    {
        bimodal.at<uchar>(0, x) =
            static_cast<uchar>(x < 3 ? 10 : 200);
    }
    equalizeHist(bimodal, output);
    EXPECT_EQ(output.at<uchar>(0, 0), 0);
    EXPECT_EQ(output.at<uchar>(0, 5), 255);

    Mat parent({3, 258}, CV_8UC1);
    parent.setTo(Scalar::all(0));
    Mat ramp = parent(Range(1, 2), Range(1, 257));
    for (int x = 0; x < 256; ++x)
    {
        ramp.at<uchar>(0, x) = static_cast<uchar>(x);
    }
    equalizeHist(ramp, ramp);
    for (int x = 0; x < 256; ++x)
    {
        EXPECT_EQ(ramp.at<uchar>(0, x), x);
    }
}
