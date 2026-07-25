#include "test/imgproc/support/intensity_filter_test_utils.hpp"

TEST(MedianBlurTest, median_blur_handles_boundaries_roi_and_in_place)
{
    Mat impulse({3, 3}, CV_8UC1);
    impulse.setTo(Scalar::all(0));
    impulse.at<uchar>(1, 1) = 255;
    Mat filtered;
    medianBlur(impulse, filtered, 3);
    EXPECT_EQ(filtered.at<uchar>(1, 1), 0);
    EXPECT_EQ(filtered.at<uchar>(0, 0), 0);

    Mat parent({6, 8}, CV_8UC3);
    fill_pattern(parent);
    Mat roi = parent(Range(1, 6), Range(2, 8));
    Mat expected;
    medianBlur(roi, expected, 5);
    Mat in_place = roi.clone();
    medianBlur(in_place, in_place, 5);
    EXPECT_EQ(
        std::memcmp(
            expected.data,
            in_place.data,
            expected.total() * expected.elemSize()),
        0);

    Mat one_row({1, 5}, CV_32FC1);
    for (int x = 0; x < 5; ++x)
    {
        one_row.at<float>(0, x) = static_cast<float>(x);
    }
    medianBlur(one_row, filtered, 3);
    EXPECT_FLOAT_EQ(filtered.at<float>(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(filtered.at<float>(0, 4), 4.0f);
    EXPECT_THROW(medianBlur(one_row, filtered, 7), Exception);
    EXPECT_THROW(medianBlur(impulse, filtered, 4), Exception);
}
