#include "test/imgproc/support/pyramid_color_test_utils.hpp"

TEST(PyramidTest, pyramid_sizes_constants_and_build_contract)
{
    Mat constant({7, 9}, CV_8UC3);
    constant.setTo(Scalar(12, 34, 210));
    Mat down;
    pyrDown(constant, down);
    EXPECT_EQ(down.shape(), MatShape({4, 5}));
    EXPECT_EQ(down.at<uchar>(2, 3, 0), 12);
    EXPECT_EQ(down.at<uchar>(2, 3, 2), 210);

    Mat up;
    pyrUp(down, up, Size(9, 7));
    EXPECT_EQ(up.shape(), constant.shape());
    EXPECT_EQ(up.at<uchar>(3, 4, 1), 34);

    std::vector<Mat> pyramid;
    buildPyramid(constant, pyramid, 3);
    ASSERT_EQ(pyramid.size(), 4u);
    for (size_t level = 1; level < pyramid.size(); ++level)
    {
        Mat expected;
        pyrDown(pyramid[level - 1], expected);
        EXPECT_EQ(expected.shape(), pyramid[level].shape());
        EXPECT_EQ(
            std::memcmp(
                expected.data,
                pyramid[level].data,
                expected.total() * expected.elemSize()),
            0);
    }
    EXPECT_THROW(pyrDown(constant, down, Size(20, 20)), Exception);
    EXPECT_THROW(pyrUp(down, up, Size(), BORDER_REPLICATE), Exception);
}
