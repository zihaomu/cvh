#include "test/imgproc/support/copy_make_border_test_utils.hpp"

TEST(CopyMakeBorderUpstreamTest, upstream_findcontours_border_preamble_semantics)
{
    Mat src({8, 10}, CV_8UC1);
    src.setTo(Scalar::all(0.0));

    Mat img;
    copyMakeBorder(src, img, 1, 1, 1, 1, BORDER_CONSTANT, Scalar::all(1.0));

    ASSERT_EQ(img.size[0], 10);
    ASSERT_EQ(img.size[1], 12);
    for (int y = 0; y < img.size[0]; ++y)
    {
        for (int x = 0; x < img.size[1]; ++x)
        {
            const bool is_border = (y == 0 || y == img.size[0] - 1 || x == 0 || x == img.size[1] - 1);
            const uchar expected = is_border ? static_cast<uchar>(1) : static_cast<uchar>(0);
            EXPECT_EQ(img.at<uchar>(y, x), expected) << "y=" << y << ", x=" << x;
        }
    }
}
