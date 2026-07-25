#include "test/imgproc/support/resize_test_utils.hpp"

TEST(ResizeUpstreamTest, nearest_regression_15075_from_upstream_imgwarp)
{
    // Ported from OpenCV: modules/imgproc/test/test_imgwarp.cpp
    // TEST(Resize, nearest_regression_15075)
    const int channels = 5;
    const int col = 5;
    const int row = 5;

    Mat src({12, 12}, CV_8UC(channels));
    src = 0;
    for (int ch = 0; ch < channels; ++ch)
    {
        src.at<uchar>(row, col, ch) = 1;
    }

    Mat dst;
    resize(src, dst, Size(11, 11), 0.0, 0.0, INTER_NEAREST);

    ASSERT_EQ(dst.type(), CV_8UC(channels));
    EXPECT_EQ(dst.size[0], 11);
    EXPECT_EQ(dst.size[1], 11);
    EXPECT_EQ(l1_u8(dst), static_cast<double>(channels));
}

TEST(ResizeUpstreamTest, nearest_exact_nearest8u_port_from_upstream)
{
    // Ported from OpenCV: modules/imgproc/test/test_resize_bitexact.cpp
    // TEST(Resize_Bitexact, Nearest8U)
    struct Case
    {
        Mat src;
        Mat expected;
    };

    const std::vector<Case> cases = {
        {mat_u8(1, 6, {0, 1, 2, 3, 4, 5}), mat_u8(1, 3, {1, 3, 5})},
        {mat_u8(1, 5, {0, 1, 2, 3, 4}), mat_u8(1, 1, {2})},
        {mat_u8(1, 5, {0, 1, 2, 3, 4}), mat_u8(1, 3, {0, 2, 4})},
        {mat_u8(1, 5, {0, 1, 2, 3, 4}), mat_u8(1, 2, {1, 3})},
        {
            mat_u8(3, 5, {
                0, 1, 2, 3, 4,
                5, 6, 7, 8, 9,
                10, 11, 12, 13, 14,
            }),
            mat_u8(5, 7, {
                0, 1, 1, 2, 3, 3, 4,
                0, 1, 1, 2, 3, 3, 4,
                5, 6, 6, 7, 8, 8, 9,
                10, 11, 11, 12, 13, 13, 14,
                10, 11, 11, 12, 13, 13, 14,
            }),
        },
        {
            mat_u8(2, 3, {
                0, 1, 2,
                3, 4, 5,
            }),
            mat_u8(4, 6, {
                0, 0, 1, 1, 2, 2,
                0, 0, 1, 1, 2, 2,
                3, 3, 4, 4, 5, 5,
                3, 3, 4, 4, 5, 5,
            }),
        },
    };

    for (size_t i = 0; i < cases.size(); ++i)
    {
        SCOPED_TRACE(i);
        Mat calc;
        resize(cases[i].src, calc, Size(cases[i].expected.size[1], cases[i].expected.size[0]), 0.0, 0.0, INTER_NEAREST_EXACT);
        EXPECT_EQ(max_abs_diff_u8(calc, cases[i].expected), 0);

        const Mat src_t = transpose_u8(cases[i].src);
        const Mat expected_t = transpose_u8(cases[i].expected);
        Mat calc_t;
        resize(src_t, calc_t, Size(expected_t.size[1], expected_t.size[0]), 0.0, 0.0, INTER_NEAREST_EXACT);
        EXPECT_EQ(max_abs_diff_u8(calc_t, expected_t), 0);
    }
}
