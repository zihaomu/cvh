#include "test/imgproc/support/resize_test_utils.hpp"

TEST(ResizeTest, nearest_matches_smoke_reference_grid)
{
    Mat src({2, 2}, CV_8UC1);
    src.at<uchar>(0, 0) = 1;
    src.at<uchar>(0, 1) = 2;
    src.at<uchar>(1, 0) = 3;
    src.at<uchar>(1, 1) = 4;

    Mat dst;
    resize(src, dst, Size(3, 3), 0.0, 0.0, INTER_NEAREST);
    ASSERT_EQ(dst.type(), CV_8UC1);
    ASSERT_EQ(dst.size[0], 3);
    ASSERT_EQ(dst.size[1], 3);

    const uchar expected[9] = {
        1, 1, 2,
        1, 1, 2,
        3, 3, 4,
    };
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            EXPECT_EQ(dst.at<uchar>(y, x), expected[y * 3 + x]);
        }
    }
}

TEST(ResizeTest, linear_matches_independent_reference_on_u8)
{
    Mat src({4, 5}, CV_8UC3);
    for (int y = 0; y < src.size[0]; ++y)
    {
        for (int x = 0; x < src.size[1]; ++x)
        {
            src.at<uchar>(y, x, 0) = static_cast<uchar>((y * 17 + x * 11) % 256);
            src.at<uchar>(y, x, 1) = static_cast<uchar>((y * 7 + x * 23) % 256);
            src.at<uchar>(y, x, 2) = static_cast<uchar>((y * 19 + x * 5) % 256);
        }
    }

    Mat expected = resize_reference_linear_u8(src, Size(7, 6), 0.0, 0.0);
    Mat actual;
    resize(src, actual, Size(7, 6), 0.0, 0.0, INTER_LINEAR);

    ASSERT_EQ(actual.type(), expected.type());
    ASSERT_EQ(actual.size[0], expected.size[0]);
    ASSERT_EQ(actual.size[1], expected.size[1]);
    EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
}

TEST(ResizeTest, dsize_takes_precedence_over_fx_fy)
{
    Mat src({8, 6}, CV_8UC1);
    src = 9;

    Mat dst;
    resize(src, dst, Size(3, 4), 10.0, 10.0, INTER_NEAREST);
    EXPECT_EQ(dst.size[0], 4);
    EXPECT_EQ(dst.size[1], 3);
}

TEST(ResizeTest, fx_fy_are_used_when_dsize_is_empty)
{
    Mat src({6, 10}, CV_8UC1);
    src = 1;

    Mat dst;
    resize(src, dst, Size(), 0.5, 0.5, INTER_NEAREST);
    EXPECT_EQ(dst.size[0], 3);
    EXPECT_EQ(dst.size[1], 5);
}

TEST(ResizeTest, throws_on_invalid_inputs)
{
    Mat dst;
    const Mat empty;
    EXPECT_THROW(resize(empty, dst, Size(2, 2), 0.0, 0.0, INTER_NEAREST), Exception);

    const Mat u16({4, 4}, CV_16UC1);
    EXPECT_THROW(resize(u16, dst, Size(2, 2), 0.0, 0.0, INTER_NEAREST), Exception);

    const Mat src({4, 4}, CV_8UC1);
    EXPECT_THROW(resize(src, dst, Size(), 0.0, 0.0, INTER_NEAREST), Exception);
    EXPECT_THROW(resize(src, dst, Size(2, 2), 0.0, 0.0, 1234), Exception);
}

TEST(ResizeTest, non_contiguous_roi_matches_reference_for_all_interpolations)
{
    Mat base({9, 13}, CV_8UC4);
    for (int y = 0; y < base.size[0]; ++y)
    {
        for (int x = 0; x < base.size[1]; ++x)
        {
            base.at<uchar>(y, x, 0) = static_cast<uchar>((y * 13 + x * 7 + 1) % 256);
            base.at<uchar>(y, x, 1) = static_cast<uchar>((y * 3 + x * 19 + 2) % 256);
            base.at<uchar>(y, x, 2) = static_cast<uchar>((y * 11 + x * 5 + 3) % 256);
            base.at<uchar>(y, x, 3) = static_cast<uchar>((y * 17 + x * 9 + 4) % 256);
        }
    }

    Mat roi = base.colRange(2, 11);
    ASSERT_FALSE(roi.isContinuous());
    ASSERT_EQ(roi.channels(), 4);

    struct Case
    {
        int interpolation;
        Size dsize;
    };

    const std::vector<Case> cases = {
        {INTER_NEAREST, Size(7, 5)},
        {INTER_NEAREST_EXACT, Size(8, 6)},
        {INTER_LINEAR, Size(6, 7)},
    };

    for (const auto& c : cases)
    {
        SCOPED_TRACE(c.interpolation);
        Mat expected;
        if (c.interpolation == INTER_NEAREST)
        {
            expected = resize_reference_nearest_u8(roi, c.dsize, 0.0, 0.0);
        }
        else if (c.interpolation == INTER_NEAREST_EXACT)
        {
            expected = resize_reference_nearest_exact_u8(roi, c.dsize, 0.0, 0.0);
        }
        else
        {
            expected = resize_reference_linear_u8(roi, c.dsize, 0.0, 0.0);
        }

        Mat actual;
        resize(roi, actual, c.dsize, 0.0, 0.0, c.interpolation);

        ASSERT_EQ(actual.type(), expected.type());
        ASSERT_EQ(actual.size[0], expected.size[0]);
        ASSERT_EQ(actual.size[1], expected.size[1]);
        EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
    }
}

TEST(ResizeTest, boundary_sizes_single_row_and_single_col_match_reference)
{
    Mat row_src({1, 9}, CV_8UC3);
    for (int x = 0; x < row_src.size[1]; ++x)
    {
        row_src.at<uchar>(0, x, 0) = static_cast<uchar>((x * 13 + 5) % 256);
        row_src.at<uchar>(0, x, 1) = static_cast<uchar>((x * 7 + 9) % 256);
        row_src.at<uchar>(0, x, 2) = static_cast<uchar>((x * 3 + 17) % 256);
    }

    Mat col_src({9, 1}, CV_8UC3);
    for (int y = 0; y < col_src.size[0]; ++y)
    {
        col_src.at<uchar>(y, 0, 0) = static_cast<uchar>((y * 5 + 3) % 256);
        col_src.at<uchar>(y, 0, 1) = static_cast<uchar>((y * 11 + 1) % 256);
        col_src.at<uchar>(y, 0, 2) = static_cast<uchar>((y * 17 + 7) % 256);
    }

    {
        Mat expected = resize_reference_linear_u8(row_src, Size(13, 1), 0.0, 0.0);
        Mat actual;
        resize(row_src, actual, Size(13, 1), 0.0, 0.0, INTER_LINEAR);
        EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
    }

    {
        Mat expected = resize_reference_nearest_exact_u8(row_src, Size(4, 1), 0.0, 0.0);
        Mat actual;
        resize(row_src, actual, Size(4, 1), 0.0, 0.0, INTER_NEAREST_EXACT);
        EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
    }

    {
        Mat expected = resize_reference_linear_u8(col_src, Size(1, 13), 0.0, 0.0);
        Mat actual;
        resize(col_src, actual, Size(1, 13), 0.0, 0.0, INTER_LINEAR);
        EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
    }

    {
        Mat expected = resize_reference_nearest_u8(col_src, Size(1, 4), 0.0, 0.0);
        Mat actual;
        resize(col_src, actual, Size(1, 4), 0.0, 0.0, INTER_NEAREST);
        EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
    }
}

TEST(ResizeTest, supports_cv32f_all_interpolations_for_c1_c3_c4)
{
    for (int cn : {1, 3, 4})
    {
        SCOPED_TRACE(cn);
        Mat src({6, 7}, CV_MAKETYPE(CV_32F, cn));
        for (int y = 0; y < src.size[0]; ++y)
        {
            for (int x = 0; x < src.size[1]; ++x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    src.at<float>(y, x, c) = static_cast<float>((y - 2) * 0.75 + (x - 3) * 0.42 + c * 1.13);
                }
            }
        }

        struct Case
        {
            int interpolation;
            Size dsize;
        };
        const std::vector<Case> cases = {
            {INTER_NEAREST, Size(5, 4)},
            {INTER_NEAREST_EXACT, Size(8, 5)},
            {INTER_LINEAR, Size(9, 6)},
        };

        for (const auto& c : cases)
        {
            SCOPED_TRACE(c.interpolation);
            Mat expected;
            if (c.interpolation == INTER_NEAREST)
            {
                expected = resize_reference_nearest_f32(src, c.dsize, 0.0, 0.0);
            }
            else if (c.interpolation == INTER_NEAREST_EXACT)
            {
                expected = resize_reference_nearest_exact_f32(src, c.dsize, 0.0, 0.0);
            }
            else
            {
                expected = resize_reference_linear_f32(src, c.dsize, 0.0, 0.0);
            }

            Mat actual;
            resize(src, actual, c.dsize, 0.0, 0.0, c.interpolation);
            ASSERT_EQ(actual.type(), expected.type());
            ASSERT_EQ(actual.size[0], expected.size[0]);
            ASSERT_EQ(actual.size[1], expected.size[1]);
            EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-5f);
        }
    }
}

TEST(ResizeTest, cv32f_non_contiguous_roi_matches_reference)
{
    Mat base({8, 11}, CV_32FC3);
    for (int y = 0; y < base.size[0]; ++y)
    {
        for (int x = 0; x < base.size[1]; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                base.at<float>(y, x, c) = static_cast<float>((y * 0.8) - (x * 0.15) + c * 1.7);
            }
        }
    }

    Mat roi = base.colRange(2, 10);
    ASSERT_FALSE(roi.isContinuous());

    Mat expected = resize_reference_linear_f32(roi, Size(7, 6), 0.0, 0.0);
    Mat actual;
    resize(roi, actual, Size(7, 6), 0.0, 0.0, INTER_LINEAR);

    ASSERT_EQ(actual.type(), expected.type());
    ASSERT_EQ(actual.size[0], expected.size[0]);
    ASSERT_EQ(actual.size[1], expected.size[1]);
    EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-5f);
}
