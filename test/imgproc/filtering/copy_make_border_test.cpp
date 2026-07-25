#include "test/imgproc/support/copy_make_border_test_utils.hpp"

TEST(CopyMakeBorderTest, u8_c3_matches_reference_across_border_modes)
{
    Mat src({6, 8}, CV_8UC3);
    fill_u8_pattern(src, 0x1234u);

    const std::vector<int> border_modes = {
        BORDER_CONSTANT,
        BORDER_REPLICATE,
        BORDER_REFLECT,
        BORDER_REFLECT_101,
        BORDER_WRAP,
        BORDER_REFLECT | BORDER_ISOLATED,
    };
    const Scalar border_value(17.0, 29.0, 43.0, 251.0);

    for (int border_mode : border_modes)
    {
        Mat actual;
        copyMakeBorder(src, actual, 2, 1, 3, 2, border_mode, border_value);
        const Mat expected = copy_make_border_reference<uchar>(src, 2, 1, 3, 2, border_mode, border_value);
        EXPECT_EQ(max_abs_diff_u8(actual, expected), 0) << "border_mode=" << border_mode;
    }
}

TEST(CopyMakeBorderTest, f32_c4_matches_reference_for_replicate_and_wrap)
{
    Mat src({5, 7}, CV_32FC4);
    fill_f32_pattern(src);

    const std::vector<int> border_modes = {BORDER_REPLICATE, BORDER_WRAP, BORDER_REFLECT_101};
    for (int border_mode : border_modes)
    {
        Mat actual;
        copyMakeBorder(src, actual, 3, 2, 1, 4, border_mode, Scalar::all(0.0));
        const Mat expected = copy_make_border_reference<float>(src, 3, 2, 1, 4, border_mode, Scalar::all(0.0));
        EXPECT_LE(max_abs_diff_f32(actual, expected), 1e-6f) << "border_mode=" << border_mode;
    }
}

TEST(CopyMakeBorderTest, roi_non_contiguous_matches_reference)
{
    Mat full({9, 12}, CV_8UC4);
    fill_u8_pattern(full, 0x5a5au);
    Mat roi = full(Range(2, 8), Range(1, 10));
    ASSERT_FALSE(roi.isContinuous());

    Mat actual;
    copyMakeBorder(roi, actual, 1, 2, 2, 1, BORDER_REFLECT, Scalar::all(0.0));
    const Mat expected = copy_make_border_reference<uchar>(roi, 1, 2, 2, 1, BORDER_REFLECT, Scalar::all(0.0));
    EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
}

TEST(CopyMakeBorderTest, in_place_same_mat_is_supported)
{
    Mat src({4, 5}, CV_8UC1);
    fill_u8_pattern(src, 0x77u);

    const Mat expected = copy_make_border_reference<uchar>(src, 1, 2, 3, 1, BORDER_WRAP, Scalar::all(0.0));
    copyMakeBorder(src, src, 1, 2, 3, 1, BORDER_WRAP, Scalar::all(0.0));

    EXPECT_EQ(src.size[0], expected.size[0]);
    EXPECT_EQ(src.size[1], expected.size[1]);
    EXPECT_EQ(max_abs_diff_u8(src, expected), 0);
}

TEST(CopyMakeBorderTest, throws_on_invalid_arguments)
{
    Mat empty;
    Mat dst;
    EXPECT_THROW(copyMakeBorder(empty, dst, 1, 1, 1, 1, BORDER_CONSTANT), Exception);

    Mat src_u16({3, 4}, CV_16UC1);
    src_u16.setTo(Scalar::all(7.0));
    EXPECT_THROW(copyMakeBorder(src_u16, dst, 1, 1, 1, 1, BORDER_CONSTANT), Exception);

    Mat src_u8({3, 4}, CV_8UC1);
    src_u8.setTo(Scalar::all(7.0));
    EXPECT_THROW(copyMakeBorder(src_u8, dst, -1, 0, 0, 0, BORDER_CONSTANT), Exception);
    EXPECT_THROW(copyMakeBorder(src_u8, dst, 1, 1, 1, 1, BORDER_TRANSPARENT), Exception);
}
