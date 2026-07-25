#include "test/core/support/transpose_test_utils.hpp"

TEST(TransposeTest, transpose_supports_non_contiguous_multichannel_roi)
{
    Mat padded({4, 7}, CV_8UC3);
    fill_with_byte_pattern(padded);
    Mat roi = padded(Range(1, 4), Range(2, 6));
    ASSERT_FALSE(roi.isContinuous());

    Mat dst = transpose(roi);
    ASSERT_EQ(dst.size[0], roi.size[1]);
    ASSERT_EQ(dst.size[1], roi.size[0]);
    ASSERT_EQ(dst.type(), roi.type());

    for (int row = 0; row < roi.size[0]; ++row)
    {
        for (int col = 0; col < roi.size[1]; ++col)
        {
            for (int ch = 0; ch < roi.channels(); ++ch)
            {
                EXPECT_EQ(dst.at<uchar>(col, row, ch), roi.at<uchar>(row, col, ch));
            }
        }
    }
}

TEST(TransposeTest, transpose2d_preserves_interleaved_bytes_for_multi_type_multi_channel)
{
    const int types[] = {
        CV_8UC1,
        CV_8UC3,
        CV_8UC4,
        CV_16SC1,
        CV_16SC3,
        CV_16FC4,
        CV_64FC3,
        CV_32SC2,
        CV_32FC3,
        CV_32FC4,
    };

    for (const int type : types)
    {
        Mat src({5, 7}, type);
        fill_with_byte_pattern(src);

        Mat dst = transpose(src);
        expect_transpose2d_bytes_equal(src, dst);
    }
}

TEST(TransposeTest, transpose3d_last_two_swap_preserves_interleaved_bytes_for_multichannel_types)
{
    const int types[] = {
        CV_8UC3,
        CV_16SC3,
        CV_32FC3,
        CV_32FC4,
    };

    for (const int type : types)
    {
        Mat src({2, 4, 6}, type);
        fill_with_byte_pattern(src);

        Mat dst = transpose(src);
        expect_transpose_last2_3d_bytes_equal(src, dst);
    }
}
