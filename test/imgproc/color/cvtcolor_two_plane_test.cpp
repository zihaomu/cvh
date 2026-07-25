#include "test/imgproc/support/pyramid_color_test_utils.hpp"

TEST(CvtColorTwoPlaneTest, two_plane_matches_existing_packed_nv12_and_nv21)
{
    constexpr int rows = 6;
    constexpr int cols = 8;
    Mat y_parent({rows + 2, cols + 3}, CV_8UC1);
    Mat uv_parent({rows / 2 + 2, cols / 2 + 3}, CV_8UC2);
    Mat y = y_parent(Range(1, rows + 1), Range(1, cols + 1));
    Mat uv = uv_parent(
        Range(1, rows / 2 + 1), Range(1, cols / 2 + 1));
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            y.at<uchar>(row, col) =
                static_cast<uchar>(16 + (row * 23 + col * 17) % 220);
        }
    }
    for (int row = 0; row < rows / 2; ++row)
    {
        for (int col = 0; col < cols / 2; ++col)
        {
            uv.at<uchar>(row, col, 0) =
                static_cast<uchar>(70 + row * 7 + col);
            uv.at<uchar>(row, col, 1) =
                static_cast<uchar>(150 + row + col * 3);
        }
    }

    for (const int code :
         {COLOR_YUV2BGR_NV12,
          COLOR_YUV2RGB_NV12,
          COLOR_YUV2BGR_NV21,
          COLOR_YUV2RGB_NV21})
    {
        Mat packed({rows + rows / 2, cols}, CV_8UC1);
        for (int row = 0; row < rows; ++row)
        {
            std::memcpy(
                packed.data + static_cast<size_t>(row) * packed.step(0),
                y.data + static_cast<size_t>(row) * y.step(0),
                cols);
        }
        for (int row = 0; row < rows / 2; ++row)
        {
            std::memcpy(
                packed.data +
                    static_cast<size_t>(rows + row) * packed.step(0),
                uv.data + static_cast<size_t>(row) * uv.step(0),
                cols);
        }
        Mat expected;
        Mat actual;
        cvtColor(packed, expected, code);
        cvtColorTwoPlane(y, uv, actual, code);
        EXPECT_EQ(
            std::memcmp(
                expected.data,
                actual.data,
                actual.total() * actual.elemSize()),
            0);
    }
}
