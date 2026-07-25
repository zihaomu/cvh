#include "test/utils/mat_load.h"
#include "gtest/gtest.h"

#include <string>

using namespace cvh;

TEST(NpyReaderTest, reads_float32_shape_and_values)
{
    const std::string path =
        std::string(M_ROOT_PATH) +
        "/test/core/data/npy/random10x12.npy";
    const Mat matrix = readMatFromNpy(path);

    ASSERT_EQ(matrix.shape(), (MatShape{10, 12}));
    ASSERT_EQ(matrix.type(), CV_32F);
    EXPECT_FLOAT_EQ(matrix.at<float>(0, 0), 0.5488135f);
    EXPECT_FLOAT_EQ(matrix.at<float>(0, 1), 0.71518934f);
    EXPECT_FLOAT_EQ(matrix.at<float>(1, 0), 0.56804454f);
}
