#include "cvh.h"
#include "gtest/gtest.h"

using namespace cvh;

TEST(MatReshapeTest, reshape_success_shares_storage_and_preserves_total)
{
    Mat src({2, 3}, CV_32F);
    src = 2.0f;

    Mat reshaped = src.reshape({3, 2});
    ASSERT_EQ(reshaped.total(), src.total());
    ASSERT_EQ(reshaped.data, src.data);

    float* reshaped_data = reinterpret_cast<float*>(reshaped.data);
    reshaped_data[5] = 11.0f;
    EXPECT_FLOAT_EQ(reinterpret_cast<float*>(src.data)[5], 11.0f);
}

TEST(MatReshapeTest, reshape_total_mismatch_throws)
{
    Mat src({2, 3}, CV_32F);
    EXPECT_THROW((void)src.reshape({5}), Exception);
}
