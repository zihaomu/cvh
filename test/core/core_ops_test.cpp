#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>

using namespace cvh;

TEST(CoreOps_TEST, convert_to_int32_roundtrip)
{
    Mat src({1, 5}, CV_32F);
    float* src_data = reinterpret_cast<float*>(src.data);
    src_data[0] = -2.0f;
    src_data[1] = -1.2f;
    src_data[2] = 0.0f;
    src_data[3] = 2.4f;
    src_data[4] = 9.9f;

    Mat as_int32;
    src.convertTo(as_int32, CV_32S);
    ASSERT_EQ(as_int32.type(), CV_32S);

    Mat back_to_fp32;
    as_int32.convertTo(back_to_fp32, CV_32F);
    ASSERT_EQ(back_to_fp32.type(), CV_32F);

    const float* out = reinterpret_cast<const float*>(back_to_fp32.data);
    EXPECT_NEAR(out[0], -2.0f, 1e-6f);
    EXPECT_NEAR(out[1], -1.0f, 1e-6f);
    EXPECT_NEAR(out[2], 0.0f, 1e-6f);
    EXPECT_NEAR(out[3], 2.0f, 1e-6f);
    EXPECT_NEAR(out[4], 10.0f, 1e-6f);
}

TEST(CoreOps_TEST, copy_to_empty_dst_keeps_source_type_and_shape)
{
    Mat src({2, 2}, CV_32S);
    int* src_data = reinterpret_cast<int*>(src.data);
    src_data[0] = 1;
    src_data[1] = -2;
    src_data[2] = 3;
    src_data[3] = -4;

    Mat dst;
    src.copyTo(dst);

    ASSERT_EQ(dst.type(), CV_32S);
    ASSERT_EQ(dst.shape(), src.shape());
    const int* dst_data = reinterpret_cast<const int*>(dst.data);
    EXPECT_EQ(dst_data[0], 1);
    EXPECT_EQ(dst_data[1], -2);
    EXPECT_EQ(dst_data[2], 3);
    EXPECT_EQ(dst_data[3], -4);
}
