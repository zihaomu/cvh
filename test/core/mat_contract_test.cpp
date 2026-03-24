#include "cvh.h"
#include "gtest/gtest.h"

#include <cstdlib>

using namespace cvh;

TEST(MatContract_TEST, clone_is_deep_copy)
{
    Mat src({2, 3}, CV_32F);
    float* src_data = reinterpret_cast<float*>(src.data);
    for (int i = 0; i < 6; ++i)
    {
        src_data[i] = static_cast<float>(i + 1);
    }

    Mat cloned = src.clone();
    ASSERT_EQ(cloned.shape(), src.shape());
    ASSERT_EQ(cloned.type(), src.type());
    ASSERT_NE(cloned.data, src.data);

    float* cloned_data = reinterpret_cast<float*>(cloned.data);
    cloned_data[0] = -100.0f;
    EXPECT_FLOAT_EQ(src_data[0], 1.0f);
}

TEST(MatContract_TEST, copy_assignment_is_shallow_copy)
{
    Mat src({2, 2}, CV_32S);
    int* src_data = reinterpret_cast<int*>(src.data);
    src_data[0] = 10;
    src_data[1] = 20;
    src_data[2] = 30;
    src_data[3] = 40;

    Mat alias = src;
    ASSERT_EQ(alias.data, src.data);

    int* alias_data = reinterpret_cast<int*>(alias.data);
    alias_data[1] = -7;
    EXPECT_EQ(src_data[1], -7);
}

TEST(MatContract_TEST, reshape_success_shares_storage_and_preserves_total)
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

TEST(MatContract_TEST, reshape_total_mismatch_throws)
{
    Mat src({2, 3}, CV_32F);
    EXPECT_THROW((void)src.reshape({5}), Exception);
}

TEST(MatContract_TEST, convert_to_uint8_preserves_shape_and_saturates)
{
    Mat src({1, 5}, CV_32F);
    float* src_data = reinterpret_cast<float*>(src.data);
    src_data[0] = -5.1f;
    src_data[1] = 0.4f;
    src_data[2] = 12.6f;
    src_data[3] = 255.0f;
    src_data[4] = 300.0f;

    Mat dst;
    src.convertTo(dst, CV_8U);

    ASSERT_EQ(dst.shape(), src.shape());
    ASSERT_EQ(dst.type(), CV_8U);

    const uchar* out = reinterpret_cast<const uchar*>(dst.data);
    EXPECT_EQ(out[0], static_cast<uchar>(0));
    EXPECT_EQ(out[1], static_cast<uchar>(0));
    EXPECT_EQ(out[2], static_cast<uchar>(13));
    EXPECT_EQ(out[3], static_cast<uchar>(255));
    EXPECT_EQ(out[4], static_cast<uchar>(255));
}

TEST(MatContract_TEST, convert_to_unsupported_type_throws)
{
    Mat src({1, 4}, CV_32F);
    Mat dst;
    EXPECT_THROW(src.convertTo(dst, CV_64F), Exception);
}

TEST(MatContract_TEST, setto_covers_all_elements_for_odd_16bit_shape)
{
    Mat m({3}, CV_16U);
    m.setTo(9.0f);

    const ushort* out = reinterpret_cast<const ushort*>(m.data);
    EXPECT_EQ(out[0], static_cast<ushort>(9));
    EXPECT_EQ(out[1], static_cast<ushort>(9));
    EXPECT_EQ(out[2], static_cast<ushort>(9));
}

TEST(MatContract_TEST, setto_covers_all_elements_for_odd_16s_shape)
{
    Mat m({5}, CV_16S);
    m.setTo(-3.0f);

    const short* out = reinterpret_cast<const short*>(m.data);
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(out[i], static_cast<short>(-3));
    }
}

TEST(MatContract_TEST, setto_covers_all_elements_for_odd_16f_shape)
{
    Mat m({3}, CV_16F);
    m.setTo(1.75f);

    const hfloat* out = reinterpret_cast<const hfloat*>(m.data);
    EXPECT_NEAR(static_cast<float>(out[0]), 1.75f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(out[1]), 1.75f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(out[2]), 1.75f, 1e-3f);
}

TEST(MatContract_TEST, empty_mat_copyto_releases_destination)
{
    Mat src;
    Mat dst({2, 2}, CV_32F);
    dst = 3.0f;

    src.copyTo(dst);
    EXPECT_TRUE(dst.empty());
}

TEST(MatContract_TEST, copyto_type_mismatch_throws)
{
    Mat src({2, 2}, CV_32F);
    src = 1.0f;

    Mat dst({2, 2}, CV_32S);
    EXPECT_THROW(src.copyTo(dst), Exception);
}

TEST(MatContract_TEST, external_memory_is_not_owned_by_mat)
{
    auto* raw = static_cast<float*>(std::malloc(4 * sizeof(float)));
    ASSERT_NE(raw, nullptr);
    raw[0] = 1.0f;
    raw[1] = 2.0f;
    raw[2] = 3.0f;
    raw[3] = 4.0f;

    {
        Mat wrapped({2, 2}, CV_32F, raw);
        wrapped.setTo(6.0f);
    }

    raw[0] = 7.0f;
    EXPECT_FLOAT_EQ(raw[0], 7.0f);
    std::free(raw);
}

TEST(MatContract_TEST, unsupported_depth_is_rejected_in_create)
{
    Mat m;
    const int sizes[2] = {2, 2};
    EXPECT_THROW(m.create(2, sizes, CV_64F), Exception);
}
