#include "cvh.h"
#include "gtest/gtest.h"

using namespace cvh;

TEST(MatOpenCVCompat_TEST, metadata_helpers_match_continuous_layout)
{
    Mat m({2, 3, 4}, CV_16U);
    ASSERT_FALSE(m.empty());

    EXPECT_EQ(m.depth(), CV_16U);
    EXPECT_EQ(m.channels(), 1);
    EXPECT_EQ(m.elemSize1(), sizeof(ushort));
    EXPECT_EQ(m.elemSize(), sizeof(ushort));
    EXPECT_TRUE(m.isContinuous());

    EXPECT_EQ(m.step(0), static_cast<size_t>(3 * 4 * sizeof(ushort)));
    EXPECT_EQ(m.step(1), static_cast<size_t>(4 * sizeof(ushort)));
    EXPECT_EQ(m.step(2), static_cast<size_t>(sizeof(ushort)));

    EXPECT_EQ(m.step1(0), static_cast<size_t>(12));
    EXPECT_EQ(m.step1(1), static_cast<size_t>(4));
    EXPECT_EQ(m.step1(2), static_cast<size_t>(1));
}

TEST(MatOpenCVCompat_TEST, create_same_shape_and_type_keeps_existing_buffer)
{
    Mat m({2, 2}, CV_32F);
    float* data = reinterpret_cast<float*>(m.data);
    data[0] = 1.5f;

    uchar* before = m.data;
    const int same_sizes[2] = {2, 2};
    m.create(2, same_sizes, CV_32F);

    EXPECT_EQ(m.data, before);
    EXPECT_FLOAT_EQ(reinterpret_cast<float*>(m.data)[0], 1.5f);
}

TEST(MatOpenCVCompat_TEST, copyto_preallocated_buffer_reuses_storage)
{
    Mat src({2, 3}, CV_32S);
    int* src_data = reinterpret_cast<int*>(src.data);
    for (int i = 0; i < 6; ++i)
    {
        src_data[i] = i + 1;
    }

    Mat dst({2, 3}, CV_32S);
    dst = 0;
    uchar* before = dst.data;

    src.copyTo(dst);

    EXPECT_EQ(dst.data, before);
    const int* dst_data = reinterpret_cast<const int*>(dst.data);
    for (int i = 0; i < 6; ++i)
    {
        EXPECT_EQ(dst_data[i], i + 1);
    }
}

TEST(MatOpenCVCompat_TEST, release_then_recreate_remains_usable)
{
    Mat m({2, 2}, CV_8U);
    m.setTo(7.0f);
    m.release();
    EXPECT_TRUE(m.empty());

    const int new_sizes[2] = {1, 5};
    m.create(2, new_sizes, CV_32F);
    ASSERT_FALSE(m.empty());
    EXPECT_EQ(m.shape(), (MatShape{1, 5}));
    EXPECT_EQ(m.type(), CV_32F);

    m = 3.0f;
    const float* out = reinterpret_cast<const float*>(m.data);
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_FLOAT_EQ(out[i], 3.0f);
    }
}
