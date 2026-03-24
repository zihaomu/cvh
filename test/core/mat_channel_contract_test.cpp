#include "cvh.h"
#include "gtest/gtest.h"

using namespace cvh;

TEST(MatChannelContract_TEST, opencv_like_type_macros_encode_decode_channels)
{
    static_assert(CV_MAT_DEPTH(CV_8UC3) == CV_8U, "depth decode mismatch");
    static_assert(CV_MAT_CN(CV_8UC3) == 3, "channel decode mismatch");
    static_assert(CV_MAKETYPE(CV_32F, 4) == CV_32FC4, "type encode mismatch");

    EXPECT_EQ(CV_MAT_DEPTH(CV_16SC2), CV_16S);
    EXPECT_EQ(CV_MAT_CN(CV_16SC2), 2);
}

TEST(MatChannelContract_TEST, create_setto_and_copyto_cover_all_channel_elements)
{
    Mat src({2, 2}, CV_8UC3);
    ASSERT_EQ(src.channels(), 3);
    ASSERT_EQ(src.elemSize(), static_cast<size_t>(3));
    ASSERT_EQ(src.step1(0), static_cast<size_t>(6));
    src.setTo(7.0f);

    const uchar* src_data = reinterpret_cast<const uchar*>(src.data);
    for (int i = 0; i < 12; ++i)
    {
        EXPECT_EQ(src_data[i], static_cast<uchar>(7));
    }

    Mat dst({2, 2}, CV_8UC3);
    src.copyTo(dst);
    ASSERT_EQ(dst.type(), CV_8UC3);
    ASSERT_EQ(dst.channels(), 3);

    const uchar* dst_data = reinterpret_cast<const uchar*>(dst.data);
    for (int i = 0; i < 12; ++i)
    {
        EXPECT_EQ(dst_data[i], static_cast<uchar>(7));
    }
}

TEST(MatChannelContract_TEST, convertto_keeps_channels_when_rtype_is_depth)
{
    Mat src({1, 2}, CV_32FC3);
    float* src_data = reinterpret_cast<float*>(src.data);
    src_data[0] = -1.2f;
    src_data[1] = 0.0f;
    src_data[2] = 2.4f;
    src_data[3] = 255.0f;
    src_data[4] = 256.1f;
    src_data[5] = 12.6f;

    Mat dst;
    src.convertTo(dst, CV_8U);

    ASSERT_EQ(dst.type(), CV_8UC3);
    ASSERT_EQ(dst.channels(), 3);
    ASSERT_EQ(dst.shape(), src.shape());

    const uchar* out = reinterpret_cast<const uchar*>(dst.data);
    EXPECT_EQ(out[0], static_cast<uchar>(0));
    EXPECT_EQ(out[1], static_cast<uchar>(0));
    EXPECT_EQ(out[2], static_cast<uchar>(2));
    EXPECT_EQ(out[3], static_cast<uchar>(255));
    EXPECT_EQ(out[4], static_cast<uchar>(255));
    EXPECT_EQ(out[5], static_cast<uchar>(13));
}
