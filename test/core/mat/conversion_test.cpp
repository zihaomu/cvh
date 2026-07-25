#include "cvh.h"
#include "gtest/gtest.h"

using namespace cvh;

TEST(MatConversionTest, convert_to_uint8_preserves_shape_and_saturates)
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

TEST(MatConversionTest, convert_to_f64_and_back_preserves_values)
{
    Mat src({1, 4}, CV_32F);
    src.at<float>(0, 0) = -3.5f;
    src.at<float>(0, 1) = 0.0f;
    src.at<float>(0, 2) = 12.25f;
    src.at<float>(0, 3) = 255.75f;

    Mat f64;
    src.convertTo(f64, CV_64F);
    ASSERT_EQ(f64.type(), CV_64FC1);
    EXPECT_DOUBLE_EQ(f64.at<double>(0, 0), -3.5);
    EXPECT_DOUBLE_EQ(f64.at<double>(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(f64.at<double>(0, 2), 12.25);
    EXPECT_DOUBLE_EQ(f64.at<double>(0, 3), 255.75);

    Mat roundtrip;
    f64.convertTo(roundtrip, CV_32F);
    ASSERT_EQ(roundtrip.type(), CV_32FC1);
    for (int x = 0; x < 4; ++x)
    {
        EXPECT_FLOAT_EQ(roundtrip.at<float>(0, x), src.at<float>(0, x));
    }
}

TEST(MatConversionTest, convert_between_u8_and_f64_preserves_and_saturates_values)
{
    Mat u8({1, 4}, CV_8U);
    u8.at<uchar>(0, 0) = 0;
    u8.at<uchar>(0, 1) = 1;
    u8.at<uchar>(0, 2) = 127;
    u8.at<uchar>(0, 3) = 255;

    Mat f64;
    u8.convertTo(f64, CV_64F);
    ASSERT_EQ(f64.type(), CV_64FC1);
    EXPECT_DOUBLE_EQ(f64.at<double>(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(f64.at<double>(0, 1), 1.0);
    EXPECT_DOUBLE_EQ(f64.at<double>(0, 2), 127.0);
    EXPECT_DOUBLE_EQ(f64.at<double>(0, 3), 255.0);

    f64.at<double>(0, 0) = -1.0;
    f64.at<double>(0, 1) = 12.6;
    f64.at<double>(0, 2) = 254.4;
    f64.at<double>(0, 3) = 300.0;

    Mat roundtrip;
    f64.convertTo(roundtrip, CV_8U);
    ASSERT_EQ(roundtrip.type(), CV_8UC1);
    EXPECT_EQ(roundtrip.at<uchar>(0, 0), static_cast<uchar>(0));
    EXPECT_EQ(roundtrip.at<uchar>(0, 1), static_cast<uchar>(13));
    EXPECT_EQ(roundtrip.at<uchar>(0, 2), static_cast<uchar>(254));
    EXPECT_EQ(roundtrip.at<uchar>(0, 3), static_cast<uchar>(255));
}

TEST(MatConversionTest, setto_covers_all_elements_for_odd_16bit_shape)
{
    Mat m({3}, CV_16U);
    m.setTo(9.0f);

    const ushort* out = reinterpret_cast<const ushort*>(m.data);
    EXPECT_EQ(out[0], static_cast<ushort>(9));
    EXPECT_EQ(out[1], static_cast<ushort>(9));
    EXPECT_EQ(out[2], static_cast<ushort>(9));
}

TEST(MatConversionTest, setto_covers_all_elements_for_odd_16s_shape)
{
    Mat m({5}, CV_16S);
    m.setTo(-3.0f);

    const short* out = reinterpret_cast<const short*>(m.data);
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(out[i], static_cast<short>(-3));
    }
}

TEST(MatConversionTest, setto_covers_all_elements_for_odd_16f_shape)
{
    Mat m({3}, CV_16F);
    m.setTo(1.75f);

    const hfloat* out = reinterpret_cast<const hfloat*>(m.data);
    EXPECT_NEAR(static_cast<float>(out[0]), 1.75f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(out[1]), 1.75f, 1e-3f);
    EXPECT_NEAR(static_cast<float>(out[2]), 1.75f, 1e-3f);
}
