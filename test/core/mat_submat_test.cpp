#include "cvh.h"
#include "gtest/gtest.h"

using namespace cvh;

TEST(MatSubmat_TEST, colrange_is_view_and_non_continuous)
{
    Mat src({3, 5}, CV_32S);
    int* src_data = reinterpret_cast<int*>(src.data);
    for (int i = 0; i < 15; ++i)
    {
        src_data[i] = i;
    }

    Mat roi = src.colRange(1, 4);
    ASSERT_EQ(roi.shape(), (MatShape{3, 3}));
    EXPECT_EQ(roi.step(0), src.step(0));
    EXPECT_FALSE(roi.isContinuous());

    roi.setTo(9.0f);
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 5; ++c)
        {
            const int idx = r * 5 + c;
            const int expected = (c >= 1 && c < 4) ? 9 : idx;
            EXPECT_EQ(src_data[idx], expected);
        }
    }
}

TEST(MatSubmat_TEST, clone_from_non_continuous_roi_is_continuous_and_deep_copy)
{
    Mat src({3, 5}, CV_32S);
    int* src_data = reinterpret_cast<int*>(src.data);
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 5; ++c)
        {
            src_data[r * 5 + c] = r * 10 + c;
        }
    }

    Mat roi = src.colRange(1, 4);
    Mat cloned = roi.clone();
    ASSERT_EQ(cloned.shape(), (MatShape{3, 3}));
    EXPECT_TRUE(cloned.isContinuous());

    const int* cloned_data = reinterpret_cast<const int*>(cloned.data);
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            EXPECT_EQ(cloned_data[r * 3 + c], r * 10 + (c + 1));
        }
    }

    cloned.setTo(-5.0f);
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 1; c < 4; ++c)
        {
            EXPECT_EQ(src_data[r * 5 + c], r * 10 + c);
        }
    }
}

TEST(MatSubmat_TEST, copyto_supports_non_continuous_dst_roi)
{
    Mat src({3, 5}, CV_32S);
    int* src_data = reinterpret_cast<int*>(src.data);
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 5; ++c)
        {
            src_data[r * 5 + c] = r * 100 + c;
        }
    }

    Mat dst_base({3, 5}, CV_32S);
    dst_base.setTo(0.0f);

    Mat src_roi = src.colRange(1, 4);
    Mat dst_roi = dst_base.colRange(1, 4);
    uchar* before = dst_roi.data;

    src_roi.copyTo(dst_roi);
    EXPECT_EQ(dst_roi.data, before);

    const int* dst_data = reinterpret_cast<const int*>(dst_base.data);
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 5; ++c)
        {
            const int idx = r * 5 + c;
            const int expected = (c >= 1 && c < 4) ? (r * 100 + c) : 0;
            EXPECT_EQ(dst_data[idx], expected);
        }
    }
}

TEST(MatSubmat_TEST, convertto_from_non_continuous_multichannel_roi)
{
    Mat src({2, 4}, CV_32FC3);
    float* src_data = reinterpret_cast<float*>(src.data);
    for (int r = 0; r < 2; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                const int idx = (r * 4 + c) * 3 + ch;
                src_data[idx] = static_cast<float>(r * 100 + c * 10 + ch);
            }
        }
    }

    Mat roi = src.colRange(1, 3);
    ASSERT_FALSE(roi.isContinuous());
    ASSERT_EQ(roi.channels(), 3);

    Mat dst;
    roi.convertTo(dst, CV_8U);
    ASSERT_EQ(dst.type(), CV_8UC3);
    ASSERT_EQ(dst.shape(), (MatShape{2, 2}));
    EXPECT_TRUE(dst.isContinuous());

    const uchar* out = reinterpret_cast<const uchar*>(dst.data);
    const uchar expected[12] = {
        10, 11, 12, 20, 21, 22,
        110, 111, 112, 120, 121, 122
    };
    for (int i = 0; i < 12; ++i)
    {
        EXPECT_EQ(out[i], expected[i]);
    }
}

TEST(MatSubmat_TEST, range_operator_creates_view)
{
    Mat src({4, 5}, CV_32S);
    int* src_data = reinterpret_cast<int*>(src.data);
    for (int i = 0; i < 20; ++i)
    {
        src_data[i] = i;
    }

    Mat roi = src(Range(1, 3), Range(2, 5));
    ASSERT_EQ(roi.shape(), (MatShape{2, 3}));
    EXPECT_FALSE(roi.isContinuous());

    roi.setTo(7.0f);
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 5; ++c)
        {
            const int idx = r * 5 + c;
            const bool in_roi = (r >= 1 && r < 3 && c >= 2 && c < 5);
            const int expected = in_roi ? 7 : idx;
            EXPECT_EQ(src_data[idx], expected);
        }
    }
}
