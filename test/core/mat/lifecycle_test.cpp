#include "cvh.h"
#include "gtest/gtest.h"

#include <cstdlib>

using namespace cvh;

TEST(MatLifecycleTest, clone_is_deep_copy)
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

TEST(MatLifecycleTest, copy_assignment_is_shallow_copy)
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

TEST(MatLifecycleTest, empty_mat_copyto_releases_destination)
{
    Mat src;
    Mat dst({2, 2}, CV_32F);
    dst = 3.0f;

    src.copyTo(dst);
    EXPECT_TRUE(dst.empty());
}

TEST(MatLifecycleTest, copyto_type_mismatch_throws)
{
    Mat src({2, 2}, CV_32F);
    src = 1.0f;

    Mat dst({2, 2}, CV_32S);
    EXPECT_THROW(src.copyTo(dst), Exception);
}

TEST(MatLifecycleTest, external_memory_is_not_owned_by_mat)
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

TEST(MatLifecycleTest, f64_multichannel_lifecycle_and_roi_are_supported)
{
    Mat base({4, 6}, CV_64FC3);
    base.setTo(Scalar(1.25, -2.5, 7.75));
    ASSERT_EQ(base.elemSize1(), sizeof(double));
    ASSERT_EQ(base.elemSize(), 3 * sizeof(double));

    Mat roi = base(Range(1, 3), Range(2, 5));
    ASSERT_FALSE(roi.isContinuous());
    EXPECT_EQ(roi.step(0), base.step(0));
    EXPECT_DOUBLE_EQ(roi.at<double>(0, 0, 0), 1.25);
    EXPECT_DOUBLE_EQ(roi.at<double>(0, 0, 1), -2.5);
    EXPECT_DOUBLE_EQ(roi.at<double>(0, 0, 2), 7.75);

    Mat cloned = roi.clone();
    ASSERT_TRUE(cloned.isContinuous());
    ASSERT_EQ(cloned.type(), CV_64FC3);
    EXPECT_DOUBLE_EQ(cloned.at<double>(1, 2, 0), 1.25);
    EXPECT_DOUBLE_EQ(cloned.at<double>(1, 2, 1), -2.5);
    EXPECT_DOUBLE_EQ(cloned.at<double>(1, 2, 2), 7.75);

    Mat copied;
    cloned.copyTo(copied);
    ASSERT_EQ(copied.type(), CV_64FC3);
    EXPECT_EQ(copied.shape(), cloned.shape());
    EXPECT_DOUBLE_EQ(copied.at<double>(1, 2, 2), 7.75);
}

TEST(MatLifecycleTest, unsupported_depth_is_rejected_in_create)
{
    Mat m;
    const int sizes[2] = {2, 2};
    EXPECT_THROW(m.create(2, sizes, CV_16BF), Exception);
}

TEST(MatLifecycleTest, at_i0_checks_upper_bound)
{
    Mat m({2, 2}, CV_8U);
    m.setTo(1.0f);

    EXPECT_NO_THROW((void)m.at<uchar>(0));
    EXPECT_NO_THROW((void)m.at<uchar>(3));
    EXPECT_THROW((void)m.at<uchar>(4), Exception);

    const Mat cm = m;
    EXPECT_NO_THROW((void)cm.at<uchar>(3));
    EXPECT_THROW((void)cm.at<uchar>(4), Exception);
}
