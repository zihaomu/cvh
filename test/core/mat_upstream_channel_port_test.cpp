#include "cvh.h"
#include "gtest/gtest.h"

using namespace cvh;

namespace
{
void expect_all_u8(const Mat& m, uchar expected)
{
    ASSERT_EQ(m.depth(), CV_8U);
    const size_t scalar_count = m.total() * static_cast<size_t>(m.channels());
    const uchar* data = reinterpret_cast<const uchar*>(m.data);
    ASSERT_NE(data, nullptr);
    for (size_t i = 0; i < scalar_count; ++i)
    {
        EXPECT_EQ(data[i], expected);
    }
}

void skip_pending(const char* upstream_case, const char* reason)
{
    GTEST_SKIP() << "Pending upstream port: " << upstream_case << " | " << reason;
}
} // namespace

// From modules/core/test/test_mat.cpp
TEST(OpenCVUpstreamChannelPort_TEST, Core_Merge_shape_operations)
{
    skip_pending("Core_Merge.shape_operations", "cvh::merge/split API is not available yet");
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Split_shape_operations)
{
    skip_pending("Core_Split.shape_operations", "cvh::merge/split API is not available yet");
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Merge_hang_12171)
{
    skip_pending("Core_Merge.hang_12171", "cvh::merge API is not available yet");
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Split_hang_12171)
{
    skip_pending("Core_Split.hang_12171", "cvh::split API is not available yet");
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Split_crash_12171)
{
    skip_pending("Core_Split.crash_12171", "cvh::split API is not available yet");
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Merge_bug_13544)
{
    skip_pending("Core_Merge.bug_13544", "cvh::merge API is not available yet");
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Mat_reinterpret_Mat_8UC3_8SC3)
{
    skip_pending("Core_Mat.reinterpret_Mat_8UC3_8SC3", "cvh::Mat::reinterpret is not implemented");
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Mat_reinterpret_Mat_8UC4_32FC1)
{
    skip_pending("Core_Mat.reinterpret_Mat_8UC4_32FC1", "cvh::Mat::reinterpret is not implemented");
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Mat_reinterpret_OutputArray_8UC3_8SC3)
{
    skip_pending("Core_Mat.reinterpret_OutputArray_8UC3_8SC3", "OutputArray compatibility is not implemented");
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_Mat_reinterpret_OutputArray_8UC4_32FC1)
{
    skip_pending("Core_Mat.reinterpret_OutputArray_8UC4_32FC1", "OutputArray compatibility is not implemented");
}

TEST(OpenCVUpstreamChannelPort_TEST, Core_MatExpr_issue_16655)
{
    skip_pending("Core_MatExpr.issue_16655", "channel-aware MatExpr compare (!=) is not implemented");
}

// From modules/core/test/test_arithm.cpp
TEST(OpenCVUpstreamChannelPort_TEST, Subtract_scalarc1_matc3)
{
    // Upstream uses cv::subtract(int scalar, Mat, dst). cvh does not expose that overload yet,
    // so this port verifies equivalent behavior through scalar Mat broadcasting by construction.
    Mat src({5, 5}, CV_8UC3);
    src.setTo(5.0f);
    Mat scalar_like(src.shape(), CV_8UC3);
    scalar_like.setTo(255.0f);

    Mat dst;
    subtract(scalar_like, src, dst);

    ASSERT_EQ(dst.type(), CV_8UC3);
    expect_all_u8(dst, static_cast<uchar>(250));
}

TEST(OpenCVUpstreamChannelPort_TEST, Subtract_scalarc4_matc4)
{
    Mat src({5, 5}, CV_8UC4);
    src.setTo(5.0f);
    Mat scalar_like(src.shape(), CV_8UC4);
    scalar_like.setTo(255.0f);

    Mat dst;
    subtract(scalar_like, src, dst);

    ASSERT_EQ(dst.type(), CV_8UC4);
    expect_all_u8(dst, static_cast<uchar>(250));
}

TEST(OpenCVUpstreamChannelPort_TEST, Compare_empty)
{
    skip_pending("Compare.empty", "cvh::compare is not implemented");
}

TEST(OpenCVUpstreamChannelPort_TEST, Compare_regression_8999)
{
    skip_pending("Compare.regression_8999", "cvh::compare is not implemented");
}

TEST(OpenCVUpstreamChannelPort_TEST, Compare_regression_16F_do_not_crash)
{
    skip_pending("Compare.regression_16F_do_not_crash", "cvh::compare is not implemented");
}

// From modules/core/test/test_operations.cpp
TEST(OpenCVUpstreamChannelPort_TEST, Core_Array_expressions)
{
    // Keep this runnable as a minimal channel-semantics checkpoint extracted from CV_OperationsTest.
    Mat m1({1, 1}, CV_8UC1);
    Mat m2({1, 1}, CV_32FC2);
    Mat m3({1, 1}, CV_16SC3);

    EXPECT_EQ(m1.channels(), 1);
    EXPECT_EQ(m2.channels(), 2);
    EXPECT_EQ(m3.channels(), 3);
}
