#include "test/core/support/floating_point_test_utils.hpp"

#include <limits>
#include <string>

namespace {

using cvh::test::compare_float_values;

TEST(FloatingPointTestUtilsTest, accepts_finite_absolute_and_relative_tolerances)
{
    EXPECT_TRUE(
        compare_float_values(0.006f, 0.001f, 0.01f, 0.02f).close);
    EXPECT_TRUE(
        compare_float_values(101.0f, 100.0f, 0.01f, 0.02f).close);
    EXPECT_FALSE(
        compare_float_values(103.0f, 100.0f, 0.01f, 0.02f).close);
}

TEST(FloatingPointTestUtilsTest, applies_absolute_or_relative_rule_per_element)
{
    cvh::Mat actual({1, 2}, CV_32FC1);
    cvh::Mat expected({1, 2}, CV_32FC1);
    actual.at<float>(0, 0) = 0.006f;
    actual.at<float>(0, 1) = 101.0f;
    expected.at<float>(0, 0) = 0.001f;
    expected.at<float>(0, 1) = 100.0f;

    EXPECT_TRUE(cvh::test::mat_close(
        actual, expected, 0.01f, 0.02f, "mixed_tolerances"));

    cvh::Mat actual_parent({2, 4}, CV_32FC3);
    cvh::Mat expected_parent({2, 5}, CV_32FC3);
    cvh::Mat actual_roi = actual_parent.colRange(1, 3);
    cvh::Mat expected_roi = expected_parent.colRange(2, 4);
    actual_roi.setTo(cvh::Scalar::all(7.0));
    expected_roi.setTo(cvh::Scalar::all(7.0));
    ASSERT_FALSE(actual_roi.isContinuous());
    ASSERT_FALSE(expected_roi.isContinuous());
    EXPECT_TRUE(cvh::test::mat_close(
        actual_roi, expected_roi, 0.0f, 0.0f, "roi_channels"));

    actual_roi.at<float>(1, 1, 2) = 8.0f;
    const testing::AssertionResult result = cvh::test::mat_close(
        actual_roi, expected_roi, 0.0f, 0.0f, "roi_channels");
    ASSERT_FALSE(result);
    EXPECT_NE(
        std::string(result.message()).find("first_bad_index=11"),
        std::string::npos);
}

TEST(FloatingPointTestUtilsTest, rejects_nan_and_mismatched_infinity)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();

    EXPECT_FALSE(compare_float_values(nan, 1.0f, 1.0f, 1.0f).close);
    EXPECT_FALSE(compare_float_values(1.0f, nan, 1.0f, 1.0f).close);
    EXPECT_FALSE(compare_float_values(nan, nan, 1.0f, 1.0f).close);
    EXPECT_FALSE(
        compare_float_values(infinity, 1.0f, 1.0f, 1.0f).close);
    EXPECT_FALSE(
        compare_float_values(infinity, -infinity, 1.0f, 1.0f).close);
}

TEST(FloatingPointTestUtilsTest, accepts_same_sign_infinity_and_signed_zero)
{
    const float infinity = std::numeric_limits<float>::infinity();

    EXPECT_TRUE(
        compare_float_values(infinity, infinity, 0.0f, 0.0f).close);
    EXPECT_TRUE(
        compare_float_values(-infinity, -infinity, 0.0f, 0.0f).close);
    EXPECT_TRUE(
        compare_float_values(-0.0f, 0.0f, 0.0f, 0.0f).close);
}

TEST(FloatingPointTestUtilsTest, reports_first_bad_index_and_values)
{
    cvh::Mat actual({1, 3}, CV_32FC1);
    cvh::Mat expected({1, 3}, CV_32FC1);
    actual.at<float>(0, 0) = 1.0f;
    actual.at<float>(0, 1) = 5.0f;
    actual.at<float>(0, 2) = 9.0f;
    expected.at<float>(0, 0) = 1.0f;
    expected.at<float>(0, 1) = 2.0f;
    expected.at<float>(0, 2) = 3.0f;

    const testing::AssertionResult result = cvh::test::mat_close(
        actual, expected, 0.0f, 0.0f, "diagnostic");
    ASSERT_FALSE(result);
    const std::string message = result.message();
    EXPECT_NE(message.find("first_bad_index=1"), std::string::npos);
    EXPECT_NE(message.find("actual=5"), std::string::npos);
    EXPECT_NE(message.find("expected=2"), std::string::npos);
}

}  // namespace
