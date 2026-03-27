#include "cvh.h"
#include "gtest/gtest.h"

using namespace cvh;

TEST(MatExprScalarCompare_TEST, scalar_arithmetic_overloads_for_mat_and_matexpr)
{
    Mat a({1, 2}, CV_32SC3);
    a.at<int>(0, 0, 0) = 1;  a.at<int>(0, 0, 1) = 2;  a.at<int>(0, 0, 2) = 3;
    a.at<int>(0, 1, 0) = 4;  a.at<int>(0, 1, 1) = 5;  a.at<int>(0, 1, 2) = 6;

    Mat plus_out = Mat(a + Scalar(10.0, 20.0, 30.0));
    EXPECT_EQ(plus_out.at<int>(0, 0, 0), 11);
    EXPECT_EQ(plus_out.at<int>(0, 0, 1), 22);
    EXPECT_EQ(plus_out.at<int>(0, 0, 2), 33);
    EXPECT_EQ(plus_out.at<int>(0, 1, 0), 14);
    EXPECT_EQ(plus_out.at<int>(0, 1, 1), 25);
    EXPECT_EQ(plus_out.at<int>(0, 1, 2), 36);

    Mat mix_out = Mat(Scalar(20.0, 40.0, 60.0) - (a + Scalar(1.0, 1.0, 1.0)));
    EXPECT_EQ(mix_out.at<int>(0, 0, 0), 18);
    EXPECT_EQ(mix_out.at<int>(0, 0, 1), 37);
    EXPECT_EQ(mix_out.at<int>(0, 0, 2), 56);
    EXPECT_EQ(mix_out.at<int>(0, 1, 0), 15);
    EXPECT_EQ(mix_out.at<int>(0, 1, 1), 34);
    EXPECT_EQ(mix_out.at<int>(0, 1, 2), 53);

    Mat mul_out = Mat((a + Scalar(1.0, 1.0, 1.0)) * Scalar(2.0, 3.0, 4.0));
    EXPECT_EQ(mul_out.at<int>(0, 0, 0), 4);
    EXPECT_EQ(mul_out.at<int>(0, 0, 1), 9);
    EXPECT_EQ(mul_out.at<int>(0, 0, 2), 16);
    EXPECT_EQ(mul_out.at<int>(0, 1, 0), 10);
    EXPECT_EQ(mul_out.at<int>(0, 1, 1), 18);
    EXPECT_EQ(mul_out.at<int>(0, 1, 2), 28);

    Mat div_out = Mat(Scalar(100.0, 120.0, 140.0) / (a + Scalar(1.0, 1.0, 1.0)));
    EXPECT_EQ(div_out.at<int>(0, 0, 0), 50);
    EXPECT_EQ(div_out.at<int>(0, 0, 1), 40);
    EXPECT_EQ(div_out.at<int>(0, 0, 2), 35);
    EXPECT_EQ(div_out.at<int>(0, 1, 0), 20);
    EXPECT_EQ(div_out.at<int>(0, 1, 1), 20);
    EXPECT_EQ(div_out.at<int>(0, 1, 2), 20);
}

TEST(MatExprScalarCompare_TEST, compare_ops_return_u8c_mask_and_keep_channels)
{
    Mat a({1, 2}, CV_32SC3);
    Mat b({1, 2}, CV_32SC3);

    a.at<int>(0, 0, 0) = 1;  a.at<int>(0, 0, 1) = 5;  a.at<int>(0, 0, 2) = 3;
    a.at<int>(0, 1, 0) = 7;  a.at<int>(0, 1, 1) = 8;  a.at<int>(0, 1, 2) = 9;
    b.at<int>(0, 0, 0) = 1;  b.at<int>(0, 0, 1) = 4;  b.at<int>(0, 0, 2) = 4;
    b.at<int>(0, 1, 0) = 8;  b.at<int>(0, 1, 1) = 8;  b.at<int>(0, 1, 2) = 1;

    MatExpr ne_expr = a != b;
    ASSERT_EQ(ne_expr.type(), CV_8UC3);
    Mat ne = ne_expr;
    ASSERT_EQ(ne.type(), CV_8UC3);
    EXPECT_EQ(ne.at<uchar>(0, 0, 0), static_cast<uchar>(0));
    EXPECT_EQ(ne.at<uchar>(0, 0, 1), static_cast<uchar>(255));
    EXPECT_EQ(ne.at<uchar>(0, 0, 2), static_cast<uchar>(255));
    EXPECT_EQ(ne.at<uchar>(0, 1, 0), static_cast<uchar>(255));
    EXPECT_EQ(ne.at<uchar>(0, 1, 1), static_cast<uchar>(0));
    EXPECT_EQ(ne.at<uchar>(0, 1, 2), static_cast<uchar>(255));

    Mat gt = a > b;
    ASSERT_EQ(gt.type(), CV_8UC3);
    EXPECT_EQ(gt.at<uchar>(0, 0, 0), static_cast<uchar>(0));
    EXPECT_EQ(gt.at<uchar>(0, 0, 1), static_cast<uchar>(255));
    EXPECT_EQ(gt.at<uchar>(0, 0, 2), static_cast<uchar>(0));
    EXPECT_EQ(gt.at<uchar>(0, 1, 0), static_cast<uchar>(0));
    EXPECT_EQ(gt.at<uchar>(0, 1, 1), static_cast<uchar>(0));
    EXPECT_EQ(gt.at<uchar>(0, 1, 2), static_cast<uchar>(255));
}

TEST(MatExprScalarCompare_TEST, compare_with_scalar_and_float_wrappers_work)
{
    Mat a({1, 2}, CV_32SC3);
    a.at<int>(0, 0, 0) = 3;  a.at<int>(0, 0, 1) = 4;  a.at<int>(0, 0, 2) = 5;
    a.at<int>(0, 1, 0) = 6;  a.at<int>(0, 1, 1) = 7;  a.at<int>(0, 1, 2) = 8;

    Mat ge_scalar = a >= Scalar(3.0, 5.0, 5.0);
    ASSERT_EQ(ge_scalar.type(), CV_8UC3);
    EXPECT_EQ(ge_scalar.at<uchar>(0, 0, 0), static_cast<uchar>(255));
    EXPECT_EQ(ge_scalar.at<uchar>(0, 0, 1), static_cast<uchar>(0));
    EXPECT_EQ(ge_scalar.at<uchar>(0, 0, 2), static_cast<uchar>(255));

    Mat lt_scalar = Scalar(5.0, 5.0, 5.0) < a;
    ASSERT_EQ(lt_scalar.type(), CV_8UC3);
    EXPECT_EQ(lt_scalar.at<uchar>(0, 0, 0), static_cast<uchar>(0));
    EXPECT_EQ(lt_scalar.at<uchar>(0, 0, 1), static_cast<uchar>(0));
    EXPECT_EQ(lt_scalar.at<uchar>(0, 0, 2), static_cast<uchar>(0));
    EXPECT_EQ(lt_scalar.at<uchar>(0, 1, 0), static_cast<uchar>(255));
    EXPECT_EQ(lt_scalar.at<uchar>(0, 1, 1), static_cast<uchar>(255));
    EXPECT_EQ(lt_scalar.at<uchar>(0, 1, 2), static_cast<uchar>(255));

    Mat eq_float = a == 6.0f;
    ASSERT_EQ(eq_float.type(), CV_8UC3);
    EXPECT_EQ(eq_float.at<uchar>(0, 1, 0), static_cast<uchar>(255));
    EXPECT_EQ(eq_float.at<uchar>(0, 1, 1), static_cast<uchar>(0));
    EXPECT_EQ(eq_float.at<uchar>(0, 1, 2), static_cast<uchar>(0));
}

TEST(MatExprScalarCompare_TEST, scalar_expr_paths_reject_more_than_four_channels)
{
    Mat src({1, 2}, CV_32SC(5));

    EXPECT_THROW({ Mat tmp = src + Scalar::all(1.0); }, Exception);
    EXPECT_THROW({ Mat tmp = Scalar::all(1.0) - src; }, Exception);
    EXPECT_THROW({ Mat tmp = src != Scalar::all(1.0); }, Exception);
}
