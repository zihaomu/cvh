#include "test/core/support/arithmetic_test_utils.hpp"
#include "test/support/dispatch_mode_guard.hpp"
#include "cvh/core/detail/dispatch_control.h"

TEST(ArithmeticDispatchInternalTest, mat_mat_add_sub_mul_int32_and_uint32_support_scalar_only_mode)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard(
        cpu::DispatchMode::ScalarOnly);

    Mat out;

    const Mat a32s = make_vec_mat<int>({10, -20, 30, -40}, CV_32SC1);
    const Mat b32s = make_vec_mat<int>({3, 5, -7, 2}, CV_32SC1);
    binaryFunc(BinaryOp::ADD, a32s, b32s, out);
    expect_vec_eq<int>(out, {13, -15, 23, -38});
    binaryFunc(BinaryOp::SUB, a32s, b32s, out);
    expect_vec_eq<int>(out, {7, -25, 37, -42});
    binaryFunc(BinaryOp::MUL, a32s, b32s, out);
    expect_vec_eq<int>(out, {30, -100, -210, -80});

    out = Mat();
    const Mat a32u = make_vec_mat<uint>({10u, 20u, 30u, 40u}, CV_32UC1);
    const Mat b32u = make_vec_mat<uint>({3u, 5u, 7u, 2u}, CV_32UC1);
    binaryFunc(BinaryOp::ADD, a32u, b32u, out);
    expect_vec_eq<uint>(out, {13u, 25u, 37u, 42u});
    binaryFunc(BinaryOp::SUB, a32u, b32u, out);
    expect_vec_eq<uint>(out, {7u, 15u, 23u, 38u});
    binaryFunc(BinaryOp::MUL, a32u, b32u, out);
    expect_vec_eq<uint>(out, {30u, 100u, 210u, 80u});

}

TEST(ArithmeticDispatchInternalTest, mat_mat_add_sub_u8_s8_u16_s16_support_scalar_only_with_saturation)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard(
        cpu::DispatchMode::ScalarOnly);

    Mat out;

    const Mat a8u = make_vec_mat<uchar>({250, 10, 5, 0}, CV_8UC1);
    const Mat b8u = make_vec_mat<uchar>({10, 250, 7, 1}, CV_8UC1);
    binaryFunc(BinaryOp::ADD, a8u, b8u, out);
    expect_vec_eq<uchar>(out, {255, 255, 12, 1});
    binaryFunc(BinaryOp::SUB, a8u, b8u, out);
    expect_vec_eq<uchar>(out, {240, 0, 0, 0});

    out = Mat();
    const Mat a8s = make_vec_mat<schar>({120, -120, 50, -50}, CV_8SC1);
    const Mat b8s = make_vec_mat<schar>({20, -20, -100, 100}, CV_8SC1);
    binaryFunc(BinaryOp::ADD, a8s, b8s, out);
    expect_vec_eq<schar>(out, {127, -128, -50, 50});
    binaryFunc(BinaryOp::SUB, a8s, b8s, out);
    expect_vec_eq<schar>(out, {100, -100, 127, -128});

    out = Mat();
    const Mat a16u = make_vec_mat<ushort>({65530, 10, 3, 0}, CV_16UC1);
    const Mat b16u = make_vec_mat<ushort>({100, 65530, 9, 1}, CV_16UC1);
    binaryFunc(BinaryOp::ADD, a16u, b16u, out);
    expect_vec_eq<ushort>(out, {65535, 65535, 12, 1});
    binaryFunc(BinaryOp::SUB, a16u, b16u, out);
    expect_vec_eq<ushort>(out, {65430, 0, 0, 0});

    out = Mat();
    const Mat a16s = make_vec_mat<short>({32760, -32760, 1000, -1000}, CV_16SC1);
    const Mat b16s = make_vec_mat<short>({100, -100, -2000, 2000}, CV_16SC1);
    binaryFunc(BinaryOp::ADD, a16s, b16s, out);
    expect_vec_eq<short>(out, {32767, -32768, -1000, 1000});
    binaryFunc(BinaryOp::SUB, a16s, b16s, out);
    expect_vec_eq<short>(out, {32660, -32660, 3000, -3000});

}

TEST(ArithmeticDispatchInternalTest, mat_scalar_add_sub_mul_int32_and_uint32_support_scalar_only_mode)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard(
        cpu::DispatchMode::ScalarOnly);

    Mat out;

    const Mat src32s = make_vec_mat<int>({10, -20, 30, -40}, CV_32SC1);
    add(src32s, Scalar(5.0), out);
    expect_vec_eq<int>(out, {15, -15, 35, -35});
    subtract(Scalar(7.0), src32s, out);
    expect_vec_eq<int>(out, {-3, 27, -23, 47});
    multiply(src32s, Scalar(-2.0), out);
    expect_vec_eq<int>(out, {-20, 40, -60, 80});

    out = Mat();
    Mat src32u({1, 2}, CV_32UC3);
    src32u.at<uint>(0, 0, 0) = 10u;
    src32u.at<uint>(0, 0, 1) = 20u;
    src32u.at<uint>(0, 0, 2) = 30u;
    src32u.at<uint>(0, 1, 0) = 40u;
    src32u.at<uint>(0, 1, 1) = 50u;
    src32u.at<uint>(0, 1, 2) = 60u;

    add(src32u, Scalar(3.0, 5.0, 7.0), out);
    EXPECT_EQ(out.at<uint>(0, 0, 0), 13u);
    EXPECT_EQ(out.at<uint>(0, 0, 1), 25u);
    EXPECT_EQ(out.at<uint>(0, 0, 2), 37u);
    EXPECT_EQ(out.at<uint>(0, 1, 0), 43u);
    EXPECT_EQ(out.at<uint>(0, 1, 1), 55u);
    EXPECT_EQ(out.at<uint>(0, 1, 2), 67u);

    subtract(Scalar(100.0, 80.0, 60.0), src32u, out);
    EXPECT_EQ(out.at<uint>(0, 0, 0), 90u);
    EXPECT_EQ(out.at<uint>(0, 0, 1), 60u);
    EXPECT_EQ(out.at<uint>(0, 0, 2), 30u);
    EXPECT_EQ(out.at<uint>(0, 1, 0), 60u);
    EXPECT_EQ(out.at<uint>(0, 1, 1), 30u);
    EXPECT_EQ(out.at<uint>(0, 1, 2), 0u);

    multiply(src32u, Scalar(2.0, 3.0, 4.0), out);
    EXPECT_EQ(out.at<uint>(0, 0, 0), 20u);
    EXPECT_EQ(out.at<uint>(0, 0, 1), 60u);
    EXPECT_EQ(out.at<uint>(0, 0, 2), 120u);
    EXPECT_EQ(out.at<uint>(0, 1, 0), 80u);
    EXPECT_EQ(out.at<uint>(0, 1, 1), 150u);
    EXPECT_EQ(out.at<uint>(0, 1, 2), 240u);

}

TEST(ArithmeticDispatchInternalTest, mat_scalar_add_sub_u8_s8_u16_s16_support_scalar_only_with_saturation)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard(
        cpu::DispatchMode::ScalarOnly);

    Mat out;

    const Mat src8u = make_vec_mat<uchar>({250, 10, 5, 0}, CV_8UC1);
    add(src8u, Scalar(10.0), out);
    expect_vec_eq<uchar>(out, {255, 20, 15, 10});
    subtract(Scalar(3.0), src8u, out);
    expect_vec_eq<uchar>(out, {0, 0, 0, 3});

    out = Mat();
    const Mat src8s = make_vec_mat<schar>({120, -120, 50, -50}, CV_8SC1);
    add(src8s, Scalar(20.0), out);
    expect_vec_eq<schar>(out, {127, -100, 70, -30});
    subtract(Scalar(-100.0), src8s, out);
    expect_vec_eq<schar>(out, {-128, 20, -128, -50});

    out = Mat();
    Mat src16u({1, 2}, CV_16UC3);
    src16u.at<ushort>(0, 0, 0) = 65530;
    src16u.at<ushort>(0, 0, 1) = 10;
    src16u.at<ushort>(0, 0, 2) = 3;
    src16u.at<ushort>(0, 1, 0) = 100;
    src16u.at<ushort>(0, 1, 1) = 200;
    src16u.at<ushort>(0, 1, 2) = 300;
    add(src16u, Scalar(10.0, 65530.0, 40.0), out);
    EXPECT_EQ(out.at<ushort>(0, 0, 0), 65535);
    EXPECT_EQ(out.at<ushort>(0, 0, 1), 65535);
    EXPECT_EQ(out.at<ushort>(0, 0, 2), 43);
    EXPECT_EQ(out.at<ushort>(0, 1, 0), 110);
    EXPECT_EQ(out.at<ushort>(0, 1, 1), 65535);
    EXPECT_EQ(out.at<ushort>(0, 1, 2), 340);

    subtract(Scalar(50.0, 20.0, 100.0), src16u, out);
    EXPECT_EQ(out.at<ushort>(0, 0, 0), 0);
    EXPECT_EQ(out.at<ushort>(0, 0, 1), 10);
    EXPECT_EQ(out.at<ushort>(0, 0, 2), 97);
    EXPECT_EQ(out.at<ushort>(0, 1, 0), 0);
    EXPECT_EQ(out.at<ushort>(0, 1, 1), 0);
    EXPECT_EQ(out.at<ushort>(0, 1, 2), 0);

    out = Mat();
    Mat src16s({1, 2}, CV_16SC3);
    src16s.at<short>(0, 0, 0) = 32760;
    src16s.at<short>(0, 0, 1) = -32760;
    src16s.at<short>(0, 0, 2) = 1000;
    src16s.at<short>(0, 1, 0) = -1000;
    src16s.at<short>(0, 1, 1) = 2000;
    src16s.at<short>(0, 1, 2) = -2000;
    add(src16s, Scalar(100.0, -100.0, 30000.0), out);
    EXPECT_EQ(out.at<short>(0, 0, 0), 32767);
    EXPECT_EQ(out.at<short>(0, 0, 1), -32768);
    EXPECT_EQ(out.at<short>(0, 0, 2), 31000);
    EXPECT_EQ(out.at<short>(0, 1, 0), -900);
    EXPECT_EQ(out.at<short>(0, 1, 1), 1900);
    EXPECT_EQ(out.at<short>(0, 1, 2), 28000);

    subtract(Scalar(-32000.0, 32000.0, -32000.0), src16s, out);
    EXPECT_EQ(out.at<short>(0, 0, 0), -32768);
    EXPECT_EQ(out.at<short>(0, 0, 1), 32767);
    EXPECT_EQ(out.at<short>(0, 0, 2), -32768);
    EXPECT_EQ(out.at<short>(0, 1, 0), -31000);
    EXPECT_EQ(out.at<short>(0, 1, 1), 30000);
    EXPECT_EQ(out.at<short>(0, 1, 2), -30000);

}

TEST(ArithmeticDispatchInternalTest, mat_mat_and_mat_scalar_mul_u8_s8_u16_s16_support_scalar_only_with_saturation)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard(
        cpu::DispatchMode::ScalarOnly);

    Mat out;

    const Mat a8u = make_vec_mat<uchar>({20, 30, 200, 255}, CV_8UC1);
    const Mat b8u = make_vec_mat<uchar>({10, 9, 2, 2}, CV_8UC1);
    multiply(a8u, b8u, out);
    expect_vec_eq<uchar>(out, {200, 255, 255, 255});
    multiply(a8u, Scalar(3.0), out);
    expect_vec_eq<uchar>(out, {60, 90, 255, 255});

    out = Mat();
    const Mat a8s = make_vec_mat<schar>({50, -50, 100, -100}, CV_8SC1);
    const Mat b8s = make_vec_mat<schar>({3, 3, 2, 2}, CV_8SC1);
    multiply(a8s, b8s, out);
    expect_vec_eq<schar>(out, {127, -128, 127, -128});
    multiply(a8s, Scalar(-3.0), out);
    expect_vec_eq<schar>(out, {-128, 127, -128, 127});

    out = Mat();
    const Mat a16u = make_vec_mat<ushort>({1000, 40000, 60000, 65535}, CV_16UC1);
    const Mat b16u = make_vec_mat<ushort>({100, 2, 2, 2}, CV_16UC1);
    multiply(a16u, b16u, out);
    expect_vec_eq<ushort>(out, {65535, 65535, 65535, 65535});
    multiply(a16u, Scalar(2.0), out);
    expect_vec_eq<ushort>(out, {2000, 65535, 65535, 65535});

    out = Mat();
    const Mat a16s = make_vec_mat<short>({1000, -1000, 20000, -20000}, CV_16SC1);
    const Mat b16s = make_vec_mat<short>({40, 40, 2, 2}, CV_16SC1);
    multiply(a16s, b16s, out);
    expect_vec_eq<short>(out, {32767, -32768, 32767, -32768});
    multiply(a16s, Scalar(-3.0), out);
    expect_vec_eq<short>(out, {-3000, 3000, -32768, 32767});

}

TEST(ArithmeticDispatchInternalTest, mat_mat_integer_compare_supports_scalar_only_mode)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard(
        cpu::DispatchMode::ScalarOnly);

    Mat out;

    const Mat a8u = make_vec_mat<uchar>({1, 5, 9, 9}, CV_8UC1);
    const Mat b8u = make_vec_mat<uchar>({1, 3, 9, 10}, CV_8UC1);
    compare(a8u, b8u, out, CV_CMP_EQ);
    expect_vec_eq<uchar>(out, {255, 0, 255, 0});

    const Mat a16s = make_vec_mat<short>({-3, 4, 7, 7}, CV_16SC1);
    const Mat b16s = make_vec_mat<short>({-4, 4, 8, 6}, CV_16SC1);
    compare(a16s, b16s, out, CV_CMP_GT);
    expect_vec_eq<uchar>(out, {255, 0, 0, 255});

    const Mat a32u = make_vec_mat<uint>({3u, 4u, 5u, 6u}, CV_32UC1);
    const Mat b32u = make_vec_mat<uint>({3u, 7u, 4u, 6u}, CV_32UC1);
    compare(a32u, b32u, out, CV_CMP_LE);
    expect_vec_eq<uchar>(out, {255, 255, 0, 255});

}

TEST(ArithmeticDispatchInternalTest, mat_scalar_integer_compare_supports_scalar_only_mode)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard(
        cpu::DispatchMode::ScalarOnly);

    Mat out;

    const Mat src8s = make_vec_mat<schar>({-2, 0, 3, 7}, CV_8SC1);
    compare(src8s, Scalar(0.0), out, CV_CMP_GT);
    expect_vec_eq<uchar>(out, {0, 0, 255, 255});

    out = Mat();
    Mat src16u({1, 2}, CV_16UC3);
    src16u.at<ushort>(0, 0, 0) = 10;
    src16u.at<ushort>(0, 0, 1) = 20;
    src16u.at<ushort>(0, 0, 2) = 30;
    src16u.at<ushort>(0, 1, 0) = 40;
    src16u.at<ushort>(0, 1, 1) = 50;
    src16u.at<ushort>(0, 1, 2) = 60;
    compare(Scalar(10.0, 25.0, 30.0), src16u, out, CV_CMP_NE);
    EXPECT_EQ(out.at<uchar>(0, 0, 0), 0);
    EXPECT_EQ(out.at<uchar>(0, 0, 1), 255);
    EXPECT_EQ(out.at<uchar>(0, 0, 2), 0);
    EXPECT_EQ(out.at<uchar>(0, 1, 0), 255);
    EXPECT_EQ(out.at<uchar>(0, 1, 1), 255);
    EXPECT_EQ(out.at<uchar>(0, 1, 2), 255);

    out = Mat();
    const Mat src32s = make_vec_mat<int>({5, -1, 0, 9}, CV_32SC1);
    compare(src32s, Scalar(0.0), out, CV_CMP_GE);
    expect_vec_eq<uchar>(out, {255, 0, 255, 255});

}

TEST(ArithmeticDispatchInternalTest, mat_mat_add_fp16_supports_scalar_only_mode)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard(
        cpu::DispatchMode::ScalarOnly);

    const Mat a = make_vec_mat_from_doubles({1.0, -2.0, 3.5, 4.0}, CV_16FC1);
    const Mat b = make_vec_mat_from_doubles({0.5, 1.0, -1.5, 2.0}, CV_16FC1);

    Mat out;
    binaryFunc(BinaryOp::ADD, a, b, out);
    expect_vec_match_by_depth(out, {1.5, -1.0, 2.0, 6.0}, 1e-6, 2e-2);

}

TEST(ArithmeticDispatchInternalTest, mat_scalar_add_fp16_supports_scalar_only_mode)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard(
        cpu::DispatchMode::ScalarOnly);

    const Mat src = make_vec_mat_from_doubles({1.0, -2.0, 3.5, 4.0}, CV_16FC1);

    Mat out;
    add(src, Scalar(0.5), out);
    expect_vec_match_by_depth(out, {1.5, -1.5, 4.0, 4.5}, 1e-6, 2e-2);

}
