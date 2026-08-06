#include "cvh.h"
#include "gtest/gtest.h"
#include "opencv_contract_backend.h"
#include "../support/dispatch_mode_guard.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cvh;

namespace {

std::uint32_t lcg_next(std::uint32_t state)
{
    return state * 1664525u + 1013904223u;
}

void fill_u8(Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int rows = mat.size[0];
    const int scalars_per_row = mat.size[1] * mat.channels();
    for (int y = 0; y < rows; ++y)
    {
        unsigned char* row = mat.data + static_cast<std::size_t>(y) * mat.step(0);
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            row[x] = static_cast<unsigned char>((state >> 24) ^ static_cast<std::uint32_t>(x + y * 17));
        }
    }
}

void fill_f32(Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int rows = mat.size[0];
    const int scalars_per_row = mat.size[1] * mat.channels();
    for (int y = 0; y < rows; ++y)
    {
        float* row = reinterpret_cast<float*>(mat.data + static_cast<std::size_t>(y) * mat.step(0));
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            row[x] = static_cast<float>(static_cast<int>(state & 0xffffu) - 32768) / 4096.0f;
        }
    }
}

void fill_f64(Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int rows = mat.size[0];
    const int scalars_per_row = mat.size[1] * mat.channels();
    for (int y = 0; y < rows; ++y)
    {
        double* row =
            reinterpret_cast<double*>(mat.data + static_cast<std::size_t>(y) * mat.step(0));
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            row[x] = static_cast<double>(static_cast<int>(state & 0xffffu) - 32768) / 4096.0;
        }
    }
}

template<typename T>
void fill_integer(Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int rows = mat.size[0];
    const int scalars_per_row = mat.size[1] * mat.channels();
    for (int y = 0; y < rows; ++y)
    {
        T* row = reinterpret_cast<T*>(mat.data + static_cast<std::size_t>(y) * mat.step(0));
        for (int x = 0; x < scalars_per_row; ++x)
        {
            state = lcg_next(state);
            if constexpr (std::is_same<T, int>::value)
            {
                row[x] = static_cast<int>(state % 2000001u) - 1000000;
            }
            else
            {
                row[x] = static_cast<T>(state ^ (state >> 16));
            }
        }
    }
}

int cvh_depth(cvh_test_opencv_contract::CoreDepthId depth)
{
    using cvh_test_opencv_contract::CoreDepthId;
    switch (depth)
    {
        case CoreDepthId::U8: return CV_8U;
        case CoreDepthId::S8: return CV_8S;
        case CoreDepthId::U16: return CV_16U;
        case CoreDepthId::S16: return CV_16S;
        case CoreDepthId::S32: return CV_32S;
        case CoreDepthId::F32: return CV_32F;
        case CoreDepthId::F64: return CV_64F;
    }
    return -1;
}

void fill_core_depth(Mat& mat,
                     cvh_test_opencv_contract::CoreDepthId depth,
                     std::uint32_t seed)
{
    using cvh_test_opencv_contract::CoreDepthId;
    switch (depth)
    {
        case CoreDepthId::U8: fill_u8(mat, seed); return;
        case CoreDepthId::S8: fill_integer<signed char>(mat, seed); return;
        case CoreDepthId::U16: fill_integer<unsigned short>(mat, seed); return;
        case CoreDepthId::S16: fill_integer<short>(mat, seed); return;
        case CoreDepthId::S32: fill_integer<int>(mat, seed); return;
        case CoreDepthId::F32: fill_f32(mat, seed); return;
        case CoreDepthId::F64: fill_f64(mat, seed); return;
    }
}

void fill_core_depth_contiguous(
    Mat& mat,
    cvh_test_opencv_contract::CoreDepthId depth,
    std::uint32_t seed)
{
    Mat flat(
        {1, static_cast<int>(mat.total())},
        mat.type(),
        mat.data);
    fill_core_depth(flat, depth, seed);
}

void fill_mask(Mat& mask)
{
    for (int y = 0; y < mask.size[0]; ++y)
    {
        unsigned char* row =
            mask.data + static_cast<std::size_t>(y) * mask.step(0);
        for (int x = 0; x < mask.size[1]; ++x)
        {
            row[x] = ((x + 2 * y) % 3) != 0 ? 255 : 0;
        }
    }
}

cvh_test_opencv_contract::CoreReductionSummary collect_reduction_summary(
    const Mat& src,
    const Mat& mask)
{
    cvh_test_opencv_contract::CoreReductionSummary result{};
    const Scalar sums = sum(src);
    const Scalar means = mean(src, mask);
    Scalar mean_from_stddev;
    Scalar stddevs;
    meanStdDev(src, mean_from_stddev, stddevs, mask);
    for (int ch = 0; ch < src.channels(); ++ch)
    {
        result.sums[ch] = sums[ch];
        result.means[ch] = means[ch];
        result.stddevs[ch] = stddevs[ch];
    }
    result.norm_inf = norm(src, NORM_INF, mask);
    result.norm_l1 = norm(src, NORM_L1, mask);
    result.norm_l2 = norm(src, NORM_L2, mask);
    if (src.channels() == 1)
    {
        Point min_location;
        Point max_location;
        minMaxLoc(
            src,
            &result.min_value,
            &result.max_value,
            &min_location,
            &max_location,
            mask);
        result.count_non_zero = countNonZero(src);
        result.min_x = min_location.x;
        result.min_y = min_location.y;
        result.max_x = max_location.x;
        result.max_y = max_location.y;
    }
    return result;
}

void run_core_array_op(cvh_test_opencv_contract::CoreArrayOpId op,
                       const Mat& a,
                       const Mat& b,
                       Mat& dst)
{
    using cvh_test_opencv_contract::CoreArrayOpId;
    switch (op)
    {
        case CoreArrayOpId::AbsDiff: absdiff(a, b, dst); return;
        case CoreArrayOpId::BitwiseAnd: bitwise_and(a, b, dst); return;
        case CoreArrayOpId::BitwiseNot: bitwise_not(a, dst); return;
        case CoreArrayOpId::BitwiseOr: bitwise_or(a, b, dst); return;
        case CoreArrayOpId::BitwiseXor: bitwise_xor(a, b, dst); return;
        case CoreArrayOpId::InRange:
            inRange(a, Scalar::all(-2.5), Scalar::all(3.5), dst);
            return;
        case CoreArrayOpId::Min: min(a, b, dst); return;
        case CoreArrayOpId::Max: max(a, b, dst); return;
    }
}

void write_f32_bits(float& value, std::uint32_t bits)
{
    std::memcpy(&value, &bits, sizeof(bits));
}

void write_f64_bits(double& value, std::uint64_t bits)
{
    std::memcpy(&value, &bits, sizeof(bits));
}

void fill_math_input(Mat& mat,
                     cvh_test_opencv_contract::CoreMathOpId op,
                     std::uint32_t seed)
{
    using cvh_test_opencv_contract::CoreMathOpId;
    std::uint32_t state = seed;
    const int scalar_count = mat.size[1] * mat.channels();
    for (int y = 0; y < mat.size[0]; ++y)
    {
        for (int x = 0; x < scalar_count; ++x)
        {
            state = lcg_next(state);
            double value =
                static_cast<double>(static_cast<int>(state % 20001u) - 10000) / 2000.0;
            if (op == CoreMathOpId::Sqrt || op == CoreMathOpId::Log)
            {
                value = std::fabs(value) + 0.01;
            }
            if (mat.depth() == CV_32F)
            {
                reinterpret_cast<float*>(
                    mat.data + static_cast<std::size_t>(y) * mat.step(0))[x] =
                    static_cast<float>(value);
            }
            else
            {
                reinterpret_cast<double*>(
                    mat.data + static_cast<std::size_t>(y) * mat.step(0))[x] = value;
            }
        }
    }
}

void run_core_math_op(cvh_test_opencv_contract::CoreMathOpId op,
                      const Mat& src,
                      Mat& dst)
{
    using cvh_test_opencv_contract::CoreMathOpId;
    switch (op)
    {
        case CoreMathOpId::Sqrt: cvh::sqrt(src, dst); return;
        case CoreMathOpId::Pow: cvh::pow(src, 1.75, dst); return;
        case CoreMathOpId::Exp: cvh::exp(src, dst); return;
        case CoreMathOpId::Log: cvh::log(src, dst); return;
    }
}

void run_core_layout_op(cvh_test_opencv_contract::CoreLayoutOpId op,
                        const Mat& src,
                        cvh_test_opencv_contract::CoreDepthId depth,
                        std::uint32_t seed,
                        Mat& dst)
{
    using cvh_test_opencv_contract::CoreLayoutOpId;
    switch (op)
    {
        case CoreLayoutOpId::CopyMask:
        {
            Mat mask(src.shape(), CV_8UC1);
            fill_mask(mask);
            copyTo(src, dst, mask);
            return;
        }
        case CoreLayoutOpId::ExtractLastChannel:
            extractChannel(src, dst, src.channels() - 1);
            return;
        case CoreLayoutOpId::FlipHorizontal: flip(src, dst, 1); return;
        case CoreLayoutOpId::FlipVertical: flip(src, dst, 0); return;
        case CoreLayoutOpId::FlipBoth: flip(src, dst, -1); return;
        case CoreLayoutOpId::RotateClockwise:
            rotate(src, dst, ROTATE_90_CLOCKWISE);
            return;
        case CoreLayoutOpId::Rotate180:
            rotate(src, dst, ROTATE_180);
            return;
        case CoreLayoutOpId::RotateCounterclockwise:
            rotate(src, dst, ROTATE_90_COUNTERCLOCKWISE);
            return;
        case CoreLayoutOpId::Repeat2x3: repeat(src, 2, 3, dst); return;
        case CoreLayoutOpId::HConcat:
        {
            Mat other(src.shape(), src.type());
            fill_core_depth(other, depth, seed ^ 0x9e3779b9u);
            hconcat(src, other, dst);
            return;
        }
        case CoreLayoutOpId::VConcat:
        {
            Mat other(src.shape(), src.type());
            fill_core_depth(other, depth, seed ^ 0x9e3779b9u);
            vconcat(src, other, dst);
            return;
        }
    }
}

}  // namespace

TEST(OpenCVContractSmoke_TEST, core_u8_to_f64_matches_upstream)
{
    constexpr int rows = 5;
    constexpr int cols = 7;
    constexpr int channels = 3;
    constexpr std::uint32_t seed = 0x12345678u;

    Mat src({rows, cols}, CV_8UC3);
    fill_u8(src, seed);

    Mat dst;
    src.convertTo(dst, CV_64F);
    ASSERT_EQ(dst.type(), CV_64FC3);
    ASSERT_TRUE(dst.isContinuous());

    EXPECT_TRUE(cvh_test_opencv_contract::validate_core_convert_u8_to_f64(
        rows,
        cols,
        channels,
        seed,
        dst.data,
        dst.total() * dst.elemSize()));
}

TEST(OpenCVContractSmoke_TEST, imgproc_resize_linear_u8_matches_upstream)
{
    constexpr int src_rows = 7;
    constexpr int src_cols = 9;
    constexpr int dst_rows = 5;
    constexpr int dst_cols = 6;
    constexpr int channels = 1;
    constexpr std::uint32_t seed = 0x9e3779b9u;

    Mat src({src_rows, src_cols}, CV_8UC1);
    fill_u8(src, seed);

    Mat dst;
    resize(src, dst, Size(dst_cols, dst_rows), 0.0, 0.0, INTER_LINEAR);
    ASSERT_EQ(dst.type(), CV_8UC1);
    ASSERT_TRUE(dst.isContinuous());

    EXPECT_TRUE(cvh_test_opencv_contract::validate_imgproc_resize_linear_u8(
        src_rows,
        src_cols,
        dst_rows,
        dst_cols,
        channels,
        seed,
        dst.data,
        dst.total() * dst.elemSize()));
}

TEST(OpenCVContractSmoke_TEST, v01_neon_hot_paths_match_upstream)
{
    using cvh_test_opencv_contract::ImgprocHotColorOpId;
    constexpr int rows = 37;
    constexpr int cols = 67;
    constexpr std::uint32_t seed = 0x51a7c39du;
    const auto expect_direct_neon_on_arm = []() {
        if (cpu::neon_runtime_available())
        {
            EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::NEON);
        }
        else
        {
            EXPECT_NE(cpu::last_dispatch_tag(), cpu::DispatchTag::NEON);
        }
    };

    struct ColorCase
    {
        ImgprocHotColorOpId op;
        int source_type;
        int code;
    };
    for (const ColorCase color :
         {ColorCase{ImgprocHotColorOpId::BgrToRgb, CV_8UC3, COLOR_BGR2RGB},
          ColorCase{ImgprocHotColorOpId::BgrToBgra, CV_8UC3, COLOR_BGR2BGRA},
          ColorCase{ImgprocHotColorOpId::BgraToGray, CV_8UC4, COLOR_BGRA2GRAY},
          ColorCase{ImgprocHotColorOpId::BgrToYuv, CV_8UC3, COLOR_BGR2YUV},
          ColorCase{ImgprocHotColorOpId::YuvToBgr, CV_8UC3, COLOR_YUV2BGR}})
    {
        Mat source({rows, cols}, color.source_type);
        fill_u8(source, seed);
        Mat actual;
        cpu::reset_last_dispatch_tag();
        cvtColor(source, actual, color.code);
        expect_direct_neon_on_arm();
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_hot_cvtcolor_u8(
                color.op,
                rows,
                cols,
                seed,
                actual.data,
                actual.total() * actual.elemSize()));
    }

    Mat resize_source({64, 96}, CV_8UC3);
    fill_u8(resize_source, seed);
    Mat resized;
    cpu::reset_last_dispatch_tag();
    resize(
        resize_source, resized, Size(72, 48),
        0.0, 0.0, INTER_LINEAR);
    expect_direct_neon_on_arm();
    EXPECT_TRUE(cvh_test_opencv_contract::validate_imgproc_resize_linear_u8(
        64, 96, 48, 72, 3, seed,
        resized.data, resized.total() * resized.elemSize()));

    for (const int channels : {1, 3, 4})
    {
        Mat source({rows, cols}, CV_MAKETYPE(CV_8U, channels));
        fill_u8(source, seed);
        for (const auto order :
             {std::pair<int, int>{1, 0}, std::pair<int, int>{0, 1}})
        {
            Mat actual;
            cpu::reset_last_dispatch_tag();
            Sobel(
                source, actual, CV_16S,
                order.first, order.second,
                3, 1.0, 0.0, BORDER_REPLICATE);
            expect_direct_neon_on_arm();
            EXPECT_TRUE(cvh_test_opencv_contract::validate_imgproc_sobel3_u8(
                rows, cols, channels, seed,
                order.first, order.second,
                actual.data, actual.total() * actual.elemSize()));
        }
    }

    Mat derivative_source({rows, cols}, CV_8UC1);
    fill_u8(derivative_source, seed);
    Mat scharr;
    Mat laplacian;
    cpu::reset_last_dispatch_tag();
    Scharr(derivative_source, scharr, CV_16S, 1, 0);
    expect_direct_neon_on_arm();
    Laplacian(derivative_source, laplacian, CV_16S, 3);
    EXPECT_TRUE(
        cvh_test_opencv_contract::validate_imgproc_derivative_filters_u8(
            rows, cols, 1, seed,
            scharr.data, scharr.total() * scharr.elemSize(),
            laplacian.data, laplacian.total() * laplacian.elemSize()));

    Mat gradient_x;
    Mat gradient_y;
    cpu::reset_last_dispatch_tag();
    spatialGradient(derivative_source, gradient_x, gradient_y);
    expect_direct_neon_on_arm();
    EXPECT_TRUE(
        cvh_test_opencv_contract::validate_imgproc_spatial_gradient_u8(
            rows, cols, seed,
            gradient_x.data, gradient_x.total() * gradient_x.elemSize(),
            gradient_y.data, gradient_y.total() * gradient_y.elemSize()));
}

TEST(OpenCVContractSmoke_TEST, core_array_ops_match_upstream_for_standard_depths)
{
    using cvh_test_opencv_contract::CoreArrayOpId;
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr int rows = 5;
    constexpr int cols = 7;
    constexpr int channels = 3;
    constexpr std::uint32_t seed_a = 0x10203040u;
    constexpr std::uint32_t seed_b = 0x55667788u;
    const CoreArrayOpId ops[] = {
        CoreArrayOpId::AbsDiff,
        CoreArrayOpId::BitwiseAnd,
        CoreArrayOpId::BitwiseNot,
        CoreArrayOpId::BitwiseOr,
        CoreArrayOpId::BitwiseXor,
        CoreArrayOpId::InRange,
        CoreArrayOpId::Min,
        CoreArrayOpId::Max,
    };

    for (const CoreDepthId depth :
         {CoreDepthId::U8,
          CoreDepthId::S8,
          CoreDepthId::U16,
          CoreDepthId::S16,
          CoreDepthId::S32,
          CoreDepthId::F32})
    {
        const int type = CV_MAKETYPE(cvh_depth(depth), channels);
        Mat a({rows, cols}, type);
        Mat b({rows, cols}, type);
        fill_core_depth(a, depth, seed_a);
        fill_core_depth(b, depth, seed_b);

        for (const CoreArrayOpId op : ops)
        {
            SCOPED_TRACE(static_cast<int>(depth));
            SCOPED_TRACE(static_cast<int>(op));
            Mat dst;
            run_core_array_op(op, a, b, dst);
            EXPECT_TRUE(cvh_test_opencv_contract::validate_core_array_op(
                op,
                depth,
                rows,
                cols,
                channels,
                seed_a,
                seed_b,
                dst.data,
                dst.total() * dst.elemSize()));
        }
    }
}

TEST(OpenCVContractSmoke_TEST, core_float_numeric_edges_match_upstream_bits)
{
    using cvh_test_opencv_contract::CoreArrayOpId;
    Mat a({1, 5}, CV_32FC1);
    Mat b({1, 5}, CV_32FC1);
    const std::uint32_t a_bits[] = {
        0x7fc12345u, 0x3f800000u, 0x7f800000u, 0x80000000u, 0x00000000u,
    };
    const std::uint32_t b_bits[] = {
        0x40000000u, 0x7fc54321u, 0x7f800000u, 0x00000000u, 0x80000000u,
    };
    for (int x = 0; x < 5; ++x)
    {
        write_f32_bits(a.at<float>(0, x), a_bits[x]);
        write_f32_bits(b.at<float>(0, x), b_bits[x]);
    }

    for (const CoreArrayOpId op :
         {CoreArrayOpId::AbsDiff, CoreArrayOpId::Min, CoreArrayOpId::Max})
    {
        SCOPED_TRACE(static_cast<int>(op));
        Mat dst;
        run_core_array_op(op, a, b, dst);
        EXPECT_TRUE(cvh_test_opencv_contract::validate_core_float_edge_op(
            op, dst.data, dst.total() * dst.elemSize()));
    }
}

TEST(OpenCVContractSmoke_TEST, core_double_numeric_edges_match_upstream_bits)
{
    using cvh_test_opencv_contract::CoreArrayOpId;
    Mat a({1, 5}, CV_64FC1);
    Mat b({1, 5}, CV_64FC1);
    const std::uint64_t a_bits[] = {
        0x7ff8123456789abcULL,
        0x3ff0000000000000ULL,
        0x7ff0000000000000ULL,
        0x8000000000000000ULL,
        0x0000000000000000ULL,
    };
    const std::uint64_t b_bits[] = {
        0x4000000000000000ULL,
        0x7ff854321abcdef0ULL,
        0x7ff0000000000000ULL,
        0x0000000000000000ULL,
        0x8000000000000000ULL,
    };
    for (int x = 0; x < 5; ++x)
    {
        write_f64_bits(a.at<double>(0, x), a_bits[x]);
        write_f64_bits(b.at<double>(0, x), b_bits[x]);
    }

    for (const CoreArrayOpId op :
         {CoreArrayOpId::AbsDiff, CoreArrayOpId::Min, CoreArrayOpId::Max})
    {
        SCOPED_TRACE(static_cast<int>(op));
        Mat dst;
        run_core_array_op(op, a, b, dst);
        EXPECT_TRUE(cvh_test_opencv_contract::validate_core_double_edge_op(
            op, dst.data, dst.total() * dst.elemSize()));
    }
}

TEST(OpenCVContractSmoke_TEST, core_convert_scale_abs_and_fp16_match_upstream_bits)
{
    Mat scale_src({1, 9}, CV_32FC1);
    const float scale_values[] = {
        -300.0f, -2.5f, -1.5f, -0.5f, 0.5f, 1.5f, 2.5f, 254.5f, 300.0f,
    };
    std::memcpy(scale_src.data, scale_values, sizeof(scale_values));
    Mat scale_dst;
    convertScaleAbs(scale_src, scale_dst);
    EXPECT_TRUE(cvh_test_opencv_contract::validate_convert_scale_abs_edges(
        scale_dst.data, scale_dst.total() * scale_dst.elemSize()));

    Mat fp32({1, 11}, CV_32FC1);
    const float denorm = std::ldexp(1.0f, -24);
    const float fp32_values[] = {
        0.0f,
        -0.0f,
        1.0f,
        -2.0f,
        65504.0f,
        std::ldexp(1.0f, -14),
        denorm,
        denorm * 0.25f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN(),
    };
    std::memcpy(fp32.data, fp32_values, sizeof(fp32_values));
    Mat fp16;
    convertFp16(fp32, fp16);
    EXPECT_TRUE(cvh_test_opencv_contract::validate_convert_fp16_edges(
        fp16.data, fp16.total() * fp16.elemSize()));
}

TEST(OpenCVContractSmoke_TEST, core_math_functions_match_upstream_tolerance)
{
    using cvh_test_opencv_contract::CoreDepthId;
    using cvh_test_opencv_contract::CoreMathOpId;
    constexpr int rows = 3;
    constexpr int cols = 7;
    constexpr int channels = 3;
    constexpr std::uint32_t seed = 0xa17c93e5u;

    for (const CoreDepthId depth : {CoreDepthId::F32, CoreDepthId::F64})
    {
        const int type = CV_MAKETYPE(cvh_depth(depth), channels);
        for (const CoreMathOpId op :
             {CoreMathOpId::Sqrt, CoreMathOpId::Pow, CoreMathOpId::Exp, CoreMathOpId::Log})
        {
            SCOPED_TRACE(static_cast<int>(depth));
            SCOPED_TRACE(static_cast<int>(op));
            Mat src({rows, cols}, type);
            fill_math_input(src, op, seed);
            Mat dst;
            run_core_math_op(op, src, dst);
            EXPECT_TRUE(cvh_test_opencv_contract::validate_core_math_op(
                op,
                depth,
                rows,
                cols,
                channels,
                seed,
                dst.data,
                dst.total() * dst.elemSize()));
        }
    }
}

TEST(OpenCVContractSmoke_TEST, core_reduction_summaries_match_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr int rows = 7;
    constexpr int cols = 11;
    constexpr std::uint32_t seed = 0x4f13c2a9u;

    for (const CoreDepthId depth :
         {CoreDepthId::U8,
          CoreDepthId::S16,
          CoreDepthId::S32,
          CoreDepthId::F32,
          CoreDepthId::F64})
    {
        SCOPED_TRACE(static_cast<int>(depth));
        Mat src({rows, cols}, CV_MAKETYPE(cvh_depth(depth), 1));
        fill_core_depth(src, depth, seed);
        Mat mask({rows, cols}, CV_8UC1);
        fill_mask(mask);
        const auto actual = collect_reduction_summary(src, mask);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_core_reduction_summary(
                depth, rows, cols, 1, seed, true, actual));

        std::vector<Point> points;
        findNonZero(src, points);
        std::vector<int> xy(points.size() * 2);
        for (std::size_t i = 0; i < points.size(); ++i)
        {
            xy[2 * i] = points[i].x;
            xy[2 * i + 1] = points[i].y;
        }
        EXPECT_TRUE(cvh_test_opencv_contract::validate_core_nonzero_locations(
            depth,
            rows,
            cols,
            seed,
            xy.empty() ? nullptr : xy.data(),
            points.size()));
    }

    for (const auto depth_and_channels :
         {std::pair<CoreDepthId, int>(CoreDepthId::U8, 3),
          std::pair<CoreDepthId, int>(CoreDepthId::F64, 4)})
    {
        const CoreDepthId depth = depth_and_channels.first;
        const int channels = depth_and_channels.second;
        SCOPED_TRACE(channels);
        Mat src(
            {rows, cols}, CV_MAKETYPE(cvh_depth(depth), channels));
        fill_core_depth(src, depth, seed);
        Mat mask({rows, cols}, CV_8UC1);
        fill_mask(mask);
        const auto actual = collect_reduction_summary(src, mask);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_core_reduction_summary(
                depth, rows, cols, channels, seed, true, actual));
    }
}

TEST(OpenCVContractSmoke_TEST, core_reduce_and_normalize_match_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr int rows = 7;
    constexpr int cols = 11;
    constexpr int channels = 3;
    constexpr std::uint32_t seed = 0x6ac19e35u;

    for (const CoreDepthId depth :
         {CoreDepthId::U8,
          CoreDepthId::S16,
          CoreDepthId::S32,
          CoreDepthId::F32,
          CoreDepthId::F64})
    {
        SCOPED_TRACE(static_cast<int>(depth));
        Mat src(
            {rows, cols}, CV_MAKETYPE(cvh_depth(depth), channels));
        fill_core_depth(src, depth, seed);
        if (depth != CoreDepthId::S32)
        {
            for (const int dim : {0, 1})
            {
                for (const int reduce_type :
                     {REDUCE_SUM, REDUCE_AVG, REDUCE_SUM2})
                {
                    SCOPED_TRACE(dim);
                    SCOPED_TRACE(reduce_type);
                    Mat reduced;
                    reduce(src, reduced, dim, reduce_type, CV_64F);
                    EXPECT_TRUE(
                        cvh_test_opencv_contract::validate_core_reduce_f64(
                            depth,
                            rows,
                            cols,
                            channels,
                            seed,
                            dim,
                            reduce_type,
                            reduced.data,
                            reduced.total() * reduced.elemSize()));
                }
            }
        }

        Mat mask({rows, cols}, CV_8UC1);
        fill_mask(mask);
        Mat normalized;
        normalize(src, normalized, 2.0, 0.0, NORM_L2, CV_64F, mask);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_core_normalize_l2_f64(
                depth,
                rows,
                cols,
                channels,
                seed,
                true,
                normalized.data,
                normalized.total() * normalized.elemSize()));
    }
}

TEST(OpenCVContractSmoke_TEST, core_reduce_arg_matches_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr int rows = 9;
    constexpr int cols = 13;
    constexpr std::uint32_t seed = 0xe31b2475u;

    for (const CoreDepthId depth :
         {CoreDepthId::U8,
          CoreDepthId::S16,
          CoreDepthId::S32,
          CoreDepthId::F32,
          CoreDepthId::F64})
    {
        Mat src({rows, cols}, CV_MAKETYPE(cvh_depth(depth), 1));
        fill_core_depth(src, depth, seed);
        for (const int axis : {0, 1})
        {
            for (const bool find_max : {false, true})
            {
                for (const bool last_index : {false, true})
                {
                    SCOPED_TRACE(static_cast<int>(depth));
                    SCOPED_TRACE(axis);
                    SCOPED_TRACE(find_max);
                    SCOPED_TRACE(last_index);
                    Mat actual;
                    if (find_max)
                    {
                        reduceArgMax(src, actual, axis, last_index);
                    }
                    else
                    {
                        reduceArgMin(src, actual, axis, last_index);
                    }
                    EXPECT_TRUE(
                        cvh_test_opencv_contract::validate_core_reduce_arg(
                            depth,
                            rows,
                            cols,
                            seed,
                            axis,
                            find_max,
                            last_index,
                            actual.data,
                            actual.total() * actual.elemSize()));
                }
            }
        }
    }
}

TEST(OpenCVContractSmoke_TEST, core_border_interpolate_matches_upstream)
{
    for (const int border_type :
         {BORDER_CONSTANT,
          BORDER_REPLICATE,
          BORDER_REFLECT,
          BORDER_WRAP,
          BORDER_REFLECT_101})
    {
        for (const int coordinate : {-13, -1, 0, 4, 5, 17})
        {
            SCOPED_TRACE(border_type);
            SCOPED_TRACE(coordinate);
            const int actual =
                borderInterpolate(coordinate, 5, border_type);
            EXPECT_TRUE(
                cvh_test_opencv_contract::validate_core_border_interpolate(
                    coordinate, 5, border_type, actual));
        }
    }
}

TEST(OpenCVContractSmoke_TEST, core_layout_ops_match_upstream_bytes)
{
    using cvh_test_opencv_contract::CoreDepthId;
    using cvh_test_opencv_contract::CoreLayoutOpId;
    constexpr int rows = 5;
    constexpr int cols = 7;
    constexpr std::uint32_t seed = 0x38a7d12fu;
    const CoreLayoutOpId operations[] = {
        CoreLayoutOpId::CopyMask,
        CoreLayoutOpId::ExtractLastChannel,
        CoreLayoutOpId::FlipHorizontal,
        CoreLayoutOpId::FlipVertical,
        CoreLayoutOpId::FlipBoth,
        CoreLayoutOpId::RotateClockwise,
        CoreLayoutOpId::Rotate180,
        CoreLayoutOpId::RotateCounterclockwise,
        CoreLayoutOpId::Repeat2x3,
        CoreLayoutOpId::HConcat,
        CoreLayoutOpId::VConcat,
    };
    for (const CoreDepthId depth :
         {CoreDepthId::U8,
          CoreDepthId::S16,
          CoreDepthId::S32,
          CoreDepthId::F32,
          CoreDepthId::F64})
    {
        for (const int channels : {1, 3, 4})
        {
            Mat src(
                {rows, cols}, CV_MAKETYPE(cvh_depth(depth), channels));
            fill_core_depth(src, depth, seed);
            for (const CoreLayoutOpId op : operations)
            {
                SCOPED_TRACE(static_cast<int>(depth));
                SCOPED_TRACE(channels);
                SCOPED_TRACE(static_cast<int>(op));
                Mat actual;
                run_core_layout_op(op, src, depth, seed, actual);
                EXPECT_TRUE(
                    cvh_test_opencv_contract::validate_core_layout_op(
                        op,
                        depth,
                        rows,
                        cols,
                        channels,
                        seed,
                        actual.data,
                        actual.total() * actual.elemSize()));
            }
        }
    }
}

TEST(OpenCVContractSmoke_TEST, core_mix_flip_nd_and_broadcast_match_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr int rows = 5;
    constexpr int cols = 7;
    constexpr std::uint32_t seed = 0x91c6e24bu;

    for (const CoreDepthId depth :
         {CoreDepthId::U8, CoreDepthId::F64})
    {
        Mat src({rows, cols}, CV_MAKETYPE(cvh_depth(depth), 4));
        fill_core_depth(src, depth, seed);
        Mat bgr({rows, cols}, CV_MAKETYPE(cvh_depth(depth), 3));
        Mat alpha({rows, cols}, CV_MAKETYPE(cvh_depth(depth), 1));
        Mat outputs[] = {bgr, alpha};
        const int routes[] = {0, 2, 1, 1, 2, 0, 3, 3};
        mixChannels(&src, 1, outputs, 2, routes, 4);
        EXPECT_TRUE(cvh_test_opencv_contract::validate_core_mix_channels(
            depth,
            rows,
            cols,
            seed,
            outputs[0].data,
            outputs[0].total() * outputs[0].elemSize(),
            outputs[1].data,
            outputs[1].total() * outputs[1].elemSize()));
    }

    for (const CoreDepthId depth :
         {CoreDepthId::U8,
          CoreDepthId::S16,
          CoreDepthId::S32,
          CoreDepthId::F32,
          CoreDepthId::F64})
    {
        Mat nd_src({2, 3, 4}, CV_MAKETYPE(cvh_depth(depth), 1));
        fill_core_depth_contiguous(nd_src, depth, seed);
        for (const int axis : {0, -1})
        {
            Mat flipped;
            flipND(nd_src, flipped, axis);
            EXPECT_TRUE(
                cvh_test_opencv_contract::validate_core_flip_nd(
                    depth,
                    seed,
                    axis,
                    flipped.data,
                    flipped.total() * flipped.elemSize()));
        }

        Mat broadcast_src(
            {2, 1, 3}, CV_MAKETYPE(cvh_depth(depth), 1));
        fill_core_depth_contiguous(broadcast_src, depth, seed);
        Mat broadcast_dst;
        broadcast(
            broadcast_src,
            std::vector<int>({4, 2, 5, 3}),
            broadcast_dst);
        EXPECT_TRUE(cvh_test_opencv_contract::validate_core_broadcast(
            depth,
            seed,
            broadcast_dst.data,
            broadcast_dst.total() * broadcast_dst.elemSize()));
    }
}

TEST(OpenCVContractSmoke_TEST, imgproc_phase1_kernels_match_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    for (const int shape : {MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE})
    {
        Mat kernel =
            getStructuringElement(shape, Size(7, 5), Point(2, 1));
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_structuring_element(
                shape,
                7,
                5,
                2,
                1,
                kernel.data,
                kernel.total() * kernel.elemSize()));
    }

    for (const CoreDepthId depth :
         {CoreDepthId::F32, CoreDepthId::F64})
    {
        const int type =
            depth == CoreDepthId::F32 ? CV_32F : CV_64F;
        for (const double sigma : {0.0, 1.7})
        {
            Mat gaussian = getGaussianKernel(7, sigma, type);
            EXPECT_TRUE(
                cvh_test_opencv_contract::validate_imgproc_gaussian_kernel(
                    7,
                    sigma,
                    depth,
                    gaussian.data,
                    gaussian.total() * gaussian.elemSize()));
        }

        Mat kx;
        Mat ky;
        getDerivKernels(kx, ky, 1, 0, 5, true, type);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_deriv_kernels(
                1,
                0,
                5,
                true,
                depth,
                kx.data,
                kx.total() * kx.elemSize(),
                ky.data,
                ky.total() * ky.elemSize()));

        Mat gabor = getGaborKernel(
            Size(7, 5), 2.0, 0.3, 4.0, 0.8, 0.0, type);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_gabor_kernel(
                7,
                5,
                depth,
                gabor.data,
                gabor.total() * gabor.elemSize()));

        Mat hanning;
        createHanningWindow(hanning, Size(7, 5), type);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_hanning_window(
                7,
                5,
                depth,
                hanning.data,
                hanning.total() * hanning.elemSize()));
    }
}

TEST(OpenCVContractSmoke_TEST, imgproc_gaussian5x5_u8_matches_upstream_bits)
{
    constexpr std::uint32_t seed = 0x5A5511u;
    for (const int channels : {1, 3, 4})
    {
        for (const int border_type :
             {BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,
              BORDER_REFLECT_101})
        {
            for (const cpu::DispatchMode mode :
                 {cpu::DispatchMode::ScalarOnly,
                  cpu::DispatchMode::OpenCVUIOnly})
            {
                SCOPED_TRACE(
                    std::string("channels=") + std::to_string(channels) +
                    ", border=" + std::to_string(border_type) +
                    ", mode=" + std::to_string(static_cast<int>(mode)));
                Mat src({17, 23}, CV_MAKETYPE(CV_8U, channels));
                fill_u8(src, seed + static_cast<std::uint32_t>(channels));
                Mat dst;
                {
                    cvh::test::DispatchModeGuard guard(mode);
                    GaussianBlur(
                        src,
                        dst,
                        Size(5, 5),
                        0.0,
                        0.0,
                        border_type);
                    EXPECT_EQ(
                        cpu::last_dispatch_tag(),
                        mode == cpu::DispatchMode::ScalarOnly
                            ? cpu::DispatchTag::Scalar
                            : cvh::test::expected_fixed_width_dispatch_tag());
                }
                EXPECT_TRUE(
                    cvh_test_opencv_contract::validate_imgproc_gaussian_blur_u8(
                        17,
                        23,
                        channels,
                        border_type,
                        seed + static_cast<std::uint32_t>(channels),
                        dst.data,
                        dst.total() * dst.elemSize()));
            }
        }
    }
}

TEST(OpenCVContractSmoke_TEST, imgproc_gaussian5x5_f32_matches_upstream)
{
    constexpr std::uint32_t seed = 0x5A5522u;
    for (const int channels : {1, 3, 4})
    {
        for (const int border_type :
             {BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,
              BORDER_REFLECT_101})
        {
            for (const cpu::DispatchMode mode :
                 {cpu::DispatchMode::ScalarOnly,
                  cpu::DispatchMode::OpenCVUIOnly})
            {
                SCOPED_TRACE(
                    std::string("channels=") + std::to_string(channels) +
                    ", border=" + std::to_string(border_type) +
                    ", mode=" + std::to_string(static_cast<int>(mode)));
                Mat src({17, 23}, CV_MAKETYPE(CV_32F, channels));
                fill_f32(src, seed + static_cast<std::uint32_t>(channels));
                Mat dst;
                {
                    cvh::test::DispatchModeGuard guard(mode);
                    GaussianBlur(
                        src,
                        dst,
                        Size(5, 5),
                        0.0,
                        0.0,
                        border_type);
                    EXPECT_EQ(
                        cpu::last_dispatch_tag(),
                        mode == cpu::DispatchMode::ScalarOnly
                            ? cpu::DispatchTag::Scalar
                            : cvh::test::expected_fixed_width_dispatch_tag());
                }
                EXPECT_TRUE(
                    cvh_test_opencv_contract::validate_imgproc_gaussian_blur_f32(
                        17,
                        23,
                        channels,
                        border_type,
                        seed + static_cast<std::uint32_t>(channels),
                        dst.data,
                        dst.total() * dst.elemSize()));
            }
        }
    }
}

TEST(OpenCVContractSmoke_TEST, imgproc_canny_matches_upstream_bits)
{
    struct Case
    {
        int rows;
        int cols;
        double threshold1;
        double threshold2;
        int aperture;
        bool l2;
    };
    constexpr std::uint32_t seed = 0xCA7711u;
    const Case cases[] = {
        {31, 37, 50.0, 130.0, 3, false},
        {31, 37, 40.0, 110.0, 3, true},
        {31, 37, 120.0, 340.0, 5, false},
        {3, 7, 130.0, 50.0, 3, false},
        {2, 9, 0.0, 1.0, 3, true},
    };
    for (const Case& current : cases)
    {
        for (const cpu::DispatchMode mode :
             {cpu::DispatchMode::ScalarOnly,
              cpu::DispatchMode::OpenCVUIOnly})
        {
            SCOPED_TRACE(
                std::string("aperture=") +
                    std::to_string(current.aperture) +
                ", l2=" + std::to_string(current.l2) +
                ", mode=" + std::to_string(static_cast<int>(mode)));
            Mat src({current.rows, current.cols}, CV_8UC1);
            fill_u8(src, seed + static_cast<std::uint32_t>(current.aperture));
            Mat dst;
            {
                cvh::test::DispatchModeGuard guard(mode);
                Canny(
                    src,
                    dst,
                    current.threshold1,
                    current.threshold2,
                    current.aperture,
                    current.l2);
            }
            EXPECT_TRUE(cvh_test_opencv_contract::validate_imgproc_canny(
                current.rows,
                current.cols,
                current.threshold1,
                current.threshold2,
                current.aperture,
                current.l2,
                seed + static_cast<std::uint32_t>(current.aperture),
                dst.data,
                dst.total() * dst.elemSize()));
        }
    }
}

TEST(OpenCVContractSmoke_TEST, imgproc_box3x3_matches_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr std::uint32_t seed = 0xB03311u;
    for (const CoreDepthId depth : {CoreDepthId::U8, CoreDepthId::F32})
    {
        const int cvh_type =
            depth == CoreDepthId::U8 ? CV_8U : CV_32F;
        for (const int channels : {1, 3, 4})
        {
            for (const int border_type :
                 {BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,
                  BORDER_REFLECT_101})
            {
                for (const cpu::DispatchMode mode :
                     {cpu::DispatchMode::ScalarOnly,
                      cpu::DispatchMode::OpenCVUIOnly})
                {
                    SCOPED_TRACE(
                        std::string("depth=") +
                        std::to_string(static_cast<int>(depth)) +
                        ", channels=" + std::to_string(channels) +
                        ", border=" + std::to_string(border_type) +
                        ", mode=" +
                        std::to_string(static_cast<int>(mode)));
                    Mat src(
                        {17, 23},
                        CV_MAKETYPE(cvh_type, channels));
                    if (depth == CoreDepthId::U8)
                    {
                        fill_u8(src, seed + channels);
                    }
                    else
                    {
                        fill_f32(src, seed + channels);
                    }
                    Mat dst;
                    {
                        cvh::test::DispatchModeGuard guard(mode);
                        boxFilter(
                            src,
                            dst,
                            -1,
                            Size(3, 3),
                            Point(-1, -1),
                            true,
                            border_type);
                        EXPECT_EQ(
                            cpu::last_dispatch_tag(),
                            mode == cpu::DispatchMode::ScalarOnly
                                ? cpu::DispatchTag::Scalar
                                : cvh::test::expected_fixed_width_dispatch_tag());
                    }
                    EXPECT_TRUE(
                        cvh_test_opencv_contract::validate_imgproc_box_filter(
                            17,
                            23,
                            channels,
                            depth,
                            border_type,
                            seed + channels,
                            dst.data,
                            dst.total() * dst.elemSize()));
                }
            }
        }
    }
}

TEST(OpenCVContractSmoke_TEST, imgproc_sep_filter3_matches_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr std::uint32_t seed = 0x5E9311u;
    Mat kernel({3, 1}, CV_32FC1);
    float* values = reinterpret_cast<float*>(kernel.data);
    values[0] = 0.25f;
    values[1] = 0.5f;
    values[2] = 0.25f;
    for (const CoreDepthId depth : {CoreDepthId::U8, CoreDepthId::F32})
    {
        const int cvh_type =
            depth == CoreDepthId::U8 ? CV_8U : CV_32F;
        for (const int channels : {1, 3, 4})
        {
            for (const int border_type :
                 {BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,
                  BORDER_REFLECT_101})
            {
                for (const cpu::DispatchMode mode :
                     {cpu::DispatchMode::ScalarOnly,
                      cpu::DispatchMode::OpenCVUIOnly})
                {
                    SCOPED_TRACE(
                        std::string("depth=") +
                        std::to_string(static_cast<int>(depth)) +
                        ", channels=" + std::to_string(channels) +
                        ", border=" + std::to_string(border_type) +
                        ", mode=" +
                        std::to_string(static_cast<int>(mode)));
                    Mat src(
                        {17, 23},
                        CV_MAKETYPE(cvh_type, channels));
                    if (depth == CoreDepthId::U8)
                    {
                        fill_u8(src, seed + channels);
                    }
                    else
                    {
                        fill_f32(src, seed + channels);
                    }
                    Mat dst;
                    {
                        cvh::test::DispatchModeGuard guard(mode);
                        sepFilter2D(
                            src,
                            dst,
                            -1,
                            kernel,
                            kernel,
                            Point(-1, -1),
                            0.0,
                            border_type);
                        EXPECT_EQ(
                            cpu::last_dispatch_tag(),
                            mode == cpu::DispatchMode::ScalarOnly
                                ? cpu::DispatchTag::Scalar
                                : cvh::test::expected_fixed_width_dispatch_tag());
                    }
                    EXPECT_TRUE(
                        cvh_test_opencv_contract::validate_imgproc_sep_filter3(
                            17,
                            23,
                            channels,
                            depth,
                            border_type,
                            seed + channels,
                            dst.data,
                            dst.total() * dst.elemSize()));
                }
            }
        }
    }
}

TEST(OpenCVContractSmoke_TEST, imgproc_filter2d_cross3_matches_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr std::uint32_t seed = 0xF11311u;
    Mat kernel({3, 3}, CV_32FC1);
    const float kernel_values[] = {
        0.0f, 0.25f, 0.0f,
        0.25f, 0.0f, 0.25f,
        0.0f, 0.25f, 0.0f};
    std::copy(
        kernel_values,
        kernel_values + 9,
        reinterpret_cast<float*>(kernel.data));
    for (const CoreDepthId depth : {CoreDepthId::U8, CoreDepthId::F32})
    {
        const int cvh_type =
            depth == CoreDepthId::U8 ? CV_8U : CV_32F;
        for (const int channels : {1, 3, 4})
        {
            for (const int border_type :
                 {BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT,
                  BORDER_REFLECT_101})
            {
                for (const cpu::DispatchMode mode :
                     {cpu::DispatchMode::ScalarOnly,
                      cpu::DispatchMode::OpenCVUIOnly})
                {
                    SCOPED_TRACE(
                        std::string("depth=") +
                        std::to_string(static_cast<int>(depth)) +
                        ", channels=" + std::to_string(channels) +
                        ", border=" + std::to_string(border_type) +
                        ", mode=" +
                        std::to_string(static_cast<int>(mode)));
                    Mat src(
                        {17, 23},
                        CV_MAKETYPE(cvh_type, channels));
                    if (depth == CoreDepthId::U8)
                    {
                        fill_u8(src, seed + channels);
                    }
                    else
                    {
                        fill_f32(src, seed + channels);
                    }
                    Mat dst;
                    {
                        cvh::test::DispatchModeGuard guard(mode);
                        filter2D(
                            src,
                            dst,
                            -1,
                            kernel,
                            Point(-1, -1),
                            0.0,
                            border_type);
                        EXPECT_EQ(
                            cpu::last_dispatch_tag(),
                            mode == cpu::DispatchMode::ScalarOnly
                                ? cpu::DispatchTag::Scalar
                                : cvh::test::expected_fixed_width_dispatch_tag());
                    }
                    EXPECT_TRUE(
                        cvh_test_opencv_contract::validate_imgproc_filter2d_cross3(
                            17,
                            23,
                            channels,
                            depth,
                            border_type,
                            seed + channels,
                            dst.data,
                            dst.total() * dst.elemSize()));
                }
            }
        }
    }
}

TEST(OpenCVContractSmoke_TEST, imgproc_phase1_integral_derivatives_and_square_box_match_upstream)
{
    using cvh_test_opencv_contract::CoreDepthId;
    constexpr int rows = 11;
    constexpr int cols = 13;
    constexpr std::uint32_t seed = 0x73ad91e5u;
    for (const int channels : {1, 3, 4})
    {
        Mat src({rows, cols}, CV_MAKETYPE(CV_8U, channels));
        fill_u8(src, seed);
        for (const auto depth_and_type :
             {std::pair<CoreDepthId, int>(CoreDepthId::S32, CV_32S),
              std::pair<CoreDepthId, int>(CoreDepthId::F64, CV_64F)})
        {
            Mat actual;
            integral(src, actual, depth_and_type.second);
            EXPECT_TRUE(
                cvh_test_opencv_contract::validate_imgproc_integral_u8(
                    rows,
                    cols,
                    channels,
                    seed,
                    depth_and_type.first,
                    actual.data,
                    actual.total() * actual.elemSize()));
        }

        Mat scharr;
        Mat laplacian;
        Scharr(src, scharr, CV_16S, 1, 0);
        Laplacian(src, laplacian, CV_16S, 3);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_derivative_filters_u8(
                rows,
                cols,
                channels,
                seed,
                scharr.data,
                scharr.total() * scharr.elemSize(),
                laplacian.data,
                laplacian.total() * laplacian.elemSize()));

        Mat squared;
        sqrBoxFilter(
            src, squared, CV_64F, Size(7, 5));
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_sqr_box_filter_u8(
                rows,
                cols,
                channels,
                seed,
                squared.data,
                squared.total() * squared.elemSize()));
    }

    Mat src({rows, cols}, CV_8UC1);
    fill_u8(src, seed);
    Mat dx;
    Mat dy;
    spatialGradient(src, dx, dy);
    EXPECT_TRUE(
        cvh_test_opencv_contract::validate_imgproc_spatial_gradient_u8(
            rows,
            cols,
            seed,
            dx.data,
            dx.total() * dx.elemSize(),
            dy.data,
            dy.total() * dy.elemSize()));
}

TEST(OpenCVContractSmoke_TEST, imgproc_phase1_intensity_ops_match_upstream)
{
    using cvh_test_opencv_contract::ImgprocIntensityOpId;
    constexpr int rows = 13;
    constexpr int cols = 17;
    constexpr std::uint32_t seed = 0x4e912bd3u;
    const auto run = [&](ImgprocIntensityOpId op) {
        const bool color_source =
            op == ImgprocIntensityOpId::BilateralU8 ||
            op == ImgprocIntensityOpId::StackBlurU8;
        Mat src(
            {rows, cols},
            color_source ? CV_8UC3 : CV_8UC1);
        fill_u8(src, seed);
        Mat actual;
        switch (op)
        {
            case ImgprocIntensityOpId::MedianU8:
                medianBlur(src, actual, 5);
                break;
            case ImgprocIntensityOpId::BilateralU8:
                bilateralFilter(
                    src,
                    actual,
                    5,
                    35.0,
                    2.0,
                    BORDER_REFLECT_101);
                break;
            case ImgprocIntensityOpId::StackBlurU8:
                stackBlur(src, actual, Size(5, 3));
                break;
            case ImgprocIntensityOpId::AdaptiveMeanU8:
                adaptiveThreshold(
                    src,
                    actual,
                    200.0,
                    ADAPTIVE_THRESH_MEAN_C,
                    THRESH_BINARY,
                    5,
                    2.25);
                break;
            case ImgprocIntensityOpId::AdaptiveGaussianU8:
                adaptiveThreshold(
                    src,
                    actual,
                    200.0,
                    ADAPTIVE_THRESH_GAUSSIAN_C,
                    THRESH_BINARY_INV,
                    5,
                    -1.25);
                break;
            case ImgprocIntensityOpId::ThresholdMaskU8:
            {
                Mat mask({rows, cols}, CV_8UC1);
                fill_mask(mask);
                actual.create(src.shape(), src.type());
                actual.setTo(Scalar::all(17));
                thresholdWithMask(
                    src,
                    actual,
                    mask,
                    110.0,
                    200.0,
                    THRESH_BINARY);
                break;
            }
            case ImgprocIntensityOpId::EqualizeHistU8:
                equalizeHist(src, actual);
                break;
            case ImgprocIntensityOpId::ColorMapJetU8:
                applyColorMap(src, actual, COLORMAP_JET);
                break;
            case ImgprocIntensityOpId::ColorMapUserU8:
            {
                Mat lookup({256, 1}, CV_8UC3);
                for (int i = 0; i < 256; ++i)
                {
                    lookup.at<uchar>(i, 0, 0) =
                        static_cast<uchar>(i);
                    lookup.at<uchar>(i, 0, 1) =
                        static_cast<uchar>(255 - i);
                    lookup.at<uchar>(i, 0, 2) = 17;
                }
                applyColorMap(src, actual, lookup);
                break;
            }
        }
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_intensity_u8(
                op,
                rows,
                cols,
                seed,
                actual.data,
                actual.total() * actual.elemSize()))
            << "intensity op id=" << static_cast<int>(op);
    };

    for (const ImgprocIntensityOpId op :
         {ImgprocIntensityOpId::MedianU8,
          ImgprocIntensityOpId::BilateralU8,
          ImgprocIntensityOpId::StackBlurU8,
          ImgprocIntensityOpId::AdaptiveMeanU8,
          ImgprocIntensityOpId::AdaptiveGaussianU8,
          ImgprocIntensityOpId::ThresholdMaskU8,
          ImgprocIntensityOpId::EqualizeHistU8,
          ImgprocIntensityOpId::ColorMapJetU8,
          ImgprocIntensityOpId::ColorMapUserU8})
    {
        run(op);
    }
}

TEST(OpenCVContractSmoke_TEST, imgproc_phase1_pyramid_color_ops_match_upstream)
{
    using cvh_test_opencv_contract::ImgprocPyramidColorOpId;
    constexpr int rows = 12;
    constexpr int cols = 16;
    constexpr std::uint32_t seed = 0x8d2346b1u;
    const auto run = [&](ImgprocPyramidColorOpId op) {
        const bool demosaic =
            op == ImgprocPyramidColorOpId::DemosaicBgU8 ||
            op == ImgprocPyramidColorOpId::DemosaicGbU8 ||
            op == ImgprocPyramidColorOpId::DemosaicRgU8 ||
            op == ImgprocPyramidColorOpId::DemosaicGrU8;
        Mat src(
            {rows, cols}, demosaic ? CV_8UC1 : CV_8UC3);
        fill_u8(src, seed);
        Mat actual;
        switch (op)
        {
            case ImgprocPyramidColorOpId::AccumulateU8:
            case ImgprocPyramidColorOpId::AccumulateSquareU8:
            case ImgprocPyramidColorOpId::AccumulateProductU8:
            case ImgprocPyramidColorOpId::AccumulateWeightedU8:
            {
                Mat mask({rows, cols}, CV_8UC1);
                fill_mask(mask);
                actual.create({rows, cols}, CV_32FC3);
                actual.setTo(Scalar::all(1.0));
                if (op == ImgprocPyramidColorOpId::AccumulateU8)
                {
                    accumulate(src, actual);
                }
                else if (
                    op == ImgprocPyramidColorOpId::AccumulateSquareU8)
                {
                    accumulateSquare(src, actual, mask);
                }
                else if (
                    op == ImgprocPyramidColorOpId::AccumulateProductU8)
                {
                    Mat second({rows, cols}, CV_8UC3);
                    fill_u8(second, seed + 17u);
                    accumulateProduct(src, second, actual, mask);
                }
                else
                {
                    accumulateWeighted(src, actual, 0.375, mask);
                }
                break;
            }
            case ImgprocPyramidColorOpId::BlendLinearU8:
            {
                Mat second({rows, cols}, CV_8UC3);
                fill_u8(second, seed + 17u);
                Mat weight1({rows, cols}, CV_32FC1);
                Mat weight2({rows, cols}, CV_32FC1);
                for (int y = 0; y < rows; ++y)
                {
                    for (int x = 0; x < cols; ++x)
                    {
                        weight1.at<float>(y, x) =
                            static_cast<float>((x + y) % 5) * 0.25f;
                        weight2.at<float>(y, x) =
                            static_cast<float>((2 * x + y + 1) % 7) *
                            0.2f;
                    }
                }
                blendLinear(src, second, weight1, weight2, actual);
                break;
            }
            case ImgprocPyramidColorOpId::PyrDownU8:
                pyrDown(src, actual);
                break;
            case ImgprocPyramidColorOpId::PyrUpU8:
                pyrUp(src, actual);
                break;
            case ImgprocPyramidColorOpId::TwoPlaneNv12U8:
            case ImgprocPyramidColorOpId::TwoPlaneNv21U8:
            {
                Mat y_plane({rows, cols}, CV_8UC1);
                Mat uv_plane({rows / 2, cols / 2}, CV_8UC2);
                fill_u8(y_plane, seed);
                fill_u8(uv_plane, seed + 17u);
                cvtColorTwoPlane(
                    y_plane,
                    uv_plane,
                    actual,
                    op == ImgprocPyramidColorOpId::TwoPlaneNv12U8
                        ? COLOR_YUV2BGR_NV12
                        : COLOR_YUV2RGB_NV21);
                break;
            }
            case ImgprocPyramidColorOpId::DemosaicBgU8:
                demosaicing(src, actual, COLOR_BayerBG2BGR);
                break;
            case ImgprocPyramidColorOpId::DemosaicGbU8:
                demosaicing(src, actual, COLOR_BayerGB2BGR);
                break;
            case ImgprocPyramidColorOpId::DemosaicRgU8:
                demosaicing(src, actual, COLOR_BayerRG2BGR);
                break;
            case ImgprocPyramidColorOpId::DemosaicGrU8:
                demosaicing(src, actual, COLOR_BayerGR2BGR);
                break;
        }
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_imgproc_pyramid_color_u8(
                op,
                rows,
                cols,
                seed,
                actual.data,
                actual.total() * actual.elemSize()))
            << "pyramid/color op id=" << static_cast<int>(op);
    };

    for (const ImgprocPyramidColorOpId op :
         {ImgprocPyramidColorOpId::AccumulateU8,
          ImgprocPyramidColorOpId::AccumulateSquareU8,
          ImgprocPyramidColorOpId::AccumulateProductU8,
          ImgprocPyramidColorOpId::AccumulateWeightedU8,
          ImgprocPyramidColorOpId::BlendLinearU8,
          ImgprocPyramidColorOpId::PyrDownU8,
          ImgprocPyramidColorOpId::PyrUpU8,
          ImgprocPyramidColorOpId::TwoPlaneNv12U8,
          ImgprocPyramidColorOpId::TwoPlaneNv21U8,
          ImgprocPyramidColorOpId::DemosaicBgU8,
          ImgprocPyramidColorOpId::DemosaicGbU8,
          ImgprocPyramidColorOpId::DemosaicRgU8,
          ImgprocPyramidColorOpId::DemosaicGrU8})
    {
        run(op);
    }
}

TEST(OpenCVContractSmoke_TEST, imgproc_geometry_matrices_match_upstream)
{
    const Point2f center(12.5f, -7.25f);
    const AffineMatrix2x3d rotation =
        getRotationMatrix2D_(center, 37.0, 0.75);

    const Point2f affine_source[] = {
        Point2f(0.0f, 0.0f),
        Point2f(4.0f, 0.0f),
        Point2f(0.0f, 3.0f)};
    const Point2f affine_target[] = {
        Point2f(5.0f, -2.0f),
        Point2f(13.0f, -6.0f),
        Point2f(6.5f, 7.0f)};
    const Mat affine =
        getAffineTransform(affine_source, affine_target);

    const Point2f perspective_source[] = {
        Point2f(0.0f, 0.0f),
        Point2f(8.0f, 0.0f),
        Point2f(8.0f, 6.0f),
        Point2f(0.0f, 6.0f)};
    const Point2f perspective_target[] = {
        Point2f(1.0f, 2.0f),
        Point2f(9.0f, 1.0f),
        Point2f(7.5f, 8.0f),
        Point2f(-0.5f, 6.5f)};
    const Mat perspective =
        getPerspectiveTransform(
            perspective_source,
            perspective_target);

    Mat inverse;
    invertAffineTransform(affine, inverse);
    EXPECT_TRUE(
        cvh_test_opencv_contract::validate_imgproc_geometry_matrices(
            rotation.val,
            reinterpret_cast<const double*>(affine.data),
            reinterpret_cast<const double*>(perspective.data),
            reinterpret_cast<const double*>(inverse.data)));
}

TEST(OpenCVContractSmoke_TEST, imgproc_geometry_sampling_matches_upstream)
{
    using cvh_test_opencv_contract::ImgprocGeometrySamplingOpId;
    constexpr std::uint32_t seed = 0x7193u;
    Mat source({9, 11}, CV_8UC3);
    fill_u8(source, seed);

    const auto validate =
        [&](ImgprocGeometrySamplingOpId op, const Mat& actual) {
            EXPECT_TRUE(
                cvh_test_opencv_contract::
                    validate_imgproc_geometry_sampling(
                        op,
                        seed,
                        actual.data,
                        actual.total() * actual.elemSize()))
                << "geometry sampling op=" << static_cast<int>(op);
        };

    Mat map_x({7, 9}, CV_32FC1);
    Mat map_y({7, 9}, CV_32FC1);
    for (int row = 0; row < 7; ++row)
    {
        for (int col = 0; col < 9; ++col)
        {
            map_x.at<float>(row, col) =
                static_cast<float>(col) + 0.28125f;
            map_y.at<float>(row, col) =
                static_cast<float>(row) - 0.34375f;
        }
    }
    Mat actual;
    remap(
        source,
        actual,
        map_x,
        map_y,
        INTER_LINEAR,
        BORDER_REFLECT_101);
    validate(ImgprocGeometrySamplingOpId::RemapFloatU8, actual);

    Mat fixed_coordinates;
    Mat fixed_fractions;
    convertMaps(
        map_x,
        map_y,
        fixed_coordinates,
        fixed_fractions,
        CV_16SC2);
    remap(
        source,
        actual,
        fixed_coordinates,
        fixed_fractions,
        INTER_LINEAR,
        BORDER_REFLECT_101);
    validate(ImgprocGeometrySamplingOpId::RemapFixedU8, actual);

    Mat affine({2, 3}, CV_32FC1);
    const float affine_values[] = {
        1.0f, 0.0f, -1.25f,
        0.0f, 1.0f, 0.75f};
    std::memcpy(affine.data, affine_values, sizeof(affine_values));
    warpAffine(
        source,
        actual,
        affine,
        Size(9, 7),
        INTER_LINEAR | WARP_INVERSE_MAP,
        BORDER_REPLICATE);
    validate(
        ImgprocGeometrySamplingOpId::WarpAffineTranslationU8,
        actual);

    Mat source_f32({9, 11}, CV_32FC4);
    fill_f32(source_f32, seed + 4u);
    warpAffine(
        source_f32,
        actual,
        affine,
        Size(9, 7),
        INTER_LINEAR | WARP_INVERSE_MAP,
        BORDER_REPLICATE);
    validate(
        ImgprocGeometrySamplingOpId::WarpAffineTranslationF32,
        actual);

    Mat perspective({3, 3}, CV_64FC1);
    perspective.setTo(Scalar::all(0.0));
    perspective.at<double>(0, 0) = 1.0;
    perspective.at<double>(0, 1) = 0.1;
    perspective.at<double>(0, 2) = 0.25;
    perspective.at<double>(1, 0) = -0.05;
    perspective.at<double>(1, 1) = 1.0;
    perspective.at<double>(1, 2) = 0.5;
    perspective.at<double>(2, 0) = 0.002;
    perspective.at<double>(2, 1) = -0.003;
    perspective.at<double>(2, 2) = 1.0;
    warpPerspective(
        source,
        actual,
        perspective,
        Size(9, 7),
        INTER_LINEAR | WARP_INVERSE_MAP,
        BORDER_REFLECT_101);
    validate(ImgprocGeometrySamplingOpId::WarpPerspectiveU8, actual);

    getRectSubPix(
        source,
        Size(7, 5),
        Point2f(0.25f, 0.75f),
        actual);
    validate(ImgprocGeometrySamplingOpId::RectSubPixU8, actual);
    getRectSubPix(
        source,
        Size(7, 5),
        Point2f(4.25f, 3.75f),
        actual,
        CV_32F);
    validate(ImgprocGeometrySamplingOpId::RectSubPixU8F32, actual);
}

TEST(OpenCVContractSmoke_TEST, phase2_core_random_and_point_transforms_match_upstream)
{
    Mat random_values({3, 5}, CV_8UC4);
    randn(random_values,
          Scalar(-10.0, 12.6, 300.0, 42.0),
          Scalar::all(0.0));
    EXPECT_TRUE(cvh_test_opencv_contract::validate_phase2_random_zero_stddev(
        random_values.data, random_values.total() * random_values.elemSize()));

    Mat source({3, 4}, CV_32FC2);
    for (int row = 0; row < source.size[0]; ++row)
    {
        for (int column = 0; column < source.size[1]; ++column)
        {
            source.at<float>(row, column, 0) = static_cast<float>(column + row * 0.25);
            source.at<float>(row, column, 1) = static_cast<float>(2 * column - row * 0.5);
        }
    }
    Mat matrix({3, 3}, CV_64FC1);
    const double matrix_values[] = {
        2.0, -1.0, 3.0, 0.5, 4.0, -2.0, -1.0, 0.25, 7.0};
    for (int index = 0; index < 9; ++index)
        matrix.at<double>(index / 3, index % 3) = matrix_values[index];
    Mat transformed;
    transform(source, transformed, matrix);
    EXPECT_TRUE(cvh_test_opencv_contract::validate_phase2_transform(
        transformed.data, transformed.total() * transformed.elemSize()));

    Mat points({1, 4}, CV_64FC2);
    points.at<double>(0, 0, 0) = 1.0;
    points.at<double>(0, 0, 1) = 2.0;
    points.at<double>(0, 1, 0) = 3.0;
    points.at<double>(0, 1, 1) = 4.0;
    points.at<double>(0, 2, 0) = -2.0;
    points.at<double>(0, 2, 1) = 5.0;
    points.at<double>(0, 3, 0) = std::numeric_limits<double>::quiet_NaN();
    points.at<double>(0, 3, 1) = 1.0;
    const double perspective_values[] = {
        2.0, 0.5, 3.0, -1.0, 3.0, -2.0, 1.0, 0.25, -1.0};
    for (int index = 0; index < 9; ++index)
        matrix.at<double>(index / 3, index % 3) = perspective_values[index];
    perspectiveTransform(points, transformed, matrix);
    EXPECT_TRUE(cvh_test_opencv_contract::validate_phase2_perspective_transform(
        transformed.data, transformed.total() * transformed.elemSize()));
}

TEST(OpenCVContractSmoke_TEST, phase2_connected_components_match_upstream)
{
    Mat image({7, 9}, CV_8UC1);
    image = 0;
    for (int row = 0; row < image.size[0]; ++row)
        for (int column = 0; column < image.size[1]; ++column)
            if (((row * column + column + 2 * row) % 7) < 2)
                image.at<uchar>(row, column) = 255;

    for (int connectivity : {4, 8})
    {
        Mat labels;
        Mat stats;
        Mat centroids;
        const int count = connectedComponentsWithStats(
            image, labels, stats, centroids, connectivity);
        EXPECT_TRUE(cvh_test_opencv_contract::validate_phase2_connected_components(
            connectivity,
            labels.data, labels.total() * labels.elemSize(),
            stats.data, stats.total() * stats.elemSize(),
            centroids.data, centroids.total() * centroids.elemSize(),
            count)) << "connectivity=" << connectivity;
    }
}

TEST(OpenCVContractSmoke_TEST, phase2_contours_match_upstream_order_and_ties)
{
    Mat image({9, 11}, CV_8UC1);
    image = 0;
    for (int row = 1; row < 7; ++row)
        for (int column = 1; column < 8; ++column)
            image.at<uchar>(row, column) = 255;
    for (int row = 2; row < 5; ++row)
        for (int column = 3; column < 6; ++column)
            image.at<uchar>(row, column) = 0;
    image.at<uchar>(7, 9) = 255;
    image.at<uchar>(8, 10) = 255;

    for (int mode : {RETR_EXTERNAL, RETR_LIST})
    {
        for (int method : {CHAIN_APPROX_NONE, CHAIN_APPROX_SIMPLE})
        {
            std::vector<std::vector<Point>> contours;
            findContours(image, contours, mode, method, Point(2, -3));
            std::vector<int> xy;
            std::vector<std::size_t> sizes;
            for (const auto& contour : contours)
            {
                sizes.push_back(contour.size());
                for (const Point& point : contour)
                {
                    xy.push_back(point.x);
                    xy.push_back(point.y);
                }
            }
            EXPECT_TRUE(cvh_test_opencv_contract::validate_phase2_contours(
                mode, method, 2, -3, xy.data(), sizes.data(), contours.size()))
                << "mode=" << mode << " method=" << method;
        }
    }
}

TEST(OpenCVContractSmoke_TEST, phase2_shapes_match_upstream)
{
    const std::vector<Point> points = {
        Point(1, 1), Point(6, 1), Point(7, 3), Point(5, 6),
        Point(3, 5), Point(1, 6), Point(2, 3)};
    const Rect rect = boundingRect(points);
    const int rect_values[] = {rect.x, rect.y, rect.width, rect.height};
    const double scalar_values[] = {
        contourArea(points, false), contourArea(points, true),
        arcLength(points, false), arcLength(points, true)};
    std::vector<Point> approximate;
    approxPolyDP(points, approximate, 0.75, true);
    std::vector<Point> hull;
    convexHull(points, hull, false);
    std::vector<int> approximate_xy;
    std::vector<int> hull_xy;
    for (const Point& point : approximate)
    {
        approximate_xy.push_back(point.x);
        approximate_xy.push_back(point.y);
    }
    for (const Point& point : hull)
    {
        hull_xy.push_back(point.x);
        hull_xy.push_back(point.y);
    }
    const Moments value = moments(points);
    const double moment_values[] = {
        value.m00, value.m10, value.m01, value.m20, value.m11, value.m02,
        value.m30, value.m21, value.m12, value.m03,
        value.mu20, value.mu11, value.mu02, value.mu30, value.mu21,
        value.mu12, value.mu03, value.nu20, value.nu11, value.nu02,
        value.nu30, value.nu21, value.nu12, value.nu03};
    EXPECT_TRUE(cvh_test_opencv_contract::validate_phase2_shapes(
        rect_values, scalar_values,
        approximate_xy.data(), approximate.size(),
        hull_xy.data(), hull.size(),
        isContourConvex(points), moment_values));
}

TEST(OpenCVContractSmoke_TEST, phase2_histogram_and_template_matching_match_upstream)
{
    Mat image({5, 7}, CV_32FC3);
    Mat mask({5, 7}, CV_8UC1);
    for (int row = 0; row < image.size[0]; ++row)
    {
        for (int column = 0; column < image.size[1]; ++column)
        {
            image.at<float>(row, column, 0) = static_cast<float>(column - row);
            image.at<float>(row, column, 1) = static_cast<float>(column * 0.75 + row * 1.25);
            image.at<float>(row, column, 2) = static_cast<float>(100 + column);
            mask.at<uchar>(row, column) = (row + 2 * column) % 3 != 0 ? 255 : 0;
        }
    }
    Mat histogram;
    calcHist(image, 1, mask, histogram, 8, 0.0f, 12.0f);
    Mat other({8, 1}, CV_32FC1);
    for (int index = 0; index < 8; ++index)
        other.at<float>(index, 0) = static_cast<float>(index + 1);
    const double comparisons[] = {
        compareHist(histogram, other, HISTCMP_CORREL),
        compareHist(histogram, other, HISTCMP_CHISQR),
        compareHist(histogram, other, HISTCMP_INTERSECT),
        compareHist(histogram, other, HISTCMP_BHATTACHARYYA)};
    EXPECT_TRUE(cvh_test_opencv_contract::validate_phase2_histogram(
        histogram.data, histogram.total() * histogram.elemSize(), comparisons));

    Mat match_image({8, 10}, CV_32FC1);
    for (int row = 0; row < match_image.size[0]; ++row)
        for (int column = 0; column < match_image.size[1]; ++column)
            match_image.at<float>(row, column) =
                static_cast<float>(std::sin(row * 0.3) + std::cos(column * 0.2) + row * column * 0.01);
    Mat templ = match_image(Range(2, 5), Range(3, 7));
    for (int method : {TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED})
    {
        Mat actual;
        matchTemplate(match_image, templ, actual, method);
        EXPECT_TRUE(cvh_test_opencv_contract::validate_phase2_template_match(
            method, actual.data, actual.total() * actual.elemSize()))
            << "method=" << method;
    }


    Mat u8_storage({9, 79}, CV_8UC1);
    for (int row = 0; row < u8_storage.size[0]; ++row)
        for (int column = 0; column < u8_storage.size[1]; ++column)
            u8_storage.at<uchar>(row, column) = static_cast<uchar>(
                (row * 31 + column * 17 + row * column * 3) & 255);
    Mat u8_image = u8_storage(Range(1, 9), Range(1, 79));
    Mat u8_template = u8_image(Range(2, 7), Range(3, 70));
    for (int method : {TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED})
    {
        Mat actual;
        matchTemplate(u8_image, u8_template, actual, method);
        EXPECT_TRUE(
            cvh_test_opencv_contract::validate_phase2_template_match_u8_roi(
                method, actual.data, actual.total() * actual.elemSize()))
            << "U8 ROI method=" << method;
    }
}
