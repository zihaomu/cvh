#include "test/imgproc/support/kernel_family_test_utils.hpp"
#include "test/support/dispatch_mode_guard.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

int max_s16_difference(const Mat& first, const Mat& second)
{
    int maximum = 0;
    for (size_t index = 0; index < first.total(); ++index)
    {
        maximum = std::max(
            maximum,
            std::abs(
                static_cast<int>(
                    reinterpret_cast<const short*>(
                        first.data)[index]) -
                static_cast<int>(
                    reinterpret_cast<const short*>(
                        second.data)[index])));
    }
    return maximum;
}

void fill_derivative_pattern(Mat& mat)
{
    for (int y = 0; y < mat.size.p[0]; ++y)
    {
        uchar* row = mat.data + static_cast<std::size_t>(y) * mat.step(0);
        for (int x = 0; x < mat.size.p[1] * mat.channels(); ++x)
        {
            row[x] = static_cast<uchar>((y * 37 + x * 19) & 255);
        }
    }
}

}  // namespace

TEST(DerivativesDispatchInternalTest,
     derivative_s16_records_scalar_ui_and_short_fallback)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    for (const int cols : {5, 67})
    {
        Mat parent({9, cols + 3}, CV_8UC1);
        fill_derivative_pattern(parent);
        const Mat source =
            parent(Range(1, 8), Range(1, cols + 1));
        ASSERT_FALSE(source.isContinuous());
        const cpu::DispatchTag auto_tag =
            cols >= 6
                ? cvh::test::expected_fixed_width_dispatch_tag()
                : cpu::DispatchTag::Scalar;

        for (const int border_type :
             {BORDER_CONSTANT,
              BORDER_REPLICATE,
              BORDER_REFLECT,
              BORDER_REFLECT_101})
        {
            Mat scalar_scharr;
            Mat scalar_laplacian;
            Mat auto_scharr;
            Mat auto_laplacian;

            cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
            cpu::reset_last_dispatch_tag();
            Scharr(
                source,
                scalar_scharr,
                CV_16S,
                1,
                0,
                0.75,
                3.0,
                border_type | BORDER_ISOLATED);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                cpu::DispatchTag::Scalar);

            cpu::reset_last_dispatch_tag();
            Laplacian(
                source,
                scalar_laplacian,
                CV_16S,
                3,
                0.75,
                3.0,
                border_type | BORDER_ISOLATED);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                cpu::DispatchTag::Scalar);

            cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
            cpu::reset_last_dispatch_tag();
            Scharr(
                source,
                auto_scharr,
                CV_16S,
                1,
                0,
                0.75,
                3.0,
                border_type | BORDER_ISOLATED);
            EXPECT_EQ(cpu::last_dispatch_tag(), auto_tag);

            cpu::reset_last_dispatch_tag();
            Laplacian(
                source,
                auto_laplacian,
                CV_16S,
                3,
                0.75,
                3.0,
                border_type | BORDER_ISOLATED);
            EXPECT_EQ(cpu::last_dispatch_tag(), auto_tag);

            EXPECT_LE(
                max_s16_difference(scalar_scharr, auto_scharr),
                1);
            EXPECT_EQ(
                std::memcmp(
                    scalar_laplacian.data,
                    auto_laplacian.data,
                    scalar_laplacian.total() *
                        scalar_laplacian.elemSize()),
                0);
        }
    }
}

TEST(DerivativesDispatchInternalTest,
     derivative_parent_roi_and_spatial_gradient_preserve_contracts)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat parent({11, 73}, CV_8UC1);
    fill_derivative_pattern(parent);
    const Mat parent_sampled_roi =
        parent(Range(1, 10), Range(2, 69));

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    Mat scalar_roi_scharr;
    Scharr(
        parent_sampled_roi,
        scalar_roi_scharr,
        CV_16S,
        1,
        0,
        1.0,
        0.0,
        BORDER_REPLICATE);
    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
    cpu::reset_last_dispatch_tag();
    Mat roi_scharr;
    Scharr(
        parent_sampled_roi,
        roi_scharr,
        CV_16S,
        1,
        0,
        1.0,
        0.0,
        BORDER_REPLICATE);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::neon_runtime_available()
            ? cpu::DispatchTag::NEON
            : cpu::DispatchTag::Scalar);
    EXPECT_EQ(
        std::memcmp(
            scalar_roi_scharr.data,
            roi_scharr.data,
            scalar_roi_scharr.total() * scalar_roi_scharr.elemSize()),
        0);

    Mat source({9, 67}, CV_8UC1);
    fill_derivative_pattern(source);
    Mat scalar_dx;
    Mat scalar_dy;
    Mat auto_dx;
    Mat auto_dy;

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    cpu::reset_last_dispatch_tag();
    spatialGradient(source, scalar_dx, scalar_dy);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::DispatchTag::Scalar);

    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
    cpu::reset_last_dispatch_tag();
    spatialGradient(source, auto_dx, auto_dy);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::neon_runtime_available()
            ? cpu::DispatchTag::NEON
            : cvh::test::expected_fixed_width_dispatch_tag());
    EXPECT_EQ(
        std::memcmp(
            scalar_dx.data,
            auto_dx.data,
            scalar_dx.total() * scalar_dx.elemSize()),
        0);
    EXPECT_EQ(
        std::memcmp(
            scalar_dy.data,
            auto_dy.data,
            scalar_dy.total() * scalar_dy.elemSize()),
        0);

    Mat short_source({9, 9}, CV_8UC1);
    fill_derivative_pattern(short_source);
    cpu::reset_last_dispatch_tag();
    spatialGradient(short_source, auto_dx, auto_dy);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::DispatchTag::Scalar);
}

TEST(DerivativesDispatchInternalTest,
     direct_neon_sobel_scharr_and_spatial_match_scalar)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    for (const int channels : {1, 3, 4})
    {
        Mat parent({39, 75}, CV_MAKETYPE(CV_8U, channels));
        fill_derivative_pattern(parent);
        const Mat source = parent(Range(1, 38), Range(3, 70));
        ASSERT_FALSE(source.isContinuous());
        for (const int border_type :
             {BORDER_REPLICATE, BORDER_REFLECT_101})
        {
            for (const int output_depth : {CV_16S, CV_32F})
            {
                for (const bool derivative_x : {false, true})
                {
                    Mat scalar;
                    Mat accelerated;
                    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
                    Sobel(
                        source, scalar, output_depth,
                        derivative_x ? 1 : 0,
                        derivative_x ? 0 : 1,
                        3, 1.0, 0.0,
                        border_type | BORDER_ISOLATED);
                    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
                    cpu::reset_last_dispatch_tag();
                    Sobel(
                        source, accelerated, output_depth,
                        derivative_x ? 1 : 0,
                        derivative_x ? 0 : 1,
                        3, 1.0, 0.0,
                        border_type | BORDER_ISOLATED);
                    EXPECT_EQ(
                        cpu::last_dispatch_tag(),
                        cpu::neon_runtime_available()
                            ? cpu::DispatchTag::NEON
                            : cvh::test::expected_fixed_width_dispatch_tag());
                    EXPECT_EQ(
                        std::memcmp(
                            scalar.data,
                            accelerated.data,
                            scalar.total() * scalar.elemSize()),
                        0);
                }
            }
        }
    }

    Mat source({37, 67}, CV_8UC1);
    fill_derivative_pattern(source);
    Mat scalar_scharr;
    Mat accelerated_scharr;
    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    Scharr(
        source, scalar_scharr, CV_32F,
        1, 0, 1.0, 0.0, BORDER_REFLECT_101);
    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
    cpu::reset_last_dispatch_tag();
    Scharr(
        source, accelerated_scharr, CV_32F,
        1, 0, 1.0, 0.0, BORDER_REFLECT_101);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::neon_runtime_available()
            ? cpu::DispatchTag::NEON
            : cvh::test::expected_fixed_width_dispatch_tag());
    EXPECT_EQ(
        std::memcmp(
            scalar_scharr.data,
            accelerated_scharr.data,
            scalar_scharr.total() * scalar_scharr.elemSize()),
        0);

    Mat scalar_dx;
    Mat scalar_dy;
    Mat accelerated_dx;
    Mat accelerated_dy;
    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    spatialGradient(
        source, scalar_dx, scalar_dy, 3, BORDER_REPLICATE);
    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
    cpu::reset_last_dispatch_tag();
    spatialGradient(
        source, accelerated_dx, accelerated_dy, 3, BORDER_REPLICATE);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::neon_runtime_available()
            ? cpu::DispatchTag::NEON
            : cvh::test::expected_fixed_width_dispatch_tag());
    EXPECT_EQ(
        std::memcmp(
            scalar_dx.data,
            accelerated_dx.data,
            scalar_dx.total() * scalar_dx.elemSize()),
        0);
    EXPECT_EQ(
        std::memcmp(
            scalar_dy.data,
            accelerated_dy.data,
            scalar_dy.total() * scalar_dy.elemSize()),
        0);

    Mat mode_output;
    cpu::set_dispatch_mode(cpu::DispatchMode::OpenCVUIOnly);
    cpu::reset_last_dispatch_tag();
    Sobel(
        source, mode_output, CV_32F,
        1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
    EXPECT_NE(cpu::last_dispatch_tag(), cpu::DispatchTag::NEON);

    cpu::set_dispatch_mode(cpu::DispatchMode::NeonOnly);
    cpu::reset_last_dispatch_tag();
    Sobel(
        source, mode_output, CV_32F,
        1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::neon_runtime_available()
            ? cpu::DispatchTag::NEON
            : cpu::DispatchTag::Scalar);
}
