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
        for (int x = 0; x < mat.size.p[1]; ++x)
        {
            mat.at<uchar>(y, x) =
                static_cast<uchar>((y * 37 + x * 19) & 255);
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
     derivative_parent_roi_and_spatial_gradient_prove_fallbacks)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat parent({11, 73}, CV_8UC1);
    fill_derivative_pattern(parent);
    const Mat parent_sampled_roi =
        parent(Range(1, 10), Range(2, 69));

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
        cpu::DispatchTag::Scalar);

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
        cvh::test::expected_fixed_width_dispatch_tag());
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
