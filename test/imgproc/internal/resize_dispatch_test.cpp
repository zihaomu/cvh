#include "test/imgproc/support/kernel_family_test_utils.hpp"
#include "test/support/dispatch_mode_guard.hpp"

#include "cvh/core/detail/cpu_features.hpp"

#include <cstring>
#include <tuple>
#include <vector>

namespace {

void fill_resize_pattern(Mat& mat)
{
    unsigned state = 0x2468ace1u;
    for (int y = 0; y < mat.size[0]; ++y)
    {
        uchar* row = mat.data + static_cast<std::size_t>(y) * mat.step(0);
        for (std::size_t x = 0;
             x < static_cast<std::size_t>(mat.size[1]) * mat.elemSize();
             ++x)
        {
            state = state * 1664525u + 1013904223u;
            row[x] = static_cast<uchar>(state >> 24);
        }
    }
}

void expect_resize_equal(const Mat& expected, const Mat& actual)
{
    ASSERT_EQ(expected.type(), actual.type());
    ASSERT_EQ(expected.shape(), actual.shape());
    const std::size_t row_bytes =
        static_cast<std::size_t>(expected.size[1]) * expected.elemSize();
    for (int y = 0; y < expected.size[0]; ++y)
    {
        const uchar* a = expected.data +
            static_cast<std::size_t>(y) * expected.step(0);
        const uchar* b = actual.data +
            static_cast<std::size_t>(y) * actual.step(0);
        if (std::memcmp(a, b, row_bytes) != 0)
        {
            for (std::size_t x = 0; x < row_bytes; ++x)
            {
                ASSERT_EQ(a[x], b[x])
                    << "row=" << y << ", byte=" << x;
            }
        }
    }
}

}  // namespace

TEST(ResizeDispatchInternalTest,
     linear_u8c3_neon_matches_scalar_for_ratios_roi_and_tails)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    const std::vector<std::tuple<int, int, int, int, bool>> cases = {
        {34, 50, 17, 25, false},
        {36, 52, 27, 39, false},
        {480, 640, 360, 480, false},
        {18, 26, 27, 39, false},
        {41, 59, 29, 39, true},
    };

    for (const auto& [rows, cols, dst_rows, dst_cols, use_roi] : cases)
    {
        Mat parent({rows + 2, cols + 7}, CV_8UC3);
        fill_resize_pattern(parent);
        const Mat source = use_roi
            ? parent(Range(1, rows + 1), Range(3, cols + 3))
            : parent(Range(0, rows), Range(0, cols));
        ASSERT_FALSE(source.isContinuous());

        cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
        cpu::reset_last_dispatch_tag();
        Mat scalar;
        resize(
            source, scalar, Size(dst_cols, dst_rows),
            0.0, 0.0, INTER_LINEAR);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

        cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        Mat accelerated;
        resize(
            source, accelerated, Size(dst_cols, dst_rows),
            0.0, 0.0, INTER_LINEAR);
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            cpu::neon_runtime_available()
                ? cpu::DispatchTag::NEON
                : cpu::DispatchTag::Scalar);
        expect_resize_equal(scalar, accelerated);
    }
}

TEST(ResizeDispatchInternalTest,
     forced_modes_and_small_workload_preserve_fallback_contract)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat source({34, 50}, CV_8UC3);
    fill_resize_pattern(source);
    Mat output;

    cpu::set_dispatch_mode(cpu::DispatchMode::OpenCVUIOnly);
    cpu::reset_last_dispatch_tag();
    resize(source, output, Size(25, 17), 0.0, 0.0, INTER_LINEAR);
    EXPECT_NE(cpu::last_dispatch_tag(), cpu::DispatchTag::NEON);

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    cpu::reset_last_dispatch_tag();
    resize(source, output, Size(25, 17), 0.0, 0.0, INTER_LINEAR);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

    cpu::set_dispatch_mode(cpu::DispatchMode::NeonOnly);
    cpu::reset_last_dispatch_tag();
    resize(source, output, Size(25, 17), 0.0, 0.0, INTER_LINEAR);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::neon_runtime_available()
            ? cpu::DispatchTag::NEON
            : cpu::DispatchTag::Scalar);

    Mat short_source({7, 13}, CV_8UC3);
    fill_resize_pattern(short_source);
    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
    cpu::reset_last_dispatch_tag();
    resize(short_source, output, Size(9, 5), 0.0, 0.0, INTER_LINEAR);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
}
