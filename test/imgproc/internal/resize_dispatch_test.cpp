#include "test/imgproc/support/kernel_family_test_utils.hpp"
#include "test/imgproc/support/resize_test_utils.hpp"
#include "test/support/dispatch_mode_guard.hpp"

#include "cvh/core/detail/cpu_features.hpp"
#include "cvh/imgproc/detail/resize_fixed_u8c3.hpp"

#include <cstring>
#include <limits>
#include <string>
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

TEST(ResizeFixedPointInternalTest,
     coordinate_rounding_and_shape_predicate_are_frozen)
{
    using cvh::detail::resize_fixed_u8c3::AxisCoordinate;
    using cvh::detail::resize_fixed_u8c3::build_axis_coordinate;
    using cvh::detail::resize_fixed_u8c3::is_exact_three_quarter_shape;

    EXPECT_TRUE(is_exact_three_quarter_shape(480, 640, 360, 480));
    EXPECT_TRUE(is_exact_three_quarter_shape(479, 641, 359, 480));
    EXPECT_FALSE(is_exact_three_quarter_shape(480, 640, 240, 320));
    EXPECT_FALSE(is_exact_three_quarter_shape(480, 640, 720, 960));

    const AxisCoordinate first = build_axis_coordinate(0, 4, 3);
    const AxisCoordinate middle = build_axis_coordinate(1, 4, 3);
    const AxisCoordinate last = build_axis_coordinate(2, 4, 3);
    EXPECT_EQ(first.first, 0);
    EXPECT_EQ(first.second, 1);
    EXPECT_EQ(first.fraction, 43);
    EXPECT_EQ(middle.first, 1);
    EXPECT_EQ(middle.second, 2);
    EXPECT_EQ(middle.fraction, 128);
    EXPECT_EQ(last.first, 2);
    EXPECT_EQ(last.second, 3);
    EXPECT_EQ(last.fraction, 213);
}

TEST(ResizeFixedPointInternalTest,
     two_stage_rounding_covers_ties_and_signed_deltas)
{
    using cvh::detail::resize_fixed_u8c3::bilinear_u8;
    using cvh::detail::resize_fixed_u8c3::lerp_u8;

    EXPECT_EQ(lerp_u8(0, 255, 0), 0);
    EXPECT_EQ(lerp_u8(255, 0, 0), 255);
    EXPECT_EQ(lerp_u8(0, 255, 128), 128);
    EXPECT_EQ(lerp_u8(255, 0, 128), 128);
    EXPECT_EQ(lerp_u8(0, 255, 255), 254);
    EXPECT_EQ(lerp_u8(255, 0, 255), 1);
    EXPECT_EQ(lerp_u8(127, 128, 128), 128);
    EXPECT_EQ(lerp_u8(128, 127, 128), 128);

    EXPECT_EQ(bilinear_u8(0, 255, 255, 0, 128, 128), 128);
    EXPECT_EQ(bilinear_u8(255, 0, 0, 255, 128, 128), 128);
    EXPECT_EQ(bilinear_u8(0, 0, 255, 255, 0, 128), 128);
}

TEST(ResizeFixedPointInternalTest,
     scalar_reference_stays_within_one_of_legacy_float_path)
{
    const std::vector<std::tuple<int, int, int, int, bool>> cases = {
        {4, 4, 3, 3, false},
        {64, 96, 48, 72, false},
        {479, 641, 359, 480, true},
    };

    for (const auto& [rows, cols, dst_rows, dst_cols, use_roi] : cases)
    {
        Mat parent({rows + 2, cols + 7}, CV_8UC3);
        fill_resize_pattern(parent);
        const Mat source = use_roi
            ? parent(Range(1, rows + 1), Range(3, cols + 3))
            : parent(Range(0, rows), Range(0, cols));

        Mat fixed;
        cvh::detail::resize_fixed_u8c3::resize_linear_scalar_reference(
            source, fixed, dst_rows, dst_cols);
        const Mat legacy = resize_reference_linear_u8(
            source, Size(dst_cols, dst_rows), 0.0, 0.0);
        EXPECT_LE(max_abs_diff_u8(fixed, legacy), 1)
            << "src=" << rows << "x" << cols
            << ", dst=" << dst_rows << "x" << dst_cols;
    }
}

TEST(ResizeFixedPointInternalTest,
     flat_block_maps_stay_inside_each_source_row)
{
    using cvh::detail::resize_fixed_u8c3::Maps;
    using cvh::detail::resize_fixed_u8c3::build_axis_coordinate;
    using cvh::detail::resize_fixed_u8c3::build_maps;

    for (int src_cols = 2; src_cols <= 257; ++src_cols)
    {
        const int src_rows = 37;
        const int dst_cols = src_cols * 3 / 4;
        const int dst_rows = src_rows * 3 / 4;
        if (dst_cols <= 0)
        {
            continue;
        }
        const Maps maps = build_maps(
            src_rows, src_cols, dst_rows, dst_cols);
        const std::size_t source_bytes =
            static_cast<std::size_t>(src_cols) * 3;
        const std::size_t output_bytes =
            static_cast<std::size_t>(dst_cols) * 3;
        EXPECT_LE(maps.vector_output_bytes(), output_bytes);
        for (const auto& block : maps.blocks)
        {
            EXPECT_GE(source_bytes, 32u);
            EXPECT_LE(block.source_byte_base + 32, source_bytes);
            for (std::size_t lane = 0; lane < 16; ++lane)
            {
                EXPECT_LE(block.left_index[lane], 28);
                EXPECT_LE(block.x_fraction[lane], 255);
            }
        }
    }

    const int maximum = std::numeric_limits<int>::max();
    const int destination = static_cast<int>(
        static_cast<std::int64_t>(maximum) * 3 / 4);
    const auto last = build_axis_coordinate(
        destination - 1, maximum, destination);
    EXPECT_GE(last.first, 0);
    EXPECT_LT(last.first, maximum);
    EXPECT_GE(last.second, last.first);
    EXPECT_LT(last.second, maximum);
    EXPECT_LE(last.fraction, 255);
}

TEST(ResizeFixedPointInternalTest,
     product_path_covers_continuous_unaligned_roi_narrow_rows_and_tails)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    struct Case
    {
        int rows;
        int cols;
        bool roi;
        bool expects_vector_path;
    };
    const std::vector<Case> cases = {
        {40, 8, false, false},
        {40, 12, false, true},
        {40, 16, false, true},
        {40, 16, true, true},
        {479, 641, true, true},
    };

    for (const Case& test_case : cases)
    {
        SCOPED_TRACE(
            std::to_string(test_case.rows) + "x" +
            std::to_string(test_case.cols) +
            (test_case.roi ? " roi" : " continuous"));
        Mat parent;
        Mat source;
        if (test_case.roi)
        {
            parent.create(
                std::vector<int>{test_case.rows + 2, test_case.cols + 5},
                CV_8UC3);
            fill_resize_pattern(parent);
            source = parent(
                Range(1, test_case.rows + 1),
                Range(1, test_case.cols + 1));
            ASSERT_FALSE(source.isContinuous());
            ASSERT_NE(
                reinterpret_cast<std::uintptr_t>(source.data) % 16,
                0u);
        }
        else
        {
            source.create(
                std::vector<int>{test_case.rows, test_case.cols},
                CV_8UC3);
            fill_resize_pattern(source);
            ASSERT_TRUE(source.isContinuous());
        }

        const int dst_rows = test_case.rows * 3 / 4;
        const int dst_cols = test_case.cols * 3 / 4;
        cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
        cpu::reset_last_dispatch_tag();
        Mat scalar;
        resize(
            source, scalar, Size(dst_cols, dst_rows),
            0.0, 0.0, INTER_LINEAR);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
        EXPECT_NE(
            std::string(cpu::last_kernel_route()).find("fixed_q16_q8"),
            std::string::npos);

        cpu::set_dispatch_mode(cpu::DispatchMode::OpenCVUIOnly);
        cpu::reset_last_dispatch_tag();
        Mat ui_only;
        resize(
            source, ui_only, Size(dst_cols, dst_rows),
            0.0, 0.0, INTER_LINEAR);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
        EXPECT_NE(
            std::string(cpu::last_kernel_route()).find("fixed_q16_q8"),
            std::string::npos);
        expect_resize_equal(scalar, ui_only);

        cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        Mat accelerated;
        resize(
            source, accelerated, Size(dst_cols, dst_rows),
            0.0, 0.0, INTER_LINEAR);
        const bool expects_neon =
            test_case.expects_vector_path && cpu::neon_runtime_available();
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            expects_neon
                ? cpu::DispatchTag::NEON
                : cpu::DispatchTag::Scalar);
        if (expects_neon)
        {
            EXPECT_NE(
                std::string(cpu::last_kernel_route()).find("layout=flat_c3"),
                std::string::npos);
            EXPECT_NE(
                std::string(cpu::last_kernel_route()).find("tail=fixed_scalar"),
                std::string::npos);
        }
        expect_resize_equal(scalar, accelerated);

        cpu::set_dispatch_mode(cpu::DispatchMode::NeonOnly);
        cpu::reset_last_dispatch_tag();
        Mat neon_only;
        resize(
            source, neon_only, Size(dst_cols, dst_rows),
            0.0, 0.0, INTER_LINEAR);
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            expects_neon
                ? cpu::DispatchTag::NEON
                : cpu::DispatchTag::Scalar);
        if (expects_neon)
        {
            EXPECT_NE(
                std::string(cpu::last_kernel_route()).find("layout=flat_c3"),
                std::string::npos);
        }
        expect_resize_equal(scalar, neon_only);
    }

    // Exercise every narrow/block/tail layout produced by exact floor-3/4
    // U8C3 widths around the first few 16-byte vector boundaries.
    for (int source_cols = 2; source_cols <= 65; ++source_cols)
    {
        constexpr int source_rows = 40;
        const int dst_rows = source_rows * 3 / 4;
        const int dst_cols = source_cols * 3 / 4;
        Mat source({source_rows, source_cols}, CV_8UC3);
        fill_resize_pattern(source);

        cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
        Mat scalar;
        resize(
            source, scalar, Size(dst_cols, dst_rows),
            0.0, 0.0, INTER_LINEAR);

        cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
        Mat accelerated;
        resize(
            source, accelerated, Size(dst_cols, dst_rows),
            0.0, 0.0, INTER_LINEAR);
        expect_resize_equal(scalar, accelerated);
    }
}

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
