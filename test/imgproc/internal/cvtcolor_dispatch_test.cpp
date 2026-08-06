#include "test/imgproc/support/kernel_family_test_utils.hpp"
#include "test/support/dispatch_mode_guard.hpp"

#include "cvh/core/detail/cpu_features.hpp"

#include <cstring>
#include <string>
#include <tuple>
#include <vector>

namespace {

void fill_color_pattern(Mat& mat)
{
    for (int y = 0; y < mat.size[0]; ++y)
    {
        uchar* row = mat.data + static_cast<std::size_t>(y) * mat.step(0);
        for (std::size_t x = 0; x <
             static_cast<std::size_t>(mat.size[1]) * mat.elemSize(); ++x)
        {
            row[x] = static_cast<uchar>((y * 37 + x * 19 + 11) & 255);
        }
    }
}

void expect_u8_mat_equal(const Mat& expected, const Mat& actual)
{
    ASSERT_EQ(expected.type(), actual.type());
    ASSERT_EQ(expected.shape(), actual.shape());
    const std::size_t row_bytes =
        static_cast<std::size_t>(expected.size[1]) * expected.elemSize();
    for (int y = 0; y < expected.size[0]; ++y)
    {
        const uchar* expected_row = expected.data +
            static_cast<std::size_t>(y) * expected.step(0);
        const uchar* actual_row = actual.data +
            static_cast<std::size_t>(y) * actual.step(0);
        if (std::memcmp(expected_row, actual_row, row_bytes) != 0)
        {
            for (std::size_t x = 0; x < row_bytes; ++x)
            {
                ASSERT_EQ(expected_row[x], actual_row[x])
                    << "row=" << y << ", byte=" << x;
            }
        }
    }
}

}  // namespace

TEST(CvtColorDispatchInternalTest,
     packed_u8_auto_matches_scalar_for_roi_and_tail)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    const std::vector<std::tuple<int, int>> cases = {
        {CV_8UC3, COLOR_BGR2RGB},
        {CV_8UC3, COLOR_BGR2BGRA},
        {CV_8UC3, COLOR_BGR2RGBA},
        {CV_8UC4, COLOR_BGRA2BGR},
        {CV_8UC4, COLOR_BGRA2RGB},
        {CV_8UC4, COLOR_BGRA2RGBA},
        {CV_8UC4, COLOR_BGRA2GRAY},
        {CV_8UC4, COLOR_RGBA2GRAY},
        {CV_8UC1, COLOR_GRAY2BGR},
        {CV_8UC1, COLOR_GRAY2BGRA},
    };

    for (const auto& [type, code] : cases)
    {
        Mat parent({13, 44}, type);
        fill_color_pattern(parent);
        const Mat source = parent(Range(1, 12), Range(3, 40));
        ASSERT_FALSE(source.isContinuous());

        cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
        cpu::reset_last_dispatch_tag();
        Mat scalar;
        cvtColor(source, scalar, code);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

        cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        Mat accelerated;
        cvtColor(source, accelerated, code);

        if (cpu::neon_runtime_available())
        {
            EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::NEON);
            EXPECT_NE(
                std::string(cpu::last_kernel_route()).find("cvtcolor_packed"),
                std::string::npos);
        }
        else
        {
            EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
        }
        expect_u8_mat_equal(scalar, accelerated);
    }
}

TEST(CvtColorDispatchInternalTest,
     forced_modes_and_small_workload_do_not_misreport_neon)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat source({9, 37}, CV_8UC3);
    fill_color_pattern(source);
    Mat output;

    cpu::set_dispatch_mode(cpu::DispatchMode::OpenCVUIOnly);
    cpu::reset_last_dispatch_tag();
    cvtColor(source, output, COLOR_BGR2RGB);
    EXPECT_NE(cpu::last_dispatch_tag(), cpu::DispatchTag::NEON);

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    cpu::reset_last_dispatch_tag();
    cvtColor(source, output, COLOR_BGR2RGB);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

    cpu::set_dispatch_mode(cpu::DispatchMode::NeonOnly);
    cpu::reset_last_dispatch_tag();
    cvtColor(source, output, COLOR_BGR2RGB);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::neon_runtime_available()
            ? cpu::DispatchTag::NEON
            : cpu::DispatchTag::Scalar);

    Mat short_source({3, 17}, CV_8UC3);
    fill_color_pattern(short_source);
    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
    cpu::reset_last_dispatch_tag();
    cvtColor(short_source, output, COLOR_BGR2RGB);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
}

TEST(CvtColorDispatchInternalTest,
     existing_bgr_gray_ui_route_is_not_replaced)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard(
        cpu::DispatchMode::Auto);
    Mat source({17, 37}, CV_8UC3);
    fill_color_pattern(source);
    Mat output;
    cpu::reset_last_dispatch_tag();
    cvtColor(source, output, COLOR_BGR2GRAY);
    EXPECT_NE(cpu::last_dispatch_tag(), cpu::DispatchTag::NEON);
}

TEST(CvtColorDispatchInternalTest,
     yuv_decode_neon_matches_scalar_for_non_contiguous_step)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat color({18, 34}, CV_8UC3);
    fill_color_pattern(color);

    const std::vector<std::tuple<int, int, int>> cases = {
        {COLOR_BGR2YUV_NV12, COLOR_YUV2BGR_NV12, CV_8UC1},
        {COLOR_BGR2YUV_I420, COLOR_YUV2BGR_I420, CV_8UC1},
        {COLOR_BGR2YUV_YUY2, COLOR_YUV2BGR_YUY2, CV_8UC2},
    };

    for (const auto& [encode_code, decode_code, encoded_type] : cases)
    {
        cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
        Mat encoded;
        cvtColor(color, encoded, encode_code);

        Mat parent(
            {encoded.size[0], encoded.size[1] + 5}, encoded_type);
        fill_color_pattern(parent);
        Mat encoded_roi = parent(
            Range::all(), Range(2, 2 + encoded.size[1]));
        ASSERT_FALSE(encoded_roi.isContinuous());
        const std::size_t row_bytes =
            static_cast<std::size_t>(encoded.size[1]) * encoded.elemSize();
        for (int y = 0; y < encoded.size[0]; ++y)
        {
            std::memcpy(
                encoded_roi.data + static_cast<std::size_t>(y) * encoded_roi.step(0),
                encoded.data + static_cast<std::size_t>(y) * encoded.step(0),
                row_bytes);
        }

        cpu::reset_last_dispatch_tag();
        Mat scalar;
        cvtColor(encoded_roi, scalar, decode_code);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

        cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        Mat accelerated;
        cvtColor(encoded_roi, accelerated, decode_code);
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            cpu::neon_runtime_available()
                ? cpu::DispatchTag::NEON
                : cpu::DispatchTag::Scalar);
        if (cpu::neon_runtime_available())
        {
            EXPECT_NE(
                std::string(cpu::last_kernel_route()).find("decode"),
                std::string::npos);
        }
        expect_u8_mat_equal(scalar, accelerated);
    }
}

TEST(CvtColorDispatchInternalTest,
     yuv_encode_neon_matches_scalar_for_roi_and_tail)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat parent({18, 41}, CV_8UC3);
    fill_color_pattern(parent);
    const Mat color = parent(Range::all(), Range(2, 36));
    ASSERT_FALSE(color.isContinuous());

    for (const int code : {
             COLOR_BGR2YUV_NV12,
             COLOR_BGR2YUV_I420,
             COLOR_BGR2YUV_YUY2})
    {
        cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
        cpu::reset_last_dispatch_tag();
        Mat scalar;
        cvtColor(color, scalar, code);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

        cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        Mat accelerated;
        cvtColor(color, accelerated, code);
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            cpu::neon_runtime_available()
                ? cpu::DispatchTag::NEON
                : cpu::DispatchTag::Scalar);
        if (cpu::neon_runtime_available())
        {
            EXPECT_NE(
                std::string(cpu::last_kernel_route()).find("encode"),
                std::string::npos);
        }
        expect_u8_mat_equal(scalar, accelerated);
    }
}

TEST(CvtColorDispatchInternalTest,
     interleaved_yuv444_neon_preserves_float_rounding_contract)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat parent({67, 263}, CV_8UC3);
    unsigned state = 0x13579bdfu;
    for (int y = 0; y < parent.size[0]; ++y)
    {
        uchar* row = parent.data + static_cast<std::size_t>(y) * parent.step(0);
        for (std::size_t x = 0;
             x < static_cast<std::size_t>(parent.size[1]) * parent.elemSize();
             ++x)
        {
            state = state * 1664525u + 1013904223u;
            row[x] = static_cast<uchar>(state >> 24);
        }
    }
    const Mat color = parent(Range::all(), Range(3, 260));
    ASSERT_FALSE(color.isContinuous());

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    Mat scalar_yuv;
    cvtColor(color, scalar_yuv, COLOR_BGR2YUV);
    Mat scalar_bgr;
    cvtColor(scalar_yuv, scalar_bgr, COLOR_YUV2BGR);

    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
    cpu::reset_last_dispatch_tag();
    Mat neon_yuv;
    cvtColor(color, neon_yuv, COLOR_BGR2YUV);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::neon_runtime_available()
            ? cpu::DispatchTag::NEON
            : cpu::DispatchTag::Scalar);
    {
        SCOPED_TRACE("BGR2YUV");
        expect_u8_mat_equal(scalar_yuv, neon_yuv);
    }

    cpu::reset_last_dispatch_tag();
    Mat neon_bgr;
    cvtColor(scalar_yuv, neon_bgr, COLOR_YUV2BGR);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::neon_runtime_available()
            ? cpu::DispatchTag::NEON
            : cpu::DispatchTag::Scalar);
    {
        SCOPED_TRACE("YUV2BGR");
        expect_u8_mat_equal(scalar_bgr, neon_bgr);
    }
}
