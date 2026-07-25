#include "test/imgproc/support/pyramid_color_test_utils.hpp"
#include "test/support/dispatch_mode_guard.hpp"

#include <cstring>
#include <vector>

namespace {

void fill_pyramid_pattern(Mat& mat)
{
    for (int y = 0; y < mat.size.p[0]; ++y)
    {
        for (int x = 0; x < mat.size.p[1]; ++x)
        {
            for (int channel = 0;
                 channel < mat.channels();
                 ++channel)
            {
                const float value = static_cast<float>(
                    3 + y * 17 + x * 11 + channel * 29);
                if (mat.depth() == CV_8U)
                {
                    mat.at<uchar>(y, x, channel) =
                        static_cast<uchar>(
                            static_cast<int>(value) & 255);
                }
                else
                {
                    mat.at<float>(y, x, channel) =
                        value * 0.125f - 7.0f;
                }
            }
        }
    }
}

void expect_pyramid_mat_equal(
    const Mat& expected,
    const Mat& actual)
{
    ASSERT_EQ(expected.shape(), actual.shape());
    ASSERT_EQ(expected.type(), actual.type());
    for (int y = 0; y < expected.size.p[0]; ++y)
    {
        for (int x = 0; x < expected.size.p[1]; ++x)
        {
            for (int channel = 0;
                 channel < expected.channels();
                 ++channel)
            {
                if (expected.depth() == CV_8U)
                {
                    EXPECT_EQ(
                        expected.at<uchar>(y, x, channel),
                        actual.at<uchar>(y, x, channel));
                }
                else
                {
                    EXPECT_NEAR(
                        expected.at<float>(y, x, channel),
                        actual.at<float>(y, x, channel),
                        1e-4f);
                }
            }
        }
    }
}

}  // namespace

TEST(PyramidDispatchInternalTest,
     pyramid_records_scalar_and_ui_for_types_borders_roi_and_tail)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    const int borders[] = {
        BORDER_REPLICATE,
        BORDER_REFLECT,
        BORDER_REFLECT_101,
        BORDER_WRAP,
    };
    const int types[] = {
        CV_8UC1,
        CV_8UC3,
        CV_8UC4,
        CV_32FC1,
        CV_32FC3,
        CV_32FC4,
    };

    for (const int type : types)
    {
        Mat parent({13, 137}, type);
        fill_pyramid_pattern(parent);
        const Mat source =
            parent(Range(1, 12), Range(3, 134));
        ASSERT_FALSE(source.isContinuous());

        for (const int border : borders)
        {
            Mat scalar_down;
            Mat auto_down;
            cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
            cpu::reset_last_dispatch_tag();
            pyrDown(source, scalar_down, Size(), border);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                cpu::DispatchTag::Scalar);

            cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
            cpu::reset_last_dispatch_tag();
            pyrDown(source, auto_down, Size(), border);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                cvh::test::expected_fixed_width_dispatch_tag());
            expect_pyramid_mat_equal(scalar_down, auto_down);
        }

        Mat scalar_up;
        Mat auto_up;
        cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
        cpu::reset_last_dispatch_tag();
        pyrUp(source, scalar_up);
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            cpu::DispatchTag::Scalar);

        cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        pyrUp(source, auto_up);
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            cvh::test::expected_fixed_width_dispatch_tag());
        expect_pyramid_mat_equal(scalar_up, auto_up);
    }
}

TEST(PyramidDispatchInternalTest,
     pyramid_short_alias_and_build_pyramid_have_deterministic_tags)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard(
        cpu::DispatchMode::Auto);

    Mat short_source({1, 1}, CV_8UC1);
    short_source.at<uchar>(0, 0) = 17;
    Mat short_output;
    cpu::reset_last_dispatch_tag();
    pyrDown(short_source, short_output);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::DispatchTag::Scalar);
    cpu::reset_last_dispatch_tag();
    pyrUp(short_source, short_output);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::DispatchTag::Scalar);

    Mat alias_source({9, 131}, CV_8UC3);
    fill_pyramid_pattern(alias_source);
    Mat expected;
    cpu::reset_last_dispatch_tag();
    pyrDown(alias_source, expected);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cvh::test::expected_fixed_width_dispatch_tag());
    Mat alias = alias_source.clone();
    cpu::reset_last_dispatch_tag();
    pyrDown(alias, alias);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cvh::test::expected_fixed_width_dispatch_tag());
    expect_pyramid_mat_equal(expected, alias);

    Mat source({13, 131}, CV_8UC1);
    fill_pyramid_pattern(source);
    std::vector<Mat> scalar_levels;
    std::vector<Mat> auto_levels;
    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    cpu::reset_last_dispatch_tag();
    buildPyramid(source, scalar_levels, 2);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::DispatchTag::Scalar);
    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
    cpu::reset_last_dispatch_tag();
    buildPyramid(source, auto_levels, 2);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cvh::test::expected_fixed_width_dispatch_tag());
    ASSERT_EQ(scalar_levels.size(), auto_levels.size());
    for (size_t level = 0; level < scalar_levels.size(); ++level)
    {
        expect_pyramid_mat_equal(
            scalar_levels[level],
            auto_levels[level]);
    }
}
