#include "test/imgproc/support/intensity_filter_test_utils.hpp"
#include "test/support/dispatch_mode_guard.hpp"

#include <cstring>

TEST(MedianBlurDispatchInternalTest,
     median_blur_records_scalar_ui_and_alias_dispatch)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    for (const int type : {CV_8UC1, CV_8UC3, CV_8UC4})
    {
        Mat parent({11, 137}, type);
        fill_pattern(parent);
        const Mat roi =
            parent(Range(1, 10), Range(3, 134));
        ASSERT_FALSE(roi.isContinuous());

        for (const int ksize : {3, 5})
        {
            Mat scalar;
            Mat accelerated;
            cpu::set_dispatch_mode(
                cpu::DispatchMode::ScalarOnly);
            cpu::reset_last_dispatch_tag();
            medianBlur(roi, scalar, ksize);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                cpu::DispatchTag::Scalar);

            cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
            cpu::reset_last_dispatch_tag();
            medianBlur(roi, accelerated, ksize);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                cvh::test::expected_fixed_width_dispatch_tag());
            ASSERT_EQ(scalar.shape(), accelerated.shape());
            EXPECT_EQ(
                std::memcmp(
                    scalar.data,
                    accelerated.data,
                    scalar.total() * scalar.elemSize()),
                0);

            Mat alias = roi.clone();
            cpu::reset_last_dispatch_tag();
            medianBlur(alias, alias, ksize);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                cvh::test::expected_fixed_width_dispatch_tag());
            EXPECT_EQ(
                std::memcmp(
                    accelerated.data,
                    alias.data,
                    accelerated.total() *
                        accelerated.elemSize()),
                0);
        }
    }
}

TEST(MedianBlurDispatchInternalTest,
     median_blur_short_and_float_inputs_record_scalar_fallback)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard(
        cpu::DispatchMode::Auto);

    Mat short_source({7, 5}, CV_8UC1);
    fill_pattern(short_source);
    Mat output;
    cpu::reset_last_dispatch_tag();
    medianBlur(short_source, output, 5);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::DispatchTag::Scalar);

    Mat float_source({7, 67}, CV_32FC1);
    for (int y = 0; y < float_source.size.p[0]; ++y)
    {
        for (int x = 0; x < float_source.size.p[1]; ++x)
        {
            float_source.at<float>(y, x) =
                static_cast<float>(y * 17 + x * 3) * 0.25f;
        }
    }
    cpu::reset_last_dispatch_tag();
    medianBlur(float_source, output, 3);
    EXPECT_EQ(
        cpu::last_dispatch_tag(),
        cpu::DispatchTag::Scalar);
}
