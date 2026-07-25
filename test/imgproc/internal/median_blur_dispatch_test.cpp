#include "test/imgproc/support/intensity_filter_test_utils.hpp"
#include "test/support/dispatch_mode_guard.hpp"

TEST(MedianBlurDispatchInternalTest, median_blur_ui_matches_scalar_for_u8_channels_and_roi)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    for (const int type : {CV_8UC1, CV_8UC3, CV_8UC4})
    {
        Mat parent({11, 19}, type);
        fill_pattern(parent);
        const Mat roi =
            parent(Range(1, 10), Range(2, 18));
        for (const int ksize : {3, 5})
        {
            Mat scalar;
            Mat accelerated;
            cpu::set_dispatch_mode(
                cpu::DispatchMode::ScalarOnly);
            medianBlur(roi, scalar, ksize);
            cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
            medianBlur(roi, accelerated, ksize);
            ASSERT_EQ(scalar.shape(), accelerated.shape());
            EXPECT_EQ(
                std::memcmp(
                    scalar.data,
                    accelerated.data,
                    scalar.total() * scalar.elemSize()),
                0);

            Mat alias = roi.clone();
            medianBlur(alias, alias, ksize);
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
