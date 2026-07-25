#include "test/imgproc/support/kernel_family_test_utils.hpp"
#include "test/support/dispatch_mode_guard.hpp"

TEST(DerivativesDispatchInternalTest, derivative_s16_ui_matches_scalar_for_borders_and_tails)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    const auto max_s16_difference = [](const Mat& first,
                                       const Mat& second) {
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
    };
    for (const int cols : {5, 8, 9, 17})
    {
        Mat parent({9, cols + 3}, CV_8UC1);
        for (int y = 0; y < parent.size.p[0]; ++y)
        {
            for (int x = 0; x < parent.size.p[1]; ++x)
            {
                parent.at<uchar>(y, x) =
                    static_cast<uchar>((y * 37 + x * 19) & 255);
            }
        }
        Mat source = parent(Range(1, 8), Range(1, cols + 1));
        ASSERT_FALSE(source.isContinuous());
        for (const int border_type :
             {BORDER_CONSTANT,
              BORDER_REPLICATE,
              BORDER_REFLECT,
              BORDER_REFLECT_101})
        {
            Mat scalar_scharr;
            Mat scalar_laplacian;
            Mat ui_scharr;
            Mat ui_laplacian;
            cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
            Scharr(
                source,
                scalar_scharr,
                CV_16S,
                1,
                0,
                0.75,
                3.0,
                border_type | BORDER_ISOLATED);
            Laplacian(
                source,
                scalar_laplacian,
                CV_16S,
                3,
                0.75,
                3.0,
                border_type | BORDER_ISOLATED);
            cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
            Scharr(
                source,
                ui_scharr,
                CV_16S,
                1,
                0,
                0.75,
                3.0,
                border_type | BORDER_ISOLATED);
            Laplacian(
                source,
                ui_laplacian,
                CV_16S,
                3,
                0.75,
                3.0,
                border_type | BORDER_ISOLATED);
            EXPECT_LE(
                max_s16_difference(scalar_scharr, ui_scharr),
                1);
            EXPECT_EQ(
                std::memcmp(
                    scalar_laplacian.data,
                    ui_laplacian.data,
                    scalar_laplacian.total() *
                        scalar_laplacian.elemSize()),
                0);
        }
    }
}
