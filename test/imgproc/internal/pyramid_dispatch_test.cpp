#include "test/imgproc/support/pyramid_color_test_utils.hpp"
#include "test/support/dispatch_mode_guard.hpp"

TEST(PyramidDispatchInternalTest, pyramid_ui_matches_scalar_for_types_channels_borders_and_roi)
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
        Mat parent({13, 21}, type);
        for (int y = 0; y < parent.size.p[0]; ++y)
        {
            for (int x = 0; x < parent.size.p[1]; ++x)
            {
                for (int ch = 0; ch < parent.channels(); ++ch)
                {
                    const float value = static_cast<float>(
                        3 + y * 17 + x * 11 + ch * 29);
                    if (parent.depth() == CV_8U)
                    {
                        parent.at<uchar>(y, x, ch) =
                            static_cast<uchar>(
                                static_cast<int>(value) & 255);
                    }
                    else
                    {
                        parent.at<float>(y, x, ch) =
                            value * 0.125f - 7.0f;
                    }
                }
            }
        }
        const Mat source =
            parent(Range(1, 12), Range(2, 19));

        for (const int border : borders)
        {
            Mat scalar_down;
            Mat ui_down;
            cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
            pyrDown(source, scalar_down, Size(), border);
            cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
            pyrDown(source, ui_down, Size(), border);
            ASSERT_EQ(scalar_down.shape(), ui_down.shape());
            ASSERT_EQ(scalar_down.type(), ui_down.type());
            for (int y = 0; y < scalar_down.size.p[0]; ++y)
            {
                for (int x = 0; x < scalar_down.size.p[1]; ++x)
                {
                    for (int ch = 0;
                         ch < scalar_down.channels();
                         ++ch)
                    {
                        if (scalar_down.depth() == CV_8U)
                        {
                            EXPECT_EQ(
                                scalar_down.at<uchar>(y, x, ch),
                                ui_down.at<uchar>(y, x, ch));
                        }
                        else
                        {
                            EXPECT_NEAR(
                                scalar_down.at<float>(y, x, ch),
                                ui_down.at<float>(y, x, ch),
                                1e-4f);
                        }
                    }
                }
            }
        }

        Mat scalar_up;
        Mat ui_up;
        cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
        pyrUp(source, scalar_up);
        cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
        pyrUp(source, ui_up);
        ASSERT_EQ(scalar_up.shape(), ui_up.shape());
        for (int y = 0; y < scalar_up.size.p[0]; ++y)
        {
            for (int x = 0; x < scalar_up.size.p[1]; ++x)
            {
                for (int ch = 0; ch < scalar_up.channels(); ++ch)
                {
                    if (scalar_up.depth() == CV_8U)
                    {
                        EXPECT_EQ(
                            scalar_up.at<uchar>(y, x, ch),
                            ui_up.at<uchar>(y, x, ch));
                    }
                    else
                    {
                        EXPECT_NEAR(
                            scalar_up.at<float>(y, x, ch),
                            ui_up.at<float>(y, x, ch),
                            1e-4f);
                    }
                }
            }
        }
    }

    Mat alias_source({9, 13}, CV_8UC3);
    for (size_t index = 0;
         index < alias_source.total() * alias_source.elemSize();
         ++index)
    {
        alias_source.data[index] =
            static_cast<uchar>((index * 37u + 11u) & 255u);
    }
    Mat expected;
    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
    pyrDown(alias_source, expected);
    Mat alias = alias_source.clone();
    pyrDown(alias, alias);
    ASSERT_EQ(alias.shape(), expected.shape());
    EXPECT_EQ(
        std::memcmp(
            alias.data,
            expected.data,
            expected.total() * expected.elemSize()),
        0);

}
