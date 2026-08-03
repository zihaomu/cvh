#include "cvh/highgui/highgui.h"
#include "gtest/gtest.h"

#include <string>

using namespace cvh;

TEST(HighguiContract_TEST, named_window_lifecycle_is_idempotent)
{
    EXPECT_NO_THROW(
        namedWindow("lifecycle", WINDOW_AUTOSIZE));
    EXPECT_NO_THROW(
        namedWindow("lifecycle", WINDOW_AUTOSIZE));
    EXPECT_NO_THROW(destroyWindow("lifecycle"));
    EXPECT_NO_THROW(destroyWindow("lifecycle"));
    EXPECT_NO_THROW(destroyAllWindows());
}

TEST(HighguiContract_TEST, imshow_accepts_u8_gray_bgr_and_bgra)
{
    Mat gray({2, 3}, CV_8UC1);
    Mat bgr({2, 3}, CV_8UC3);
    Mat bgra({2, 3}, CV_8UC4);
    gray = 1;
    bgr = 2;
    bgra = 3;

    EXPECT_NO_THROW(imshow("gray", gray));
    EXPECT_NO_THROW(imshow("bgr", bgr));
    EXPECT_NO_THROW(imshow("bgra", bgra));
    destroyAllWindows();
}

TEST(HighguiContract_TEST, imshow_rejects_invalid_images)
{
    Mat empty;
    Mat unsupported_depth({2, 2}, CV_32FC1);
    Mat unsupported_channels({2, 2}, CV_8UC2);

    EXPECT_THROW(imshow("empty", empty), Exception);
    EXPECT_THROW(
        imshow("unsupported_depth", unsupported_depth),
        Exception);
    EXPECT_THROW(
        imshow("unsupported_channels", unsupported_channels),
        Exception);
}

TEST(HighguiContract_TEST, window_names_and_flags_are_validated)
{
    EXPECT_THROW(namedWindow(""), Exception);
    EXPECT_THROW(namedWindow("bad_flags", 17), Exception);

    Mat image({1, 1}, CV_8UC1);
    EXPECT_THROW(imshow("", image), Exception);
    EXPECT_THROW(destroyWindow(""), Exception);
}

TEST(HighguiContract_TEST, wait_key_is_noninteractive_in_headless_mode)
{
    EXPECT_EQ(waitKey(1), -1);
}
