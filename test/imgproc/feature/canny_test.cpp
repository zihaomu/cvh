#include "test/imgproc/support/canny_test_utils.hpp"

TEST(CannyTest, canny_u8_matches_reference_for_aperture3_modes)
{
    Mat src({63, 97}, CV_8UC1);
    fill_u8_pattern(src, 0x12345678u);

    for (bool l2 : {false, true})
    {
        Mat actual;
        Canny(src, actual, 50.0, 130.0, 3, l2);
        const Mat expected = canny_reference(src, 50.0, 130.0, 3, l2);
        EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
    }
}

TEST(CannyTest, canny_u8_matches_reference_for_aperture5_modes)
{
    Mat src({65, 89}, CV_8UC1);
    fill_u8_pattern(src, 0x89abcdefu);

    for (bool l2 : {false, true})
    {
        Mat actual;
        Canny(src, actual, 120.0, 340.0, 5, l2);
        const Mat expected = canny_reference(src, 120.0, 340.0, 5, l2);
        EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
    }
}

TEST(CannyTest, canny_derivative_overload_matches_reference)
{
    Mat src({57, 91}, CV_8UC1);
    fill_u8_pattern(src, 0xfeedbeefu);

    Mat dx;
    Mat dy;
    sobel_reference_u8_to_s16(src, dx, dy, 3);

    Mat actual;
    Canny(dx, dy, actual, 40.0, 110.0, false);
    const Mat expected = canny_reference_from_derivatives(dx, dy, 40.0, 110.0, false);
    EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
}

TEST(CannyTest, non_contiguous_roi_matches_reference)
{
    Mat src_full({80, 120}, CV_8UC1);
    fill_u8_pattern(src_full, 0x11223344u);
    Mat roi = src_full(Range(7, 71), Range(11, 109));

    Mat actual;
    Canny(roi, actual, 70.0, 160.0, 3, true);
    const Mat expected = canny_reference(roi, 70.0, 160.0, 3, true);
    EXPECT_EQ(max_abs_diff_u8(actual, expected), 0);
}

TEST(CannyTest, invalid_arguments_throw)
{
    Mat src_u8({8, 9}, CV_8UC1);
    fill_u8_pattern(src_u8, 0x42u);
    Mat src_u8c3({8, 9}, CV_8UC3);
    Mat src_u16({8, 9}, CV_16UC1);
    Mat empty;
    Mat out;

    EXPECT_THROW(Canny(empty, out, 10.0, 20.0, 3, false), Exception);
    EXPECT_THROW(Canny(src_u8c3, out, 10.0, 20.0, 3, false), Exception);
    EXPECT_THROW(Canny(src_u16, out, 10.0, 20.0, 3, false), Exception);
    EXPECT_THROW(Canny(src_u8, out, 10.0, 20.0, 7, false), Exception);

    Mat dx({8, 9}, CV_16SC1);
    Mat dy_bad_type({8, 9}, CV_8UC1);
    Mat dy_bad_size({9, 8}, CV_16SC1);
    Mat dx_c3({8, 9}, CV_16SC3);

    EXPECT_THROW(Canny(dx, dy_bad_type, out, 10.0, 20.0, false), Exception);
    EXPECT_THROW(Canny(dx, dy_bad_size, out, 10.0, 20.0, false), Exception);
    EXPECT_THROW(Canny(dx_c3, dx, out, 10.0, 20.0, false), Exception);
}
