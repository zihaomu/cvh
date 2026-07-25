#include "test/imgproc/support/canny_test_utils.hpp"

TEST(CannyUpstreamTest, Canny_Modes_accuracy_subset_fixed)
{
    require_fixture("cv/shared/fruits.png");
    const Mat original = imread(fixture_path("cv/shared/fruits.png"), IMREAD_GRAYSCALE);
    ASSERT_FALSE(original.empty());
    ASSERT_EQ(original.type(), CV_8UC1);

    struct ModeCase
    {
        int aperture = 3;
        bool l2 = false;
        double t1 = 0.0;
        double t2 = 0.0;
        Size size;
    };

    const std::vector<ModeCase> cases = {
        {3, false, 60.0, 150.0, Size(320, 240)},
        {3, true, 90.0, 180.0, Size(317, 233)},
        {5, false, 220.0, 520.0, Size(401, 257)},
        {5, true, 180.0, 480.0, Size(257, 193)},
    };

    for (const ModeCase& c : cases)
    {
        SCOPED_TRACE(cvh::format("aperture=%d l2=%d size=%dx%d", c.aperture, c.l2 ? 1 : 0, c.size.width, c.size.height));
        Mat img;
        resize(original, img, c.size, 0.0, 0.0, INTER_LINEAR);
        GaussianBlur(img, img, Size(5, 5), 0.0, 0.0, BORDER_REPLICATE);

        Mat result;
        Canny(img, result, c.t1, c.t2, c.aperture, c.l2);

        Mat dx;
        Mat dy;
        Sobel(img, dx, CV_16S, 1, 0, c.aperture, 1.0, 0.0, BORDER_REPLICATE);
        Sobel(img, dy, CV_16S, 0, 1, c.aperture, 1.0, 0.0, BORDER_REPLICATE);
        Mat custom_result;
        Canny(dx, dy, custom_result, c.t1, c.t2, c.l2);

        const Mat reference = canny_reference(img, c.t1, c.t2, c.aperture, c.l2);
        EXPECT_EQ(max_abs_diff_u8(result, reference), 0);
        EXPECT_EQ(max_abs_diff_u8(custom_result, reference), 0);
    }
}
