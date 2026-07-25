#include "test/imgproc/support/geometric_sampling_test_utils.hpp"

TEST(WarpPerspectiveTest, warp_perspective_identity_inverse_and_alias)
{
    Mat source({7, 9}, CV_8UC1);
    fill_u8(source);
    const Mat identity = identity_perspective(CV_64F);
    Mat output;
    warpPerspective(
        source,
        output,
        identity,
        Size(9, 7),
        INTER_NEAREST);
    EXPECT_EQ(max_u8_difference(source, output), 0);

    Mat inverse_map = identity.clone();
    inverse_map.at<double>(0, 2) = -1.0;
    Mat from_inverse;
    warpPerspective(
        source,
        from_inverse,
        inverse_map,
        Size(9, 7),
        INTER_NEAREST | WARP_INVERSE_MAP,
        BORDER_REPLICATE);
    Mat forward = identity.clone();
    forward.at<double>(0, 2) = 1.0;
    Mat from_forward;
    warpPerspective(
        source,
        from_forward,
        forward,
        Size(9, 7),
        INTER_NEAREST,
        BORDER_REPLICATE);
    EXPECT_EQ(max_u8_difference(from_inverse, from_forward), 0);

    warpPerspective(
        source,
        source,
        identity,
        Size(9, 7),
        INTER_LINEAR,
        BORDER_REFLECT);
    EXPECT_EQ(max_u8_difference(source, output), 0);
}

TEST(WarpPerspectiveTest, warp_perspective_handles_true_projective_map)
{
    Mat source({8, 10}, CV_32FC3);
    for (int row = 0; row < source.size[0]; ++row)
    {
        for (int col = 0; col < source.size[1]; ++col)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                source.at<float>(row, col, channel) =
                    static_cast<float>(row * 10 + col + channel);
            }
        }
    }
    Mat inverse = identity_perspective(CV_32F);
    inverse.at<float>(0, 1) = 0.1f;
    inverse.at<float>(0, 2) = 0.25f;
    inverse.at<float>(1, 0) = -0.05f;
    inverse.at<float>(1, 2) = 0.5f;
    inverse.at<float>(2, 0) = 0.002f;
    inverse.at<float>(2, 1) = -0.003f;
    Mat output;
    warpPerspective(
        source,
        output,
        inverse,
        Size(7, 5),
        INTER_LINEAR | WARP_INVERSE_MAP,
        BORDER_REFLECT_101);
    EXPECT_EQ(output.type(), CV_32FC3);
    EXPECT_EQ(output.shape(), MatShape({5, 7}));
    EXPECT_TRUE(std::isfinite(output.at<float>(4, 6, 2)));
}

TEST(WarpPerspectiveTest, rejects_singular_transform)
{
    Mat source({4, 5}, CV_8UC1);
    Mat singular({3, 3}, CV_64FC1);
    singular.setTo(Scalar::all(0.0));
    Mat output;

    EXPECT_THROW(
        warpPerspective(
            source,
            output,
            singular,
            Size(5, 4)),
        Exception);
}
