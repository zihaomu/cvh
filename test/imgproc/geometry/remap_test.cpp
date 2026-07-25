#include "test/imgproc/support/geometric_sampling_test_utils.hpp"

TEST(RemapTest, remap_identity_and_alias_are_exact)
{
    Mat source({5, 7}, CV_8UC3);
    fill_u8(source);
    Mat map({5, 7}, CV_32FC2);
    for (int row = 0; row < 5; ++row)
    {
        for (int col = 0; col < 7; ++col)
        {
            map.at<float>(row, col, 0) = static_cast<float>(col);
            map.at<float>(row, col, 1) = static_cast<float>(row);
        }
    }
    Mat output;
    remap(
        source,
        output,
        map,
        Mat(),
        INTER_NEAREST,
        BORDER_CONSTANT);
    EXPECT_EQ(max_u8_difference(source, output), 0);

    remap(
        source,
        source,
        map,
        Mat(),
        INTER_NEAREST,
        BORDER_CONSTANT);
    EXPECT_EQ(max_u8_difference(source, output), 0);
}

TEST(RemapTest, map_representations_produce_same_linear_result)
{
    Mat parent({9, 12}, CV_8UC4);
    fill_u8(parent);
    Mat source = parent(Range(1, 8), Range(2, 11));
    ASSERT_FALSE(source.isContinuous());
    Mat map_x;
    Mat map_y;
    make_maps(map_x, map_y, 7, 9);

    Mat interleaved;
    Mat unused;
    convertMaps(
        map_x,
        map_y,
        interleaved,
        unused,
        CV_32FC2);
    ASSERT_TRUE(unused.empty());

    Mat fixed_coordinates;
    Mat fixed_fractions;
    convertMaps(
        interleaved,
        Mat(),
        fixed_coordinates,
        fixed_fractions,
        CV_16SC2);
    EXPECT_EQ(fixed_coordinates.type(), CV_16SC2);
    EXPECT_EQ(fixed_fractions.type(), CV_16UC1);

    Mat from_pair;
    Mat from_interleaved;
    Mat from_fixed;
    remap(
        source,
        from_pair,
        map_x,
        map_y,
        INTER_LINEAR,
        BORDER_REFLECT_101);
    remap(
        source,
        from_interleaved,
        interleaved,
        Mat(),
        INTER_LINEAR,
        BORDER_REFLECT_101);
    remap(
        source,
        from_fixed,
        fixed_coordinates,
        fixed_fractions,
        INTER_LINEAR,
        BORDER_REFLECT_101);
    EXPECT_EQ(max_u8_difference(from_pair, from_interleaved), 0);
    EXPECT_EQ(max_u8_difference(from_pair, from_fixed), 0);

    Mat restored_x;
    Mat restored_y;
    convertMaps(
        fixed_coordinates,
        fixed_fractions,
        restored_x,
        restored_y,
        CV_32FC1);
    for (int row = 0; row < map_x.size[0]; ++row)
    {
        for (int col = 0; col < map_x.size[1]; ++col)
        {
            EXPECT_NEAR(
                restored_x.at<float>(row, col),
                map_x.at<float>(row, col),
                1.0f / INTER_TAB_SIZE);
            EXPECT_NEAR(
                restored_y.at<float>(row, col),
                map_y.at<float>(row, col),
                1.0f / INTER_TAB_SIZE);
        }
    }
}

TEST(RemapTest, shared_fixed_sampler_covers_channels_roi_and_borders)
{
    for (const int channels : {1, 3, 4})
    {
        Mat parent({9, 12}, CV_MAKETYPE(CV_8U, channels));
        fill_u8(parent);
        Mat source = parent(Range(1, 8), Range(2, 11));
        ASSERT_FALSE(source.isContinuous());

        Mat map_x;
        Mat map_y;
        make_maps(map_x, map_y, 7, 9);
        Mat fixed_coordinates;
        Mat fixed_fractions;
        convertMaps(
            map_x,
            map_y,
            fixed_coordinates,
            fixed_fractions,
            CV_16SC2);

        for (const int border_type :
             {BORDER_CONSTANT,
              BORDER_REPLICATE,
              BORDER_REFLECT,
              BORDER_REFLECT_101})
        {
            Mat from_float;
            Mat from_fixed;
            const Scalar border_value(7.0, 11.0, 19.0, 23.0);
            remap(
                source,
                from_float,
                map_x,
                map_y,
                INTER_LINEAR,
                border_type,
                border_value);
            remap(
                source,
                from_fixed,
                fixed_coordinates,
                fixed_fractions,
                INTER_LINEAR,
                border_type,
                border_value);
            EXPECT_EQ(
                max_u8_difference(from_float, from_fixed),
                0);
        }
    }
}

TEST(RemapTest, nearest_fixed_map_and_border_modes_are_defined)
{
    Mat source({2, 3}, CV_32FC1);
    for (int row = 0; row < 2; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            source.at<float>(row, col) =
                static_cast<float>(row * 10 + col);
        }
    }
    Mat map_x({1, 4}, CV_32FC1);
    Mat map_y({1, 4}, CV_32FC1);
    const float coordinates[] = {-1.0f, 0.49f, 1.51f, 4.0f};
    for (int col = 0; col < 4; ++col)
    {
        map_x.at<float>(0, col) = coordinates[col];
        map_y.at<float>(0, col) = 0.0f;
    }
    Mat fixed;
    Mat fractions;
    convertMaps(
        map_x,
        map_y,
        fixed,
        fractions,
        CV_16SC2,
        true);
    EXPECT_TRUE(fractions.empty());

    Mat output;
    remap(
        source,
        output,
        fixed,
        fractions,
        INTER_NEAREST,
        BORDER_CONSTANT,
        Scalar::all(99.0));
    EXPECT_FLOAT_EQ(output.at<float>(0, 0), 99.0f);
    EXPECT_FLOAT_EQ(output.at<float>(0, 1), 0.0f);
    EXPECT_FLOAT_EQ(output.at<float>(0, 2), 2.0f);
    EXPECT_FLOAT_EQ(output.at<float>(0, 3), 99.0f);
}

TEST(RemapTest, rejects_unsupported_map_type)
{
    Mat source({4, 5}, CV_8UC1);
    Mat bad_map({4, 5}, CV_8UC2);
    Mat output;

    EXPECT_THROW(
        remap(
            source,
            output,
            bad_map,
            Mat(),
            INTER_LINEAR),
        Exception);
}
