#include "test/imgproc/support/geometric_sampling_test_utils.hpp"

TEST(ConvertMapsTest, rejects_mismatched_float_map_shapes)
{
    Mat map_x({4, 5}, CV_32FC1);
    Mat map_y({3, 5}, CV_32FC1);
    Mat coordinates;
    Mat fractions;

    EXPECT_THROW(
        convertMaps(
            map_x,
            map_y,
            coordinates,
            fractions,
            CV_16SC2),
        Exception);
}
