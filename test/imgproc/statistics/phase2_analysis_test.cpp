#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <vector>

using namespace cvh;

TEST(ConnectedComponentsPhase2Test, labels_stats_and_connectivity_are_stable)
{
    Mat image({4, 6}, CV_8UC1);
    image = 0;
    image.at<uchar>(0, 0) = 255;
    image.at<uchar>(1, 1) = 255;
    image.at<uchar>(1, 4) = 255;
    image.at<uchar>(2, 4) = 255;
    image.at<uchar>(3, 5) = 255;

    Mat labels4;
    EXPECT_EQ(connectedComponents(image, labels4, 4), 5);
    EXPECT_EQ(labels4.at<int>(0, 0), 1);
    EXPECT_EQ(labels4.at<int>(1, 1), 2);
    EXPECT_EQ(labels4.at<int>(1, 4), 3);
    EXPECT_EQ(labels4.at<int>(2, 4), 3);
    EXPECT_EQ(labels4.at<int>(3, 5), 4);

    Mat labels8;
    Mat stats;
    Mat centroids;
    ASSERT_EQ(connectedComponentsWithStats(image, labels8, stats, centroids, 8), 3);
    EXPECT_EQ(labels8.at<int>(0, 0), 1);
    EXPECT_EQ(labels8.at<int>(1, 1), 1);
    EXPECT_EQ(labels8.at<int>(1, 4), 2);
    EXPECT_EQ(labels8.at<int>(3, 5), 2);
    EXPECT_EQ(stats.at<int>(1, CC_STAT_LEFT), 0);
    EXPECT_EQ(stats.at<int>(1, CC_STAT_TOP), 0);
    EXPECT_EQ(stats.at<int>(1, CC_STAT_WIDTH), 2);
    EXPECT_EQ(stats.at<int>(1, CC_STAT_HEIGHT), 2);
    EXPECT_EQ(stats.at<int>(1, CC_STAT_AREA), 2);
    EXPECT_DOUBLE_EQ(centroids.at<double>(1, 0), 0.5);
    EXPECT_DOUBLE_EQ(centroids.at<double>(1, 1), 0.5);

    Mat full_storage({5, 6}, CV_8UC1);
    full_storage = 0;
    Mat full = full_storage(Range(1, 4), Range(1, 5));
    full = 255;
    ASSERT_EQ(connectedComponentsWithStats(full, labels8, stats, centroids, 8), 2);
    EXPECT_EQ(stats.at<int>(0, CC_STAT_LEFT), -1);
    EXPECT_EQ(stats.at<int>(0, CC_STAT_TOP), std::numeric_limits<int>::max());
    EXPECT_EQ(stats.at<int>(0, CC_STAT_AREA), 0);
    EXPECT_TRUE(std::isnan(centroids.at<double>(0, 0)));
    EXPECT_EQ(stats.at<int>(1, CC_STAT_AREA), 12);
    EXPECT_THROW(connectedComponents(full, labels8, 6), Exception);
}

TEST(ContoursPhase2Test, list_and_external_modes_trace_without_modifying_roi)
{
    Mat storage({9, 10}, CV_8UC1);
    storage = 0;
    Mat image = storage(Range(1, 8), Range(1, 9));
    for (int y = 1; y <= 5; ++y)
        for (int x = 1; x <= 6; ++x)
            image.at<uchar>(y, x) = 255;
    for (int y = 2; y <= 4; ++y)
        for (int x = 2; x <= 5; ++x)
            image.at<uchar>(y, x) = 0;
    const Mat before = image.clone();

    std::vector<std::vector<Point>> list;
    findContours(image, list, RETR_LIST, CHAIN_APPROX_SIMPLE, Point(3, -2));
    ASSERT_EQ(list.size(), 2u);
    EXPECT_EQ(list[0].front(), Point(4, 0));
    EXPECT_EQ(contourArea(list[0]), 18.0);
    EXPECT_EQ(contourArea(list[1]), 20.0);

    std::vector<std::vector<Point>> external;
    findContours(image, external, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    ASSERT_EQ(external.size(), 1u);
    EXPECT_EQ(external[0].size(), 18u);
    for (int y = 0; y < image.size[0]; ++y)
        for (int x = 0; x < image.size[1]; ++x)
            EXPECT_EQ(image.at<uchar>(y, x), before.at<uchar>(y, x));

    Mat nested({11, 11}, CV_8UC1);
    nested = 0;
    for (int y = 1; y < 10; ++y)
        for (int x = 1; x < 10; ++x)
            nested.at<uchar>(y, x) = 255;
    for (int y = 3; y < 8; ++y)
        for (int x = 3; x < 8; ++x)
            nested.at<uchar>(y, x) = 0;
    nested.at<uchar>(5, 5) = 255;
    findContours(nested, external, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    EXPECT_EQ(external.size(), 1u);
    findContours(nested, list, RETR_LIST, CHAIN_APPROX_SIMPLE);
    EXPECT_EQ(list.size(), 3u);
    EXPECT_THROW(findContours(nested, list, RETR_TREE, CHAIN_APPROX_SIMPLE), Exception);
}

TEST(ShapePhase2Test, point_geometry_and_moments_cover_degenerate_inputs)
{
    const std::vector<Point> contour = {
        Point(1, 1), Point(5, 1), Point(5, 4), Point(3, 4), Point(1, 4)};
    EXPECT_EQ(boundingRect(contour), Rect(1, 1, 5, 4));
    EXPECT_DOUBLE_EQ(contourArea(contour), 12.0);
    EXPECT_DOUBLE_EQ(contourArea(contour, true), 12.0);
    EXPECT_DOUBLE_EQ(arcLength(contour, true), 14.0);
    EXPECT_FALSE(isContourConvex(contour));

    std::vector<Point> approximate;
    approxPolyDP(contour, approximate, 0.01, true);
    EXPECT_EQ(approximate.size(), 4u);
    std::vector<Point> hull;
    convexHull(contour, hull);
    EXPECT_EQ(hull.size(), 4u);
    EXPECT_DOUBLE_EQ(contourArea(hull), 12.0);

    const Moments contour_moments = moments(contour);
    EXPECT_DOUBLE_EQ(contour_moments.m00, 12.0);
    EXPECT_DOUBLE_EQ(contour_moments.m10 / contour_moments.m00, 3.0);
    EXPECT_DOUBLE_EQ(contour_moments.m01 / contour_moments.m00, 2.5);
    EXPECT_FALSE(isContourConvex(std::vector<Point>()));
    EXPECT_EQ(boundingRect(std::vector<Point>()), Rect());

    const std::vector<Point2f> float_points = {
        Point2f(-1.25f, 2.75f), Point2f(3.99f, 2.1f), Point2f(2.0f, 5.01f)};
    EXPECT_EQ(boundingRect(float_points), Rect(-2, 2, 6, 4));
    EXPECT_GT(contourArea(float_points), 0.0);
    std::vector<Point2f> float_hull;
    convexHull(float_points, float_hull, true);
    EXPECT_EQ(float_hull.size(), 3u);

    const std::vector<Point> bow_tie = {
        Point(0, 0), Point(4, 4), Point(0, 4), Point(4, 0)};
    EXPECT_DOUBLE_EQ(contourArea(bow_tie, true), 0.0);
    EXPECT_FALSE(isContourConvex(bow_tie));

    const std::vector<Point2f> nonfinite = {
        Point2f(std::numeric_limits<float>::quiet_NaN(), 0.0f), Point2f(1.0f, 1.0f)};
    EXPECT_THROW(boundingRect(nonfinite), Exception);
    EXPECT_THROW(contourArea(nonfinite), Exception);
    EXPECT_THROW(arcLength(nonfinite, true), Exception);
    EXPECT_THROW(approxPolyDP(nonfinite, float_hull, 1.0, true), Exception);
    EXPECT_THROW(convexHull(nonfinite, float_hull), Exception);
    EXPECT_THROW(isContourConvex(nonfinite), Exception);
    EXPECT_THROW(moments(nonfinite), Exception);

    const std::vector<Point> overflowing_bounds = {
        Point(std::numeric_limits<int>::min(), 0),
        Point(std::numeric_limits<int>::max(), 1)};
    EXPECT_THROW(boundingRect(overflowing_bounds), Exception);
}

TEST(HistogramPhase2Test, mask_accumulate_and_compare_methods_match_formulas)
{
    Mat image({2, 4}, CV_32FC3);
    Mat mask({2, 4}, CV_8UC1);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            image.at<float>(y, x, 0) = static_cast<float>(x);
            image.at<float>(y, x, 1) = static_cast<float>(x + 4 * y);
            image.at<float>(y, x, 2) = 100.0f;
            mask.at<uchar>(y, x) = (x + y) % 2 == 0 ? 255 : 0;
        }
    }
    Mat histogram;
    calcHist(image, 1, mask, histogram, 4, 0.0f, 8.0f);
    EXPECT_FLOAT_EQ(histogram.at<float>(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(histogram.at<float>(1, 0), 1.0f);
    EXPECT_FLOAT_EQ(histogram.at<float>(2, 0), 1.0f);
    EXPECT_FLOAT_EQ(histogram.at<float>(3, 0), 1.0f);
    calcHist(image, 1, mask, histogram, 4, 0.0f, 8.0f, true);
    for (int bin = 0; bin < 4; ++bin) EXPECT_FLOAT_EQ(histogram.at<float>(bin, 0), 2.0f);

    Mat other({4, 1}, CV_32FC1);
    other.at<float>(0, 0) = 1.0f;
    other.at<float>(1, 0) = 2.0f;
    other.at<float>(2, 0) = 3.0f;
    other.at<float>(3, 0) = 4.0f;
    EXPECT_NEAR(compareHist(histogram, other, HISTCMP_INTERSECT), 7.0, 1e-12);
    EXPECT_NEAR(compareHist(other, other, HISTCMP_CORREL), 1.0, 1e-12);
    EXPECT_NEAR(compareHist(other, other, HISTCMP_CHISQR), 0.0, 1e-12);
    EXPECT_NEAR(compareHist(other, other, HISTCMP_BHATTACHARYYA), 0.0, 1e-12);

    Mat u8({1, 5}, CV_8UC1);
    const uchar samples[] = {0, 63, 64, 255, 255};
    for (int index = 0; index < 5; ++index) u8.at<uchar>(0, index) = samples[index];
    const int channel = 0;
    const int bins = 4;
    const float range_values[] = {0.0f, 256.0f};
    const float* ranges[] = {range_values};
    calcHist(&u8, 1, &channel, Mat(), histogram, 1, &bins, ranges);
    EXPECT_FLOAT_EQ(histogram.at<float>(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(histogram.at<float>(1, 0), 1.0f);
    EXPECT_FLOAT_EQ(histogram.at<float>(2, 0), 0.0f);
    EXPECT_FLOAT_EQ(histogram.at<float>(3, 0), 2.0f);
    EXPECT_THROW(calcHist(&u8, 1, &channel, Mat(), histogram, 1, &bins, ranges, false), Exception);
}

TEST(TemplateMatchPhase2Test, four_methods_support_roi_and_expected_output_shape)
{
    Mat storage({7, 8}, CV_8UC1);
    for (int y = 0; y < 7; ++y)
        for (int x = 0; x < 8; ++x)
            storage.at<uchar>(y, x) = static_cast<uchar>((x * 13 + y * 17) % 251);
    Mat image = storage(Range(1, 7), Range(1, 8));
    Mat templ = image(Range(2, 5), Range(3, 6));
    for (int method : {TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED})
    {
        Mat result;
        matchTemplate(image, templ, result, method);
        EXPECT_EQ(result.type(), CV_32FC1);
        EXPECT_EQ(result.shape(), MatShape({4, 5}));
        if (method == TM_SQDIFF) EXPECT_FLOAT_EQ(result.at<float>(2, 3), 0.0f);
        if (method == TM_SQDIFF_NORMED) EXPECT_FLOAT_EQ(result.at<float>(2, 3), 0.0f);
        if (method == TM_CCORR_NORMED) EXPECT_FLOAT_EQ(result.at<float>(2, 3), 1.0f);
    }

    Mat floating({4, 5}, CV_32FC1);
    floating = 0.0f;
    Mat constant_template({2, 2}, CV_32FC1);
    constant_template = 0.0f;
    Mat normalized;
    matchTemplate(floating, constant_template, normalized, TM_CCORR_NORMED);
    for (size_t index = 0; index < normalized.total(); ++index)
        EXPECT_FLOAT_EQ(normalized.at<float>(static_cast<int>(index)), 0.0f);
    matchTemplate(floating, constant_template, normalized, TM_SQDIFF_NORMED);
    for (size_t index = 0; index < normalized.total(); ++index)
        EXPECT_FLOAT_EQ(normalized.at<float>(static_cast<int>(index)), 1.0f);
    Mat too_large({8, 8}, CV_32FC1);
    EXPECT_THROW(matchTemplate(floating, too_large, normalized, TM_SQDIFF), Exception);
}
