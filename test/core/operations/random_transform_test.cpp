#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>
#include <limits>

using namespace cvh;

TEST(RandomPhase2Test, uniform_respects_channel_ranges_and_roi_stride)
{
    Mat storage({5, 7}, CV_16SC3);
    storage = Scalar::all(-999.0);
    Mat roi = storage(Range(1, 4), Range(2, 6));
    randu(roi, Scalar(-10, 100, 1000), Scalar(10, 120, 1030));
    for (int y = 0; y < storage.size[0]; ++y)
    {
        for (int x = 0; x < storage.size[1]; ++x)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                const short value = storage.at<short>(y, x, channel);
                if (y >= 1 && y < 4 && x >= 2 && x < 6)
                {
                    const int low[3] = {-10, 100, 1000};
                    const int high[3] = {10, 120, 1030};
                    EXPECT_GE(value, low[channel]);
                    EXPECT_LT(value, high[channel]);
                }
                else
                {
                    EXPECT_EQ(value, -999);
                }
            }
        }
    }

    Mat constant({2, 3}, CV_8UC4);
    randu(constant,
          Scalar(-300.0, 12.6, 300.0, 42.0),
          Scalar(-300.0, 12.6, 300.0, 42.0));
    for (int y = 0; y < constant.size[0]; ++y)
    {
        for (int x = 0; x < constant.size[1]; ++x)
        {
            EXPECT_EQ(constant.at<uchar>(y, x, 0), 0);
            EXPECT_EQ(constant.at<uchar>(y, x, 1), 13);
            EXPECT_EQ(constant.at<uchar>(y, x, 2), 255);
            EXPECT_EQ(constant.at<uchar>(y, x, 3), 42);
        }
    }

    Mat floating({100, 100}, CV_32FC3);
    randu(floating, Scalar(-1.0, 14.0, 100.0), Scalar(1.0, 10.0, 101.0));
    double sums[3] = {0.0, 0.0, 0.0};
    for (int y = 0; y < floating.size[0]; ++y)
    {
        for (int x = 0; x < floating.size[1]; ++x)
        {
            for (int channel = 0; channel < 3; ++channel)
                sums[channel] += floating.at<float>(y, x, channel);
        }
    }
    const double sample_count = static_cast<double>(floating.total());
    EXPECT_NEAR(sums[0] / sample_count, 0.0, 0.05);
    EXPECT_NEAR(sums[1] / sample_count, 12.0, 0.08);
    EXPECT_NEAR(sums[2] / sample_count, 100.5, 0.03);
}

TEST(RandomPhase2Test, normal_zero_deviation_matches_opencv_conversion_contract)
{
    Mat integers({2, 4}, CV_8UC4);
    randn(integers, Scalar(-10.0, 12.6, 300.0, 42.0), Scalar::all(0.0));
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            EXPECT_EQ(integers.at<uchar>(y, x, 0), 0);
            EXPECT_EQ(integers.at<uchar>(y, x, 1), 13);
            EXPECT_EQ(integers.at<uchar>(y, x, 2), 255);
            EXPECT_EQ(integers.at<uchar>(y, x, 3), 42);
        }
    }

    const int sizes[] = {2, 3, 4};
    Mat nd(3, sizes, CV_64FC1);
    randn(nd, Scalar(1.25), Scalar(0.0));
    for (size_t index = 0; index < nd.total(); ++index)
    {
        EXPECT_DOUBLE_EQ(nd.at<double>(static_cast<int>(index)), 1.25);
    }

    Mat signed8({1, 1}, CV_8SC1);
    Mat unsigned16({1, 1}, CV_16UC1);
    Mat signed16({1, 1}, CV_16SC1);
    Mat signed32({1, 1}, CV_32SC1);
    Mat floating32({1, 1}, CV_32FC1);
    randn(signed8, Scalar(-200.0), Scalar(0.0));
    randn(unsigned16, Scalar(70000.0), Scalar(0.0));
    randn(signed16, Scalar(-40000.0), Scalar(0.0));
    randn(signed32, Scalar(12.6), Scalar(0.0));
    randn(floating32, Scalar(-2.25), Scalar(0.0));
    EXPECT_EQ(signed8.at<schar>(), -128);
    EXPECT_EQ(unsigned16.at<ushort>(), 65535);
    EXPECT_EQ(signed16.at<short>(), -32768);
    EXPECT_EQ(signed32.at<int>(), 13);
    EXPECT_FLOAT_EQ(floating32.at<float>(), -2.25f);

    Mat normal_samples({100, 100}, CV_32FC1);
    randn(normal_samples, Scalar(5.0), Scalar(2.0));
    double sum = 0.0;
    double square_sum = 0.0;
    for (size_t index = 0; index < normal_samples.total(); ++index)
    {
        const double value = normal_samples.at<float>(static_cast<int>(index));
        sum += value;
        square_sum += value * value;
    }
    const double count = static_cast<double>(normal_samples.total());
    const double observed_mean = sum / count;
    const double observed_variance = square_sum / count - observed_mean * observed_mean;
    EXPECT_NEAR(observed_mean, 5.0, 0.10);
    EXPECT_NEAR(observed_variance, 4.0, 0.25);

    Mat mixed_storage({4, 6}, CV_32FC4);
    mixed_storage = Scalar::all(-999.0);
    Mat mixed_roi = mixed_storage(Range(1, 3), Range(1, 5));
    randn(mixed_roi, Scalar(3.5, 10.0, -2.25, 20.0),
          Scalar(0.0, 1.0, 0.0, 2.0));
    for (int y = 0; y < mixed_storage.size[0]; ++y)
    {
        for (int x = 0; x < mixed_storage.size[1]; ++x)
        {
            const bool inside = y >= 1 && y < 3 && x >= 1 && x < 5;
            if (inside)
            {
                EXPECT_FLOAT_EQ(mixed_storage.at<float>(y, x, 0), 3.5f);
                EXPECT_FLOAT_EQ(mixed_storage.at<float>(y, x, 2), -2.25f);
            }
            else
            {
                for (int channel = 0; channel < 4; ++channel)
                    EXPECT_FLOAT_EQ(
                        mixed_storage.at<float>(y, x, channel), -999.0f);
            }
        }
    }

    Mat empty;
    EXPECT_THROW(randu(empty, Scalar(0), Scalar(1)), Exception);
    Mat unsupported({1, 1}, CV_32UC1);
    EXPECT_THROW(randn(unsupported, Scalar(0), Scalar(1)), Exception);
}

TEST(TransformPhase2Test, affine_channel_transform_supports_roi_and_alias)
{
    Mat storage({3, 4}, CV_32FC2);
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            storage.at<float>(y, x, 0) = static_cast<float>(x + 10 * y);
            storage.at<float>(y, x, 1) = static_cast<float>(2 * x - y);
        }
    }
    Mat source = storage(Range(1, 3), Range(1, 4));
    Mat matrix({3, 3}, CV_64FC1);
    const double coefficients[9] = {2.0, -1.0, 3.0, 0.5, 4.0, -2.0, -1.0, 0.25, 7.0};
    for (int index = 0; index < 9; ++index)
    {
        matrix.at<double>(index / 3, index % 3) = coefficients[index];
    }
    Mat destination;
    transform(source, destination, matrix);
    ASSERT_EQ(destination.type(), CV_32FC3);
    for (int y = 0; y < source.size[0]; ++y)
    {
        for (int x = 0; x < source.size[1]; ++x)
        {
            const double a = source.at<float>(y, x, 0);
            const double b = source.at<float>(y, x, 1);
            EXPECT_FLOAT_EQ(destination.at<float>(y, x, 0), static_cast<float>(2 * a - b + 3));
            EXPECT_FLOAT_EQ(destination.at<float>(y, x, 1), static_cast<float>(0.5 * a + 4 * b - 2));
            EXPECT_FLOAT_EQ(destination.at<float>(y, x, 2), static_cast<float>(-a + 0.25 * b + 7));
        }
    }

    Mat identity({2, 3}, CV_32FC1);
    identity = 0.0f;
    identity.at<float>(0, 0) = 1.0f;
    identity.at<float>(1, 1) = 1.0f;
    source = source.clone();
    const Mat expected = source.clone();
    transform(source, source, identity);
    for (size_t index = 0; index < expected.total() * 2; ++index)
    {
        EXPECT_FLOAT_EQ(source.at<float>(static_cast<int>(index)), expected.at<float>(static_cast<int>(index)));
    }

    for (int source_channels = 1; source_channels <= 4; ++source_channels)
    {
        Mat values({1, 3}, CV_MAKETYPE(CV_64F, source_channels));
        for (int point = 0; point < 3; ++point)
            for (int input_channel = 0;
                 input_channel < source_channels;
                 ++input_channel)
                values.at<double>(0, point, input_channel) =
                    static_cast<double>(point * 5 + input_channel + 1);

        for (int destination_channels = 1;
             destination_channels <= 4;
             ++destination_channels)
        {
            Mat coefficients_matrix(
                {destination_channels, source_channels + 1},
                CV_32FC1);
            for (int output_channel = 0;
                 output_channel < destination_channels;
                 ++output_channel)
            {
                for (int input_channel = 0;
                     input_channel < source_channels;
                     ++input_channel)
                {
                    coefficients_matrix.at<float>(
                        output_channel, input_channel) =
                        static_cast<float>(
                            (output_channel + 1) * (input_channel + 1)) *
                        0.25f;
                }
                coefficients_matrix.at<float>(
                    output_channel, source_channels) =
                    static_cast<float>(output_channel - 1);
            }

            transform(values, destination, coefficients_matrix);
            ASSERT_EQ(
                destination.type(),
                CV_MAKETYPE(CV_64F, destination_channels));
            for (int point = 0; point < 3; ++point)
            {
                for (int output_channel = 0;
                     output_channel < destination_channels;
                     ++output_channel)
                {
                    double expected_value =
                        static_cast<double>(output_channel - 1);
                    for (int input_channel = 0;
                         input_channel < source_channels;
                         ++input_channel)
                    {
                        expected_value +=
                            values.at<double>(0, point, input_channel) *
                            coefficients_matrix.at<float>(
                                output_channel, input_channel);
                    }
                    EXPECT_DOUBLE_EQ(
                        destination.at<double>(
                            0, point, output_channel),
                        expected_value);
                }
            }
        }
    }
}

TEST(TransformPhase2Test, perspective_matches_zero_w_and_nonfinite_rules)
{
    Mat points({1, 3}, CV_64FC2);
    points.at<double>(0, 0, 0) = 1.0;
    points.at<double>(0, 0, 1) = 2.0;
    points.at<double>(0, 1, 0) = 3.0;
    points.at<double>(0, 1, 1) = 4.0;
    points.at<double>(0, 2, 0) = std::numeric_limits<double>::quiet_NaN();
    points.at<double>(0, 2, 1) = 1.0;
    Mat matrix({3, 3}, CV_64FC1);
    matrix = 0.0f;
    matrix.at<double>(0, 0) = 2.0;
    matrix.at<double>(1, 1) = 3.0;
    matrix.at<double>(2, 0) = 1.0;
    matrix.at<double>(2, 2) = -1.0;

    Mat transformed;
    perspectiveTransform(points, transformed, matrix);
    EXPECT_DOUBLE_EQ(transformed.at<double>(0, 0, 0), 0.0);
    EXPECT_DOUBLE_EQ(transformed.at<double>(0, 0, 1), 0.0);
    EXPECT_DOUBLE_EQ(transformed.at<double>(0, 1, 0), 3.0);
    EXPECT_DOUBLE_EQ(transformed.at<double>(0, 1, 1), 6.0);
    EXPECT_DOUBLE_EQ(transformed.at<double>(0, 2, 0), 0.0);
    EXPECT_DOUBLE_EQ(transformed.at<double>(0, 2, 1), 0.0);

    Mat floating_points({5, 1}, CV_32FC3);
    for (int point = 0; point < 5; ++point)
    {
        floating_points.at<float>(point, 0, 0) =
            static_cast<float>(point + 1);
        floating_points.at<float>(point, 0, 1) =
            static_cast<float>(2 * point - 1);
        floating_points.at<float>(point, 0, 2) =
            static_cast<float>(3 - point);
    }
    Mat identity4({4, 4}, CV_32FC1);
    identity4 = 0.0f;
    for (int index = 0; index < 4; ++index)
        identity4.at<float>(index, index) = 1.0f;
    const Mat floating_expected = floating_points.clone();
    perspectiveTransform(floating_points, floating_points, identity4);
    for (std::size_t index = 0; index < floating_expected.total() * 3; ++index)
    {
        EXPECT_FLOAT_EQ(
            floating_points.at<float>(static_cast<int>(index)),
            floating_expected.at<float>(static_cast<int>(index)));
    }

    identity4.at<float>(3, 3) =
        std::numeric_limits<float>::epsilon() * 0.5f;
    perspectiveTransform(floating_points, transformed, identity4);
    for (std::size_t index = 0; index < transformed.total() * 3; ++index)
    {
        EXPECT_FLOAT_EQ(transformed.at<float>(static_cast<int>(index)), 0.0f);
    }

    Mat source4({2, 1}, CV_64FC4);
    source4 = Scalar(1.0, 2.0, 3.0, 4.0);
    Mat reduce_matrix({1, 4}, CV_32FC1);
    for (int index = 0; index < 4; ++index)
        reduce_matrix.at<float>(0, index) = static_cast<float>(index + 1);
    transform(source4, transformed, reduce_matrix);
    ASSERT_EQ(transformed.type(), CV_64FC1);
    EXPECT_DOUBLE_EQ(transformed.at<double>(0, 0), 30.0);

    Mat wrong_matrix({2, 2}, CV_32FC1);
    EXPECT_THROW(transform(source4, transformed, wrong_matrix), Exception);
    Mat bad_points({2, 2}, CV_32FC2);
    Mat identity3({3, 3}, CV_32FC1);
    EXPECT_THROW(perspectiveTransform(bad_points, transformed, identity3), Exception);
}
