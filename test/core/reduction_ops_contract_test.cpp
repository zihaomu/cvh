#include "cvh.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace cvh;

namespace {

class DispatchModeGuard
{
public:
    explicit DispatchModeGuard(cpu::DispatchMode mode)
        : previous_(cpu::dispatch_mode())
    {
        cpu::set_dispatch_mode(mode);
    }

    ~DispatchModeGuard()
    {
        cpu::set_dispatch_mode(previous_);
    }

private:
    cpu::DispatchMode previous_;
};

void set_nonzero_at(Mat& mat, size_t index)
{
    switch (mat.depth())
    {
        case CV_8U:
            reinterpret_cast<uchar*>(mat.data)[index] = 1;
            return;
        case CV_8S:
            reinterpret_cast<schar*>(mat.data)[index] = -1;
            return;
        case CV_16U:
            reinterpret_cast<ushort*>(mat.data)[index] = 1;
            return;
        case CV_16S:
            reinterpret_cast<short*>(mat.data)[index] = -1;
            return;
        case CV_32S:
            reinterpret_cast<int*>(mat.data)[index] = -1;
            return;
        case CV_32U:
            reinterpret_cast<uint*>(mat.data)[index] = 1;
            return;
        case CV_16F:
            reinterpret_cast<hfloat*>(mat.data)[index] = hfloat(1.0f);
            return;
        case CV_32F:
            reinterpret_cast<float*>(mat.data)[index] = -1.0f;
            return;
        case CV_64F:
            reinterpret_cast<double*>(mat.data)[index] = -1.0;
            return;
        default:
            FAIL() << "unsupported test depth";
    }
}

void expect_nonzero_results(const Mat& src,
                            int expected_count,
                            cpu::DispatchTag expected_auto_tag)
{
    std::vector<Point> scalar_points;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        cpu::reset_last_dispatch_tag();
        EXPECT_EQ(countNonZero(src), expected_count);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
        cpu::reset_last_dispatch_tag();
        EXPECT_EQ(hasNonZero(src), expected_count != 0);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
        cpu::reset_last_dispatch_tag();
        findNonZero(src, scalar_points);
        EXPECT_EQ(
            scalar_points.size(),
            static_cast<size_t>(expected_count));
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    }

    DispatchModeGuard guard(cpu::DispatchMode::Auto);
    cpu::reset_last_dispatch_tag();
    EXPECT_EQ(countNonZero(src), expected_count);
    EXPECT_EQ(cpu::last_dispatch_tag(), expected_auto_tag);
    cpu::reset_last_dispatch_tag();
    EXPECT_EQ(hasNonZero(src), expected_count != 0);
    EXPECT_EQ(cpu::last_dispatch_tag(), expected_auto_tag);
    cpu::reset_last_dispatch_tag();
    std::vector<Point> auto_points;
    findNonZero(src, auto_points);
    EXPECT_EQ(auto_points, scalar_points);
    EXPECT_EQ(cpu::last_dispatch_tag(), expected_auto_tag);

    cpu::reset_last_dispatch_tag();
    Mat point_mat;
    findNonZero(src, point_mat);
    EXPECT_EQ(cpu::last_dispatch_tag(), expected_auto_tag);
    if (scalar_points.empty())
    {
        EXPECT_TRUE(point_mat.empty());
        return;
    }

    ASSERT_EQ(point_mat.type(), CV_32SC2);
    ASSERT_EQ(
        point_mat.shape(),
        MatShape({static_cast<int>(scalar_points.size()), 1}));
    const int* coordinates =
        reinterpret_cast<const int*>(point_mat.data);
    for (size_t index = 0; index < scalar_points.size(); ++index)
    {
        EXPECT_EQ(coordinates[index * 2], scalar_points[index].x);
        EXPECT_EQ(coordinates[index * 2 + 1], scalar_points[index].y);
    }
}

cpu::DispatchTag expected_nonzero_auto_tag(int depth, bool ui_enabled)
{
    if (!ui_enabled || depth == CV_16F)
    {
        return cpu::DispatchTag::Scalar;
    }
    if (depth == CV_64F)
    {
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        return cpu::DispatchTag::OpenCVUI;
#else
        return cpu::DispatchTag::Scalar;
#endif
    }
    return cpu::DispatchTag::OpenCVUI;
}

cpu::DispatchTag expected_statistics_auto_tag(int depth, bool ui_enabled)
{
    if (!ui_enabled || depth == CV_16F)
    {
        return cpu::DispatchTag::Scalar;
    }
    if (depth == CV_64F)
    {
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        return cpu::DispatchTag::OpenCVUI;
#else
        return cpu::DispatchTag::Scalar;
#endif
    }
    return cpu::DispatchTag::OpenCVUI;
}

cpu::DispatchTag expected_norm_auto_tag(int depth, bool ui_enabled)
{
    if (!ui_enabled)
    {
        return cpu::DispatchTag::Scalar;
    }
    if (depth == CV_8U)
    {
        return cpu::DispatchTag::OpenCVUI;
    }
    if (depth == CV_32F)
    {
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        return cpu::DispatchTag::OpenCVUI;
#else
        return cpu::DispatchTag::Scalar;
#endif
    }
    return cpu::DispatchTag::Scalar;
}

void set_test_value(Mat& mat,
                    int row,
                    int column,
                    int channel,
                    double value)
{
    uchar* row_data = mat.data + static_cast<size_t>(row) * mat.step(0);
    const size_t index =
        static_cast<size_t>(column * mat.channels() + channel);
    switch (mat.depth())
    {
        case CV_8U:
            reinterpret_cast<uchar*>(row_data)[index] =
                static_cast<uchar>(value);
            return;
        case CV_8S:
            reinterpret_cast<schar*>(row_data)[index] =
                static_cast<schar>(value);
            return;
        case CV_16U:
            reinterpret_cast<ushort*>(row_data)[index] =
                static_cast<ushort>(value);
            return;
        case CV_16S:
            reinterpret_cast<short*>(row_data)[index] =
                static_cast<short>(value);
            return;
        case CV_32S:
            reinterpret_cast<int*>(row_data)[index] =
                static_cast<int>(value);
            return;
        case CV_32U:
            reinterpret_cast<uint*>(row_data)[index] =
                static_cast<uint>(value);
            return;
        case CV_16F:
            reinterpret_cast<hfloat*>(row_data)[index] =
                hfloat(static_cast<float>(value));
            return;
        case CV_32F:
            reinterpret_cast<float*>(row_data)[index] =
                static_cast<float>(value);
            return;
        case CV_64F:
            reinterpret_cast<double*>(row_data)[index] = value;
            return;
        default:
            FAIL() << "unsupported test depth";
    }
}

void expect_scalar_close(const Scalar& actual,
                         const Scalar& expected,
                         int channels,
                         double absolute_tolerance,
                         double relative_tolerance)
{
    for (int channel = 0; channel < channels; ++channel)
    {
        if (std::isnan(expected[channel]))
        {
            EXPECT_TRUE(std::isnan(actual[channel]));
            continue;
        }
        if (std::isinf(expected[channel]))
        {
            EXPECT_EQ(actual[channel], expected[channel]);
            continue;
        }
        const double scale =
            std::max(std::fabs(actual[channel]), std::fabs(expected[channel]));
        EXPECT_NEAR(
            actual[channel],
            expected[channel],
            absolute_tolerance + relative_tolerance * scale);
    }
}

void expect_index_mat_equal(const Mat& actual, const Mat& expected)
{
    ASSERT_EQ(actual.shape(), expected.shape());
    ASSERT_EQ(actual.type(), CV_32SC1);
    ASSERT_EQ(expected.type(), CV_32SC1);
    for (size_t index = 0; index < actual.total(); ++index)
    {
        EXPECT_EQ(
            reinterpret_cast<const int*>(actual.data)[index],
            reinterpret_cast<const int*>(expected.data)[index]);
    }
}

void expect_f32_mat_close(const Mat& actual,
                          const Mat& expected,
                          float tolerance)
{
    ASSERT_EQ(actual.shape(), expected.shape());
    ASSERT_EQ(actual.type(), expected.type());
    ASSERT_EQ(actual.depth(), CV_32F);
    const size_t scalars =
        actual.total() * static_cast<size_t>(actual.channels());
    for (size_t index = 0; index < scalars; ++index)
    {
        EXPECT_NEAR(
            reinterpret_cast<const float*>(actual.data)[index],
            reinterpret_cast<const float*>(expected.data)[index],
            tolerance);
    }
}

void expect_reduce_mat_close(const Mat& actual,
                             const Mat& expected,
                             double absolute_tolerance,
                             double relative_tolerance)
{
    ASSERT_EQ(actual.shape(), expected.shape());
    ASSERT_EQ(actual.type(), expected.type());
    for (int row = 0; row < actual.size.p[0]; ++row)
    {
        const uchar* actual_row =
            actual.data + static_cast<size_t>(row) * actual.step(0);
        const uchar* expected_row =
            expected.data + static_cast<size_t>(row) * expected.step(0);
        const size_t row_scalars =
            static_cast<size_t>(actual.size.p[1]) *
            static_cast<size_t>(actual.channels());
        for (size_t scalar = 0; scalar < row_scalars; ++scalar)
        {
            const double actual_value =
                reduce_detail::read_scalar(
                    actual_row, scalar, actual.depth());
            const double expected_value =
                reduce_detail::read_scalar(
                    expected_row, scalar, expected.depth());
            if (std::isnan(expected_value))
            {
                EXPECT_TRUE(std::isnan(actual_value));
                continue;
            }
            if (std::isinf(expected_value))
            {
                EXPECT_EQ(actual_value, expected_value);
                continue;
            }
            const double scale = std::max(
                std::fabs(actual_value), std::fabs(expected_value));
            EXPECT_NEAR(
                actual_value,
                expected_value,
                absolute_tolerance + relative_tolerance * scale);
        }
    }
}

int reduce_test_lanes(int depth)
{
#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return depth == CV_8U
        ? cv::VTraits<cv::v_uint8>::vlanes()
        : cv::VTraits<cv::v_float32>::vlanes();
#else
    return depth == CV_8U ? 16 : 4;
#endif
}

cpu::DispatchTag expected_reduce_auto_tag(int depth,
                                          int axis,
                                          int rtype,
                                          bool ui_enabled)
{
    if (!ui_enabled)
    {
        return cpu::DispatchTag::Scalar;
    }
    if (depth == CV_32F && axis == 0 &&
        rtype != REDUCE_MAX && rtype != REDUCE_MIN)
    {
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        return cpu::DispatchTag::OpenCVUI;
#else
        return cpu::DispatchTag::Scalar;
#endif
    }
    return cpu::DispatchTag::OpenCVUI;
}

}  // namespace

TEST(ReductionOpsContract_TEST, norm_values_mask_and_two_input_forms)
{
    EXPECT_EQ(NORM_INF, 1);
    EXPECT_EQ(NORM_L1, 2);
    EXPECT_EQ(NORM_L2, 4);
    EXPECT_EQ(NORM_MINMAX, 32);

    Mat src({1, 3}, CV_32FC2);
    const float values[] = {3.0f, 4.0f, -5.0f, 0.0f, 0.0f, 12.0f};
    for (int x = 0; x < 3; ++x)
    {
        src.at<float>(0, x, 0) = values[2 * x];
        src.at<float>(0, x, 1) = values[2 * x + 1];
    }

    EXPECT_DOUBLE_EQ(norm(src, NORM_INF), 12.0);
    EXPECT_DOUBLE_EQ(norm(src, NORM_L1), 24.0);
    EXPECT_DOUBLE_EQ(norm(src, NORM_L2), std::sqrt(194.0));

    Mat mask({1, 3}, CV_8UC1);
    mask.at<uchar>(0, 0) = 255;
    mask.at<uchar>(0, 1) = 0;
    mask.at<uchar>(0, 2) = 255;
    EXPECT_DOUBLE_EQ(norm(src, NORM_L1, mask), 19.0);

    Mat zeros(src.shape(), src.type());
    zeros.setTo(Scalar::all(0.0));
    EXPECT_DOUBLE_EQ(norm(src, zeros, NORM_L2), std::sqrt(194.0));
}

TEST(ReductionOpsContract_TEST, sum_mean_stddev_cover_c3_mask_and_roi)
{
    Mat parent({2, 6}, CV_32FC3);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 6; ++x)
        {
            parent.at<float>(y, x, 0) = static_cast<float>(y * 10 + x);
            parent.at<float>(y, x, 1) = static_cast<float>(2 * (y * 10 + x));
            parent.at<float>(y, x, 2) = 5.0f;
        }
    }
    Mat src = parent.colRange(1, 5);
    ASSERT_FALSE(src.isContinuous());

    const Scalar sums = sum(src);
    EXPECT_DOUBLE_EQ(sums[0], 60.0);
    EXPECT_DOUBLE_EQ(sums[1], 120.0);
    EXPECT_DOUBLE_EQ(sums[2], 40.0);

    Mat mask({2, 4}, CV_8UC1);
    mask.setTo(Scalar::all(0.0));
    mask.at<uchar>(0, 0) = 255;
    mask.at<uchar>(0, 2) = 255;
    mask.at<uchar>(1, 1) = 255;
    mask.at<uchar>(1, 3) = 255;

    const Scalar means = mean(src, mask);
    EXPECT_DOUBLE_EQ(means[0], 7.5);
    EXPECT_DOUBLE_EQ(means[1], 15.0);
    EXPECT_DOUBLE_EQ(means[2], 5.0);

    Scalar mean_value;
    Scalar stddev_value;
    meanStdDev(src, mean_value, stddev_value, mask);
    EXPECT_EQ(mean_value, means);
    EXPECT_NEAR(stddev_value[0], std::sqrt(31.25), 1e-12);
    EXPECT_NEAR(stddev_value[1], std::sqrt(125.0), 1e-12);
    EXPECT_DOUBLE_EQ(stddev_value[2], 0.0);

    mask.setTo(Scalar::all(0.0));
    EXPECT_EQ(mean(src, mask), Scalar());
    meanStdDev(src, mean_value, stddev_value, mask);
    EXPECT_EQ(mean_value, Scalar());
    EXPECT_EQ(stddev_value, Scalar());
}

TEST(ReductionOpsContract_TEST, statistics_cover_c4_single_and_identical_values)
{
    Mat c4({1, 2}, CV_64FC4);
    for (int ch = 0; ch < 4; ++ch)
    {
        c4.at<double>(0, 0, ch) = static_cast<double>(ch + 1);
        c4.at<double>(0, 1, ch) = static_cast<double>(2 * (ch + 1));
    }
    const Scalar sums = sum(c4);
    EXPECT_DOUBLE_EQ(sums[0], 3.0);
    EXPECT_DOUBLE_EQ(sums[1], 6.0);
    EXPECT_DOUBLE_EQ(sums[2], 9.0);
    EXPECT_DOUBLE_EQ(sums[3], 12.0);

    Mat identical({3, 5}, CV_64FC1);
    identical.setTo(Scalar::all(1.0e12));
    Scalar mean_value;
    Scalar stddev_value;
    meanStdDev(identical, mean_value, stddev_value);
    EXPECT_DOUBLE_EQ(mean_value[0], 1.0e12);
    EXPECT_DOUBLE_EQ(stddev_value[0], 0.0);

    Mat single({1, 1}, CV_32FC1);
    single.at<float>(0, 0) = -7.0f;
    meanStdDev(single, mean_value, stddev_value);
    EXPECT_DOUBLE_EQ(mean_value[0], -7.0);
    EXPECT_DOUBLE_EQ(stddev_value[0], 0.0);
}

TEST(ReductionOpsContract_TEST,
     statistics_ui_matches_scalar_across_depths_channels_roi_and_tail)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }
    const int depths[] = {
        CV_8U,
        CV_8S,
        CV_16U,
        CV_16S,
        CV_32S,
        CV_32U,
        CV_16F,
        CV_32F,
        CV_64F,
    };

    for (const int depth : depths)
    {
        for (int channels = 1; channels <= 4; ++channels)
        {
            Mat parent({3, 43}, CV_MAKETYPE(depth, channels));
            for (int row = 0; row < parent.size.p[0]; ++row)
            {
                for (int column = 0; column < parent.size.p[1]; ++column)
                {
                    for (int channel = 0; channel < channels; ++channel)
                    {
                        const int seed =
                            row * 19 + column * 7 + channel * 3;
                        double value = static_cast<double>(seed % 23 + 1);
                        if (depth == CV_8S || depth == CV_16S ||
                            depth == CV_32S)
                        {
                            value = static_cast<double>(seed % 23 - 11);
                        }
                        if (depth == CV_16F || depth == CV_32F ||
                            depth == CV_64F)
                        {
                            value = static_cast<double>(seed % 23 - 11) *
                                    0.25;
                        }
                        set_test_value(
                            parent, row, column, channel, value);
                    }
                }
            }
            Mat src = parent.colRange(1, 42);
            ASSERT_FALSE(src.isContinuous());

            Scalar scalar_sum;
            Scalar scalar_mean;
            Scalar scalar_mean_value;
            Scalar scalar_stddev;
            {
                DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
                scalar_sum = sum(src);
                scalar_mean = mean(src);
                meanStdDev(src, scalar_mean_value, scalar_stddev);
                EXPECT_EQ(
                    cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
            }

            DispatchModeGuard guard(cpu::DispatchMode::Auto);
            cpu::reset_last_dispatch_tag();
            const Scalar auto_sum = sum(src);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                expected_statistics_auto_tag(depth, ui_enabled));
            cpu::reset_last_dispatch_tag();
            const Scalar auto_mean = mean(src);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                expected_statistics_auto_tag(depth, ui_enabled));
            Scalar auto_mean_value;
            Scalar auto_stddev;
            cpu::reset_last_dispatch_tag();
            meanStdDev(src, auto_mean_value, auto_stddev);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                expected_statistics_auto_tag(depth, ui_enabled));

            const bool floating =
                depth == CV_16F || depth == CV_32F || depth == CV_64F;
            const double absolute_tolerance =
                depth == CV_64F ? 1e-12 : (floating ? 1e-6 : 0.0);
            const double relative_tolerance =
                depth == CV_64F ? 1e-12 : (floating ? 1e-6 : 0.0);
            expect_scalar_close(
                auto_sum,
                scalar_sum,
                channels,
                absolute_tolerance,
                relative_tolerance);
            const double statistics_absolute_tolerance =
                floating ? absolute_tolerance : 1e-12;
            const double statistics_relative_tolerance =
                floating ? relative_tolerance : 1e-12;
            expect_scalar_close(
                auto_mean,
                scalar_mean,
                channels,
                statistics_absolute_tolerance,
                statistics_relative_tolerance);
            expect_scalar_close(
                auto_mean_value,
                scalar_mean_value,
                channels,
                statistics_absolute_tolerance,
                statistics_relative_tolerance);
            expect_scalar_close(
                auto_stddev,
                scalar_stddev,
                channels,
                statistics_absolute_tolerance,
                statistics_relative_tolerance);
        }
    }
}

TEST(ReductionOpsContract_TEST,
     statistics_ui_masks_cover_empty_full_sparse_and_c1_to_c4)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }

    for (int channels = 1; channels <= 4; ++channels)
    {
        Mat src_parent({3, 47}, CV_MAKETYPE(CV_32F, channels));
        Mat mask_parent({3, 47}, CV_8UC1);
        for (int row = 0; row < src_parent.size.p[0]; ++row)
        {
            for (int column = 0; column < src_parent.size.p[1]; ++column)
            {
                for (int channel = 0; channel < channels; ++channel)
                {
                    set_test_value(
                        src_parent,
                        row,
                        column,
                        channel,
                        (row * 47 + column + channel * 0.25) * 0.5);
                }
            }
        }
        Mat src = src_parent.colRange(1, 46);
        Mat mask = mask_parent.colRange(1, 46);
        ASSERT_FALSE(src.isContinuous());
        ASSERT_FALSE(mask.isContinuous());

        for (int distribution = 0; distribution < 3; ++distribution)
        {
            mask.setTo(Scalar::all(distribution == 1 ? 255.0 : 0.0));
            if (distribution == 2)
            {
                for (int row = 0; row < mask.size.p[0]; ++row)
                {
                    for (int column = 0; column < mask.size.p[1]; ++column)
                    {
                        mask.at<uchar>(row, column) =
                            ((row * mask.size.p[1] + column) % 5) == 0
                            ? 255
                            : 0;
                    }
                }
            }

            Scalar scalar_mean;
            Scalar scalar_mean_value;
            Scalar scalar_stddev;
            {
                DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
                scalar_mean = mean(src, mask);
                meanStdDev(
                    src, scalar_mean_value, scalar_stddev, mask);
            }

            DispatchModeGuard guard(cpu::DispatchMode::Auto);
            cpu::reset_last_dispatch_tag();
            const Scalar auto_mean = mean(src, mask);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                expected_statistics_auto_tag(CV_32F, ui_enabled));
            Scalar auto_mean_value;
            Scalar auto_stddev;
            cpu::reset_last_dispatch_tag();
            meanStdDev(
                src, auto_mean_value, auto_stddev, mask);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                expected_statistics_auto_tag(CV_32F, ui_enabled));

            expect_scalar_close(
                auto_mean, scalar_mean, channels, 1e-6, 1e-6);
            expect_scalar_close(
                auto_mean_value,
                scalar_mean_value,
                channels,
                1e-6,
                1e-6);
            expect_scalar_close(
                auto_stddev,
                scalar_stddev,
                channels,
                1e-6,
                1e-6);
        }
    }
}

TEST(ReductionOpsContract_TEST, statistics_ui_integer_sum_is_exact)
{
    Mat unsigned_values({1, 257}, CV_32UC1);
    Mat signed_values({1, 257}, CV_32SC1);
    for (int column = 0; column < 257; ++column)
    {
        unsigned_values.at<uint>(0, column) =
            std::numeric_limits<uint>::max() -
            static_cast<uint>(column);
        signed_values.at<int>(0, column) =
            (column % 2 == 0)
            ? std::numeric_limits<int>::max() - column
            : std::numeric_limits<int>::min() + column;
    }

    Scalar scalar_unsigned;
    Scalar scalar_signed;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        scalar_unsigned = sum(unsigned_values);
        scalar_signed = sum(signed_values);
    }
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        EXPECT_DOUBLE_EQ(sum(unsigned_values)[0], scalar_unsigned[0]);
        EXPECT_DOUBLE_EQ(sum(signed_values)[0], scalar_signed[0]);
    }
}

TEST(ReductionOpsContract_TEST,
     statistics_ui_preserves_stability_special_values_and_fallback)
{
    Mat stable({1, 257}, CV_64FC1);
    for (int column = 0; column < stable.size.p[1]; ++column)
    {
        stable.at<double>(0, column) =
            1.0e12 + static_cast<double>(column % 5 - 2) * 0.001;
    }

    Scalar scalar_mean;
    Scalar scalar_stddev;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        meanStdDev(stable, scalar_mean, scalar_stddev);
    }
    Scalar auto_mean;
    Scalar auto_stddev;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        meanStdDev(stable, auto_mean, auto_stddev);
    }
    EXPECT_NEAR(auto_mean[0], scalar_mean[0], 2e-4);
    EXPECT_NEAR(auto_stddev[0], scalar_stddev[0], 2e-5);
    EXPECT_GT(auto_stddev[0], 0.0);

    Mat special({1, 33}, CV_32FC1);
    special.setTo(Scalar::all(1.0));
    special.at<float>(0, 7) = std::numeric_limits<float>::infinity();
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        const Scalar special_sum = sum(special);
        EXPECT_TRUE(std::isinf(special_sum[0]));
        meanStdDev(special, auto_mean, auto_stddev);
        EXPECT_FALSE(std::isfinite(auto_stddev[0]));
    }
    special.at<float>(0, 7) =
        std::numeric_limits<float>::quiet_NaN();
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        EXPECT_TRUE(std::isnan(sum(special)[0]));
        meanStdDev(special, auto_mean, auto_stddev);
        EXPECT_TRUE(std::isnan(auto_mean[0]));
        EXPECT_TRUE(std::isnan(auto_stddev[0]));
    }

    Mat short_row({1, 1}, CV_32FC4);
    short_row.setTo(Scalar(1.0, 2.0, 3.0, 4.0));
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        EXPECT_EQ(sum(short_row), Scalar(1.0, 2.0, 3.0, 4.0));
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    }
}

TEST(ReductionOpsContract_TEST, nonzero_predicates_and_locations_are_row_major)
{
    Mat src({3, 4}, CV_16SC1);
    src.setTo(Scalar::all(0.0));
    src.at<short>(0, 3) = 2;
    src.at<short>(1, 1) = -4;
    src.at<short>(2, 0) = 7;

    EXPECT_TRUE(hasNonZero(src));
    EXPECT_EQ(countNonZero(src), 3);

    std::vector<Point> points;
    findNonZero(src, points);
    ASSERT_EQ(points.size(), 3u);
    EXPECT_EQ(points[0], Point(3, 0));
    EXPECT_EQ(points[1], Point(1, 1));
    EXPECT_EQ(points[2], Point(0, 2));

    Mat point_mat;
    findNonZero(src, point_mat);
    ASSERT_EQ(point_mat.type(), CV_32SC2);
    ASSERT_EQ(point_mat.shape(), MatShape({3, 1}));
    EXPECT_EQ(point_mat.at<int>(1, 0, 0), 1);
    EXPECT_EQ(point_mat.at<int>(1, 0, 1), 1);

    src.setTo(Scalar::all(0.0));
    EXPECT_FALSE(hasNonZero(src));
    EXPECT_EQ(countNonZero(src), 0);
    findNonZero(src, point_mat);
    EXPECT_TRUE(point_mat.empty());
}

TEST(ReductionOpsContract_TEST, nonzero_apis_reject_multichannel_input)
{
    Mat src({1, 3}, CV_8UC3);
    Mat out;
    std::vector<Point> points;
    EXPECT_THROW(countNonZero(src), Exception);
    EXPECT_THROW(hasNonZero(src), Exception);
    EXPECT_THROW(findNonZero(src, points), Exception);
    EXPECT_THROW(findNonZero(src, out), Exception);
}

TEST(ReductionOpsContract_TEST, nonzero_ui_covers_public_depths_and_fallbacks)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }
    const int depths[] = {
        CV_8U,
        CV_8S,
        CV_16U,
        CV_16S,
        CV_32S,
        CV_32U,
        CV_16F,
        CV_32F,
        CV_64F,
    };

    for (const int depth : depths)
    {
        Mat src({2, 259}, CV_MAKETYPE(depth, 1));
        src.setTo(Scalar::all(0.0));
        set_nonzero_at(src, 0);
        set_nonzero_at(src, 258);
        expect_nonzero_results(
            src,
            2,
            expected_nonzero_auto_tag(depth, ui_enabled));
    }

    for (const int depth : depths)
    {
        Mat src({3, 259}, CV_MAKETYPE(depth, 1));
        src.setTo(Scalar::all(0.0));
        int expected_count = 0;
        for (size_t index = 1; index < src.total(); index += 3)
        {
            set_nonzero_at(src, index);
            ++expected_count;
        }
        expect_nonzero_results(
            src,
            expected_count,
            expected_nonzero_auto_tag(depth, ui_enabled));
    }

    Mat short_row({1, 3}, CV_32FC1);
    short_row.setTo(Scalar::all(0.0));
    set_nonzero_at(short_row, 2);
    expect_nonzero_results(short_row, 1, cpu::DispatchTag::Scalar);
}

TEST(ReductionOpsContract_TEST, nonzero_ui_preserves_float_special_values)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }
    Mat src({1, 259}, CV_32FC1);
    src.setTo(Scalar::all(0.0));
    float* values = reinterpret_cast<float*>(src.data);
    values[1] = -0.0f;
    expect_nonzero_results(
        src,
        0,
        expected_nonzero_auto_tag(CV_32F, ui_enabled));

    values[17] = std::numeric_limits<float>::quiet_NaN();
    values[258] = std::numeric_limits<float>::infinity();
    expect_nonzero_results(
        src,
        2,
        expected_nonzero_auto_tag(CV_32F, ui_enabled));

    Mat src64({1, 259}, CV_64FC1);
    src64.setTo(Scalar::all(0.0));
    double* values64 = reinterpret_cast<double*>(src64.data);
    values64[2] = -0.0;
    values64[33] = std::numeric_limits<double>::quiet_NaN();
    values64[258] = -std::numeric_limits<double>::infinity();
    expect_nonzero_results(
        src64,
        2,
        expected_nonzero_auto_tag(CV_64F, ui_enabled));
}

TEST(ReductionOpsContract_TEST, nonzero_ui_handles_roi_tail_and_hit_positions)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }
    const cpu::DispatchTag expected_tag = ui_enabled
        ? cpu::DispatchTag::OpenCVUI
        : cpu::DispatchTag::Scalar;

    Mat parent({3, 263}, CV_32FC1);
    parent.setTo(Scalar::all(0.0));
    Mat roi = parent.colRange(2, 261);
    ASSERT_FALSE(roi.isContinuous());

    const size_t positions[] = {0, 16, 257, 258};
    for (const size_t position : positions)
    {
        roi.setTo(Scalar::all(0.0));
        const size_t row = position / 259;
        const size_t column = position % 259;
        roi.at<float>(static_cast<int>(row), static_cast<int>(column)) = 1.0f;
        expect_nonzero_results(roi, 1, expected_tag);
    }

    roi.setTo(Scalar::all(0.0));
    expect_nonzero_results(roi, 0, expected_tag);
}

TEST(ReductionOpsContract_TEST, find_nonzero_handles_vector_boundaries)
{
#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }
    const int lanes = cv::VTraits<cv::v_uint8>::vlanes();
    const int widths[] = {lanes - 1, lanes, lanes + 1};
    for (const int width : widths)
    {
        Mat src({2, width}, CV_8UC1);
        src.setTo(Scalar::all(0.0));
        src.at<uchar>(0, 0) = 1;
        src.at<uchar>(0, width - 1) = 2;
        src.at<uchar>(1, width / 2) = 3;
        const cpu::DispatchTag expected_tag =
            ui_enabled && width >= lanes
            ? cpu::DispatchTag::OpenCVUI
            : cpu::DispatchTag::Scalar;
        const std::vector<Point> expected = {
            Point(0, 0),
            Point(width - 1, 0),
            Point(width / 2, 1),
        };

        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        std::vector<Point> points;
        findNonZero(src, points);
        EXPECT_EQ(points, expected);
        EXPECT_EQ(cpu::last_dispatch_tag(), expected_tag);

        cpu::reset_last_dispatch_tag();
        Mat point_mat;
        findNonZero(src, point_mat);
        EXPECT_EQ(cpu::last_dispatch_tag(), expected_tag);
        ASSERT_EQ(point_mat.type(), CV_32SC2);
        ASSERT_EQ(point_mat.shape(), MatShape({3, 1}));
        const int* coordinates =
            reinterpret_cast<const int*>(point_mat.data);
        for (size_t index = 0; index < expected.size(); ++index)
        {
            EXPECT_EQ(coordinates[index * 2], expected[index].x);
            EXPECT_EQ(coordinates[index * 2 + 1], expected[index].y);
        }
    }
#else
    GTEST_SKIP() << "OpenCV UI fixed-width backend is unavailable";
#endif
}

TEST(ReductionOpsContract_TEST, minmax_ties_use_first_row_major_location)
{
    Mat src({2, 4}, CV_32SC1);
    const int values[] = {5, -2, 9, -2, 9, 3, 9, 4};
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            src.at<int>(y, x) = values[y * 4 + x];
        }
    }

    double min_value = 0.0;
    double max_value = 0.0;
    Point min_location;
    Point max_location;
    minMaxLoc(src, &min_value, &max_value, &min_location, &max_location);
    EXPECT_DOUBLE_EQ(min_value, -2.0);
    EXPECT_DOUBLE_EQ(max_value, 9.0);
    EXPECT_EQ(min_location, Point(1, 0));
    EXPECT_EQ(max_location, Point(2, 0));

    Mat mask({2, 4}, CV_8UC1);
    mask.setTo(Scalar::all(0.0));
    mask.at<uchar>(0, 3) = 255;
    mask.at<uchar>(1, 1) = 255;
    minMaxLoc(src, &min_value, &max_value, &min_location, &max_location, mask);
    EXPECT_DOUBLE_EQ(min_value, -2.0);
    EXPECT_DOUBLE_EQ(max_value, 3.0);
    EXPECT_EQ(min_location, Point(3, 0));
    EXPECT_EQ(max_location, Point(1, 1));
}

TEST(ReductionOpsContract_TEST, minmaxidx_reports_nd_coordinates)
{
    Mat src({2, 3, 4}, CV_64FC1);
    double* values = reinterpret_cast<double*>(src.data);
    for (int i = 0; i < 24; ++i)
    {
        values[i] = static_cast<double>(i);
    }
    values[17] = -10.0;
    values[22] = 100.0;

    double min_value = 0.0;
    double max_value = 0.0;
    int min_index[3] = {-1, -1, -1};
    int max_index[3] = {-1, -1, -1};
    minMaxIdx(src, &min_value, &max_value, min_index, max_index);
    EXPECT_DOUBLE_EQ(min_value, -10.0);
    EXPECT_DOUBLE_EQ(max_value, 100.0);
    EXPECT_EQ(min_index[0], 1);
    EXPECT_EQ(min_index[1], 1);
    EXPECT_EQ(min_index[2], 1);
    EXPECT_EQ(max_index[0], 1);
    EXPECT_EQ(max_index[1], 2);
    EXPECT_EQ(max_index[2], 2);
}

TEST(ReductionOpsContract_TEST,
     minmax_ui_matches_scalar_across_depths_roi_tail_and_ties)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }
    const int depths[] = {
        CV_8U,
        CV_8S,
        CV_16U,
        CV_16S,
        CV_32S,
        CV_32U,
        CV_16F,
        CV_32F,
        CV_64F,
    };

    for (const int depth : depths)
    {
        Mat parent({3, 263}, CV_MAKETYPE(depth, 1));
        for (int row = 0; row < 3; ++row)
        {
            for (int column = 0; column < 263; ++column)
            {
                const int seed = row * 263 + column;
                double value = static_cast<double>(seed % 101 + 10);
                if (depth == CV_8S || depth == CV_16S ||
                    depth == CV_32S || depth == CV_16F ||
                    depth == CV_32F || depth == CV_64F)
                {
                    value = static_cast<double>(seed % 101 - 50);
                }
                set_test_value(parent, row, column, 0, value);
            }
        }
        Mat src = parent.colRange(2, 261);
        ASSERT_FALSE(src.isContinuous());
        set_test_value(src, 0, 17, 0, depth == CV_8U || depth == CV_16U ||
                                          depth == CV_32U
                                      ? 0.0
                                      : -100.0);
        set_test_value(src, 1, 19, 0, depth == CV_8U || depth == CV_16U ||
                                          depth == CV_32U
                                      ? 0.0
                                      : -100.0);
        set_test_value(src, 0, 31, 0, depth == CV_8U ? 255.0 : 120.0);
        set_test_value(src, 2, 258, 0, depth == CV_8U ? 255.0 : 120.0);

        double scalar_min = 0.0;
        double scalar_max = 0.0;
        Point scalar_min_location;
        Point scalar_max_location;
        {
            DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
            minMaxLoc(
                src,
                &scalar_min,
                &scalar_max,
                &scalar_min_location,
                &scalar_max_location);
        }

        double auto_min = 0.0;
        double auto_max = 0.0;
        Point auto_min_location;
        Point auto_max_location;
        {
            DispatchModeGuard guard(cpu::DispatchMode::Auto);
            cpu::reset_last_dispatch_tag();
            minMaxLoc(
                src,
                &auto_min,
                &auto_max,
                &auto_min_location,
                &auto_max_location);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                expected_statistics_auto_tag(depth, ui_enabled));
        }
        EXPECT_DOUBLE_EQ(auto_min, scalar_min);
        EXPECT_DOUBLE_EQ(auto_max, scalar_max);
        EXPECT_EQ(auto_min_location, scalar_min_location);
        EXPECT_EQ(auto_max_location, scalar_max_location);
    }
}

TEST(ReductionOpsContract_TEST,
     minmax_ui_preserves_mask_nan_inf_and_signed_zero_semantics)
{
    Mat parent({2, 67}, CV_32FC1);
    parent.setTo(Scalar::all(10.0));
    Mat src = parent.colRange(1, 66);
    Mat mask_parent({2, 67}, CV_8UC1);
    Mat mask = mask_parent.colRange(1, 66);
    ASSERT_FALSE(src.isContinuous());
    ASSERT_FALSE(mask.isContinuous());

    src.at<float>(0, 2) = std::numeric_limits<float>::quiet_NaN();
    src.at<float>(0, 5) = 0.0f;
    src.at<float>(0, 9) = -0.0f;
    src.at<float>(1, 7) = std::numeric_limits<float>::infinity();
    src.at<float>(1, 11) = -std::numeric_limits<float>::infinity();

    mask.setTo(Scalar::all(0.0));
    mask.at<uchar>(0, 2) = 255;
    mask.at<uchar>(0, 5) = 255;
    mask.at<uchar>(0, 9) = 255;
    mask.at<uchar>(1, 7) = 255;
    mask.at<uchar>(1, 11) = 255;

    double scalar_min = 0.0;
    double scalar_max = 0.0;
    Point scalar_min_location;
    Point scalar_max_location;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        minMaxLoc(
            src,
            &scalar_min,
            &scalar_max,
            &scalar_min_location,
            &scalar_max_location,
            mask);
    }
    double auto_min = 0.0;
    double auto_max = 0.0;
    Point auto_min_location;
    Point auto_max_location;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        minMaxLoc(
            src,
            &auto_min,
            &auto_max,
            &auto_min_location,
            &auto_max_location,
            mask);
    }
    EXPECT_EQ(auto_min, scalar_min);
    EXPECT_EQ(auto_max, scalar_max);
    EXPECT_EQ(auto_min_location, scalar_min_location);
    EXPECT_EQ(auto_max_location, scalar_max_location);
    EXPECT_TRUE(std::isinf(auto_min) && auto_min < 0.0);
    EXPECT_TRUE(std::isinf(auto_max) && auto_max > 0.0);

    mask.setTo(Scalar::all(0.0));
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        minMaxLoc(
            src,
            &auto_min,
            &auto_max,
            &auto_min_location,
            &auto_max_location,
            mask);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    }
    EXPECT_DOUBLE_EQ(auto_min, 0.0);
    EXPECT_DOUBLE_EQ(auto_max, 0.0);
    EXPECT_EQ(auto_min_location, Point(-1, -1));
    EXPECT_EQ(auto_max_location, Point(-1, -1));

    Mat zeros({1, 33}, CV_32FC1);
    zeros.setTo(Scalar::all(1.0));
    zeros.at<float>(0, 5) = 0.0f;
    zeros.at<float>(0, 9) = -0.0f;
    minMaxLoc(
        zeros,
        &auto_min,
        &auto_max,
        &auto_min_location,
        &auto_max_location);
    EXPECT_FALSE(std::signbit(auto_min));
    EXPECT_EQ(auto_min_location, Point(5, 0));

    Mat all_nan({1, 33}, CV_32FC1);
    all_nan.setTo(Scalar::all(
        std::numeric_limits<float>::quiet_NaN()));
    minMaxLoc(
        all_nan,
        &auto_min,
        &auto_max,
        &auto_min_location,
        &auto_max_location);
    EXPECT_DOUBLE_EQ(auto_min, 0.0);
    EXPECT_DOUBLE_EQ(auto_max, 0.0);
    EXPECT_EQ(auto_min_location, Point(-1, -1));
    EXPECT_EQ(auto_max_location, Point(-1, -1));
}

TEST(ReductionOpsContract_TEST, reduce_axis_shape_type_and_values_match_contract)
{
    Mat src({2, 3}, CV_16SC2);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 3; ++x)
        {
            src.at<short>(y, x, 0) = static_cast<short>(10 * y + x + 1);
            src.at<short>(y, x, 1) = static_cast<short>(100 + 10 * y + x);
        }
    }

    Mat dst;
    reduce(src, dst, 0, REDUCE_SUM, CV_64F);
    ASSERT_EQ(dst.shape(), MatShape({1, 3}));
    ASSERT_EQ(dst.type(), CV_64FC2);
    EXPECT_DOUBLE_EQ(dst.at<double>(0, 0, 0), 12.0);
    EXPECT_DOUBLE_EQ(dst.at<double>(0, 2, 1), 214.0);

    reduce(src, dst, 1, REDUCE_AVG, CV_32F);
    ASSERT_EQ(dst.shape(), MatShape({2, 1}));
    ASSERT_EQ(dst.type(), CV_32FC2);
    EXPECT_FLOAT_EQ(dst.at<float>(0, 0, 0), 2.0f);
    EXPECT_FLOAT_EQ(dst.at<float>(1, 0, 1), 111.0f);

    reduce(src, dst, 0, REDUCE_MAX);
    ASSERT_EQ(dst.type(), CV_16SC2);
    EXPECT_EQ(dst.at<short>(0, 1, 0), 12);

    reduce(src, dst, 1, REDUCE_SUM2, CV_64F);
    EXPECT_DOUBLE_EQ(dst.at<double>(0, 0, 0), 1.0 + 4.0 + 9.0);

    Mat alias = src.clone();
    reduce(alias, alias, 0, REDUCE_SUM, CV_64F);
    ASSERT_EQ(alias.shape(), MatShape({1, 3}));
    EXPECT_DOUBLE_EQ(alias.at<double>(0, 0, 0), 12.0);
}

TEST(ReductionOpsContract_TEST,
     reduce_ui_matches_scalar_across_axes_rtypes_channels_roi_and_tail)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }

    const int depths[] = {CV_8U, CV_32F};
    const int channel_counts[] = {1, 3};
    const int rtypes[] = {
        REDUCE_SUM,
        REDUCE_AVG,
        REDUCE_MAX,
        REDUCE_MIN,
        REDUCE_SUM2,
    };
    for (int depth : depths)
    {
        const int lanes = reduce_test_lanes(depth);
        for (int channels : channel_counts)
        {
            Mat parent(
                {7, 2 * lanes + 5},
                CV_MAKETYPE(depth, channels));
            for (int row = 0; row < parent.size.p[0]; ++row)
            {
                for (int column = 0;
                     column < parent.size.p[1];
                     ++column)
                {
                    for (int channel = 0;
                         channel < channels;
                         ++channel)
                    {
                        const int seed =
                            17 * row + 7 * column + 3 * channel;
                        const double value = depth == CV_8U
                            ? static_cast<double>(seed % 31)
                            : static_cast<double>(
                                  (seed % 37) - 18) /
                                  8.0;
                        set_test_value(
                            parent,
                            row,
                            column,
                            channel,
                            value);
                    }
                }
            }
            Mat src = parent.colRange(1, parent.size.p[1] - 1);
            ASSERT_FALSE(src.isContinuous());

            for (int axis = 0; axis <= 1; ++axis)
            {
                for (int rtype : rtypes)
                {
                    Mat expected;
                    {
                        DispatchModeGuard guard(
                            cpu::DispatchMode::ScalarOnly);
                        reduce(
                            src,
                            expected,
                            axis,
                            rtype,
                            CV_64F);
                        EXPECT_EQ(
                            cpu::last_dispatch_tag(),
                            cpu::DispatchTag::Scalar);
                    }

                    Mat actual;
                    {
                        DispatchModeGuard guard(
                            cpu::DispatchMode::Auto);
                        cpu::reset_last_dispatch_tag();
                        reduce(
                            src,
                            actual,
                            axis,
                            rtype,
                            CV_64F);
                        EXPECT_EQ(
                            cpu::last_dispatch_tag(),
                            expected_reduce_auto_tag(
                                depth,
                                axis,
                                rtype,
                                ui_enabled));
                    }
                    expect_reduce_mat_close(
                        actual, expected, 1e-12, 1e-12);
                }
            }
        }
    }
}

TEST(ReductionOpsContract_TEST,
     reduce_ui_preserves_saturation_special_values_alias_and_fallbacks)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }

    const int u8_lanes = reduce_test_lanes(CV_8U);
    Mat saturated({5, u8_lanes + 3}, CV_8UC1);
    saturated.setTo(Scalar::all(255.0));
    Mat saturated_sum;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        reduce(
            saturated,
            saturated_sum,
            1,
            REDUCE_SUM,
            CV_8U);
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            ui_enabled
                ? cpu::DispatchTag::OpenCVUI
                : cpu::DispatchTag::Scalar);
    }
    for (int row = 0; row < saturated_sum.size.p[0]; ++row)
    {
        EXPECT_EQ(saturated_sum.at<uchar>(row, 0), 255);
    }

    const int f32_lanes = reduce_test_lanes(CV_32F);
    Mat special({3, f32_lanes + 3}, CV_32FC1);
    for (int row = 0; row < special.size.p[0]; ++row)
    {
        for (int column = 0; column < special.size.p[1]; ++column)
        {
            special.at<float>(row, column) =
                static_cast<float>(row * 10 + column + 1);
        }
    }
    special.at<float>(0, 0) =
        std::numeric_limits<float>::quiet_NaN();
    special.at<float>(1, 1) =
        std::numeric_limits<float>::quiet_NaN();
    special.at<float>(2, 2) =
        std::numeric_limits<float>::infinity();
    special.at<float>(2, 3) =
        -std::numeric_limits<float>::infinity();
    for (int axis = 0; axis <= 1; ++axis)
    {
        for (int rtype : {REDUCE_MAX, REDUCE_MIN, REDUCE_SUM})
        {
            Mat expected;
            {
                DispatchModeGuard guard(
                    cpu::DispatchMode::ScalarOnly);
                reduce(
                    special,
                    expected,
                    axis,
                    rtype,
                    CV_64F);
            }
            Mat actual;
            {
                DispatchModeGuard guard(cpu::DispatchMode::Auto);
                reduce(
                    special,
                    actual,
                    axis,
                    rtype,
                    CV_64F);
            }
            expect_reduce_mat_close(
                actual, expected, 0.0, 0.0);
        }
    }

    Mat alias_source({9, u8_lanes + 3}, CV_8UC3);
    for (int row = 0; row < alias_source.size.p[0]; ++row)
    {
        for (int column = 0;
             column < alias_source.size.p[1];
             ++column)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                alias_source.at<uchar>(row, column, channel) =
                    static_cast<uchar>(
                        (row + column + channel) % 23);
            }
        }
    }
    Mat expected_alias;
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        reduce(
            alias_source,
            expected_alias,
            0,
            REDUCE_SUM2,
            CV_64F);
    }
    Mat alias = alias_source.clone();
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        reduce(alias, alias, 0, REDUCE_SUM2, CV_64F);
    }
    expect_reduce_mat_close(
        alias, expected_alias, 0.0, 0.0);

    Mat unsupported({3, 2 * u8_lanes + 1}, CV_16SC1);
    unsupported.setTo(Scalar::all(3.0));
    Mat unsupported_result;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        reduce(
            unsupported,
            unsupported_result,
            1,
            REDUCE_SUM,
            CV_64F);
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            cpu::DispatchTag::Scalar);
    }

    Mat short_row({3, std::max(1, u8_lanes - 1)}, CV_8UC1);
    short_row.setTo(Scalar::all(4.0));
    Mat short_result;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        reduce(
            short_row,
            short_result,
            1,
            REDUCE_SUM,
            CV_64F);
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            cpu::DispatchTag::Scalar);
    }
}

TEST(ReductionOpsContract_TEST, reduce_arg_ties_support_first_and_last_index)
{
    Mat src({3, 4}, CV_32FC1);
    const float values[] = {
        1.0f, 9.0f, 3.0f, 9.0f,
        1.0f, 5.0f, 3.0f, 9.0f,
        2.0f, 5.0f, 0.0f, 8.0f,
    };
    for (int y = 0; y < 3; ++y)
    {
        for (int x = 0; x < 4; ++x)
        {
            src.at<float>(y, x) = values[y * 4 + x];
        }
    }

    Mat indices;
    reduceArgMin(src, indices, 0, false);
    EXPECT_EQ(indices.at<int>(0, 0), 0);
    EXPECT_EQ(indices.at<int>(0, 1), 1);
    EXPECT_EQ(indices.at<int>(0, 2), 2);

    reduceArgMin(src, indices, 0, true);
    EXPECT_EQ(indices.at<int>(0, 0), 1);
    EXPECT_EQ(indices.at<int>(0, 1), 2);

    reduceArgMax(src, indices, 1, false);
    EXPECT_EQ(indices.at<int>(0, 0), 1);
    EXPECT_EQ(indices.at<int>(1, 0), 3);

    reduceArgMax(src, indices, 1, true);
    EXPECT_EQ(indices.at<int>(0, 0), 3);
    EXPECT_EQ(indices.at<int>(1, 0), 3);
}

TEST(ReductionOpsContract_TEST,
     reduce_arg_ui_matches_scalar_across_depths_axes_roi_and_tail)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }
    const int depths[] = {
        CV_8U,
        CV_8S,
        CV_16U,
        CV_16S,
        CV_32S,
        CV_32U,
        CV_16F,
        CV_32F,
        CV_64F,
    };

    for (const int depth : depths)
    {
        Mat parent({17, 263}, CV_MAKETYPE(depth, 1));
        for (int row = 0; row < 17; ++row)
        {
            for (int column = 0; column < 263; ++column)
            {
                const int seed = row * 13 + column * 7;
                double value = static_cast<double>(seed % 29 + 30);
                if (depth == CV_8S || depth == CV_16S ||
                    depth == CV_32S || depth == CV_16F ||
                    depth == CV_32F || depth == CV_64F)
                {
                    value = static_cast<double>(seed % 29 - 14);
                }
                set_test_value(parent, row, column, 0, value);
            }
        }
        Mat src = parent.colRange(2, 261);
        ASSERT_FALSE(src.isContinuous());

        for (int axis = 0; axis <= 1; ++axis)
        {
            for (int last = 0; last <= 1; ++last)
            {
                Mat scalar_min;
                Mat scalar_max;
                {
                    DispatchModeGuard guard(
                        cpu::DispatchMode::ScalarOnly);
                    reduceArgMin(src, scalar_min, axis, last != 0);
                    reduceArgMax(src, scalar_max, axis, last != 0);
                }

                Mat auto_min;
                Mat auto_max;
                {
                    DispatchModeGuard guard(cpu::DispatchMode::Auto);
                    cpu::reset_last_dispatch_tag();
                    reduceArgMin(src, auto_min, axis, last != 0);
                    EXPECT_EQ(
                        cpu::last_dispatch_tag(),
                        expected_statistics_auto_tag(
                            depth, ui_enabled));
                    cpu::reset_last_dispatch_tag();
                    reduceArgMax(src, auto_max, axis, last != 0);
                    EXPECT_EQ(
                        cpu::last_dispatch_tag(),
                        expected_statistics_auto_tag(
                            depth, ui_enabled));
                }
                expect_index_mat_equal(auto_min, scalar_min);
                expect_index_mat_equal(auto_max, scalar_max);
            }
        }
    }
}

TEST(ReductionOpsContract_TEST,
     reduce_arg_ui_preserves_nan_signed_zero_constant_and_alias_semantics)
{
    Mat rows({3, 33}, CV_32FC1);
    rows.setTo(Scalar::all(5.0));
    rows.at<float>(0, 0) =
        std::numeric_limits<float>::quiet_NaN();
    rows.at<float>(1, 5) = 0.0f;
    rows.at<float>(1, 9) = -0.0f;
    rows.at<float>(1, 17) =
        std::numeric_limits<float>::quiet_NaN();

    Mat indices;
    reduceArgMin(rows, indices, 1, false);
    EXPECT_EQ(indices.at<int>(0, 0), 0);
    EXPECT_EQ(indices.at<int>(1, 0), 5);
    EXPECT_EQ(indices.at<int>(2, 0), 0);
    reduceArgMin(rows, indices, 1, true);
    EXPECT_EQ(indices.at<int>(0, 0), 0);
    EXPECT_EQ(indices.at<int>(1, 0), 9);
    EXPECT_EQ(indices.at<int>(2, 0), 32);
    reduceArgMax(rows, indices, 1, false);
    EXPECT_EQ(indices.at<int>(0, 0), 0);
    EXPECT_EQ(indices.at<int>(2, 0), 0);
    reduceArgMax(rows, indices, 1, true);
    EXPECT_EQ(indices.at<int>(0, 0), 0);
    EXPECT_EQ(indices.at<int>(2, 0), 32);

    Mat columns({17, 33}, CV_32FC1);
    columns.setTo(Scalar::all(2.0));
    reduceArgMin(columns, indices, 0, false);
    EXPECT_EQ(indices.at<int>(0, 0), 0);
    EXPECT_EQ(indices.at<int>(0, 32), 0);
    reduceArgMin(columns, indices, 0, true);
    EXPECT_EQ(indices.at<int>(0, 0), 16);
    EXPECT_EQ(indices.at<int>(0, 32), 16);

    Mat expected;
    reduceArgMax(columns, expected, 1, true);
    Mat alias = columns.clone();
    reduceArgMax(alias, alias, 1, true);
    expect_index_mat_equal(alias, expected);

    Mat short_row({1, 1}, CV_32FC1);
    short_row.at<float>(0, 0) = 3.0f;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        reduceArgMin(short_row, indices, 1, false);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    }
    EXPECT_EQ(indices.at<int>(0, 0), 0);
}

TEST(ReductionOpsContract_TEST,
     norm_ui_matches_scalar_for_single_diff_mask_channels_roi_and_tail)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }
    const int depths[] = {CV_8U, CV_16S, CV_32F};
    const int norm_types[] = {NORM_INF, NORM_L1, NORM_L2};

    for (const int depth : depths)
    {
        for (int channels = 1; channels <= 4; ++channels)
        {
            Mat first_parent({3, 71}, CV_MAKETYPE(depth, channels));
            Mat second_parent({3, 71}, CV_MAKETYPE(depth, channels));
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 71; ++column)
                {
                    for (int channel = 0; channel < channels; ++channel)
                    {
                        const int seed =
                            row * 31 + column * 7 + channel * 5;
                        double first_value =
                            static_cast<double>(seed % 41 + 1);
                        double second_value =
                            static_cast<double>((seed * 3) % 37 + 2);
                        if (depth != CV_8U)
                        {
                            first_value =
                                static_cast<double>(seed % 41 - 20);
                            second_value =
                                static_cast<double>((seed * 3) % 37 - 18);
                        }
                        if (depth == CV_32F)
                        {
                            first_value *= 0.25;
                            second_value *= 0.125;
                        }
                        set_test_value(
                            first_parent,
                            row,
                            column,
                            channel,
                            first_value);
                        set_test_value(
                            second_parent,
                            row,
                            column,
                            channel,
                            second_value);
                    }
                }
            }
            Mat first = first_parent.colRange(2, 69);
            Mat second = second_parent.colRange(2, 69);
            ASSERT_FALSE(first.isContinuous());
            ASSERT_FALSE(second.isContinuous());

            Mat mask({3, 67}, CV_8UC1);
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < 67; ++column)
                {
                    mask.at<uchar>(row, column) =
                        (column < 29 || column >= 36) ? 255 : 0;
                }
            }

            for (const int norm_type : norm_types)
            {
                for (int masked = 0; masked <= 1; ++masked)
                {
                    const Mat& active_mask = masked != 0 ? mask : Mat();
                    double scalar_single = 0.0;
                    double scalar_diff = 0.0;
                    {
                        DispatchModeGuard guard(
                            cpu::DispatchMode::ScalarOnly);
                        scalar_single =
                            norm(first, norm_type, active_mask);
                        scalar_diff =
                            norm(first, second, norm_type, active_mask);
                    }

                    DispatchModeGuard guard(cpu::DispatchMode::Auto);
                    cpu::reset_last_dispatch_tag();
                    const double auto_single =
                        norm(first, norm_type, active_mask);
                    EXPECT_EQ(
                        cpu::last_dispatch_tag(),
                        expected_norm_auto_tag(depth, ui_enabled));
                    cpu::reset_last_dispatch_tag();
                    const double auto_diff =
                        norm(first, second, norm_type, active_mask);
                    EXPECT_EQ(
                        cpu::last_dispatch_tag(),
                        expected_norm_auto_tag(depth, ui_enabled));

                    const double relative_tolerance =
                        depth == CV_32F ? 1e-6 : 0.0;
                    EXPECT_NEAR(
                        auto_single,
                        scalar_single,
                        relative_tolerance *
                            std::max(1.0, std::fabs(scalar_single)));
                    EXPECT_NEAR(
                        auto_diff,
                        scalar_diff,
                        relative_tolerance *
                            std::max(1.0, std::fabs(scalar_diff)));
                }
            }
        }
    }

    Mat short_row({1, 3}, CV_8UC1);
    short_row.setTo(Scalar::all(2.0));
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        EXPECT_DOUBLE_EQ(norm(short_row, NORM_L1), 6.0);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    }
}

TEST(ReductionOpsContract_TEST,
     norm_ui_preserves_nan_inf_and_wide_difference_semantics)
{
    Mat first({1, 33}, CV_32FC1);
    Mat second({1, 33}, CV_32FC1);
    first.setTo(Scalar::all(2.0));
    second.setTo(Scalar::all(-3.0));
    first.at<float>(0, 7) =
        std::numeric_limits<float>::infinity();
    second.at<float>(0, 7) =
        std::numeric_limits<float>::infinity();
    first.at<float>(0, 19) =
        std::numeric_limits<float>::quiet_NaN();

    EXPECT_TRUE(std::isnan(norm(first, NORM_INF)));
    EXPECT_TRUE(std::isnan(norm(first, NORM_L1)));
    EXPECT_TRUE(std::isnan(norm(first, NORM_L2)));
    EXPECT_TRUE(std::isnan(norm(first, second, NORM_INF)));
    EXPECT_TRUE(std::isnan(norm(first, second, NORM_L1)));
    EXPECT_TRUE(std::isnan(norm(first, second, NORM_L2)));

    Mat u8_low({1, 65}, CV_8UC1);
    Mat u8_high({1, 65}, CV_8UC1);
    u8_low.setTo(Scalar::all(0.0));
    u8_high.setTo(Scalar::all(255.0));
    EXPECT_DOUBLE_EQ(norm(u8_low, u8_high, NORM_INF), 255.0);
    EXPECT_DOUBLE_EQ(norm(u8_low, u8_high, NORM_L1), 65.0 * 255.0);
    EXPECT_DOUBLE_EQ(
        norm(u8_low, u8_high, NORM_L2),
        std::sqrt(65.0 * 255.0 * 255.0));

    Mat f32_high({1, 33}, CV_32FC1);
    Mat f32_low({1, 33}, CV_32FC1);
    f32_high.setTo(Scalar::all(
        std::numeric_limits<float>::max()));
    f32_low.setTo(Scalar::all(
        -std::numeric_limits<float>::max()));
    const double wide_difference =
        2.0 * static_cast<double>(
                  std::numeric_limits<float>::max());
    EXPECT_DOUBLE_EQ(
        norm(f32_high, f32_low, NORM_INF),
        wide_difference);
    EXPECT_NEAR(
        norm(f32_high, f32_low, NORM_L1),
        33.0 * wide_difference,
        1e-12 * 33.0 * wide_difference);
    EXPECT_NEAR(
        norm(f32_high, f32_low, NORM_L2),
        std::sqrt(33.0) * wide_difference,
        1e-12 * std::sqrt(33.0) * wide_difference);
}

TEST(ReductionOpsContract_TEST,
     normalize_ui_matches_scalar_for_mask_alias_dtype_and_constant_minmax)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }

    Mat parent({3, 71}, CV_32FC3);
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 71; ++column)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                set_test_value(
                    parent,
                    row,
                    column,
                    channel,
                    static_cast<double>(
                        row * 17 + column * 3 + channel - 40) *
                        0.125);
            }
        }
    }
    Mat src = parent.colRange(2, 69);
    ASSERT_FALSE(src.isContinuous());
    Mat mask({3, 67}, CV_8UC1);
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 67; ++column)
        {
            mask.at<uchar>(row, column) =
                (column < 31 || column >= 39) ? 1 : 0;
        }
    }

    for (const int norm_type : {NORM_INF, NORM_L1, NORM_L2})
    {
        Mat scalar_dst(src.shape(), src.type());
        Mat auto_dst(src.shape(), src.type());
        scalar_dst.setTo(Scalar::all(9.0));
        auto_dst.setTo(Scalar::all(9.0));
        {
            DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
            normalize(
                src,
                scalar_dst,
                3.0,
                0.0,
                norm_type,
                -1,
                mask);
        }
        {
            DispatchModeGuard guard(cpu::DispatchMode::Auto);
            cpu::reset_last_dispatch_tag();
            normalize(
                src,
                auto_dst,
                3.0,
                0.0,
                norm_type,
                -1,
                mask);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                expected_norm_auto_tag(CV_32F, ui_enabled));
        }
        expect_f32_mat_close(auto_dst, scalar_dst, 2e-6f);
    }

    Mat scalar_alias = src.clone();
    Mat auto_alias = src.clone();
    {
        DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
        normalize(
            scalar_alias,
            scalar_alias,
            -2.0,
            5.0,
            NORM_MINMAX);
    }
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        normalize(
            auto_alias,
            auto_alias,
            -2.0,
            5.0,
            NORM_MINMAX);
        EXPECT_EQ(
            cpu::last_dispatch_tag(),
            expected_norm_auto_tag(CV_32F, ui_enabled));
    }
    expect_f32_mat_close(auto_alias, scalar_alias, 2e-6f);

    Mat converted;
    normalize(src, converted, 2.0, 0.0, NORM_L2, CV_64F);
    EXPECT_EQ(converted.type(), CV_64FC3);

    Mat constant({2, 65}, CV_32FC1);
    constant.setTo(Scalar::all(7.0));
    Mat constant_dst;
    normalize(
        constant,
        constant_dst,
        -4.0,
        9.0,
        NORM_MINMAX);
    for (size_t index = 0; index < constant_dst.total(); ++index)
    {
        EXPECT_FLOAT_EQ(
            reinterpret_cast<const float*>(constant_dst.data)[index],
            -4.0f);
    }
}

TEST(ReductionOpsContract_TEST, normalize_supports_norms_minmax_dtype_mask_and_alias)
{
    Mat src({1, 4}, CV_32FC1);
    src.at<float>(0, 0) = 1.0f;
    src.at<float>(0, 1) = 2.0f;
    src.at<float>(0, 2) = 3.0f;
    src.at<float>(0, 3) = 4.0f;

    Mat dst;
    normalize(src, dst, 1.0, 0.0, NORM_L1);
    EXPECT_NEAR(norm(dst, NORM_L1), 1.0, 1e-6);
    normalize(src, dst, 2.0, 0.0, NORM_L2);
    EXPECT_NEAR(norm(dst, NORM_L2), 2.0, 1e-6);
    normalize(src, dst, 5.0, 0.0, NORM_INF);
    EXPECT_NEAR(norm(dst, NORM_INF), 5.0, 1e-6);

    normalize(src, dst, 10.0, 20.0, NORM_MINMAX, CV_64F);
    ASSERT_EQ(dst.type(), CV_64FC1);
    EXPECT_DOUBLE_EQ(dst.at<double>(0, 0), 10.0);
    EXPECT_DOUBLE_EQ(dst.at<double>(0, 3), 20.0);

    Mat mask({1, 4}, CV_8UC1);
    mask.at<uchar>(0, 0) = 0;
    mask.at<uchar>(0, 1) = 255;
    mask.at<uchar>(0, 2) = 255;
    mask.at<uchar>(0, 3) = 0;
    dst = Mat();
    normalize(src, dst, 1.0, 0.0, NORM_L1, -1, mask);
    EXPECT_FLOAT_EQ(dst.at<float>(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(dst.at<float>(0, 1), 0.4f);
    EXPECT_FLOAT_EQ(dst.at<float>(0, 2), 0.6f);
    EXPECT_FLOAT_EQ(dst.at<float>(0, 3), 0.0f);

    normalize(src, src, 0.0, 1.0, NORM_MINMAX);
    EXPECT_FLOAT_EQ(src.at<float>(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(src.at<float>(0, 3), 1.0f);
}

TEST(ReductionOpsContract_TEST, empty_zero_nan_inf_and_thread_settings_are_stable)
{
    Mat empty;
    EXPECT_EQ(sum(empty), Scalar());
    EXPECT_EQ(mean(empty), Scalar());
    EXPECT_DOUBLE_EQ(norm(empty), 0.0);
    EXPECT_EQ(countNonZero(empty), 0);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    EXPECT_FALSE(hasNonZero(empty));
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    std::vector<Point> empty_points = {Point(1, 2)};
    findNonZero(empty, empty_points);
    EXPECT_TRUE(empty_points.empty());
    Mat empty_indices({1, 1}, CV_32SC2);
    findNonZero(empty, empty_indices);
    EXPECT_TRUE(empty_indices.empty());

    double empty_min = -1.0;
    double empty_max = -1.0;
    Point empty_min_location;
    Point empty_max_location;
    minMaxLoc(
        empty,
        &empty_min,
        &empty_max,
        &empty_min_location,
        &empty_max_location);
    EXPECT_DOUBLE_EQ(empty_min, 0.0);
    EXPECT_DOUBLE_EQ(empty_max, 0.0);
    EXPECT_EQ(empty_min_location, Point(-1, -1));
    EXPECT_EQ(empty_max_location, Point(-1, -1));

    Mat zero({20, 30}, CV_32FC1);
    zero.setTo(Scalar::all(0.0));
    EXPECT_DOUBLE_EQ(norm(zero, NORM_L2), 0.0);
    Mat normalized;
    normalize(zero, normalized, 1.0, 0.0, NORM_L2);
    EXPECT_DOUBLE_EQ(norm(normalized, NORM_L2), 0.0);

    Mat special({1, 3}, CV_64FC1);
    special.at<double>(0, 0) = 1.0;
    special.at<double>(0, 1) = std::numeric_limits<double>::infinity();
    special.at<double>(0, 2) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(sum(special)[0]));
    EXPECT_TRUE(std::isnan(norm(special, NORM_L2)));

    Mat large({200, 300}, CV_32FC1);
    for (size_t i = 0; i < large.total(); ++i)
    {
        reinterpret_cast<float*>(large.data)[i] =
            static_cast<float>(static_cast<int>(i % 37) - 18) * 0.25f;
    }
    const int previous_threads = getNumThreads();
    setNumThreads(1);
    const double single_thread = norm(large, NORM_L2);
    setNumThreads(4);
    const double configured_multi = norm(large, NORM_L2);
    setNumThreads(previous_threads);
    EXPECT_DOUBLE_EQ(single_thread, configured_multi);
}

TEST(ReductionOpsContract_TEST, invalid_axes_types_and_masks_throw)
{
    Mat c3({2, 3}, CV_8UC3);
    Mat c1({2, 3}, CV_8UC1);
    Mat bad_mask({2, 3}, CV_8UC3);
    Mat out;

    EXPECT_THROW(mean(c3, bad_mask), Exception);
    EXPECT_THROW(reduce(c1, out, 2, REDUCE_SUM), Exception);
    EXPECT_THROW(reduce(c1, out, 0, 99), Exception);
    EXPECT_THROW(reduceArgMin(c3, out, 0), Exception);
    EXPECT_THROW(normalize(c1, out, 1.0, 0.0, 999), Exception);
}
