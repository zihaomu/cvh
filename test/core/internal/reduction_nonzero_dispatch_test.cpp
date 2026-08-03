#include "test/core/support/reduction_internal_test_utils.hpp"

TEST(ReductionNonZeroDispatchInternalTest, nonzero_ui_covers_public_depths_and_fallbacks)
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

TEST(ReductionNonZeroDispatchInternalTest, nonzero_ui_preserves_float_special_values)
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

TEST(ReductionNonZeroDispatchInternalTest, nonzero_ui_handles_roi_tail_and_hit_positions)
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

TEST(ReductionNonZeroDispatchInternalTest, find_nonzero_handles_vector_boundaries)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
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
