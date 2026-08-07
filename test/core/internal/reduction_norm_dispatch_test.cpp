#include "test/core/support/reduction_internal_test_utils.hpp"

TEST(ReductionNormDispatchInternalTest,
     norm_ui_matches_scalar_for_single_diff_mask_channels_roi_and_tail)
{
    bool ui_enabled = false;
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        ui_enabled = detail::reduce_ui::enabled();
    }
    const int depths[] = {CV_8U, CV_16S, CV_32F};
    const int norm_types[] = {NORM_INF, NORM_L1, NORM_L2};
    const int selected_run =
        cvh::test::fixed_width_opencv_ui_lanes<uchar>() + 3;
    const int mask_gap = 7;
    const int roi_columns = 2 * selected_run + mask_gap;
    const int parent_columns = roi_columns + 4;

    for (const int depth : depths)
    {
        for (int channels = 1; channels <= 4; ++channels)
        {
            Mat first_parent(
                {3, parent_columns},
                CV_MAKETYPE(depth, channels));
            Mat second_parent(
                {3, parent_columns},
                CV_MAKETYPE(depth, channels));
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < parent_columns; ++column)
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
            Mat first = first_parent.colRange(2, 2 + roi_columns);
            Mat second = second_parent.colRange(2, 2 + roi_columns);
            ASSERT_FALSE(first.isContinuous());
            ASSERT_FALSE(second.isContinuous());

            Mat mask({3, roi_columns}, CV_8UC1);
            for (int row = 0; row < 3; ++row)
            {
                for (int column = 0; column < roi_columns; ++column)
                {
                    mask.at<uchar>(row, column) =
                        (column < selected_run ||
                         column >= selected_run + mask_gap)
                        ? 255
                        : 0;
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
                    const cpu::DispatchTag expected_auto_tag =
                        depth == CV_32F && channels == 1 && masked == 0 &&
                                cpu::neon_runtime_available()
                            ? cpu::DispatchTag::NEON
                            : expected_norm_auto_tag(depth, ui_enabled);
                    cpu::reset_last_dispatch_tag();
                    const double auto_single =
                        norm(first, norm_type, active_mask);
                    EXPECT_EQ(
                        cpu::last_dispatch_tag(),
                        expected_auto_tag);
                    cpu::reset_last_dispatch_tag();
                    const double auto_diff =
                        norm(first, second, norm_type, active_mask);
                    EXPECT_EQ(
                        cpu::last_dispatch_tag(),
                        expected_auto_tag);

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

    const int fragmented_columns =
        cvh::test::accepted_fixed_width_test_length<uchar>();
    Mat fragmented({1, fragmented_columns}, CV_8UC1);
    Mat fragmented_mask({1, fragmented_columns}, CV_8UC1);
    fragmented.setTo(Scalar::all(2.0));
    int selected_count = 0;
    for (int column = 0; column < fragmented_columns; ++column)
    {
        const bool selected = column % 2 == 0;
        fragmented_mask.at<uchar>(0, column) = selected ? 255 : 0;
        selected_count += selected ? 1 : 0;
    }
    {
        DispatchModeGuard guard(cpu::DispatchMode::Auto);
        cpu::reset_last_dispatch_tag();
        EXPECT_DOUBLE_EQ(
            norm(fragmented, NORM_L1, fragmented_mask),
            2.0 * selected_count);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    }
}

TEST(ReductionNormDispatchInternalTest,
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

TEST(ReductionNormDispatchInternalTest,
     norm_f32_c1_direct_neon_covers_forced_modes_roi_and_tail)
{
    Mat first_owner({5, 75}, CV_32FC1);
    Mat second_owner({5, 75}, CV_32FC1);
    for (int row = 0; row < 5; ++row)
    {
        for (int column = 0; column < 75; ++column)
        {
            first_owner.at<float>(row, column) =
                static_cast<float>((row * 31 + column * 7) % 43 - 21) *
                0.125f;
            second_owner.at<float>(row, column) =
                static_cast<float>((row * 19 + column * 11) % 37 - 18) *
                0.25f;
        }
    }
    Mat first = first_owner.colRange(2, 73);
    Mat second = second_owner.colRange(2, 73);
    ASSERT_FALSE(first.isContinuous());

    for (const int norm_type : {NORM_INF, NORM_L1, NORM_L2})
    {
        double expected_single = 0.0;
        double expected_diff = 0.0;
        {
            DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
            expected_single = norm(first, norm_type);
            expected_diff = norm(first, second, norm_type);
        }
        {
            DispatchModeGuard guard(cpu::DispatchMode::NeonOnly);
            cpu::reset_last_dispatch_tag();
            const double actual_single = norm(first, norm_type);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                cpu::neon_runtime_available()
                    ? cpu::DispatchTag::NEON
                    : cpu::DispatchTag::Scalar);
            if (cpu::neon_runtime_available())
            {
                EXPECT_NE(
                    std::string(cpu::last_kernel_route()).find("merge=f64"),
                    std::string::npos);
            }
            cpu::reset_last_dispatch_tag();
            const double actual_diff = norm(first, second, norm_type);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                cpu::neon_runtime_available()
                    ? cpu::DispatchTag::NEON
                    : cpu::DispatchTag::Scalar);
            const double tolerance = 1e-12 *
                std::max({1.0, std::fabs(expected_single),
                          std::fabs(expected_diff)});
            EXPECT_NEAR(actual_single, expected_single, tolerance);
            EXPECT_NEAR(actual_diff, expected_diff, tolerance);
        }
        {
            DispatchModeGuard guard(cpu::DispatchMode::OpenCVUIOnly);
            cpu::reset_last_dispatch_tag();
            (void)norm(first, norm_type);
            EXPECT_NE(cpu::last_dispatch_tag(), cpu::DispatchTag::NEON);
        }

        Mat expected_normalized;
        Mat actual_normalized;
        {
            DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
            normalize(
                first,
                expected_normalized,
                3.0,
                0.0,
                norm_type,
                CV_32F);
        }
        {
            DispatchModeGuard guard(cpu::DispatchMode::NeonOnly);
            cpu::reset_last_dispatch_tag();
            normalize(
                first,
                actual_normalized,
                3.0,
                0.0,
                norm_type,
                CV_32F);
            EXPECT_EQ(
                cpu::last_dispatch_tag(),
                cpu::neon_runtime_available()
                    ? cpu::DispatchTag::NEON
                    : cpu::DispatchTag::Scalar);
        }
        expect_f32_mat_close(
            actual_normalized, expected_normalized, 2e-6f);
    }
}

TEST(ReductionNormDispatchInternalTest,
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

TEST(ReductionNormDispatchInternalTest,
     norm_f32_block_merge_stays_within_contract_across_multiple_chunks)
{
    Mat first({1, 4099}, CV_32FC1);
    Mat second({1, 4099}, CV_32FC1);
    for (int column = 0; column < first.size[1]; ++column)
    {
        first.at<float>(0, column) =
            static_cast<float>((column * 37) % 997 - 498) / 31.0f;
        second.at<float>(0, column) =
            static_cast<float>((column * 53) % 991 - 495) / 29.0f;
    }
    for (const int norm_type : {NORM_INF, NORM_L1, NORM_L2})
    {
        double expected_single = 0.0;
        double expected_diff = 0.0;
        {
            DispatchModeGuard guard(cpu::DispatchMode::ScalarOnly);
            expected_single = norm(first, norm_type);
            expected_diff = norm(first, second, norm_type);
        }
        double actual_single = 0.0;
        double actual_diff = 0.0;
        {
            DispatchModeGuard guard(cpu::DispatchMode::NeonOnly);
            actual_single = norm(first, norm_type);
            actual_diff = norm(first, second, norm_type);
        }
        const double single_tolerance = 1e-6 *
            std::max(1.0, std::fabs(expected_single));
        const double diff_tolerance = 1e-6 *
            std::max(1.0, std::fabs(expected_diff));
        EXPECT_NEAR(actual_single, expected_single, single_tolerance);
        EXPECT_NEAR(actual_diff, expected_diff, diff_tolerance);
    }
}
