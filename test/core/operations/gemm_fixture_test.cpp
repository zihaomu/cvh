#include "test/core/support/core_fixture_test_utils.hpp"
#include "test/utils/mat_load.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

using namespace cvh;

namespace
{

struct GemmCase
{
    const char* name;
    bool transpose_a;
    bool transpose_b;
    float abs_tolerance;
    float rel_tolerance;
};

void fill_deterministic(Mat& matrix, float first_scale, float second_scale)
{
    float* data = reinterpret_cast<float*>(matrix.data);
    for (size_t index = 0; index < matrix.total(); ++index)
    {
        const float value = static_cast<float>(index);
        data[index] =
            std::sin(value * first_scale) * 0.8f +
            std::cos(value * second_scale) * 0.4f;
    }
}

void quantize_int8_per_row_for_test(
    const Mat& source,
    Mat& quantized,
    Mat& scales)
{
    ASSERT_EQ(source.dims, 2);
    ASSERT_EQ(source.type(), CV_32F);

    const int rows = source.size[0];
    const int cols = source.size[1];
    quantized.create({rows, cols}, CV_8S);
    scales.create({rows}, CV_32F);

    for (int row = 0; row < rows; ++row)
    {
        const float* source_row =
            reinterpret_cast<const float*>(source.data) +
            static_cast<size_t>(row) * static_cast<size_t>(cols);
        std::int8_t* quantized_row =
            reinterpret_cast<std::int8_t*>(quantized.data) +
            static_cast<size_t>(row) * static_cast<size_t>(cols);
        float maximum = 0.0f;
        for (int col = 0; col < cols; ++col)
            maximum = std::max(maximum, std::abs(source_row[col]));

        const float scale =
            maximum > std::numeric_limits<float>::epsilon()
                ? maximum / 127.0f
                : 1.0f;
        scales.at<float>(row) = scale;
        for (int col = 0; col < cols; ++col)
        {
            const float rounded =
                std::round(source_row[col] / scale);
            quantized_row[col] =
                static_cast<std::int8_t>(
                    std::clamp(rounded, -127.0f, 127.0f));
        }
    }
}

}  // namespace

TEST(GemmFixtureTest, generated_shape_and_transpose_cases_match_numpy)
{
    const std::vector<GemmCase> cases = {
        {"nn_small_odd", false, false, 1e-3f, 1e-4f},
        {"nn_tail_rect", false, false, 1e-3f, 1e-4f},
        {"nn_rank3", false, false, 1e-3f, 1e-4f},
        {"nn_rank4", false, false, 1e-3f, 1e-4f},
        {"nn_broadcast_a", false, false, 1e-3f, 1e-4f},
        {"nn_broadcast_b", false, false, 1e-3f, 1e-4f},
        {"nn_rank_mismatch_a", false, false, 1e-3f, 1e-4f},
        {"nn_rank_mismatch_b", false, false, 1e-3f, 1e-4f},
        {"nt_small_odd", false, true, 1e-3f, 1e-4f},
        {"nt_tail_rect", false, true, 1e-3f, 1e-4f},
        {"nt_rank3", false, true, 1e-3f, 1e-4f},
        {"nt_broadcast_a", false, true, 1e-3f, 1e-4f},
        {"nt_broadcast_b", false, true, 1e-3f, 1e-4f},
        {"nt_rank_mismatch", false, true, 1e-3f, 1e-4f},
        {"tn_basic", true, false, 1e-3f, 1e-4f},
        {"tn_rank3", true, false, 1e-3f, 1e-4f},
        {"tn_broadcast", true, false, 1e-3f, 1e-4f},
        {"tt_basic", true, true, 1e-3f, 1e-4f},
        {"tt_rank3", true, true, 1e-3f, 1e-4f},
        {"tt_broadcast", true, true, 1e-3f, 1e-4f},
        {"nn_large_value", false, false, 5e-2f, 5e-4f},
        {"nt_small_value", false, true, 1e-6f, 5e-4f},
    };

    for (const GemmCase& test_case : cases)
    {
        SCOPED_TRACE(test_case.name);
        const std::string prefix =
            std::string("gemm_") + test_case.name;
        const Mat first =
            readMatFromNpy(test::core_data_path(prefix + "_a.npy"));
        const Mat second =
            readMatFromNpy(test::core_data_path(prefix + "_b.npy"));
        const Mat expected =
            readMatFromNpy(test::core_data_path(prefix + "_o.npy"));
        const Mat actual =
            gemm(
                first,
                second,
                test_case.transpose_a,
                test_case.transpose_b);
        test::expect_mat_close(
            actual,
            expected,
            test_case.abs_tolerance,
            test_case.rel_tolerance,
            test_case.name);
    }
}

TEST(GemmFixtureTest, fp16_weight_matrix_matches_fp32_reference)
{
    Mat first({1, 3, 8}, CV_32F);
    Mat weights({6, 8}, CV_32F);
    fill_deterministic(first, 0.19f, 0.07f);
    fill_deterministic(weights, 0.13f, 0.05f);

    Mat fp16_weights;
    weights.convertTo(fp16_weights, CV_16F);
    const Mat expected = gemm(first, weights, false, true);
    const Mat actual = gemm(first, fp16_weights, false, true);
    test::expect_mat_close(
        actual,
        expected,
        5e-2f,
        1e-2f,
        "gemm_fp16_weight_matrix");
}

TEST(GemmFixtureTest, int8_weight_matrix_matches_fp32_reference)
{
    Mat first({1, 4, 8}, CV_32F);
    Mat weights({7, 8}, CV_32F);
    fill_deterministic(first, 0.17f, 0.03f);
    fill_deterministic(weights, 0.11f, 0.09f);

    Mat quantized_weights;
    Mat scales;
    quantize_int8_per_row_for_test(
        weights,
        quantized_weights,
        scales);

    const Mat expected = gemm(first, weights, false, true);
    const Mat actual =
        gemm(
            first,
            quantized_weights,
            scales,
            false,
            true);
    test::expect_mat_close(
        actual,
        expected,
        2.5e-1f,
        8e-2f,
        "gemm_int8_weight_matrix");
}
