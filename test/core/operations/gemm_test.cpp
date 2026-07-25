#include "test/core/support/gemm_test_utils.hpp"

TEST(GemmTest, fp32_packed_matches_reference)
{
    Mat a({2, 17, 31}, CV_32F);
    Mat b({31, 19}, CV_32F);
    fill_mat_deterministic(a, 0.9f, 0.2f);
    fill_mat_deterministic(b, 0.8f, -0.1f);

    const Mat ref = gemm(a, b, false, false);
    const GemmPackedB packed = gemm_pack_b(b);
    const Mat out = gemm(a, packed);

    expect_mat_close(out, ref, 1e-3f, 1e-4f);
}

TEST(GemmTest, fp32_packed_transposed_input_matches_reference)
{
    Mat a({3, 13, 29}, CV_32F);
    Mat b_nt({11, 29}, CV_32F);
    fill_mat_deterministic(a, 0.7f, 0.0f);
    fill_mat_deterministic(b_nt, 0.6f, 0.1f);

    const Mat ref = gemm(a, b_nt, false, true);
    const GemmPackedB packed = gemm_pack_b(b_nt, true);
    const Mat out = gemm(a, packed);

    expect_mat_close(out, ref, 1e-3f, 1e-4f);
}

TEST(GemmTest, fp16_packed_matches_reference)
{
    Mat a({2, 19, 23}, CV_32F);
    Mat b_fp32({23, 15}, CV_32F);
    fill_mat_deterministic(a, 1.0f, -0.3f);
    fill_mat_deterministic(b_fp32, 0.5f, 0.4f);

    Mat b_fp16;
    b_fp32.convertTo(b_fp16, CV_16F);

    const Mat ref = gemm(a, b_fp16, false, false);
    const GemmPackedB packed = gemm_pack_b(b_fp16);
    const Mat out = gemm(a, packed);

    expect_mat_close(out, ref, 3e-2f, 8e-3f);
}

TEST(GemmTest, packed_weights_can_be_reused_across_calls)
{
    Mat b({27, 21}, CV_32F);
    fill_mat_deterministic(b, 0.9f, 0.2f);
    const GemmPackedB packed = gemm_pack_b(b);

    Mat a0({1, 9, 27}, CV_32F);
    Mat a1({2, 7, 27}, CV_32F);
    fill_mat_deterministic(a0, 0.6f, -0.1f);
    fill_mat_deterministic(a1, 1.1f, 0.3f);

    const Mat ref0 = gemm(a0, b, false, false);
    const Mat ref1 = gemm(a1, b, false, false);

    const Mat out0 = gemm(a0, packed);
    const Mat out1 = gemm(a1, packed);

    expect_mat_close(out0, ref0, 1e-3f, 1e-4f);
    expect_mat_close(out1, ref1, 1e-3f, 1e-4f);
}

TEST(GemmTest, unsupported_weight_type_throws)
{
    Mat b({9, 13}, CV_8S);
    EXPECT_THROW((void)gemm_pack_b(b), Exception);
}
