#include "test/core/support/gemm_test_utils.hpp"
#include "test/support/dispatch_mode_guard.hpp"

#include "cvh/core/detail/cpu_features.hpp"

namespace {

cvh::cpu::DispatchMode available_isa_mode()
{
    if (cvh::cpu::neon_runtime_available())
    {
        return cvh::cpu::DispatchMode::NeonOnly;
    }
    return cvh::cpu::DispatchMode::Avx2Only;
}

cvh::cpu::DispatchTag available_isa_tag()
{
    if (cvh::cpu::neon_runtime_available())
    {
        return cvh::cpu::DispatchTag::NEON;
    }
    return cvh::cpu::DispatchTag::AVX2;
}

bool isa_available()
{
    return cvh::cpu::neon_runtime_available() ||
           cvh::cpu::avx2_fma_runtime_available();
}

}  // namespace

TEST(GemmIsaDispatchTest, forced_isa_nn_matches_scalar_with_tails)
{
    if (!isa_available())
    {
        GTEST_SKIP() << "No specialized GEMM ISA is available";
    }

    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat a({17, 65}, CV_32F);
    Mat b({65, 33}, CV_32F);
    fill_mat_deterministic(a, 0.7f, -0.2f);
    fill_mat_deterministic(b, 0.5f, 0.1f);

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    const Mat reference = gemm(a, b);

    cpu::set_dispatch_mode(available_isa_mode());
    const Mat actual = gemm(a, b);

    expect_mat_close(actual, reference, 2e-3f, 2e-4f);
    EXPECT_EQ(cpu::last_dispatch_tag(), available_isa_tag());
}

TEST(GemmIsaDispatchTest, auto_selects_isa_for_dense_fp32)
{
    if (!isa_available())
    {
        GTEST_SKIP() << "No specialized GEMM ISA is available";
    }

    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat a({24, 65}, CV_32F);
    Mat b({65, 35}, CV_32F);
    fill_mat_deterministic(a, 0.6f, -0.2f);
    fill_mat_deterministic(b, 0.8f, 0.1f);

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    const Mat reference = gemm(a, b);

    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
    const Mat actual = gemm(a, b);

    expect_mat_close(actual, reference, 2e-3f, 2e-4f);
    EXPECT_EQ(cpu::last_dispatch_tag(), available_isa_tag());
}

TEST(GemmIsaDispatchTest, forced_isa_nt_matches_scalar)
{
    if (!isa_available())
    {
        GTEST_SKIP() << "No specialized GEMM ISA is available";
    }

    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat a({19, 67}, CV_32F);
    Mat b_transposed({35, 67}, CV_32F);
    fill_mat_deterministic(a, 0.8f, -0.1f);
    fill_mat_deterministic(b_transposed, 0.6f, 0.2f);

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    const Mat reference = gemm(a, b_transposed, false, true);

    cpu::set_dispatch_mode(available_isa_mode());
    const Mat actual = gemm(a, b_transposed, false, true);

    expect_mat_close(actual, reference, 2e-3f, 2e-4f);
    EXPECT_EQ(cpu::last_dispatch_tag(), available_isa_tag());
}

TEST(GemmIsaDispatchTest, packed_b_reuses_isa_panel)
{
    if (!cvh::cpu::neon_runtime_available())
    {
        GTEST_SKIP() << "Persistent ISA B panels are NEON-specific";
    }

    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat a({23, 61}, CV_32F);
    Mat b({61, 37}, CV_32F);
    fill_mat_deterministic(a, 0.9f, -0.3f);
    fill_mat_deterministic(b, 0.4f, 0.2f);

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    const Mat reference = gemm(a, b);
    const GemmPackedB packed = gemm_pack_b(b);
    ASSERT_FALSE(packed.isa_packed_fp32.empty());

    cpu::set_dispatch_mode(cpu::DispatchMode::NeonOnly);
    const Mat actual = gemm(a, packed);

    expect_mat_close(actual, reference, 2e-3f, 2e-4f);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::NEON);
}

TEST(GemmIsaDispatchTest, neon_packed_fp16_matches_scalar)
{
    if (!cvh::cpu::neon_runtime_available())
    {
        GTEST_SKIP() << "FP16 ISA packing is NEON-specific";
    }

    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat a({21, 63}, CV_32F);
    Mat b_fp32({63, 39}, CV_32F);
    fill_mat_deterministic(a, 0.7f, -0.1f);
    fill_mat_deterministic(b_fp32, 0.5f, 0.3f);
    Mat b_fp16;
    b_fp32.convertTo(b_fp16, CV_16F);

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    const Mat reference = gemm(a, b_fp16);
    const GemmPackedB packed = gemm_pack_b(b_fp16);

    cpu::set_dispatch_mode(cpu::DispatchMode::NeonOnly);
    const Mat actual = gemm(a, packed);

    expect_mat_close(actual, reference, 3e-2f, 8e-3f);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::NEON);
}

TEST(GemmIsaDispatchTest, scalar_ui_and_isa_modes_are_distinct)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat a({8, 33}, CV_32F);
    Mat b({33, 17}, CV_32F);
    fill_mat_deterministic(a, 0.8f, 0.1f);
    fill_mat_deterministic(b, 0.6f, -0.2f);

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    const Mat scalar = gemm(a, b);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

    cpu::set_dispatch_mode(cpu::DispatchMode::OpenCVUIOnly);
    const Mat ui = gemm(a, b);
    expect_mat_close(ui, scalar, 1e-3f, 1e-4f);
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
#endif
}
