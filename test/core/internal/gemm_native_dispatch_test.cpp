#include "test/core/support/gemm_test_utils.hpp"
#include "test/support/dispatch_mode_guard.hpp"

#include "cvh/core/detail/cpu_features.hpp"

namespace {

cvh::cpu::DispatchMode available_native_mode()
{
    if (cvh::cpu::native_neon_runtime_available())
    {
        return cvh::cpu::DispatchMode::NeonOnly;
    }
    return cvh::cpu::DispatchMode::Avx2Only;
}

cvh::cpu::DispatchTag available_native_tag()
{
    if (cvh::cpu::native_neon_runtime_available())
    {
        return cvh::cpu::DispatchTag::NativeNEON;
    }
    return cvh::cpu::DispatchTag::NativeAVX2;
}

bool native_available()
{
    return cvh::cpu::native_neon_runtime_available() ||
           cvh::cpu::native_avx2_runtime_available();
}

}  // namespace

TEST(GemmNativeDispatchTest, forced_native_nn_matches_scalar_with_tails)
{
    if (!native_available())
    {
        GTEST_SKIP() << "No native GEMM ISA is available";
    }

    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat a({17, 65}, CV_32F);
    Mat b({65, 33}, CV_32F);
    fill_mat_deterministic(a, 0.7f, -0.2f);
    fill_mat_deterministic(b, 0.5f, 0.1f);

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    const Mat reference = gemm(a, b);

    cpu::set_dispatch_mode(available_native_mode());
    const Mat actual = gemm(a, b);

    expect_mat_close(actual, reference, 2e-3f, 2e-4f);
    EXPECT_EQ(cpu::last_dispatch_tag(), available_native_tag());
}

TEST(GemmNativeDispatchTest, auto_selects_native_for_dense_fp32)
{
    if (!native_available())
    {
        GTEST_SKIP() << "No native GEMM ISA is available";
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
    EXPECT_EQ(cpu::last_dispatch_tag(), available_native_tag());
}

TEST(GemmNativeDispatchTest, forced_native_nt_matches_scalar)
{
    if (!native_available())
    {
        GTEST_SKIP() << "No native GEMM ISA is available";
    }

    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat a({19, 67}, CV_32F);
    Mat b_transposed({35, 67}, CV_32F);
    fill_mat_deterministic(a, 0.8f, -0.1f);
    fill_mat_deterministic(b_transposed, 0.6f, 0.2f);

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    const Mat reference = gemm(a, b_transposed, false, true);

    cpu::set_dispatch_mode(available_native_mode());
    const Mat actual = gemm(a, b_transposed, false, true);

    expect_mat_close(actual, reference, 2e-3f, 2e-4f);
    EXPECT_EQ(cpu::last_dispatch_tag(), available_native_tag());
}

TEST(GemmNativeDispatchTest, packed_b_reuses_native_panel)
{
    if (!cvh::cpu::native_neon_runtime_available())
    {
        GTEST_SKIP() << "Persistent native B panels are NEON-specific";
    }

    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat a({23, 61}, CV_32F);
    Mat b({61, 37}, CV_32F);
    fill_mat_deterministic(a, 0.9f, -0.3f);
    fill_mat_deterministic(b, 0.4f, 0.2f);

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    const Mat reference = gemm(a, b);
    const GemmPackedB packed = gemm_pack_b(b);
    ASSERT_FALSE(packed.native_packed_fp32.empty());

    cpu::set_dispatch_mode(cpu::DispatchMode::NeonOnly);
    const Mat actual = gemm(a, packed);

    expect_mat_close(actual, reference, 2e-3f, 2e-4f);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::NativeNEON);
}

TEST(GemmNativeDispatchTest, neon_packed_fp16_matches_scalar)
{
    if (!cvh::cpu::native_neon_runtime_available())
    {
        GTEST_SKIP() << "FP16 native packing is NEON-specific";
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
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::NativeNEON);
}

TEST(GemmNativeDispatchTest, scalar_ui_and_native_modes_are_distinct)
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
#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
#endif
}
