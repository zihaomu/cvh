#include "test/core/support/gemm_test_utils.hpp"
#include "test/support/dispatch_mode_guard.hpp"

TEST(GemmDispatchInternalTest, fp32_ui_matches_scalar_for_nn_nt_and_vector_tail)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat a({13, 29}, CV_32F);
    Mat b({29, 19}, CV_32F);
    Mat b_nt({19, 29}, CV_32F);
    fill_mat_deterministic(a, 0.7f, -0.2f);
    fill_mat_deterministic(b, 0.5f, 0.1f);
    fill_mat_deterministic(b_nt, 0.5f, 0.1f);

    cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
    const Mat scalar_nn = gemm(a, b, false, false);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);
    const Mat scalar_nt = gemm(a, b_nt, false, true);
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
    const Mat ui_nn = gemm(a, b, false, false);
    expect_mat_close(ui_nn, scalar_nn, 1e-3f, 1e-4f);
    const Mat ui_nt = gemm(a, b_nt, false, true);
    expect_mat_close(ui_nt, scalar_nt, 1e-3f, 1e-4f);
#if CVH_ENABLE_OPENCV_INTRIN && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
#endif
    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
}
