#include "test/core/support/array_layout_test_utils.hpp"
#include "test/support/dispatch_mode_guard.hpp"

TEST(ArrayLayoutDispatchInternalTest, masked_copy_matches_scalar_for_common_u8_layouts_and_tail)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    const int types[] = {CV_8UC1, CV_8SC3, CV_8UC4};
    for (int type : types)
    {
        SCOPED_TRACE(type);
        Mat source_parent({5, 41}, type);
        Mat mask_parent({5, 41}, CV_8UC1);
        fill_incrementing_bytes(source_parent, 17);
        fill_incrementing_bytes(mask_parent, 31);
        Mat source =
            source_parent(Range(1, 4), Range(2, 39));
        Mat mask =
            mask_parent(Range(1, 4), Range(2, 39));
        ASSERT_FALSE(source.isContinuous());
        ASSERT_FALSE(mask.isContinuous());

        Mat scalar_dst(source.shape(), source.type());
        scalar_dst.setTo(Scalar(3, 5, 7, 11));
        cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
        copyTo(source, scalar_dst, mask);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

        Mat auto_dst(source.shape(), source.type());
        auto_dst.setTo(Scalar(3, 5, 7, 11));
        cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
        copyTo(source, auto_dst, mask);
        expect_same_bytes(scalar_dst, auto_dst);
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
#endif
    }
    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
}

TEST(ArrayLayoutDispatchInternalTest, channel_ui_matches_scalar_for_c3_c4_roi_and_tail)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    for (int type : {CV_8UC3, CV_8SC4})
    {
        SCOPED_TRACE(type);
        Mat parent({6, 43}, type);
        fill_incrementing_bytes(parent, 23);
        Mat source = parent(Range(1, 5), Range(3, 40));
        ASSERT_FALSE(source.isContinuous());
        const int channels = source.channels();

        Mat scalar_extract;
        cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
        extractChannel(source, scalar_extract, channels - 1);
        Mat auto_extract;
        cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
        extractChannel(source, auto_extract, channels - 1);
        expect_same_bytes(scalar_extract, auto_extract);
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
#endif

        Mat scalar_insert = source.clone();
        cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
        insertChannel(scalar_extract, scalar_insert, 1);
        Mat auto_insert = source.clone();
        cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
        insertChannel(auto_extract, auto_insert, 1);
        expect_same_bytes(scalar_insert, auto_insert);

        std::vector<int> routes;
        for (int channel = 0; channel < channels; ++channel)
        {
            routes.push_back(channel);
            routes.push_back(channels - 1 - channel);
        }
        Mat scalar_reordered(source.shape(), source.type());
        cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
        mixChannels(
            &source,
            1,
            &scalar_reordered,
            1,
            routes.data(),
            static_cast<size_t>(channels));
        Mat auto_reordered(source.shape(), source.type());
        cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
        mixChannels(
            &source,
            1,
            &auto_reordered,
            1,
            routes.data(),
            static_cast<size_t>(channels));
        expect_same_bytes(scalar_reordered, auto_reordered);
    }
    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
}

TEST(ArrayLayoutDispatchInternalTest, flip_and_rotate_ui_match_scalar_for_roi_and_tail)
{
    const cvh::test::DispatchModeGuard dispatch_mode_guard;
    Mat parent({7, 43}, CV_8UC3);
    fill_incrementing_bytes(parent, 19);
    Mat source = parent(Range(1, 6), Range(3, 40));
    ASSERT_FALSE(source.isContinuous());

    for (int flip_code : {1, -1})
    {
        SCOPED_TRACE(flip_code);
        Mat scalar_dst;
        cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
        flip(source, scalar_dst, flip_code);
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::Scalar);

        Mat auto_dst;
        cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
        flip(source, auto_dst, flip_code);
        expect_same_bytes(scalar_dst, auto_dst);
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
        EXPECT_EQ(cpu::last_dispatch_tag(), cpu::DispatchTag::OpenCVUI);
#endif
    }

    for (int rotate_code :
         {ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE})
    {
        SCOPED_TRACE(rotate_code);
        Mat scalar_dst;
        cpu::set_dispatch_mode(cpu::DispatchMode::ScalarOnly);
        rotate(source, scalar_dst, rotate_code);

        Mat auto_dst;
        cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
        rotate(source, auto_dst, rotate_code);
        expect_same_bytes(scalar_dst, auto_dst);

        Mat alias = source.clone();
        rotate(alias, alias, rotate_code);
        expect_same_bytes(auto_dst, alias);
    }
    cpu::set_dispatch_mode(cpu::DispatchMode::Auto);
}
