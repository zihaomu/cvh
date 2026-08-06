#ifndef CVH_IMGPROC_DETAIL_CVTCOLOR_NEON_HPP
#define CVH_IMGPROC_DETAIL_CVTCOLOR_NEON_HPP

#include "fastpath_common.hpp"
#include "../../core/detail/cpu_features.hpp"

namespace cvh
{
namespace detail
{
namespace cvtcolor_neon
{

inline bool direct_neon_allowed()
{
    const cpu::DispatchMode mode = cpu::dispatch_mode();
    return cpu::neon_runtime_available() &&
           (mode == cpu::DispatchMode::Auto ||
            mode == cpu::DispatchMode::NeonOnly);
}

inline bool packed_workload_is_large_enough(int rows, int cols)
{
    // A vector loop must execute, and very small images keep the existing
    // scalar path to avoid paying dispatch/setup cost for a handful of bytes.
    return rows > 0 && cols >= 16 &&
           static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) >= 256;
}

#if CVH_DETAIL_HAVE_NEON_KERNEL

inline uint8x16_t gray_from_bgr_u8(
    uint8x16_t bb,
    uint8x16_t gg,
    uint8x16_t rr)
{
    constexpr unsigned short kB = 7471;
    constexpr unsigned short kG = 38470;
    constexpr unsigned short kR = 19595;

    const uint16x8_t b0 = vmovl_u8(vget_low_u8(bb));
    const uint16x8_t b1 = vmovl_u8(vget_high_u8(bb));
    const uint16x8_t g0 = vmovl_u8(vget_low_u8(gg));
    const uint16x8_t g1 = vmovl_u8(vget_high_u8(gg));
    const uint16x8_t r0 = vmovl_u8(vget_low_u8(rr));
    const uint16x8_t r1 = vmovl_u8(vget_high_u8(rr));

    auto gray8 = [](uint16x8_t b, uint16x8_t g, uint16x8_t r) {
        uint32x4_t lo = vmull_n_u16(vget_low_u16(b), kB);
        lo = vmlal_n_u16(lo, vget_low_u16(g), kG);
        lo = vmlal_n_u16(lo, vget_low_u16(r), kR);
        uint32x4_t hi = vmull_n_u16(vget_high_u16(b), kB);
        hi = vmlal_n_u16(hi, vget_high_u16(g), kG);
        hi = vmlal_n_u16(hi, vget_high_u16(r), kR);
        return vcombine_u16(vrshrn_n_u32(lo, 16), vrshrn_n_u32(hi, 16));
    };

    return vmovn_high_u16(vmovn_u16(gray8(b0, g0, r0)),
                          gray8(b1, g1, r1));
}

inline void swap_rb_3ch(const Mat& src, Mat& dst)
{
    const bool do_parallel = should_parallelize_cvtcolor(
        src.size[0], src.size[1], 3);
    parallel_for_index_if(do_parallel, src.size[0], [&](int y) {
        const uchar* src_row = src.data + static_cast<std::size_t>(y) * src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        int x = 0;
        for (; x + 16 <= src.size[1]; x += 16)
        {
            uint8x16x3_t pixels = vld3q_u8(src_row + static_cast<std::size_t>(x) * 3);
            const uint8x16_t tmp = pixels.val[0];
            pixels.val[0] = pixels.val[2];
            pixels.val[2] = tmp;
            vst3q_u8(dst_row + static_cast<std::size_t>(x) * 3, pixels);
        }
        for (; x < src.size[1]; ++x)
        {
            const std::size_t offset = static_cast<std::size_t>(x) * 3;
            const uchar c0 = src_row[offset + 0];
            const uchar c1 = src_row[offset + 1];
            const uchar c2 = src_row[offset + 2];
            dst_row[offset + 0] = c2;
            dst_row[offset + 1] = c1;
            dst_row[offset + 2] = c0;
        }
    });
}

inline void color3_to_color4(const Mat& src, Mat& dst, bool swap_rb)
{
    const uint8x16_t alpha = vdupq_n_u8(255);
    const bool do_parallel = should_parallelize_cvtcolor(
        src.size[0], src.size[1], 3);
    parallel_for_index_if(do_parallel, src.size[0], [&](int y) {
        const uchar* src_row = src.data + static_cast<std::size_t>(y) * src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        int x = 0;
        for (; x + 16 <= src.size[1]; x += 16)
        {
            const uint8x16x3_t input =
                vld3q_u8(src_row + static_cast<std::size_t>(x) * 3);
            uint8x16x4_t output;
            output.val[0] = swap_rb ? input.val[2] : input.val[0];
            output.val[1] = input.val[1];
            output.val[2] = swap_rb ? input.val[0] : input.val[2];
            output.val[3] = alpha;
            vst4q_u8(dst_row + static_cast<std::size_t>(x) * 4, output);
        }
        for (; x < src.size[1]; ++x)
        {
            const std::size_t sx = static_cast<std::size_t>(x) * 3;
            const std::size_t dx = static_cast<std::size_t>(x) * 4;
            dst_row[dx + 0] = src_row[sx + (swap_rb ? 2 : 0)];
            dst_row[dx + 1] = src_row[sx + 1];
            dst_row[dx + 2] = src_row[sx + (swap_rb ? 0 : 2)];
            dst_row[dx + 3] = 255;
        }
    });
}

inline void color4_to_color3(const Mat& src, Mat& dst, bool swap_rb)
{
    const bool do_parallel = should_parallelize_cvtcolor(
        src.size[0], src.size[1], 4);
    parallel_for_index_if(do_parallel, src.size[0], [&](int y) {
        const uchar* src_row = src.data + static_cast<std::size_t>(y) * src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        int x = 0;
        for (; x + 16 <= src.size[1]; x += 16)
        {
            const uint8x16x4_t input =
                vld4q_u8(src_row + static_cast<std::size_t>(x) * 4);
            uint8x16x3_t output;
            output.val[0] = swap_rb ? input.val[2] : input.val[0];
            output.val[1] = input.val[1];
            output.val[2] = swap_rb ? input.val[0] : input.val[2];
            vst3q_u8(dst_row + static_cast<std::size_t>(x) * 3, output);
        }
        for (; x < src.size[1]; ++x)
        {
            const std::size_t sx = static_cast<std::size_t>(x) * 4;
            const std::size_t dx = static_cast<std::size_t>(x) * 3;
            dst_row[dx + 0] = src_row[sx + (swap_rb ? 2 : 0)];
            dst_row[dx + 1] = src_row[sx + 1];
            dst_row[dx + 2] = src_row[sx + (swap_rb ? 0 : 2)];
        }
    });
}

inline void swap_rb_4ch(const Mat& src, Mat& dst)
{
    const bool do_parallel = should_parallelize_cvtcolor(
        src.size[0], src.size[1], 4);
    parallel_for_index_if(do_parallel, src.size[0], [&](int y) {
        const uchar* src_row = src.data + static_cast<std::size_t>(y) * src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        int x = 0;
        for (; x + 16 <= src.size[1]; x += 16)
        {
            uint8x16x4_t pixels = vld4q_u8(src_row + static_cast<std::size_t>(x) * 4);
            const uint8x16_t tmp = pixels.val[0];
            pixels.val[0] = pixels.val[2];
            pixels.val[2] = tmp;
            vst4q_u8(dst_row + static_cast<std::size_t>(x) * 4, pixels);
        }
        for (; x < src.size[1]; ++x)
        {
            const std::size_t offset = static_cast<std::size_t>(x) * 4;
            const uchar c0 = src_row[offset + 0];
            const uchar c1 = src_row[offset + 1];
            const uchar c2 = src_row[offset + 2];
            const uchar c3 = src_row[offset + 3];
            dst_row[offset + 0] = c2;
            dst_row[offset + 1] = c1;
            dst_row[offset + 2] = c0;
            dst_row[offset + 3] = c3;
        }
    });
}

inline void color4_to_gray(const Mat& src, Mat& dst, bool rgba_order)
{
    constexpr int kB = 7471;
    constexpr int kG = 38470;
    constexpr int kR = 19595;
    constexpr int kRound = 1 << 15;
    const bool do_parallel = should_parallelize_cvtcolor(
        src.size[0], src.size[1], 4);
    parallel_for_index_if(do_parallel, src.size[0], [&](int y) {
        const uchar* src_row = src.data + static_cast<std::size_t>(y) * src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        int x = 0;
        for (; x + 16 <= src.size[1]; x += 16)
        {
            const uint8x16x4_t input =
                vld4q_u8(src_row + static_cast<std::size_t>(x) * 4);
            const uint8x16_t bb = rgba_order ? input.val[2] : input.val[0];
            const uint8x16_t rr = rgba_order ? input.val[0] : input.val[2];
            vst1q_u8(dst_row + x, gray_from_bgr_u8(bb, input.val[1], rr));
        }
        for (; x < src.size[1]; ++x)
        {
            const std::size_t sx = static_cast<std::size_t>(x) * 4;
            dst_row[x] = static_cast<uchar>(
                (kB * src_row[sx + (rgba_order ? 2 : 0)] +
                 kG * src_row[sx + 1] +
                 kR * src_row[sx + (rgba_order ? 0 : 2)] + kRound) >> 16);
        }
    });
}

inline void gray_to_color3(const Mat& src, Mat& dst)
{
    const bool do_parallel = should_parallelize_cvtcolor(
        src.size[0], src.size[1], 1);
    parallel_for_index_if(do_parallel, src.size[0], [&](int y) {
        const uchar* src_row = src.data + static_cast<std::size_t>(y) * src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        int x = 0;
        for (; x + 16 <= src.size[1]; x += 16)
        {
            const uint8x16_t gray = vld1q_u8(src_row + x);
            const uint8x16x3_t output = {{gray, gray, gray}};
            vst3q_u8(dst_row + static_cast<std::size_t>(x) * 3, output);
        }
        for (; x < src.size[1]; ++x)
        {
            const uchar gray = src_row[x];
            const std::size_t dx = static_cast<std::size_t>(x) * 3;
            dst_row[dx + 0] = gray;
            dst_row[dx + 1] = gray;
            dst_row[dx + 2] = gray;
        }
    });
}

inline void gray_to_color4(const Mat& src, Mat& dst)
{
    const uint8x16_t alpha = vdupq_n_u8(255);
    const bool do_parallel = should_parallelize_cvtcolor(
        src.size[0], src.size[1], 1);
    parallel_for_index_if(do_parallel, src.size[0], [&](int y) {
        const uchar* src_row = src.data + static_cast<std::size_t>(y) * src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        int x = 0;
        for (; x + 16 <= src.size[1]; x += 16)
        {
            const uint8x16_t gray = vld1q_u8(src_row + x);
            const uint8x16x4_t output = {{gray, gray, gray, alpha}};
            vst4q_u8(dst_row + static_cast<std::size_t>(x) * 4, output);
        }
        for (; x < src.size[1]; ++x)
        {
            const uchar gray = src_row[x];
            const std::size_t dx = static_cast<std::size_t>(x) * 4;
            dst_row[dx + 0] = gray;
            dst_row[dx + 1] = gray;
            dst_row[dx + 2] = gray;
            dst_row[dx + 3] = 255;
        }
    });
}

inline void yuv_limited_to_bgr8(
    uint8x8_t yy,
    uint8x8_t uu,
    uint8x8_t vv,
    uint8x8_t& bb,
    uint8x8_t& gg,
    uint8x8_t& rr)
{
    const uint16x8_t y16 = vmovl_u8(yy);
    const uint16x8_t c_unsigned = vqsubq_u16(y16, vdupq_n_u16(16));
    const int16x8_t cc = vreinterpretq_s16_u16(c_unsigned);
    const int16x8_t dd = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(uu)), vdupq_n_s16(128));
    const int16x8_t ee = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(vv)), vdupq_n_s16(128));

    auto finish = [](int32x4_t lo, int32x4_t hi) {
        lo = vaddq_s32(lo, vdupq_n_s32(128));
        hi = vaddq_s32(hi, vdupq_n_s32(128));
        return vqmovun_s16(vcombine_s16(
            vqshrn_n_s32(lo, 8), vqshrn_n_s32(hi, 8)));
    };

    int32x4_t b_lo = vmull_n_s16(vget_low_s16(cc), 298);
    int32x4_t b_hi = vmull_n_s16(vget_high_s16(cc), 298);
    b_lo = vmlal_n_s16(b_lo, vget_low_s16(dd), 516);
    b_hi = vmlal_n_s16(b_hi, vget_high_s16(dd), 516);
    bb = finish(b_lo, b_hi);

    int32x4_t g_lo = vmull_n_s16(vget_low_s16(cc), 298);
    int32x4_t g_hi = vmull_n_s16(vget_high_s16(cc), 298);
    g_lo = vmlsl_n_s16(g_lo, vget_low_s16(dd), 100);
    g_hi = vmlsl_n_s16(g_hi, vget_high_s16(dd), 100);
    g_lo = vmlsl_n_s16(g_lo, vget_low_s16(ee), 208);
    g_hi = vmlsl_n_s16(g_hi, vget_high_s16(ee), 208);
    gg = finish(g_lo, g_hi);

    int32x4_t r_lo = vmull_n_s16(vget_low_s16(cc), 298);
    int32x4_t r_hi = vmull_n_s16(vget_high_s16(cc), 298);
    r_lo = vmlal_n_s16(r_lo, vget_low_s16(ee), 409);
    r_hi = vmlal_n_s16(r_hi, vget_high_s16(ee), 409);
    rr = finish(r_lo, r_hi);
}

inline void store_yuv_limited_16(
    uint8x16_t y16,
    uint8x8_t chroma_u,
    uint8x8_t chroma_v,
    uchar* dst_ptr,
    bool rgb_order)
{
    const uint8x8x2_t duplicated_u = vzip_u8(chroma_u, chroma_u);
    const uint8x8x2_t duplicated_v = vzip_u8(chroma_v, chroma_v);
    uint8x8_t b0;
    uint8x8_t g0;
    uint8x8_t r0;
    uint8x8_t b1;
    uint8x8_t g1;
    uint8x8_t r1;
    yuv_limited_to_bgr8(
        vget_low_u8(y16), duplicated_u.val[0], duplicated_v.val[0],
        b0, g0, r0);
    yuv_limited_to_bgr8(
        vget_high_u8(y16), duplicated_u.val[1], duplicated_v.val[1],
        b1, g1, r1);
    uint8x16x3_t output;
    output.val[rgb_order ? 2 : 0] = vcombine_u8(b0, b1);
    output.val[1] = vcombine_u8(g0, g1);
    output.val[rgb_order ? 0 : 2] = vcombine_u8(r0, r1);
    vst3q_u8(dst_ptr, output);
}

inline const uchar* logical_plane_ptr(
    const uchar* data,
    std::size_t step,
    int rows,
    int cols,
    int plane_offset,
    int index)
{
    const int logical_offset = plane_offset + index;
    return data +
           static_cast<std::size_t>(rows + logical_offset / cols) * step +
           static_cast<std::size_t>(logical_offset % cols);
}

inline uchar* logical_plane_ptr(
    uchar* data,
    std::size_t step,
    int rows,
    int cols,
    int plane_offset,
    int index)
{
    const int logical_offset = plane_offset + index;
    return data +
           static_cast<std::size_t>(rows + logical_offset / cols) * step +
           static_cast<std::size_t>(logical_offset % cols);
}

inline uint8x8_t limited_color_component8(
    uint8x8_t bb,
    uint8x8_t gg,
    uint8x8_t rr,
    int coefficient_b,
    int coefficient_g,
    int coefficient_r,
    int delta)
{
    const int16x8_t b = vreinterpretq_s16_u16(vmovl_u8(bb));
    const int16x8_t g = vreinterpretq_s16_u16(vmovl_u8(gg));
    const int16x8_t r = vreinterpretq_s16_u16(vmovl_u8(rr));
    int32x4_t lo = vmull_n_s16(vget_low_s16(b), coefficient_b);
    int32x4_t hi = vmull_n_s16(vget_high_s16(b), coefficient_b);
    lo = vmlal_n_s16(lo, vget_low_s16(g), coefficient_g);
    hi = vmlal_n_s16(hi, vget_high_s16(g), coefficient_g);
    lo = vmlal_n_s16(lo, vget_low_s16(r), coefficient_r);
    hi = vmlal_n_s16(hi, vget_high_s16(r), coefficient_r);
    lo = vaddq_s32(lo, vdupq_n_s32(128));
    hi = vaddq_s32(hi, vdupq_n_s32(128));
    const int16x8_t shifted = vcombine_s16(
        vshrn_n_s32(lo, 8), vshrn_n_s32(hi, 8));
    return vqmovun_s16(vaddq_s16(shifted, vdupq_n_s16(delta)));
}

inline void color_to_yuv_limited8(
    uint8x8_t bb,
    uint8x8_t gg,
    uint8x8_t rr,
    uint8x8_t& yy,
    uint8x8_t& uu,
    uint8x8_t& vv)
{
    yy = limited_color_component8(bb, gg, rr, 25, 129, 66, 16);
    uu = limited_color_component8(bb, gg, rr, 112, -74, -38, 128);
    vv = limited_color_component8(bb, gg, rr, -18, -94, 112, 128);
}

inline uint8x8_t average_adjacent_pairs(uint8x16_t values)
{
    const uint16x8_t sums = vpaddlq_u8(values);
    return vshrn_n_u16(vaddq_u16(sums, vdupq_n_u16(1)), 1);
}

inline uint8x8_t average_2x2(uint8x16_t row0, uint8x16_t row1)
{
    uint16x8_t sums = vaddq_u16(vpaddlq_u8(row0), vpaddlq_u8(row1));
    sums = vaddq_u16(sums, vdupq_n_u16(2));
    return vshrn_n_u16(sums, 2);
}

inline void color_to_yuv_limited16(
    uint8x16_t bb,
    uint8x16_t gg,
    uint8x16_t rr,
    uint8x16_t& yy)
{
    uint8x8_t y0;
    uint8x8_t u0;
    uint8x8_t v0;
    uint8x8_t y1;
    uint8x8_t u1;
    uint8x8_t v1;
    color_to_yuv_limited8(
        vget_low_u8(bb), vget_low_u8(gg), vget_low_u8(rr),
        y0, u0, v0);
    color_to_yuv_limited8(
        vget_high_u8(bb), vget_high_u8(gg), vget_high_u8(rr),
        y1, u1, v1);
    yy = vcombine_u8(y0, y1);
}

inline void color3_to_yuv420(
    const Mat& src,
    Mat& dst,
    bool rgb_order,
    bool planar,
    bool reversed_uv)
{
    const int rows = src.size[0];
    const int cols = src.size[1];
    const int uv_size = rows * cols / 4;
    const int u_plane_offset = reversed_uv ? uv_size : 0;
    const int v_plane_offset = reversed_uv ? 0 : uv_size;
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if_step(do_parallel, 0, rows, 2, [&](int y) {
        const uchar* src_row0 = src.data + static_cast<std::size_t>(y) * src.step(0);
        const uchar* src_row1 = src_row0 + src.step(0);
        uchar* y_row0 = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        uchar* y_row1 = y_row0 + dst.step(0);
        uchar* uv_row = dst.data +
            static_cast<std::size_t>(rows + y / 2) * dst.step(0);
        int x = 0;
        for (; x + 16 <= cols; x += 16)
        {
            const uint8x16x3_t input0 =
                vld3q_u8(src_row0 + static_cast<std::size_t>(x) * 3);
            const uint8x16x3_t input1 =
                vld3q_u8(src_row1 + static_cast<std::size_t>(x) * 3);
            const uint8x16_t b0 = input0.val[rgb_order ? 2 : 0];
            const uint8x16_t g0 = input0.val[1];
            const uint8x16_t r0 = input0.val[rgb_order ? 0 : 2];
            const uint8x16_t b1 = input1.val[rgb_order ? 2 : 0];
            const uint8x16_t g1 = input1.val[1];
            const uint8x16_t r1 = input1.val[rgb_order ? 0 : 2];
            uint8x16_t yy0;
            uint8x16_t yy1;
            color_to_yuv_limited16(b0, g0, r0, yy0);
            color_to_yuv_limited16(b1, g1, r1, yy1);
            vst1q_u8(y_row0 + x, yy0);
            vst1q_u8(y_row1 + x, yy1);

            const uint8x8_t avg_b = average_2x2(b0, b1);
            const uint8x8_t avg_g = average_2x2(g0, g1);
            const uint8x8_t avg_r = average_2x2(r0, r1);
            uint8x8_t unused_y;
            uint8x8_t uu;
            uint8x8_t vv;
            color_to_yuv_limited8(
                avg_b, avg_g, avg_r, unused_y, uu, vv);
            if (planar)
            {
                const int chroma_index = (y / 2) * (cols / 2) + x / 2;
                vst1_u8(logical_plane_ptr(
                    dst.data, dst.step(0), rows, cols,
                    u_plane_offset, chroma_index), uu);
                vst1_u8(logical_plane_ptr(
                    dst.data, dst.step(0), rows, cols,
                    v_plane_offset, chroma_index), vv);
            }
            else
            {
                uint8x8x2_t uv;
                uv.val[reversed_uv ? 1 : 0] = uu;
                uv.val[reversed_uv ? 0 : 1] = vv;
                vst2_u8(uv_row + x, uv);
            }
        }
        for (; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;
            for (int dy = 0; dy < 2; ++dy)
            {
                const uchar* src_row = dy == 0 ? src_row0 : src_row1;
                uchar* y_row = dy == 0 ? y_row0 : y_row1;
                for (int dx = 0; dx < 2; ++dx)
                {
                    const int sx = (x + dx) * 3;
                    const int b = src_row[sx + (rgb_order ? 2 : 0)];
                    const int g = src_row[sx + 1];
                    const int r = src_row[sx + (rgb_order ? 0 : 2)];
                    y_row[x + dx] =
                        cvtcolor_color3_to_yuv_limited_u8(b, g, r, 0);
                    sum_b += b;
                    sum_g += g;
                    sum_r += r;
                }
            }
            const int avg_b = (sum_b + 2) >> 2;
            const int avg_g = (sum_g + 2) >> 2;
            const int avg_r = (sum_r + 2) >> 2;
            const uchar u = cvtcolor_color3_to_yuv_limited_u8(
                avg_b, avg_g, avg_r, 1);
            const uchar v = cvtcolor_color3_to_yuv_limited_u8(
                avg_b, avg_g, avg_r, 2);
            if (planar)
            {
                const int chroma_index = (y / 2) * (cols / 2) + x / 2;
                *logical_plane_ptr(dst.data, dst.step(0), rows, cols,
                                   u_plane_offset, chroma_index) = u;
                *logical_plane_ptr(dst.data, dst.step(0), rows, cols,
                                   v_plane_offset, chroma_index) = v;
            }
            else
            {
                uv_row[x + (reversed_uv ? 1 : 0)] = u;
                uv_row[x + (reversed_uv ? 0 : 1)] = v;
            }
        }
    });
}

inline void color3_to_yuv422packed(
    const Mat& src,
    Mat& dst,
    bool rgb_order,
    bool uyvy_layout)
{
    const int rows = src.size[0];
    const int cols = src.size[1];
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 3);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src.data + static_cast<std::size_t>(y) * src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        int x = 0;
        for (; x + 16 <= cols; x += 16)
        {
            const uint8x16x3_t input =
                vld3q_u8(src_row + static_cast<std::size_t>(x) * 3);
            const uint8x16_t bb = input.val[rgb_order ? 2 : 0];
            const uint8x16_t gg = input.val[1];
            const uint8x16_t rr = input.val[rgb_order ? 0 : 2];
            uint8x16_t yy;
            color_to_yuv_limited16(bb, gg, rr, yy);
            const uint8x8_t avg_b = average_adjacent_pairs(bb);
            const uint8x8_t avg_g = average_adjacent_pairs(gg);
            const uint8x8_t avg_r = average_adjacent_pairs(rr);
            uint8x8_t unused_y;
            uint8x8_t uu;
            uint8x8_t vv;
            color_to_yuv_limited8(
                avg_b, avg_g, avg_r, unused_y, uu, vv);
            const uint8x8x2_t uv_halves = vzip_u8(uu, vv);
            const uint8x16_t uv =
                vcombine_u8(uv_halves.val[0], uv_halves.val[1]);
            uint8x16x2_t packed;
            packed.val[uyvy_layout ? 1 : 0] = yy;
            packed.val[uyvy_layout ? 0 : 1] = uv;
            vst2q_u8(dst_row + static_cast<std::size_t>(x) * 2, packed);
        }
        for (; x < cols; x += 2)
        {
            int sum_b = 0;
            int sum_g = 0;
            int sum_r = 0;
            uchar yy[2];
            for (int i = 0; i < 2; ++i)
            {
                const int sx = (x + i) * 3;
                const int b = src_row[sx + (rgb_order ? 2 : 0)];
                const int g = src_row[sx + 1];
                const int r = src_row[sx + (rgb_order ? 0 : 2)];
                yy[i] = cvtcolor_color3_to_yuv_limited_u8(b, g, r, 0);
                sum_b += b;
                sum_g += g;
                sum_r += r;
            }
            const int avg_b = (sum_b + 1) >> 1;
            const int avg_g = (sum_g + 1) >> 1;
            const int avg_r = (sum_r + 1) >> 1;
            const uchar u = cvtcolor_color3_to_yuv_limited_u8(
                avg_b, avg_g, avg_r, 1);
            const uchar v = cvtcolor_color3_to_yuv_limited_u8(
                avg_b, avg_g, avg_r, 2);
            const int base = x * 2;
            if (uyvy_layout)
            {
                dst_row[base + 0] = u;
                dst_row[base + 1] = yy[0];
                dst_row[base + 2] = v;
                dst_row[base + 3] = yy[1];
            }
            else
            {
                dst_row[base + 0] = yy[0];
                dst_row[base + 1] = u;
                dst_row[base + 2] = yy[1];
                dst_row[base + 3] = v;
            }
        }
    });
}

inline uint8x8_t round_saturate_float8(float32x4_t low, float32x4_t high)
{
    // cvh::saturate_cast<uchar>(float) uses std::round: halfway cases are
    // rounded away from zero. AArch64 FCVTA has the same rounding rule.
    const int32x4_t low_i32 = vcvtaq_s32_f32(low);
    const int32x4_t high_i32 = vcvtaq_s32_f32(high);
    return vqmovun_s16(vcombine_s16(
        vqmovn_s32(low_i32), vqmovn_s32(high_i32)));
}

inline void expand_u8_to_float8(
    uint8x8_t values,
    float32x4_t& low,
    float32x4_t& high)
{
    const uint16x8_t u16 = vmovl_u8(values);
    low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(u16)));
    high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(u16)));
}

inline void color_to_yuv_float8(
    uint8x8_t bb,
    uint8x8_t gg,
    uint8x8_t rr,
    uint8x8_t& yy,
    uint8x8_t& uu,
    uint8x8_t& vv)
{
    float32x4_t b0;
    float32x4_t b1;
    float32x4_t g0;
    float32x4_t g1;
    float32x4_t r0;
    float32x4_t r1;
    expand_u8_to_float8(bb, b0, b1);
    expand_u8_to_float8(gg, g0, g1);
    expand_u8_to_float8(rr, r0, r1);

    // Match the AArch64 scalar compiler's FP contraction and evaluation
    // order. A non-fused rewrite differs at exact half-way U8 cases.
    float32x4_t y0 = vmulq_n_f32(g0, 0.587f);
    float32x4_t y1 = vmulq_n_f32(g1, 0.587f);
    y0 = vfmaq_n_f32(y0, r0, 0.299f);
    y1 = vfmaq_n_f32(y1, r1, 0.299f);
    y0 = vfmaq_n_f32(y0, b0, 0.114f);
    y1 = vfmaq_n_f32(y1, b1, 0.114f);
    const float32x4_t u0 = vfmaq_n_f32(
        vdupq_n_f32(128.0f), vsubq_f32(b0, y0), 0.492f);
    const float32x4_t u1 = vfmaq_n_f32(
        vdupq_n_f32(128.0f), vsubq_f32(b1, y1), 0.492f);
    const float32x4_t v0 = vfmaq_n_f32(
        vdupq_n_f32(128.0f), vsubq_f32(r0, y0), 0.877f);
    const float32x4_t v1 = vfmaq_n_f32(
        vdupq_n_f32(128.0f), vsubq_f32(r1, y1), 0.877f);
    yy = round_saturate_float8(y0, y1);
    uu = round_saturate_float8(u0, u1);
    vv = round_saturate_float8(v0, v1);
}

inline void yuv_float_to_color8(
    uint8x8_t yy,
    uint8x8_t uu,
    uint8x8_t vv,
    uint8x8_t& bb,
    uint8x8_t& gg,
    uint8x8_t& rr)
{
    float32x4_t y0;
    float32x4_t y1;
    float32x4_t u0;
    float32x4_t u1;
    float32x4_t v0;
    float32x4_t v1;
    expand_u8_to_float8(yy, y0, y1);
    expand_u8_to_float8(uu, u0, u1);
    expand_u8_to_float8(vv, v0, v1);
    u0 = vsubq_f32(u0, vdupq_n_f32(128.0f));
    u1 = vsubq_f32(u1, vdupq_n_f32(128.0f));
    v0 = vsubq_f32(v0, vdupq_n_f32(128.0f));
    v1 = vsubq_f32(v1, vdupq_n_f32(128.0f));

    const float32x4_t b0 = vfmaq_n_f32(y0, u0, 2.032f);
    const float32x4_t b1 = vfmaq_n_f32(y1, u1, 2.032f);
    float32x4_t g0 = vfmaq_n_f32(y0, u0, -0.395f);
    float32x4_t g1 = vfmaq_n_f32(y1, u1, -0.395f);
    g0 = vfmaq_n_f32(g0, v0, -0.581f);
    g1 = vfmaq_n_f32(g1, v1, -0.581f);
    const float32x4_t r0 = vfmaq_n_f32(y0, v0, 1.140f);
    const float32x4_t r1 = vfmaq_n_f32(y1, v1, 1.140f);
    bb = round_saturate_float8(b0, b1);
    gg = round_saturate_float8(g0, g1);
    rr = round_saturate_float8(r0, r1);
}

inline void color3_to_yuv_interleaved(
    const Mat& src,
    Mat& dst,
    bool rgb_order)
{
    const bool do_parallel = should_parallelize_cvtcolor(
        src.size[0], src.size[1], 3);
    parallel_for_index_if(do_parallel, src.size[0], [&](int y) {
        const uchar* src_row = src.data + static_cast<std::size_t>(y) * src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        int x = 0;
        for (; x + 16 <= src.size[1]; x += 16)
        {
            const uint8x16x3_t input =
                vld3q_u8(src_row + static_cast<std::size_t>(x) * 3);
            const uint8x16_t bb = input.val[rgb_order ? 2 : 0];
            const uint8x16_t gg = input.val[1];
            const uint8x16_t rr = input.val[rgb_order ? 0 : 2];
            uint8x8_t y0;
            uint8x8_t u0;
            uint8x8_t v0;
            uint8x8_t y1;
            uint8x8_t u1;
            uint8x8_t v1;
            color_to_yuv_float8(
                vget_low_u8(bb), vget_low_u8(gg), vget_low_u8(rr),
                y0, u0, v0);
            color_to_yuv_float8(
                vget_high_u8(bb), vget_high_u8(gg), vget_high_u8(rr),
                y1, u1, v1);
            const uint8x16x3_t output = {{
                vcombine_u8(y0, y1),
                vcombine_u8(u0, u1),
                vcombine_u8(v0, v1)}};
            vst3q_u8(dst_row + static_cast<std::size_t>(x) * 3, output);
        }
        for (; x < src.size[1]; ++x)
        {
            const int offset = x * 3;
            const float r = src_row[offset + (rgb_order ? 0 : 2)];
            const float g = src_row[offset + 1];
            const float b = src_row[offset + (rgb_order ? 2 : 0)];
            const float yv = 0.299f * r + 0.587f * g + 0.114f * b;
            dst_row[offset + 0] = saturate_cast<uchar>(yv);
            dst_row[offset + 1] = saturate_cast<uchar>(
                0.492f * (b - yv) + 128.0f);
            dst_row[offset + 2] = saturate_cast<uchar>(
                0.877f * (r - yv) + 128.0f);
        }
    });
}

inline void yuv_interleaved_to_color3(
    const Mat& src,
    Mat& dst,
    bool rgb_order)
{
    const bool do_parallel = should_parallelize_cvtcolor(
        src.size[0], src.size[1], 3);
    parallel_for_index_if(do_parallel, src.size[0], [&](int y) {
        const uchar* src_row = src.data + static_cast<std::size_t>(y) * src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        int x = 0;
        for (; x + 16 <= src.size[1]; x += 16)
        {
            const uint8x16x3_t input =
                vld3q_u8(src_row + static_cast<std::size_t>(x) * 3);
            uint8x8_t b0;
            uint8x8_t g0;
            uint8x8_t r0;
            uint8x8_t b1;
            uint8x8_t g1;
            uint8x8_t r1;
            yuv_float_to_color8(
                vget_low_u8(input.val[0]),
                vget_low_u8(input.val[1]),
                vget_low_u8(input.val[2]),
                b0, g0, r0);
            yuv_float_to_color8(
                vget_high_u8(input.val[0]),
                vget_high_u8(input.val[1]),
                vget_high_u8(input.val[2]),
                b1, g1, r1);
            uint8x16x3_t output;
            output.val[rgb_order ? 2 : 0] = vcombine_u8(b0, b1);
            output.val[1] = vcombine_u8(g0, g1);
            output.val[rgb_order ? 0 : 2] = vcombine_u8(r0, r1);
            vst3q_u8(dst_row + static_cast<std::size_t>(x) * 3, output);
        }
        for (; x < src.size[1]; ++x)
        {
            const int offset = x * 3;
            const float yv = src_row[offset + 0];
            const float u = static_cast<float>(src_row[offset + 1]) - 128.0f;
            const float v = static_cast<float>(src_row[offset + 2]) - 128.0f;
            const uchar b = saturate_cast<uchar>(yv + 2.032f * u);
            const uchar g = saturate_cast<uchar>(
                yv - 0.395f * u - 0.581f * v);
            const uchar r = saturate_cast<uchar>(yv + 1.140f * v);
            dst_row[offset + (rgb_order ? 0 : 2)] = r;
            dst_row[offset + 1] = g;
            dst_row[offset + (rgb_order ? 2 : 0)] = b;
        }
    });
}

inline void yuv420sp_to_color3(
    const Mat& src,
    Mat& dst,
    int rows,
    bool nv21_layout,
    bool rgb_order)
{
    const int cols = src.size[1];
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* y_row = src.data + static_cast<std::size_t>(y) * src.step(0);
        const uchar* uv_row = src.data +
            static_cast<std::size_t>(rows + y / 2) * src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        int x = 0;
        for (; x + 16 <= cols; x += 16)
        {
            const uint8x8x2_t uv = vld2_u8(uv_row + x);
            store_yuv_limited_16(
                vld1q_u8(y_row + x),
                nv21_layout ? uv.val[1] : uv.val[0],
                nv21_layout ? uv.val[0] : uv.val[1],
                dst_row + static_cast<std::size_t>(x) * 3,
                rgb_order);
        }
        for (; x < cols; x += 2)
        {
            const int first = uv_row[x + 0];
            const int second = uv_row[x + 1];
            const int u = nv21_layout ? second : first;
            const int v = nv21_layout ? first : second;
            for (int i = 0; i < 2; ++i)
            {
                const int dx = (x + i) * 3;
                const int yv = y_row[x + i];
                const uchar b = cvtcolor_yuv420sp_channel_u8(yv, u, v, 0);
                const uchar g = cvtcolor_yuv420sp_channel_u8(yv, u, v, 1);
                const uchar r = cvtcolor_yuv420sp_channel_u8(yv, u, v, 2);
                dst_row[dx + (rgb_order ? 0 : 2)] = r;
                dst_row[dx + 1] = g;
                dst_row[dx + (rgb_order ? 2 : 0)] = b;
            }
        }
    });
}

inline void yuv420p_to_color3(
    const Mat& src,
    Mat& dst,
    int rows,
    bool yv12_layout,
    bool rgb_order)
{
    const int cols = src.size[1];
    const int uv_size = rows * cols / 4;
    const int u_plane_offset = yv12_layout ? uv_size : 0;
    const int v_plane_offset = yv12_layout ? 0 : uv_size;
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 1);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* y_row = src.data + static_cast<std::size_t>(y) * src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        int x = 0;
        for (; x + 16 <= cols; x += 16)
        {
            const int chroma_index = (y / 2) * (cols / 2) + x / 2;
            const uint8x8_t u = vld1_u8(logical_plane_ptr(
                src.data, src.step(0), rows, cols,
                u_plane_offset, chroma_index));
            const uint8x8_t v = vld1_u8(logical_plane_ptr(
                src.data, src.step(0), rows, cols,
                v_plane_offset, chroma_index));
            store_yuv_limited_16(
                vld1q_u8(y_row + x), u, v,
                dst_row + static_cast<std::size_t>(x) * 3,
                rgb_order);
        }
        for (; x < cols; ++x)
        {
            const int chroma_index = (y / 2) * (cols / 2) + x / 2;
            const int u = *logical_plane_ptr(
                src.data, src.step(0), rows, cols,
                u_plane_offset, chroma_index);
            const int v = *logical_plane_ptr(
                src.data, src.step(0), rows, cols,
                v_plane_offset, chroma_index);
            const int yv = y_row[x];
            const uchar b = cvtcolor_yuv420sp_channel_u8(yv, u, v, 0);
            const uchar g = cvtcolor_yuv420sp_channel_u8(yv, u, v, 1);
            const uchar r = cvtcolor_yuv420sp_channel_u8(yv, u, v, 2);
            const int dx = x * 3;
            dst_row[dx + (rgb_order ? 0 : 2)] = r;
            dst_row[dx + 1] = g;
            dst_row[dx + (rgb_order ? 2 : 0)] = b;
        }
    });
}

inline void yuv422packed_to_color3(
    const Mat& src,
    Mat& dst,
    bool uyvy_layout,
    bool rgb_order)
{
    const int rows = src.size[0];
    const int cols = src.size[1];
    const bool do_parallel = should_parallelize_cvtcolor(rows, cols, 2);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const uchar* src_row = src.data + static_cast<std::size_t>(y) * src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        int x = 0;
        for (; x + 16 <= cols; x += 16)
        {
            const uint8x16x2_t packed =
                vld2q_u8(src_row + static_cast<std::size_t>(x) * 2);
            const uint8x16_t y16 = uyvy_layout ? packed.val[1] : packed.val[0];
            const uint8x16_t uv16 = uyvy_layout ? packed.val[0] : packed.val[1];
            const uint8x8x2_t uv = vuzp_u8(
                vget_low_u8(uv16), vget_high_u8(uv16));
            store_yuv_limited_16(
                y16, uv.val[0], uv.val[1],
                dst_row + static_cast<std::size_t>(x) * 3,
                rgb_order);
        }
        for (; x < cols; x += 2)
        {
            const int base = x * 2;
            const int first0 = src_row[base + 0];
            const int first1 = src_row[base + 1];
            const int second0 = src_row[base + 2];
            const int second1 = src_row[base + 3];
            const int y0 = uyvy_layout ? first1 : first0;
            const int u = uyvy_layout ? first0 : first1;
            const int y1 = uyvy_layout ? second1 : second0;
            const int v = uyvy_layout ? second0 : second1;
            for (int i = 0; i < 2; ++i)
            {
                const int dx = (x + i) * 3;
                const int yv = i == 0 ? y0 : y1;
                const uchar b = cvtcolor_yuv420sp_channel_u8(yv, u, v, 0);
                const uchar g = cvtcolor_yuv420sp_channel_u8(yv, u, v, 1);
                const uchar r = cvtcolor_yuv420sp_channel_u8(yv, u, v, 2);
                dst_row[dx + (rgb_order ? 0 : 2)] = r;
                dst_row[dx + 1] = g;
                dst_row[dx + (rgb_order ? 2 : 0)] = b;
            }
        }
    });
}

#endif  // CVH_DETAIL_HAVE_NEON_KERNEL

inline bool try_cvtcolor_packed_u8(const Mat& src, Mat& dst, int code)
{
#if CVH_DETAIL_HAVE_NEON_KERNEL
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U ||
        !direct_neon_allowed() ||
        !packed_workload_is_large_enough(src.size[0], src.size[1]))
    {
        return false;
    }

    const int rows = src.size[0];
    const int cols = src.size[1];

    if ((code == COLOR_BGR2RGB || code == COLOR_RGB2BGR) &&
        src.channels() == 3)
    {
        dst.create(std::vector<int>{rows, cols}, CV_8UC3);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_packed:load=neon;shuffle=neon;store=neon;tail=scalar");
        swap_rb_3ch(src, dst);
        return true;
    }

    if ((code == COLOR_BGR2BGRA || code == COLOR_RGB2RGBA ||
         code == COLOR_BGR2RGBA || code == COLOR_RGB2BGRA) &&
        src.channels() == 3)
    {
        const bool swap_rb =
            code == COLOR_BGR2RGBA || code == COLOR_RGB2BGRA;
        dst.create(std::vector<int>{rows, cols}, CV_8UC4);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_packed:load=neon;alpha=neon;store=neon;tail=scalar");
        color3_to_color4(src, dst, swap_rb);
        return true;
    }

    if ((code == COLOR_BGRA2BGR || code == COLOR_RGBA2RGB ||
         code == COLOR_BGRA2RGB || code == COLOR_RGBA2BGR) &&
        src.channels() == 4)
    {
        const bool swap_rb =
            code == COLOR_BGRA2RGB || code == COLOR_RGBA2BGR;
        dst.create(std::vector<int>{rows, cols}, CV_8UC3);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_packed:load=neon;drop_alpha=neon;store=neon;tail=scalar");
        color4_to_color3(src, dst, swap_rb);
        return true;
    }

    if ((code == COLOR_BGRA2RGBA || code == COLOR_RGBA2BGRA) &&
        src.channels() == 4)
    {
        dst.create(std::vector<int>{rows, cols}, CV_8UC4);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_packed:load=neon;shuffle=neon;store=neon;tail=scalar");
        swap_rb_4ch(src, dst);
        return true;
    }

    if ((code == COLOR_BGRA2GRAY || code == COLOR_RGBA2GRAY) &&
        src.channels() == 4)
    {
        dst.create(std::vector<int>{rows, cols}, CV_8UC1);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_packed:load=neon;gray=neon;store=neon;tail=scalar");
        color4_to_gray(src, dst, code == COLOR_RGBA2GRAY);
        return true;
    }

    if (code == COLOR_GRAY2BGR && src.channels() == 1)
    {
        dst.create(std::vector<int>{rows, cols}, CV_8UC3);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_packed:load=neon;replicate=neon;store=neon;tail=scalar");
        gray_to_color3(src, dst);
        return true;
    }

    if ((code == COLOR_GRAY2BGRA || code == COLOR_GRAY2RGBA) &&
        src.channels() == 1)
    {
        dst.create(std::vector<int>{rows, cols}, CV_8UC4);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_packed:load=neon;replicate_alpha=neon;store=neon;tail=scalar");
        gray_to_color4(src, dst);
        return true;
    }
#else
    (void)src;
    (void)dst;
    (void)code;
#endif
    return false;
}

inline bool try_cvtcolor_yuv_decode_u8(const Mat& src, Mat& dst, int code)
{
#if CVH_DETAIL_HAVE_NEON_KERNEL
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U ||
        !direct_neon_allowed())
    {
        return false;
    }

    const int cols = src.size[1];
    if (code == COLOR_YUV2BGR_NV12 || code == COLOR_YUV2RGB_NV12 ||
        code == COLOR_YUV2BGR_NV21 || code == COLOR_YUV2RGB_NV21)
    {
        if (src.channels() != 1)
        {
            return false;
        }
        const int rows = cvtcolor_validate_yuv420sp_layout_u8(src);
        if (!packed_workload_is_large_enough(rows, cols))
        {
            return false;
        }
        dst.create(std::vector<int>{rows, cols}, CV_8UC3);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_yuv420sp_decode:load=neon;chroma=neon;convert=neon;store=neon;tail=scalar");
        yuv420sp_to_color3(
            src, dst, rows,
            code == COLOR_YUV2BGR_NV21 || code == COLOR_YUV2RGB_NV21,
            code == COLOR_YUV2RGB_NV12 || code == COLOR_YUV2RGB_NV21);
        return true;
    }

    if (code == COLOR_YUV2BGR_I420 || code == COLOR_YUV2RGB_I420 ||
        code == COLOR_YUV2BGR_YV12 || code == COLOR_YUV2RGB_YV12)
    {
        if (src.channels() != 1)
        {
            return false;
        }
        const int rows = cvtcolor_validate_yuv420sp_layout_u8(src);
        if (!packed_workload_is_large_enough(rows, cols))
        {
            return false;
        }
        dst.create(std::vector<int>{rows, cols}, CV_8UC3);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_yuv420p_decode:load=neon;chroma=neon;convert=neon;store=neon;tail=scalar");
        yuv420p_to_color3(
            src, dst, rows,
            code == COLOR_YUV2BGR_YV12 || code == COLOR_YUV2RGB_YV12,
            code == COLOR_YUV2RGB_I420 || code == COLOR_YUV2RGB_YV12);
        return true;
    }

    if (code == COLOR_YUV2BGR_YUY2 || code == COLOR_YUV2RGB_YUY2 ||
        code == COLOR_YUV2BGR_UYVY || code == COLOR_YUV2RGB_UYVY)
    {
        if (src.channels() != 2 ||
            !packed_workload_is_large_enough(src.size[0], cols))
        {
            return false;
        }
        cvtcolor_validate_yuv422packed_layout_u8(src);
        dst.create(std::vector<int>{src.size[0], cols}, CV_8UC3);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_yuv422packed_decode:load=neon;chroma=neon;convert=neon;store=neon;tail=scalar");
        yuv422packed_to_color3(
            src, dst,
            code == COLOR_YUV2BGR_UYVY || code == COLOR_YUV2RGB_UYVY,
            code == COLOR_YUV2RGB_YUY2 || code == COLOR_YUV2RGB_UYVY);
        return true;
    }
#else
    (void)src;
    (void)dst;
    (void)code;
#endif
    return false;
}

inline bool try_cvtcolor_yuv_encode_u8(const Mat& src, Mat& dst, int code)
{
#if CVH_DETAIL_HAVE_NEON_KERNEL
    if (src.empty() || src.dims != 2 || src.type() != CV_8UC3 ||
        !direct_neon_allowed() ||
        !packed_workload_is_large_enough(src.size[0], src.size[1]))
    {
        return false;
    }
    const int rows = src.size[0];
    const int cols = src.size[1];

    if (code == COLOR_BGR2YUV_NV12 || code == COLOR_RGB2YUV_NV12 ||
        code == COLOR_BGR2YUV_NV21 || code == COLOR_RGB2YUV_NV21)
    {
        if ((rows & 1) != 0 || (cols & 1) != 0)
        {
            return false;
        }
        dst.create(std::vector<int>{rows * 3 / 2, cols}, CV_8UC1);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_yuv420sp_encode:load=neon;luma=neon;chroma=neon;store=neon;tail=scalar");
        color3_to_yuv420(
            src, dst,
            code == COLOR_RGB2YUV_NV12 || code == COLOR_RGB2YUV_NV21,
            false,
            code == COLOR_BGR2YUV_NV21 || code == COLOR_RGB2YUV_NV21);
        return true;
    }

    if (code == COLOR_BGR2YUV_I420 || code == COLOR_RGB2YUV_I420 ||
        code == COLOR_BGR2YUV_YV12 || code == COLOR_RGB2YUV_YV12)
    {
        if ((rows & 1) != 0 || (cols & 1) != 0)
        {
            return false;
        }
        dst.create(std::vector<int>{rows * 3 / 2, cols}, CV_8UC1);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_yuv420p_encode:load=neon;luma=neon;chroma=neon;store=neon;tail=scalar");
        color3_to_yuv420(
            src, dst,
            code == COLOR_RGB2YUV_I420 || code == COLOR_RGB2YUV_YV12,
            true,
            code == COLOR_BGR2YUV_YV12 || code == COLOR_RGB2YUV_YV12);
        return true;
    }

    if (code == COLOR_BGR2YUV_YUY2 || code == COLOR_RGB2YUV_YUY2 ||
        code == COLOR_BGR2YUV_UYVY || code == COLOR_RGB2YUV_UYVY)
    {
        if ((cols & 1) != 0)
        {
            return false;
        }
        dst.create(std::vector<int>{rows, cols}, CV_8UC2);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_yuv422packed_encode:load=neon;luma=neon;chroma=neon;store=neon;tail=scalar");
        color3_to_yuv422packed(
            src, dst,
            code == COLOR_RGB2YUV_YUY2 || code == COLOR_RGB2YUV_UYVY,
            code == COLOR_BGR2YUV_UYVY || code == COLOR_RGB2YUV_UYVY);
        return true;
    }
#else
    (void)src;
    (void)dst;
    (void)code;
#endif
    return false;
}

inline bool try_cvtcolor_yuv_interleaved_u8(
    const Mat& src,
    Mat& dst,
    int code)
{
#if CVH_DETAIL_HAVE_NEON_KERNEL
    if (src.empty() || src.dims != 2 || src.type() != CV_8UC3 ||
        !direct_neon_allowed() ||
        !packed_workload_is_large_enough(src.size[0], src.size[1]))
    {
        return false;
    }
    if (code == COLOR_BGR2YUV || code == COLOR_RGB2YUV)
    {
        dst.create(src.shape(), CV_8UC3);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_yuv444_encode:load=neon;convert=neon;store=neon;tail=scalar");
        color3_to_yuv_interleaved(src, dst, code == COLOR_RGB2YUV);
        return true;
    }
    if (code == COLOR_YUV2BGR || code == COLOR_YUV2RGB)
    {
        dst.create(src.shape(), CV_8UC3);
        cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
        cpu::set_last_kernel_route(
            "cvtcolor_yuv444_decode:load=neon;convert=neon;store=neon;tail=scalar");
        yuv_interleaved_to_color3(src, dst, code == COLOR_YUV2RGB);
        return true;
    }
#else
    (void)src;
    (void)dst;
    (void)code;
#endif
    return false;
}

}  // namespace cvtcolor_neon
}  // namespace detail
}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_CVTCOLOR_NEON_HPP
