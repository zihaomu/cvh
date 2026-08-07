#ifndef CVH_IMGPROC_DETAIL_RESIZE_NEON_HPP
#define CVH_IMGPROC_DETAIL_RESIZE_NEON_HPP

#include "fastpath_common.hpp"
#include "resize_fixed_u8c3.hpp"
#include "../../core/detail/cpu_features.hpp"

#include <array>

namespace cvh
{
namespace detail
{
namespace resize_neon
{

inline bool direct_neon_allowed()
{
    const cpu::DispatchMode mode = cpu::dispatch_mode();
    return cpu::neon_runtime_available() &&
           (mode == cpu::DispatchMode::Auto ||
            mode == cpu::DispatchMode::NeonOnly);
}

#if CVH_DETAIL_HAVE_NEON_KERNEL

inline uint8x8_t round_saturate_float8(
    float32x4_t low,
    float32x4_t high)
{
    return vqmovun_s16(vcombine_s16(
        vqmovn_s32(vcvtaq_s32_f32(low)),
        vqmovn_s32(vcvtaq_s32_f32(high))));
}

inline void resize_linear_half_u8c3(
    const Mat& src,
    Mat& dst)
{
    const int dst_rows = dst.size[0];
    const int dst_cols = dst.size[1];
    const bool do_parallel = should_parallelize_resize(dst_rows, dst_cols, 3);
    parallel_for_index_if(do_parallel, dst_rows, [&](int y) {
        const uchar* row0 = src.data +
            static_cast<std::size_t>(y * 2) * src.step(0);
        const uchar* row1 = row0 + src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        int x = 0;
        for (; x + 8 <= dst_cols; x += 8)
        {
            const uint8x16x3_t pixels0 =
                vld3q_u8(row0 + static_cast<std::size_t>(x) * 6);
            const uint8x16x3_t pixels1 =
                vld3q_u8(row1 + static_cast<std::size_t>(x) * 6);
            uint8x8x3_t output;
            for (int c = 0; c < 3; ++c)
            {
                uint16x8_t sum = vaddq_u16(
                    vpaddlq_u8(pixels0.val[c]),
                    vpaddlq_u8(pixels1.val[c]));
                sum = vaddq_u16(sum, vdupq_n_u16(2));
                output.val[c] = vshrn_n_u16(sum, 2);
            }
            vst3_u8(dst_row + static_cast<std::size_t>(x) * 3, output);
        }
        for (; x < dst_cols; ++x)
        {
            const int sx = x * 6;
            const int dx = x * 3;
            for (int c = 0; c < 3; ++c)
            {
                dst_row[dx + c] = static_cast<uchar>(
                    (static_cast<int>(row0[sx + c]) +
                     static_cast<int>(row0[sx + 3 + c]) +
                     static_cast<int>(row1[sx + c]) +
                     static_cast<int>(row1[sx + 3 + c]) + 2) >> 2);
            }
        }
    });
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

inline uint8x8_t interpolate_u8x8(
    uint8x8_t p00,
    uint8x8_t p01,
    uint8x8_t p10,
    uint8x8_t p11,
    float32x4_t wx_low,
    float32x4_t wx_high,
    float32x4_t wy)
{
    float32x4_t p00_low;
    float32x4_t p00_high;
    float32x4_t p01_low;
    float32x4_t p01_high;
    float32x4_t p10_low;
    float32x4_t p10_high;
    float32x4_t p11_low;
    float32x4_t p11_high;
    expand_u8_to_float8(p00, p00_low, p00_high);
    expand_u8_to_float8(p01, p01_low, p01_high);
    expand_u8_to_float8(p10, p10_low, p10_high);
    expand_u8_to_float8(p11, p11_low, p11_high);
    const float32x4_t top_low = vfmaq_f32(
        p00_low, vsubq_f32(p01_low, p00_low), wx_low);
    const float32x4_t top_high = vfmaq_f32(
        p00_high, vsubq_f32(p01_high, p00_high), wx_high);
    const float32x4_t bottom_low = vfmaq_f32(
        p10_low, vsubq_f32(p11_low, p10_low), wx_low);
    const float32x4_t bottom_high = vfmaq_f32(
        p10_high, vsubq_f32(p11_high, p10_high), wx_high);
    return round_saturate_float8(
        vfmaq_f32(top_low, vsubq_f32(bottom_low, top_low), wy),
        vfmaq_f32(top_high, vsubq_f32(bottom_high, top_high), wy));
}

inline void resize_linear_gather_u8c3(
    const Mat& src,
    Mat& dst)
{
    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_rows = dst.size[0];
    const int dst_cols = dst.size[1];
    const float scale_x =
        static_cast<float>(src_cols) / static_cast<float>(dst_cols);
    const float scale_y =
        static_cast<float>(src_rows) / static_cast<float>(dst_rows);
    std::vector<int> x0(static_cast<std::size_t>(dst_cols));
    std::vector<int> x1(static_cast<std::size_t>(dst_cols));
    std::vector<float> wx(static_cast<std::size_t>(dst_cols));
    for (int x = 0; x < dst_cols; ++x)
    {
        const float source_x =
            (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
        const int ix0 = std::clamp(
            static_cast<int>(std::floor(source_x)), 0, src_cols - 1);
        x0[static_cast<std::size_t>(x)] = ix0;
        x1[static_cast<std::size_t>(x)] =
            std::min(ix0 + 1, src_cols - 1);
        wx[static_cast<std::size_t>(x)] = source_x - static_cast<float>(ix0);
    }

    std::vector<int> block_source_base;
    std::vector<std::array<uchar, 16>> block_left_indices;
    std::vector<std::array<uchar, 16>> block_right_indices;
    block_source_base.reserve(static_cast<std::size_t>(dst_cols / 8));
    block_left_indices.reserve(static_cast<std::size_t>(dst_cols / 8));
    block_right_indices.reserve(static_cast<std::size_t>(dst_cols / 8));
    for (int x = 0; x + 8 <= dst_cols; x += 8)
    {
        const int source_base = x0[static_cast<std::size_t>(x)];
        const int source_last = x1[static_cast<std::size_t>(x + 7)];
        if (source_base + 16 > src_cols ||
            source_last - source_base >= 16)
        {
            break;
        }
        std::array<uchar, 16> left_indices = {};
        std::array<uchar, 16> right_indices = {};
        for (int i = 0; i < 8; ++i)
        {
            left_indices[static_cast<std::size_t>(i)] =
                static_cast<uchar>(
                    x0[static_cast<std::size_t>(x + i)] - source_base);
            right_indices[static_cast<std::size_t>(i)] =
                static_cast<uchar>(
                    x1[static_cast<std::size_t>(x + i)] - source_base);
        }
        block_source_base.push_back(source_base);
        block_left_indices.push_back(left_indices);
        block_right_indices.push_back(right_indices);
    }

    std::vector<int> y0(static_cast<std::size_t>(dst_rows));
    std::vector<int> y1(static_cast<std::size_t>(dst_rows));
    std::vector<float> wy(static_cast<std::size_t>(dst_rows));
    for (int y = 0; y < dst_rows; ++y)
    {
        const float source_y =
            (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int iy0 = std::clamp(
            static_cast<int>(std::floor(source_y)), 0, src_rows - 1);
        y0[static_cast<std::size_t>(y)] = iy0;
        y1[static_cast<std::size_t>(y)] = std::min(iy0 + 1, src_rows - 1);
        wy[static_cast<std::size_t>(y)] = source_y - static_cast<float>(iy0);
    }

    const bool do_parallel = should_parallelize_resize(dst_rows, dst_cols, 3);
    parallel_for_index_if(do_parallel, dst_rows, [&](int y) {
        const uchar* row0 = src.data +
            static_cast<std::size_t>(y0[static_cast<std::size_t>(y)]) *
                src.step(0);
        const uchar* row1 = src.data +
            static_cast<std::size_t>(y1[static_cast<std::size_t>(y)]) *
                src.step(0);
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        const float32x4_t wyv =
            vdupq_n_f32(wy[static_cast<std::size_t>(y)]);
        std::size_t block = 0;
        for (; block < block_source_base.size(); ++block)
        {
            const int x = static_cast<int>(block) * 8;
            const int source_base = block_source_base[block];
            const uint8x16_t left_index =
                vld1q_u8(block_left_indices[block].data());
            const uint8x16_t right_index =
                vld1q_u8(block_right_indices[block].data());
            const uint8x16x3_t source0 = vld3q_u8(
                row0 + static_cast<std::size_t>(source_base) * 3);
            const uint8x16x3_t source1 = vld3q_u8(
                row1 + static_cast<std::size_t>(source_base) * 3);
            const float32x4_t wx0 = vld1q_f32(wx.data() + x);
            const float32x4_t wx1 = vld1q_f32(wx.data() + x + 4);
            uint8x8x3_t output0;
            for (int c = 0; c < 3; ++c)
            {
                const uint8x16_t p00 =
                    vqtbl1q_u8(source0.val[c], left_index);
                const uint8x16_t p01 =
                    vqtbl1q_u8(source0.val[c], right_index);
                const uint8x16_t p10 =
                    vqtbl1q_u8(source1.val[c], left_index);
                const uint8x16_t p11 =
                    vqtbl1q_u8(source1.val[c], right_index);
                output0.val[c] = interpolate_u8x8(
                    vget_low_u8(p00), vget_low_u8(p01),
                    vget_low_u8(p10), vget_low_u8(p11),
                    wx0, wx1, wyv);
            }
            vst3_u8(dst_row + static_cast<std::size_t>(x) * 3, output0);
        }

        int x = static_cast<int>(block) * 8;
        for (; x < dst_cols; ++x)
        {
            const int left = x0[static_cast<std::size_t>(x)] * 3;
            const int right = x1[static_cast<std::size_t>(x)] * 3;
            const float wxv = wx[static_cast<std::size_t>(x)];
            const float wy_scalar = wy[static_cast<std::size_t>(y)];
            const int dx = x * 3;
            for (int c = 0; c < 3; ++c)
            {
                const float top = lerp(
                    static_cast<float>(row0[left + c]),
                    static_cast<float>(row0[right + c]), wxv);
                const float bottom = lerp(
                    static_cast<float>(row1[left + c]),
                    static_cast<float>(row1[right + c]), wxv);
                dst_row[dx + c] = saturate_cast<uchar>(
                    lerp(top, bottom, wy_scalar));
            }
        }
    });
}

inline uint8x16_t lerp_fixed_u8x16(
    uint8x16_t first,
    uint8x16_t second,
    uint16x8_t fraction_low,
    uint16x8_t fraction_high)
{
    const uint8x8_t low = vraddhn_u16(
        vshll_n_u8(vget_low_u8(first), 8),
        vmulq_u16(
            vsubl_u8(vget_low_u8(second), vget_low_u8(first)),
            fraction_low));
    const uint8x8_t high = vraddhn_u16(
        vshll_high_n_u8(first, 8),
        vmulq_u16(
            vsubl_high_u8(second, first),
            fraction_high));
    return vcombine_u8(low, high);
}

inline void resize_linear_fixed_u8c3(
    const Mat& src,
    Mat& dst)
{
    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_rows = dst.size[0];
    const int dst_cols = dst.size[1];
    const resize_fixed_u8c3::Maps maps =
        resize_fixed_u8c3::build_maps(
            src_rows, src_cols, dst_rows, dst_cols);
    const std::size_t output_bytes =
        static_cast<std::size_t>(dst_cols) * 3;

    const bool do_parallel = should_parallelize_resize(dst_rows, dst_cols, 3);
    parallel_for_index_if(do_parallel, dst_rows, [&](int y) {
        const resize_fixed_u8c3::AxisCoordinate& vertical =
            maps.y[static_cast<std::size_t>(y)];
        const uchar* top = src.data +
            static_cast<std::size_t>(vertical.first) * src.step(0);
        const uchar* bottom = src.data +
            static_cast<std::size_t>(vertical.second) * src.step(0);
        uchar* output = dst.data + static_cast<std::size_t>(y) * dst.step(0);
        const uint16x8_t y_fraction =
            vdupq_n_u16(vertical.fraction);

        std::size_t block_index = 0;
        for (; block_index < maps.blocks.size(); ++block_index)
        {
            const resize_fixed_u8c3::FlatBlock& block =
                maps.blocks[block_index];
            uint8x16x2_t top_source;
            uint8x16x2_t bottom_source;
            top_source.val[0] =
                vld1q_u8(top + block.source_byte_base);
            top_source.val[1] =
                vld1q_u8(top + block.source_byte_base + 16);
            bottom_source.val[0] =
                vld1q_u8(bottom + block.source_byte_base);
            bottom_source.val[1] =
                vld1q_u8(bottom + block.source_byte_base + 16);

            const uint8x16_t left_index =
                vld1q_u8(block.left_index.data());
            const uint8x16_t right_index =
                vaddq_u8(left_index, vdupq_n_u8(3));
            const uint8x16_t top_left =
                vqtbl2q_u8(top_source, left_index);
            const uint8x16_t top_right =
                vqtbl2q_u8(top_source, right_index);
            const uint8x16_t bottom_left =
                vqtbl2q_u8(bottom_source, left_index);
            const uint8x16_t bottom_right =
                vqtbl2q_u8(bottom_source, right_index);

            const uint8x16_t left = lerp_fixed_u8x16(
                top_left, bottom_left, y_fraction, y_fraction);
            const uint8x16_t right = lerp_fixed_u8x16(
                top_right, bottom_right, y_fraction, y_fraction);
            const uint16x8_t x_fraction_low =
                vld1q_u16(block.x_fraction.data());
            const uint16x8_t x_fraction_high =
                vld1q_u16(block.x_fraction.data() + 8);
            const uint8x16_t result = lerp_fixed_u8x16(
                left,
                right,
                x_fraction_low,
                x_fraction_high);
            vst1q_u8(output + block_index * 16, result);
        }

        for (std::size_t output_element = maps.vector_output_bytes();
             output_element < output_bytes;
             ++output_element)
        {
            output[output_element] =
                resize_fixed_u8c3::interpolate_output_byte(
                    top,
                    bottom,
                    maps,
                    output_element,
                    vertical.fraction);
        }
    });
}

#endif  // CVH_DETAIL_HAVE_NEON_KERNEL

inline bool try_resize_linear_u8c3(
    const Mat& src,
    Mat& dst,
    Size dsize,
    double fx,
    double fy,
    int interpolation)
{
#if CVH_DETAIL_HAVE_NEON_KERNEL
    if (src.empty() || src.dims != 2 || src.type() != CV_8UC3 ||
        interpolation != INTER_LINEAR || !direct_neon_allowed())
    {
        return false;
    }
    const int dst_cols = resolve_resize_dim(src.size[1], dsize.width, fx);
    const int dst_rows = resolve_resize_dim(src.size[0], dsize.height, fy);
    if (dst_cols < 8 || dst_rows <= 0 ||
        static_cast<std::size_t>(dst_cols) *
                static_cast<std::size_t>(dst_rows) <
            256)
    {
        return false;
    }

    dst.create(std::vector<int>{dst_rows, dst_cols}, CV_8UC3);
    cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
    if (src.size[0] == dst_rows * 2 && src.size[1] == dst_cols * 2)
    {
        cpu::set_last_kernel_route(
            "resize_linear_u8c3:map=ratio_half;load=neon;interpolate=neon;store=neon;tail=scalar");
        resize_linear_half_u8c3(src, dst);
    }
    else if (resize_fixed_u8c3::is_exact_three_quarter_shape(
                 src.size[0], src.size[1], dst_rows, dst_cols))
    {
        cpu::set_last_kernel_route(
            "resize_linear_u8c3:map=fixed_q16_q8;layout=flat_c3;load=neon_contiguous;gather=tbl2;interpolate=fixed8_vertical_horizontal;store=neon_contiguous;tail=fixed_scalar");
        resize_linear_fixed_u8c3(src, dst);
    }
    else
    {
        cpu::set_last_kernel_route(
            "resize_linear_u8c3:map=scalar;gather=neon_table;interpolate=neon;store=neon;tail=scalar");
        resize_linear_gather_u8c3(src, dst);
    }
    return true;
#else
    (void)src;
    (void)dst;
    (void)dsize;
    (void)fx;
    (void)fy;
    (void)interpolation;
    return false;
#endif
}

}  // namespace resize_neon
}  // namespace detail
}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_RESIZE_NEON_HPP
