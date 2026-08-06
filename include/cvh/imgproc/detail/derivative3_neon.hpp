#ifndef CVH_IMGPROC_DETAIL_DERIVATIVE3_NEON_HPP
#define CVH_IMGPROC_DETAIL_DERIVATIVE3_NEON_HPP

#include "fastpath_common.hpp"
#include "../../core/detail/cpu_features.hpp"

namespace cvh
{
namespace detail
{
namespace derivative3_neon
{

inline bool direct_neon_allowed()
{
    const cpu::DispatchMode mode = cpu::dispatch_mode();
    return cpu::neon_runtime_available() &&
           (mode == cpu::DispatchMode::Auto ||
            mode == cpu::DispatchMode::NeonOnly);
}

#if CVH_DETAIL_HAVE_NEON_KERNEL

inline int16x8_t derivative_x_half(
    uint8x8_t top_left,
    uint8x8_t top_right,
    uint8x8_t middle_left,
    uint8x8_t middle_right,
    uint8x8_t bottom_left,
    uint8x8_t bottom_right,
    int outer_weight,
    int middle_weight)
{
    // For U8 input, Sobel is bounded by 4*255=1020 and Scharr by
    // (3+10+3)*255=4080. Both signed derivatives fit exactly in S16.
    const int16x8_t top = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(top_right)),
        vreinterpretq_s16_u16(vmovl_u8(top_left)));
    const int16x8_t middle = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(middle_right)),
        vreinterpretq_s16_u16(vmovl_u8(middle_left)));
    const int16x8_t bottom = vsubq_s16(
        vreinterpretq_s16_u16(vmovl_u8(bottom_right)),
        vreinterpretq_s16_u16(vmovl_u8(bottom_left)));
    return vaddq_s16(
        vmulq_n_s16(vaddq_s16(top, bottom), outer_weight),
        vmulq_n_s16(middle, middle_weight));
}

inline int16x8_t derivative_y_half(
    uint8x8_t top_left,
    uint8x8_t top_middle,
    uint8x8_t top_right,
    uint8x8_t bottom_left,
    uint8x8_t bottom_middle,
    uint8x8_t bottom_right,
    int outer_weight,
    int middle_weight)
{
    const uint16x8_t top = vaddq_u16(
        vmulq_n_u16(
            vaddq_u16(vmovl_u8(top_left), vmovl_u8(top_right)),
            outer_weight),
        vmulq_n_u16(vmovl_u8(top_middle), middle_weight));
    const uint16x8_t bottom = vaddq_u16(
        vmulq_n_u16(
            vaddq_u16(vmovl_u8(bottom_left), vmovl_u8(bottom_right)),
            outer_weight),
        vmulq_n_u16(vmovl_u8(bottom_middle), middle_weight));
    return vsubq_s16(
        vreinterpretq_s16_u16(bottom),
        vreinterpretq_s16_u16(top));
}

inline void compute_derivative16(
    const uchar* top,
    const uchar* middle,
    const uchar* bottom,
    int center,
    int channel_stride,
    int outer_weight,
    int middle_weight,
    int16x8_t& dx_low,
    int16x8_t& dx_high,
    int16x8_t& dy_low,
    int16x8_t& dy_high)
{
    const uint8x16_t top_left =
        vld1q_u8(top + center - channel_stride);
    const uint8x16_t top_middle = vld1q_u8(top + center);
    const uint8x16_t top_right =
        vld1q_u8(top + center + channel_stride);
    const uint8x16_t middle_left =
        vld1q_u8(middle + center - channel_stride);
    const uint8x16_t middle_right =
        vld1q_u8(middle + center + channel_stride);
    const uint8x16_t bottom_left =
        vld1q_u8(bottom + center - channel_stride);
    const uint8x16_t bottom_middle = vld1q_u8(bottom + center);
    const uint8x16_t bottom_right =
        vld1q_u8(bottom + center + channel_stride);

    dx_low = derivative_x_half(
        vget_low_u8(top_left), vget_low_u8(top_right),
        vget_low_u8(middle_left), vget_low_u8(middle_right),
        vget_low_u8(bottom_left), vget_low_u8(bottom_right),
        outer_weight, middle_weight);
    dx_high = derivative_x_half(
        vget_high_u8(top_left), vget_high_u8(top_right),
        vget_high_u8(middle_left), vget_high_u8(middle_right),
        vget_high_u8(bottom_left), vget_high_u8(bottom_right),
        outer_weight, middle_weight);
    dy_low = derivative_y_half(
        vget_low_u8(top_left), vget_low_u8(top_middle),
        vget_low_u8(top_right), vget_low_u8(bottom_left),
        vget_low_u8(bottom_middle), vget_low_u8(bottom_right),
        outer_weight, middle_weight);
    dy_high = derivative_y_half(
        vget_high_u8(top_left), vget_high_u8(top_middle),
        vget_high_u8(top_right), vget_high_u8(bottom_left),
        vget_high_u8(bottom_middle), vget_high_u8(bottom_right),
        outer_weight, middle_weight);
}

inline void store_f32_from_s16(float* destination, int16x8_t values)
{
    const int32x4_t low = vmovl_s16(vget_low_s16(values));
    const int32x4_t high = vmovl_s16(vget_high_s16(values));
    vst1q_f32(destination, vcvtq_f32_s32(low));
    vst1q_f32(destination + 4, vcvtq_f32_s32(high));
}

inline int scalar_derivative(
    const uchar* base,
    std::size_t step,
    const SobelSamplingWindow& window,
    int output_y,
    int output_x,
    int channel,
    int channels,
    int border_type,
    bool derivative_x,
    int outer_weight,
    int middle_weight)
{
    int pixels[3][3];
    for (int ky = 0; ky < 3; ++ky)
    {
        const int sy = border_interpolate(
            output_y + window.row_offset + ky - 1,
            window.rows,
            border_type);
        const uchar* row = base + static_cast<std::size_t>(sy) * step;
        for (int kx = 0; kx < 3; ++kx)
        {
            const int sx = border_interpolate(
                output_x + window.col_offset + kx - 1,
                window.cols,
                border_type);
            pixels[ky][kx] = row[sx * channels + channel];
        }
    }
    if (derivative_x)
    {
        return outer_weight *
                   ((pixels[0][2] - pixels[0][0]) +
                    (pixels[2][2] - pixels[2][0])) +
               middle_weight * (pixels[1][2] - pixels[1][0]);
    }
    return outer_weight *
               ((pixels[2][0] + pixels[2][2]) -
                (pixels[0][0] + pixels[0][2])) +
               middle_weight * (pixels[2][1] - pixels[0][1]);
}

inline void scalar_sobel_pair(
    const uchar* base,
    std::size_t step,
    const SobelSamplingWindow& window,
    int output_y,
    int output_x,
    int border_type,
    short& dx,
    short& dy)
{
    int pixels[3][3];
    for (int ky = 0; ky < 3; ++ky)
    {
        const int sy = border_interpolate(
            output_y + window.row_offset + ky - 1,
            window.rows,
            border_type);
        const uchar* row = base + static_cast<std::size_t>(sy) * step;
        for (int kx = 0; kx < 3; ++kx)
        {
            const int sx = border_interpolate(
                output_x + window.col_offset + kx - 1,
                window.cols,
                border_type);
            pixels[ky][kx] = row[sx];
        }
    }
    dx = static_cast<short>(
        (pixels[0][2] - pixels[0][0]) +
        2 * (pixels[1][2] - pixels[1][0]) +
        (pixels[2][2] - pixels[2][0]));
    dy = static_cast<short>(
        (pixels[2][0] + 2 * pixels[2][1] + pixels[2][2]) -
        (pixels[0][0] + 2 * pixels[0][1] + pixels[0][2]));
}

inline void vector_byte_range(
    const SobelSamplingWindow& window,
    int output_cols,
    int channels,
    int& begin,
    int& end)
{
    const int first_pixel = std::max(0, 1 - window.col_offset);
    const int last_pixel = std::min(
        output_cols,
        window.cols - 1 - window.col_offset);
    begin = first_pixel * channels;
    end = std::max(first_pixel, last_pixel) * channels;
}

inline bool run_single(
    const Mat& src,
    Mat& dst,
    int output_depth,
    bool derivative_x,
    int outer_weight,
    int middle_weight,
    int border_type,
    bool isolated)
{
    const int rows = src.size[0];
    const int cols = src.size[1];
    const int channels = src.channels();
    const SobelSamplingWindow window =
        resolve_sobel_sampling_window(src, isolated);
    int vector_begin = 0;
    int vector_end = 0;
    vector_byte_range(window, cols, channels, vector_begin, vector_end);
    if (vector_end - vector_begin < 16)
    {
        return false;
    }

    dst.create(src.shape(), CV_MAKETYPE(output_depth, channels));
    const bool do_parallel =
        should_parallelize_filter_rows(rows, cols, channels, 9);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const int top_y = border_interpolate(
            y + window.row_offset - 1, window.rows, border_type);
        const int middle_y = border_interpolate(
            y + window.row_offset, window.rows, border_type);
        const int bottom_y = border_interpolate(
            y + window.row_offset + 1, window.rows, border_type);
        const uchar* top =
            window.base_data + static_cast<std::size_t>(top_y) * src.step(0);
        const uchar* middle =
            window.base_data + static_cast<std::size_t>(middle_y) * src.step(0);
        const uchar* bottom =
            window.base_data + static_cast<std::size_t>(bottom_y) * src.step(0);
        const int source_base = window.col_offset * channels;
        int byte = 0;

        auto write_scalar = [&](int byte_index) {
            const int x = byte_index / channels;
            const int channel = byte_index % channels;
            const int value = scalar_derivative(
                window.base_data, src.step(0), window, y, x, channel,
                channels, border_type, derivative_x,
                outer_weight, middle_weight);
            if (output_depth == CV_16S)
            {
                reinterpret_cast<short*>(
                    dst.data + static_cast<std::size_t>(y) * dst.step(0))
                    [byte_index] = static_cast<short>(value);
            }
            else
            {
                reinterpret_cast<float*>(
                    dst.data + static_cast<std::size_t>(y) * dst.step(0))
                    [byte_index] = static_cast<float>(value);
            }
        };

        for (; byte < vector_begin; ++byte)
        {
            write_scalar(byte);
        }
        for (; byte + 16 <= vector_end; byte += 16)
        {
            int16x8_t dx_low;
            int16x8_t dx_high;
            int16x8_t dy_low;
            int16x8_t dy_high;
            compute_derivative16(
                top, middle, bottom, source_base + byte, channels,
                outer_weight, middle_weight,
                dx_low, dx_high, dy_low, dy_high);
            const int16x8_t low = derivative_x ? dx_low : dy_low;
            const int16x8_t high = derivative_x ? dx_high : dy_high;
            if (output_depth == CV_16S)
            {
                short* output = reinterpret_cast<short*>(
                    dst.data + static_cast<std::size_t>(y) * dst.step(0));
                vst1q_s16(output + byte, low);
                vst1q_s16(output + byte + 8, high);
            }
            else
            {
                float* output = reinterpret_cast<float*>(
                    dst.data + static_cast<std::size_t>(y) * dst.step(0));
                store_f32_from_s16(output + byte, low);
                store_f32_from_s16(output + byte + 8, high);
            }
        }
        if (byte < vector_end)
        {
            byte = vector_end - 16;
            int16x8_t dx_low;
            int16x8_t dx_high;
            int16x8_t dy_low;
            int16x8_t dy_high;
            compute_derivative16(
                top, middle, bottom, source_base + byte, channels,
                outer_weight, middle_weight,
                dx_low, dx_high, dy_low, dy_high);
            const int16x8_t low = derivative_x ? dx_low : dy_low;
            const int16x8_t high = derivative_x ? dx_high : dy_high;
            if (output_depth == CV_16S)
            {
                short* output = reinterpret_cast<short*>(
                    dst.data + static_cast<std::size_t>(y) * dst.step(0));
                vst1q_s16(output + byte, low);
                vst1q_s16(output + byte + 8, high);
            }
            else
            {
                float* output = reinterpret_cast<float*>(
                    dst.data + static_cast<std::size_t>(y) * dst.step(0));
                store_f32_from_s16(output + byte, low);
                store_f32_from_s16(output + byte + 8, high);
            }
            byte = vector_end;
        }
        for (; byte < cols * channels; ++byte)
        {
            write_scalar(byte);
        }
    });
    return true;
}

inline bool run_pair_sobel_c1(
    const Mat& src,
    Mat& dx,
    Mat& dy,
    int border_type,
    bool isolated)
{
    const int rows = src.size[0];
    const int cols = src.size[1];
    const SobelSamplingWindow window =
        resolve_sobel_sampling_window(src, isolated);
    int vector_begin = 0;
    int vector_end = 0;
    vector_byte_range(window, cols, 1, vector_begin, vector_end);
    if (vector_end - vector_begin < 16)
    {
        return false;
    }
    dx.create(src.shape(), CV_16SC1);
    dy.create(src.shape(), CV_16SC1);
    const bool do_parallel = should_parallelize_filter_rows(rows, cols, 1, 9);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const int top_y = border_interpolate(
            y + window.row_offset - 1, window.rows, border_type);
        const int middle_y = border_interpolate(
            y + window.row_offset, window.rows, border_type);
        const int bottom_y = border_interpolate(
            y + window.row_offset + 1, window.rows, border_type);
        const uchar* top =
            window.base_data + static_cast<std::size_t>(top_y) * src.step(0);
        const uchar* middle =
            window.base_data + static_cast<std::size_t>(middle_y) * src.step(0);
        const uchar* bottom =
            window.base_data + static_cast<std::size_t>(bottom_y) * src.step(0);
        short* output_dx = reinterpret_cast<short*>(
            dx.data + static_cast<std::size_t>(y) * dx.step(0));
        short* output_dy = reinterpret_cast<short*>(
            dy.data + static_cast<std::size_t>(y) * dy.step(0));
        auto write_scalar_pair = [&](int x) {
            scalar_sobel_pair(
                window.base_data, src.step(0), window, y, x, border_type,
                output_dx[x], output_dy[x]);
        };
        int x = 0;
        for (; x < vector_begin; ++x)
        {
            write_scalar_pair(x);
        }
        for (; x + 16 <= vector_end; x += 16)
        {
            int16x8_t dx_low;
            int16x8_t dx_high;
            int16x8_t dy_low;
            int16x8_t dy_high;
            compute_derivative16(
                top, middle, bottom, window.col_offset + x, 1, 1, 2,
                dx_low, dx_high, dy_low, dy_high);
            vst1q_s16(output_dx + x, dx_low);
            vst1q_s16(output_dx + x + 8, dx_high);
            vst1q_s16(output_dy + x, dy_low);
            vst1q_s16(output_dy + x + 8, dy_high);
        }
        if (x < vector_end)
        {
            x = vector_end - 16;
            int16x8_t dx_low;
            int16x8_t dx_high;
            int16x8_t dy_low;
            int16x8_t dy_high;
            compute_derivative16(
                top, middle, bottom, window.col_offset + x, 1, 1, 2,
                dx_low, dx_high, dy_low, dy_high);
            vst1q_s16(output_dx + x, dx_low);
            vst1q_s16(output_dx + x + 8, dx_high);
            vst1q_s16(output_dy + x, dy_low);
            vst1q_s16(output_dy + x + 8, dy_high);
            x = vector_end;
        }
        for (; x < cols; ++x)
        {
            write_scalar_pair(x);
        }
    });
    return true;
}

#endif  // CVH_DETAIL_HAVE_NEON_KERNEL

inline bool try_single(
    const Mat& src,
    Mat& dst,
    int ddepth,
    int dx,
    int dy,
    int outer_weight,
    int middle_weight,
    double scale,
    double delta,
    int border_type_with_flags,
    const char* operation)
{
#if CVH_DETAIL_HAVE_NEON_KERNEL
    const int output_depth = ddepth < 0 ? CV_32F : CV_MAT_DEPTH(ddepth);
    const int border_type = normalize_border_type(border_type_with_flags);
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U ||
        (src.channels() != 1 && src.channels() != 3 && src.channels() != 4) ||
        (output_depth != CV_16S && output_depth != CV_32F) ||
        !((dx == 1 && dy == 0) || (dx == 0 && dy == 1)) ||
        scale != 1.0 || delta != 0.0 ||
        (border_type != BORDER_REPLICATE && border_type != BORDER_REFLECT_101) ||
        !direct_neon_allowed() ||
        static_cast<std::size_t>(src.size[0]) *
                static_cast<std::size_t>(src.size[1]) *
                static_cast<std::size_t>(src.channels()) <
            256)
    {
        return false;
    }
    const bool isolated = (border_type_with_flags & BORDER_ISOLATED) != 0;
    Mat source = src.data == dst.data ? src.clone() : src;
    if (!run_single(
            source, dst, output_depth, dx == 1, outer_weight, middle_weight,
            border_type, isolated))
    {
        return false;
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
    const bool is_scharr = std::strcmp(operation, "scharr3_u8") == 0;
    if (is_scharr)
    {
        cpu::set_last_kernel_route(
            output_depth == CV_16S
                ? "scharr3_u8:border=scalar;interior=neon;store=s16;tail=neon_overlap"
                : "scharr3_u8:border=scalar;interior=neon;store=f32;tail=neon_overlap");
    }
    else
    {
        cpu::set_last_kernel_route(
            output_depth == CV_16S
                ? "sobel3_u8:border=scalar;interior=neon;store=s16;tail=neon_overlap"
                : "sobel3_u8:border=scalar;interior=neon;store=f32;tail=neon_overlap");
    }
    return true;
#else
    (void)src;
    (void)dst;
    (void)ddepth;
    (void)dx;
    (void)dy;
    (void)outer_weight;
    (void)middle_weight;
    (void)scale;
    (void)delta;
    (void)border_type_with_flags;
    (void)operation;
    return false;
#endif
}

inline bool try_spatial_gradient(
    const Mat& src,
    Mat& dx,
    Mat& dy,
    int border_type_with_flags)
{
#if CVH_DETAIL_HAVE_NEON_KERNEL
    const int border_type = normalize_border_type(border_type_with_flags);
    if (src.empty() || src.dims != 2 || src.type() != CV_8UC1 ||
        (border_type != BORDER_REPLICATE && border_type != BORDER_REFLECT_101) ||
        !direct_neon_allowed() ||
        static_cast<std::size_t>(src.size[0]) *
                static_cast<std::size_t>(src.size[1]) <
            256)
    {
        return false;
    }
    const bool isolated = (border_type_with_flags & BORDER_ISOLATED) != 0;
    if (!run_pair_sobel_c1(src, dx, dy, border_type, isolated))
    {
        return false;
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::NEON);
    cpu::set_last_kernel_route(
        "spatial_gradient3_u8c1:border=scalar;interior=neon;store=dx_s16+dy_s16;tail=neon_overlap");
    return true;
#else
    (void)src;
    (void)dx;
    (void)dy;
    (void)border_type_with_flags;
    return false;
#endif
}

}  // namespace derivative3_neon
}  // namespace detail
}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_DERIVATIVE3_NEON_HPP
