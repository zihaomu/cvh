#ifndef CVH_IMGPROC_DETAIL_FILTER2D_IMPL_HPP
#define CVH_IMGPROC_DETAIL_FILTER2D_IMPL_HPP

#include "fastpath_common.hpp"
#include "filter_ui.hpp"

namespace cvh
{
namespace detail
{

namespace filter2d_fastpath
{
inline thread_local const char* g_last_filter2d_algorithm_path = "fallback";

inline bool is_cross3_kernel(const std::vector<float>& kernel)
{
    if (kernel.size() != 9u)
    {
        return false;
    }
    constexpr float expected[9] = {
        0.0f, 0.25f, 0.0f,
        0.25f, 0.0f, 0.25f,
        0.0f, 0.25f, 0.0f};
    return std::equal(kernel.begin(), kernel.end(), expected);
}

inline bool try_filter2d_cross3(const Mat& src,
                                Mat& dst,
                                int out_depth,
                                const std::vector<float>& kernel,
                                int anchor_x,
                                int anchor_y,
                                double delta,
                                int border_type)
{
    if (!is_cross3_kernel(kernel) || anchor_x != 1 || anchor_y != 1 ||
        delta != 0.0 || out_depth != src.depth() ||
        (src.depth() != CV_8U && src.depth() != CV_32F) ||
        !is_u8_fastpath_channels(src.channels()))
    {
        return false;
    }
    const int rows = src.size[0];
    const int cols = src.size[1];
    const int channels = src.channels();
    if (rows <= 0 || cols <= 0)
    {
        return false;
    }
    const int row_stride = cols * channels;
    const std::size_t src_step = src.step(0);
    dst.create(src.shape(), src.type());
    const std::size_t dst_step = dst.step(0);

#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const bool use_ui = cpu::opencv_ui_allowed();
#else
    const bool use_ui = false;
#endif

    const bool do_parallel =
        should_parallelize_filter_rows(rows, cols, channels, 5);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const int sy0 = border_interpolate(y - 1, rows, border_type);
        const int sy2 = border_interpolate(y + 1, rows, border_type);

        if (src.depth() == CV_8U)
        {
            const uchar* center =
                src.data + static_cast<std::size_t>(y) * src_step;
            const uchar* upper =
                sy0 >= 0
                    ? src.data + static_cast<std::size_t>(sy0) * src_step
                    : nullptr;
            const uchar* lower =
                sy2 >= 0
                    ? src.data + static_cast<std::size_t>(sy2) * src_step
                    : nullptr;
            uchar* output =
                dst.data + static_cast<std::size_t>(y) * dst_step;

            const auto scalar_pixel = [&](int x, int c) {
                const int sx0 =
                    border_interpolate(x - 1, cols, border_type);
                const int sx2 =
                    border_interpolate(x + 1, cols, border_type);
                unsigned sum = 0;
                if (upper)
                {
                    sum += upper[x * channels + c];
                }
                if (lower)
                {
                    sum += lower[x * channels + c];
                }
                if (sx0 >= 0)
                {
                    sum += center[sx0 * channels + c];
                }
                if (sx2 >= 0)
                {
                    sum += center[sx2 * channels + c];
                }
                output[x * channels + c] = static_cast<uchar>(
                    (sum + 1u + ((sum >> 2) & 1u)) >> 2);
            };
            for (int c = 0; c < channels; ++c)
            {
                scalar_pixel(0, c);
            }
            int offset = channels;
            const int interior_end =
                std::max(channels, row_stride - channels);
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
            if (use_ui)
            {
                const cv::v_uint16x8 one = cv::v_setall_u16(1);
                const cv::v_uint8x16 zero8 = cv::v_setzero_u8();
                const auto rounded8 = [&](int lane) {
                    const cv::v_uint16x8 vertical = cv::v_add(
                        upper
                            ? cv::v_load_expand(upper + offset + lane)
                            : cv::v_expand_low(zero8),
                        lower
                            ? cv::v_load_expand(lower + offset + lane)
                            : cv::v_expand_low(zero8));
                    const cv::v_uint16x8 sum = cv::v_add(
                        vertical,
                        cv::v_add(
                            cv::v_load_expand(
                                center + offset + lane - channels),
                            cv::v_load_expand(
                                center + offset + lane + channels)));
                    return cv::v_shr<2>(cv::v_add(
                        cv::v_add(sum, one),
                        cv::v_and(cv::v_shr<2>(sum), one)));
                };
                for (; offset + 16 <= interior_end; offset += 16)
                {
                    cv::v_store(
                        output + offset,
                        cv::v_pack(rounded8(0), rounded8(8)));
                }
            }
#endif
            for (; offset < interior_end; ++offset)
            {
                unsigned sum =
                    center[offset - channels] +
                    center[offset + channels];
                if (upper)
                {
                    sum += upper[offset];
                }
                if (lower)
                {
                    sum += lower[offset];
                }
                output[offset] = static_cast<uchar>(
                    (sum + 1u + ((sum >> 2) & 1u)) >> 2);
            }
            if (cols > 1)
            {
                for (int c = 0; c < channels; ++c)
                {
                    scalar_pixel(cols - 1, c);
                }
            }
            return;
        }

        const float* center = reinterpret_cast<const float*>(
            src.data + static_cast<std::size_t>(y) * src_step);
        const float* upper =
            sy0 >= 0
                ? reinterpret_cast<const float*>(
                      src.data + static_cast<std::size_t>(sy0) * src_step)
                : nullptr;
        const float* lower =
            sy2 >= 0
                ? reinterpret_cast<const float*>(
                      src.data + static_cast<std::size_t>(sy2) * src_step)
                : nullptr;
        float* output = reinterpret_cast<float*>(
            dst.data + static_cast<std::size_t>(y) * dst_step);
        const auto scalar_pixel = [&](int x, int c) {
            const int sx0 = border_interpolate(x - 1, cols, border_type);
            const int sx2 = border_interpolate(x + 1, cols, border_type);
            float sum = 0.0f;
            if (upper)
            {
                sum += upper[x * channels + c];
            }
            if (lower)
            {
                sum += lower[x * channels + c];
            }
            if (sx0 >= 0)
            {
                sum += center[sx0 * channels + c];
            }
            if (sx2 >= 0)
            {
                sum += center[sx2 * channels + c];
            }
            output[x * channels + c] = sum * 0.25f;
        };
        for (int c = 0; c < channels; ++c)
        {
            scalar_pixel(0, c);
        }
        int offset = channels;
        const int interior_end =
            std::max(channels, row_stride - channels);
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
        if (use_ui)
        {
            const cv::v_float32x4 zero = cv::v_setzero_f32();
            const cv::v_float32x4 scale = cv::v_setall_f32(0.25f);
            for (; offset + 8 <= interior_end; offset += 8)
            {
                const auto result4 = [&](int lane) {
                    return cv::v_mul(
                        cv::v_add(
                            cv::v_add(
                                upper ? cv::v_load(upper + offset + lane)
                                      : zero,
                                lower ? cv::v_load(lower + offset + lane)
                                      : zero),
                            cv::v_add(
                                cv::v_load(
                                    center + offset + lane - channels),
                                cv::v_load(
                                    center + offset + lane + channels))),
                        scale);
                };
                cv::v_store(output + offset, result4(0));
                cv::v_store(output + offset + 4, result4(4));
            }
            for (; offset + 4 <= interior_end; offset += 4)
            {
                cv::v_store(
                    output + offset,
                    cv::v_mul(
                        cv::v_add(
                            cv::v_add(
                                upper ? cv::v_load(upper + offset) : zero,
                                lower ? cv::v_load(lower + offset) : zero),
                            cv::v_add(
                                cv::v_load(center + offset - channels),
                                cv::v_load(center + offset + channels))),
                        scale));
            }
        }
#endif
        for (; offset < interior_end; ++offset)
        {
            float sum = center[offset - channels] +
                        center[offset + channels];
            if (upper)
            {
                sum += upper[offset];
            }
            if (lower)
            {
                sum += lower[offset];
            }
            output[offset] = sum * 0.25f;
        }
        if (cols > 1)
        {
            for (int c = 0; c < channels; ++c)
            {
                scalar_pixel(cols - 1, c);
            }
        }
    });

    cpu::set_last_dispatch_tag(
        use_ui && row_stride >= (src.depth() == CV_8U ? 16 : 4)
            ? cpu::DispatchTag::OpenCVUI
            : cpu::DispatchTag::Scalar);
    return true;
}

inline bool try_filter2d_fastpath(const Mat& src,
                           Mat& dst,
                           int ddepth,
                           const Mat& kernel,
                           Point anchor,
                           double delta,
                           int borderType)
{
    if (src.empty() || src.dims != 2)
    {
        return false;
    }

    const int src_depth = src.depth();
    if (src_depth != CV_8U && src_depth != CV_32F)
    {
        return false;
    }

    if (kernel.empty() || kernel.dims != 2 || kernel.channels() != 1 || kernel.depth() != CV_32F)
    {
        return false;
    }

    const int krows = kernel.size[0];
    const int kcols = kernel.size[1];
    if (krows <= 0 || kcols <= 0)
    {
        return false;
    }

    const int ax = anchor.x >= 0 ? anchor.x : (kcols / 2);
    const int ay = anchor.y >= 0 ? anchor.y : (krows / 2);
    if (ax < 0 || ax >= kcols || ay < 0 || ay >= krows)
    {
        return false;
    }

    int out_depth = ddepth;
    if (out_depth == -1)
    {
        out_depth = src_depth;
    }
    if (out_depth != CV_8U && out_depth != CV_32F)
    {
        return false;
    }

    const int border_type = normalize_border_type(borderType);
    if (!is_supported_filter_border(border_type))
    {
        return false;
    }

    Mat src_local;
    const Mat* src_ref = &src;
    if (src.data == dst.data)
    {
        src_local = src.clone();
        src_ref = &src_local;
    }

    const int rows = src_ref->size[0];
    const int cols = src_ref->size[1];
    const int channels = src_ref->channels();
    if (rows <= 0 || cols <= 0 || channels <= 0)
    {
        return false;
    }

    std::vector<float> kernel_coeffs(static_cast<std::size_t>(krows) * static_cast<std::size_t>(kcols), 0.0f);
    for (int ky = 0; ky < krows; ++ky)
    {
        for (int kx = 0; kx < kcols; ++kx)
        {
            kernel_coeffs[static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols) + static_cast<std::size_t>(kx)] =
                kernel.at<float>(ky, kx);
        }
    }

    if (try_filter2d_cross3(
            *src_ref,
            dst,
            out_depth,
            kernel_coeffs,
            ax,
            ay,
            delta,
            border_type))
    {
        g_last_filter2d_algorithm_path = "cross3_direct";
        return true;
    }

    g_last_filter2d_algorithm_path = "generic_filter2d";

    if (filter_ui::filter2d_c1(*src_ref,
                               dst,
                               out_depth,
                               kernel_coeffs,
                               krows,
                               kcols,
                               ax,
                               ay,
                               delta,
                               border_type))
    {
        return true;
    }

    std::vector<int> x_offsets(static_cast<std::size_t>(cols) * static_cast<std::size_t>(kcols), -1);
    for (int x = 0; x < cols; ++x)
    {
        int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kcols);
        for (int kx = 0; kx < kcols; ++kx)
        {
            const int sx = border_interpolate(x + kx - ax, cols, border_type);
            x_ofs[kx] = sx >= 0 ? sx * channels : -1;
        }
    }

    std::vector<int> y_indices(static_cast<std::size_t>(rows) * static_cast<std::size_t>(krows), -1);
    for (int y = 0; y < rows; ++y)
    {
        int* y_idx = y_indices.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(krows);
        for (int ky = 0; ky < krows; ++ky)
        {
            y_idx[ky] = border_interpolate(y + ky - ay, rows, border_type);
        }
    }

    dst.create(std::vector<int>{rows, cols}, CV_MAKETYPE(out_depth, channels));
    const std::size_t src_step = src_ref->step(0);
    const std::size_t dst_step = dst.step(0);
    const bool do_parallel = should_parallelize_filter_rows(rows, cols, channels, krows * kcols);

    if (src_depth == CV_8U)
    {
        const uchar* src_data = src_ref->data;
        parallel_for_index_if(do_parallel, rows, [&](int y) {
            const int* y_idx = y_indices.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(krows);
            uchar* dst_row_u8 = out_depth == CV_8U ? (dst.data + static_cast<std::size_t>(y) * dst_step) : nullptr;
            float* dst_row_f32 =
                out_depth == CV_32F ? reinterpret_cast<float*>(dst.data + static_cast<std::size_t>(y) * dst_step) : nullptr;

            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kcols);
                const std::size_t out_base = static_cast<std::size_t>(x) * static_cast<std::size_t>(channels);

                if (channels == 1)
                {
                    double acc0 = delta;
                    for (int ky = 0; ky < krows; ++ky)
                    {
                        const int sy = y_idx[ky];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const uchar* src_row = src_data + static_cast<std::size_t>(sy) * src_step;
                        const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                        for (int kx = 0; kx < kcols; ++kx)
                        {
                            const int sx = x_ofs[kx];
                            if (sx < 0)
                            {
                                continue;
                            }
                            acc0 += static_cast<double>(krow[kx]) * static_cast<double>(src_row[sx]);
                        }
                    }

                    if (dst_row_f32)
                    {
                        dst_row_f32[out_base] = static_cast<float>(acc0);
                    }
                    else
                    {
                        dst_row_u8[out_base] = saturate_cast<uchar>(acc0);
                    }
                    continue;
                }

                if (channels == 3)
                {
                    double acc0 = delta;
                    double acc1 = delta;
                    double acc2 = delta;
                    for (int ky = 0; ky < krows; ++ky)
                    {
                        const int sy = y_idx[ky];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const uchar* src_row = src_data + static_cast<std::size_t>(sy) * src_step;
                        const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                        for (int kx = 0; kx < kcols; ++kx)
                        {
                            const int sx = x_ofs[kx];
                            if (sx < 0)
                            {
                                continue;
                            }
                            const float w = krow[kx];
                            acc0 += static_cast<double>(w) * static_cast<double>(src_row[sx + 0]);
                            acc1 += static_cast<double>(w) * static_cast<double>(src_row[sx + 1]);
                            acc2 += static_cast<double>(w) * static_cast<double>(src_row[sx + 2]);
                        }
                    }

                    if (dst_row_f32)
                    {
                        dst_row_f32[out_base + 0] = static_cast<float>(acc0);
                        dst_row_f32[out_base + 1] = static_cast<float>(acc1);
                        dst_row_f32[out_base + 2] = static_cast<float>(acc2);
                    }
                    else
                    {
                        dst_row_u8[out_base + 0] = saturate_cast<uchar>(acc0);
                        dst_row_u8[out_base + 1] = saturate_cast<uchar>(acc1);
                        dst_row_u8[out_base + 2] = saturate_cast<uchar>(acc2);
                    }
                    continue;
                }

                if (channels == 4)
                {
                    double acc0 = delta;
                    double acc1 = delta;
                    double acc2 = delta;
                    double acc3 = delta;
                    for (int ky = 0; ky < krows; ++ky)
                    {
                        const int sy = y_idx[ky];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const uchar* src_row = src_data + static_cast<std::size_t>(sy) * src_step;
                        const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                        for (int kx = 0; kx < kcols; ++kx)
                        {
                            const int sx = x_ofs[kx];
                            if (sx < 0)
                            {
                                continue;
                            }
                            const float w = krow[kx];
                            acc0 += static_cast<double>(w) * static_cast<double>(src_row[sx + 0]);
                            acc1 += static_cast<double>(w) * static_cast<double>(src_row[sx + 1]);
                            acc2 += static_cast<double>(w) * static_cast<double>(src_row[sx + 2]);
                            acc3 += static_cast<double>(w) * static_cast<double>(src_row[sx + 3]);
                        }
                    }

                    if (dst_row_f32)
                    {
                        dst_row_f32[out_base + 0] = static_cast<float>(acc0);
                        dst_row_f32[out_base + 1] = static_cast<float>(acc1);
                        dst_row_f32[out_base + 2] = static_cast<float>(acc2);
                        dst_row_f32[out_base + 3] = static_cast<float>(acc3);
                    }
                    else
                    {
                        dst_row_u8[out_base + 0] = saturate_cast<uchar>(acc0);
                        dst_row_u8[out_base + 1] = saturate_cast<uchar>(acc1);
                        dst_row_u8[out_base + 2] = saturate_cast<uchar>(acc2);
                        dst_row_u8[out_base + 3] = saturate_cast<uchar>(acc3);
                    }
                    continue;
                }

                for (int c = 0; c < channels; ++c)
                {
                    double acc = delta;
                    for (int ky = 0; ky < krows; ++ky)
                    {
                        const int sy = y_idx[ky];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const uchar* src_row = src_data + static_cast<std::size_t>(sy) * src_step;
                        const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                        for (int kx = 0; kx < kcols; ++kx)
                        {
                            const int sx = x_ofs[kx];
                            if (sx < 0)
                            {
                                continue;
                            }
                            acc += static_cast<double>(krow[kx]) * static_cast<double>(src_row[sx + c]);
                        }
                    }

                    if (dst_row_f32)
                    {
                        dst_row_f32[out_base + static_cast<std::size_t>(c)] = static_cast<float>(acc);
                    }
                    else
                    {
                        dst_row_u8[out_base + static_cast<std::size_t>(c)] = saturate_cast<uchar>(acc);
                    }
                }
            }
        });
        return true;
    }

    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const int* y_idx = y_indices.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(krows);
        uchar* dst_row_u8 = out_depth == CV_8U ? (dst.data + static_cast<std::size_t>(y) * dst_step) : nullptr;
        float* dst_row_f32 =
            out_depth == CV_32F ? reinterpret_cast<float*>(dst.data + static_cast<std::size_t>(y) * dst_step) : nullptr;

        for (int x = 0; x < cols; ++x)
        {
            const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kcols);
            const std::size_t out_base = static_cast<std::size_t>(x) * static_cast<std::size_t>(channels);

            if (channels == 1)
            {
                double acc0 = delta;
                for (int ky = 0; ky < krows; ++ky)
                {
                    const int sy = y_idx[ky];
                    if (sy < 0)
                    {
                        continue;
                    }
                    const float* src_row =
                        reinterpret_cast<const float*>(src_ref->data + static_cast<std::size_t>(sy) * src_step);
                    const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                    for (int kx = 0; kx < kcols; ++kx)
                    {
                        const int sx = x_ofs[kx];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc0 += static_cast<double>(krow[kx]) * static_cast<double>(src_row[sx]);
                    }
                }

                if (dst_row_f32)
                {
                    dst_row_f32[out_base] = static_cast<float>(acc0);
                }
                else
                {
                    dst_row_u8[out_base] = saturate_cast<uchar>(acc0);
                }
                continue;
            }

            if (channels == 3)
            {
                double acc0 = delta;
                double acc1 = delta;
                double acc2 = delta;
                for (int ky = 0; ky < krows; ++ky)
                {
                    const int sy = y_idx[ky];
                    if (sy < 0)
                    {
                        continue;
                    }
                    const float* src_row =
                        reinterpret_cast<const float*>(src_ref->data + static_cast<std::size_t>(sy) * src_step);
                    const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                    for (int kx = 0; kx < kcols; ++kx)
                    {
                        const int sx = x_ofs[kx];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = krow[kx];
                        acc0 += static_cast<double>(w) * static_cast<double>(src_row[sx + 0]);
                        acc1 += static_cast<double>(w) * static_cast<double>(src_row[sx + 1]);
                        acc2 += static_cast<double>(w) * static_cast<double>(src_row[sx + 2]);
                    }
                }

                if (dst_row_f32)
                {
                    dst_row_f32[out_base + 0] = static_cast<float>(acc0);
                    dst_row_f32[out_base + 1] = static_cast<float>(acc1);
                    dst_row_f32[out_base + 2] = static_cast<float>(acc2);
                }
                else
                {
                    dst_row_u8[out_base + 0] = saturate_cast<uchar>(acc0);
                    dst_row_u8[out_base + 1] = saturate_cast<uchar>(acc1);
                    dst_row_u8[out_base + 2] = saturate_cast<uchar>(acc2);
                }
                continue;
            }

            if (channels == 4)
            {
                double acc0 = delta;
                double acc1 = delta;
                double acc2 = delta;
                double acc3 = delta;
                for (int ky = 0; ky < krows; ++ky)
                {
                    const int sy = y_idx[ky];
                    if (sy < 0)
                    {
                        continue;
                    }
                    const float* src_row =
                        reinterpret_cast<const float*>(src_ref->data + static_cast<std::size_t>(sy) * src_step);
                    const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                    for (int kx = 0; kx < kcols; ++kx)
                    {
                        const int sx = x_ofs[kx];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = krow[kx];
                        acc0 += static_cast<double>(w) * static_cast<double>(src_row[sx + 0]);
                        acc1 += static_cast<double>(w) * static_cast<double>(src_row[sx + 1]);
                        acc2 += static_cast<double>(w) * static_cast<double>(src_row[sx + 2]);
                        acc3 += static_cast<double>(w) * static_cast<double>(src_row[sx + 3]);
                    }
                }

                if (dst_row_f32)
                {
                    dst_row_f32[out_base + 0] = static_cast<float>(acc0);
                    dst_row_f32[out_base + 1] = static_cast<float>(acc1);
                    dst_row_f32[out_base + 2] = static_cast<float>(acc2);
                    dst_row_f32[out_base + 3] = static_cast<float>(acc3);
                }
                else
                {
                    dst_row_u8[out_base + 0] = saturate_cast<uchar>(acc0);
                    dst_row_u8[out_base + 1] = saturate_cast<uchar>(acc1);
                    dst_row_u8[out_base + 2] = saturate_cast<uchar>(acc2);
                    dst_row_u8[out_base + 3] = saturate_cast<uchar>(acc3);
                }
                continue;
            }

            for (int c = 0; c < channels; ++c)
            {
                double acc = delta;
                for (int ky = 0; ky < krows; ++ky)
                {
                    const int sy = y_idx[ky];
                    if (sy < 0)
                    {
                        continue;
                    }
                    const float* src_row =
                        reinterpret_cast<const float*>(src_ref->data + static_cast<std::size_t>(sy) * src_step);
                    const float* krow = kernel_coeffs.data() + static_cast<std::size_t>(ky) * static_cast<std::size_t>(kcols);
                    for (int kx = 0; kx < kcols; ++kx)
                    {
                        const int sx = x_ofs[kx];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc += static_cast<double>(krow[kx]) * static_cast<double>(src_row[sx + c]);
                    }
                }

                if (dst_row_f32)
                {
                    dst_row_f32[out_base + static_cast<std::size_t>(c)] = static_cast<float>(acc);
                }
                else
                {
                    dst_row_u8[out_base + static_cast<std::size_t>(c)] = saturate_cast<uchar>(acc);
                }
            }
        }
    });

    return true;
}


} // namespace filter2d_fastpath

inline const char* last_filter2d_algorithm_path()
{
    return filter2d_fastpath::g_last_filter2d_algorithm_path;
}

inline void filter2D_fast_impl(const Mat& src,
                           Mat& dst,
                           int ddepth,
                           const Mat& kernel,
                           Point anchor,
                           double delta,
                           int borderType)
{
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    filter2d_fastpath::g_last_filter2d_algorithm_path = "fallback";
    if (filter2d_fastpath::try_filter2d_fastpath(src, dst, ddepth, kernel, anchor, delta, borderType))
    {
        return;
    }

    filter2D_fallback(src, dst, ddepth, kernel, anchor, delta, borderType);
}

} // namespace detail
} // namespace cvh

#endif // CVH_IMGPROC_DETAIL_FILTER2D_IMPL_HPP
