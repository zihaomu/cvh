#ifndef CVH_IMGPROC_DETAIL_FILTER_UI_HPP
#define CVH_IMGPROC_DETAIL_FILTER_UI_HPP

#include "fastpath_common.hpp"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/simd/opencv_ui.h"

#include <cstddef>
#include <vector>

namespace cvh {
namespace detail {
namespace filter_ui {

inline bool enabled()
{
#if CVH_ENABLE_OPENCV_INTRIN && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return cpu::dispatch_mode() != cpu::DispatchMode::ScalarOnly;
#else
    return false;
#endif
}

inline float read_c1(const Mat& src, int y, int x)
{
    const uchar* row =
        src.data + static_cast<size_t>(y) * src.step(0);
    return src.depth() == CV_8U
               ? static_cast<float>(row[x])
               : reinterpret_cast<const float*>(row)[x];
}

inline void write_c1(Mat& dst, int y, int x, float value)
{
    uchar* row = dst.data + static_cast<size_t>(y) * dst.step(0);
    if (dst.depth() == CV_8U)
    {
        row[x] = saturate_cast<uchar>(value);
    }
    else if (dst.depth() == CV_16S)
    {
        reinterpret_cast<short*>(row)[x] =
            saturate_cast<short>(value);
    }
    else
    {
        reinterpret_cast<float*>(row)[x] = value;
    }
}

#if CVH_ENABLE_OPENCV_INTRIN && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)

inline cv::v_float32x4 load_c1_f32x4(const Mat& src, int y, int x)
{
    const uchar* row =
        src.data + static_cast<size_t>(y) * src.step(0);
    if (src.depth() == CV_8U)
    {
        return cv::v_cvt_f32(
            cv::v_reinterpret_as_s32(cv::v_load_expand_q(row + x)));
    }
    return cv::v_load(
        reinterpret_cast<const float*>(row) + x);
}

inline void store_c1_f32x4(Mat& dst,
                           int y,
                           int x,
                           const cv::v_float32x4& values)
{
    uchar* row = dst.data + static_cast<size_t>(y) * dst.step(0);
    if (dst.depth() == CV_8U)
    {
        int rounded[4];
        cv::v_store(rounded, cv::v_round(values));
        for (int lane = 0; lane < 4; ++lane)
        {
            row[x + lane] = saturate_cast<uchar>(rounded[lane]);
        }
    }
    else if (dst.depth() == CV_16S)
    {
        int rounded[4];
        cv::v_store(rounded, cv::v_round(values));
        short* output = reinterpret_cast<short*>(row) + x;
        for (int lane = 0; lane < 4; ++lane)
        {
            output[lane] = saturate_cast<short>(rounded[lane]);
        }
    }
    else
    {
        cv::v_store(
            reinterpret_cast<float*>(row) + x,
            values);
    }
}

inline void store_c1_u8x16(Mat& dst,
                           int y,
                           int x,
                           const cv::v_float32x4& values0,
                           const cv::v_float32x4& values1,
                           const cv::v_float32x4& values2,
                           const cv::v_float32x4& values3)
{
    uchar* row = dst.data + static_cast<size_t>(y) * dst.step(0);
    const cv::v_int16x8 packed0 =
        cv::v_pack(cv::v_round(values0), cv::v_round(values1));
    const cv::v_int16x8 packed1 =
        cv::v_pack(cv::v_round(values2), cv::v_round(values3));
    cv::v_store(row + x, cv::v_pack_u(packed0, packed1));
}

inline void store_c1_s16x8(Mat& dst,
                           int y,
                           int x,
                           const cv::v_float32x4& values0,
                           const cv::v_float32x4& values1)
{
    short* row = reinterpret_cast<short*>(
        dst.data + static_cast<size_t>(y) * dst.step(0));
    cv::v_store(
        row + x,
        cv::v_pack(cv::v_round(values0), cv::v_round(values1)));
}

#endif

inline bool filter2d_c1(const Mat& src,
                        Mat& dst,
                        int out_depth,
                        const std::vector<float>& kernel,
                        int kernel_rows,
                        int kernel_cols,
                        int anchor_x,
                        int anchor_y,
                        double delta,
                        int border_type)
{
#if CVH_ENABLE_OPENCV_INTRIN && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    if (!enabled() || src.channels() != 1 ||
        (src.depth() != CV_8U && src.depth() != CV_32F) ||
        (out_depth != CV_8U &&
         out_depth != CV_16S &&
         out_depth != CV_32F))
    {
        return false;
    }
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    constexpr int lanes = 4;
    const int vector_begin = anchor_x;
    const int vector_end =
        cols - (kernel_cols - anchor_x - 1);
    if (vector_end - vector_begin < lanes)
    {
        return false;
    }

    dst.create(src.shape(), CV_MAKETYPE(out_depth, 1));
    const std::vector<int> y_map = build_extended_index_map(
        rows,
        anchor_y,
        kernel_rows - anchor_y - 1,
        border_type);
    const bool do_parallel =
        should_parallelize_filter_rows(
            rows, cols, 1, kernel_rows * kernel_cols);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        int x = 0;
        for (; x < vector_begin; ++x)
        {
            double sum = delta;
            for (int ky = 0; ky < kernel_rows; ++ky)
            {
                const int sy =
                    y_map[static_cast<size_t>(y + ky)];
                if (sy < 0)
                {
                    continue;
                }
                for (int kx = 0; kx < kernel_cols; ++kx)
                {
                    const int sx = border_interpolate(
                        x + kx - anchor_x, cols, border_type);
                    if (sx >= 0)
                    {
                        sum += static_cast<double>(
                                   kernel[static_cast<size_t>(ky) *
                                              static_cast<size_t>(kernel_cols) +
                                          static_cast<size_t>(kx)]) *
                               read_c1(src, sy, sx);
                    }
                }
            }
            write_c1(dst, y, x, static_cast<float>(sum));
        }
        if (src.depth() == CV_8U && out_depth == CV_8U)
        {
            for (; x + 16 <= vector_end; x += 16)
            {
                cv::v_float32x4 sum0 =
                    cv::v_setall_f32(static_cast<float>(delta));
                cv::v_float32x4 sum1 = sum0;
                cv::v_float32x4 sum2 = sum0;
                cv::v_float32x4 sum3 = sum0;
                for (int ky = 0; ky < kernel_rows; ++ky)
                {
                    const int sy =
                        y_map[static_cast<size_t>(y + ky)];
                    if (sy < 0)
                    {
                        continue;
                    }
                    for (int kx = 0; kx < kernel_cols; ++kx)
                    {
                        const int source_x = x + kx - anchor_x;
                        const cv::v_float32x4 weight =
                            cv::v_setall_f32(
                                kernel[static_cast<size_t>(ky) *
                                           static_cast<size_t>(kernel_cols) +
                                       static_cast<size_t>(kx)]);
                        sum0 = cv::v_fma(
                            load_c1_f32x4(src, sy, source_x),
                            weight,
                            sum0);
                        sum1 = cv::v_fma(
                            load_c1_f32x4(src, sy, source_x + 4),
                            weight,
                            sum1);
                        sum2 = cv::v_fma(
                            load_c1_f32x4(src, sy, source_x + 8),
                            weight,
                            sum2);
                        sum3 = cv::v_fma(
                            load_c1_f32x4(src, sy, source_x + 12),
                            weight,
                            sum3);
                    }
                }
                store_c1_u8x16(
                    dst, y, x, sum0, sum1, sum2, sum3);
            }
        }
        if (out_depth == CV_16S)
        {
            for (; x + 8 <= vector_end; x += 8)
            {
                cv::v_float32x4 sum0 =
                    cv::v_setall_f32(static_cast<float>(delta));
                cv::v_float32x4 sum1 = sum0;
                for (int ky = 0; ky < kernel_rows; ++ky)
                {
                    const int sy =
                        y_map[static_cast<size_t>(y + ky)];
                    if (sy < 0)
                    {
                        continue;
                    }
                    for (int kx = 0; kx < kernel_cols; ++kx)
                    {
                        const int source_x = x + kx - anchor_x;
                        const cv::v_float32x4 weight =
                            cv::v_setall_f32(
                                kernel[static_cast<size_t>(ky) *
                                           static_cast<size_t>(kernel_cols) +
                                       static_cast<size_t>(kx)]);
                        sum0 = cv::v_fma(
                            load_c1_f32x4(src, sy, source_x),
                            weight,
                            sum0);
                        sum1 = cv::v_fma(
                            load_c1_f32x4(src, sy, source_x + 4),
                            weight,
                            sum1);
                    }
                }
                store_c1_s16x8(dst, y, x, sum0, sum1);
            }
        }
        for (; x + lanes <= vector_end; x += lanes)
        {
            cv::v_float32x4 sum =
                cv::v_setall_f32(static_cast<float>(delta));
            for (int ky = 0; ky < kernel_rows; ++ky)
            {
                const int sy =
                    y_map[static_cast<size_t>(y + ky)];
                if (sy < 0)
                {
                    continue;
                }
                for (int kx = 0; kx < kernel_cols; ++kx)
                {
                    const cv::v_float32x4 values =
                        load_c1_f32x4(
                            src, sy, x + kx - anchor_x);
                    sum = cv::v_fma(
                        values,
                        cv::v_setall_f32(
                            kernel[static_cast<size_t>(ky) *
                                       static_cast<size_t>(kernel_cols) +
                                   static_cast<size_t>(kx)]),
                        sum);
                }
            }
            store_c1_f32x4(dst, y, x, sum);
        }
        for (; x < cols; ++x)
        {
            double sum = delta;
            for (int ky = 0; ky < kernel_rows; ++ky)
            {
                const int sy =
                    y_map[static_cast<size_t>(y + ky)];
                if (sy < 0)
                {
                    continue;
                }
                for (int kx = 0; kx < kernel_cols; ++kx)
                {
                    const int sx = border_interpolate(
                        x + kx - anchor_x, cols, border_type);
                    if (sx >= 0)
                    {
                        sum += static_cast<double>(
                                   kernel[static_cast<size_t>(ky) *
                                              static_cast<size_t>(kernel_cols) +
                                          static_cast<size_t>(kx)]) *
                               read_c1(src, sy, sx);
                    }
                }
            }
            write_c1(dst, y, x, static_cast<float>(sum));
        }
    });
    cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
    return true;
#else
    (void)src;
    (void)dst;
    (void)out_depth;
    (void)kernel;
    (void)kernel_rows;
    (void)kernel_cols;
    (void)anchor_x;
    (void)anchor_y;
    (void)delta;
    (void)border_type;
    return false;
#endif
}

inline bool separable_c1(const Mat& src,
                         Mat& dst,
                         int out_depth,
                         const std::vector<float>& kernel_x,
                         const std::vector<float>& kernel_y,
                         int anchor_x,
                         int anchor_y,
                         double delta,
                         int border_type)
{
#if CVH_ENABLE_OPENCV_INTRIN && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    if (!enabled() || src.channels() != 1 ||
        (src.depth() != CV_8U && src.depth() != CV_32F) ||
        (out_depth != CV_8U && out_depth != CV_32F))
    {
        return false;
    }
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    constexpr int lanes = 4;
    const int kernel_cols = static_cast<int>(kernel_x.size());
    const int kernel_rows = static_cast<int>(kernel_y.size());
    const int vector_begin = anchor_x;
    const int vector_end =
        cols - (kernel_cols - anchor_x - 1);
    if (vector_end - vector_begin < lanes)
    {
        return false;
    }

    std::vector<float> temporary(
        static_cast<size_t>(rows) * static_cast<size_t>(cols),
        0.0f);
    const bool do_parallel =
        should_parallelize_filter_rows(
            rows, cols, 1, kernel_cols + kernel_rows);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        float* output =
            temporary.data() + static_cast<size_t>(y) *
                                   static_cast<size_t>(cols);
        int x = 0;
        for (; x < vector_begin; ++x)
        {
            float sum = 0.0f;
            for (int kx = 0; kx < kernel_cols; ++kx)
            {
                const int sx = border_interpolate(
                    x + kx - anchor_x, cols, border_type);
                if (sx >= 0)
                {
                    sum += kernel_x[static_cast<size_t>(kx)] *
                           read_c1(src, y, sx);
                }
            }
            output[x] = sum;
        }
        if (src.depth() == CV_8U)
        {
            for (; x + 16 <= vector_end; x += 16)
            {
                cv::v_float32x4 sum0 = cv::v_setzero_f32();
                cv::v_float32x4 sum1 = cv::v_setzero_f32();
                cv::v_float32x4 sum2 = cv::v_setzero_f32();
                cv::v_float32x4 sum3 = cv::v_setzero_f32();
                for (int kx = 0; kx < kernel_cols; ++kx)
                {
                    const int source_x = x + kx - anchor_x;
                    const cv::v_float32x4 weight =
                        cv::v_setall_f32(
                            kernel_x[static_cast<size_t>(kx)]);
                    sum0 = cv::v_fma(
                        load_c1_f32x4(src, y, source_x),
                        weight,
                        sum0);
                    sum1 = cv::v_fma(
                        load_c1_f32x4(src, y, source_x + 4),
                        weight,
                        sum1);
                    sum2 = cv::v_fma(
                        load_c1_f32x4(src, y, source_x + 8),
                        weight,
                        sum2);
                    sum3 = cv::v_fma(
                        load_c1_f32x4(src, y, source_x + 12),
                        weight,
                        sum3);
                }
                cv::v_store(output + x, sum0);
                cv::v_store(output + x + 4, sum1);
                cv::v_store(output + x + 8, sum2);
                cv::v_store(output + x + 12, sum3);
            }
        }
        for (; x + lanes <= vector_end; x += lanes)
        {
            cv::v_float32x4 sum = cv::v_setzero_f32();
            for (int kx = 0; kx < kernel_cols; ++kx)
            {
                sum = cv::v_fma(
                    load_c1_f32x4(src, y, x + kx - anchor_x),
                    cv::v_setall_f32(
                        kernel_x[static_cast<size_t>(kx)]),
                    sum);
            }
            cv::v_store(output + x, sum);
        }
        for (; x < cols; ++x)
        {
            float sum = 0.0f;
            for (int kx = 0; kx < kernel_cols; ++kx)
            {
                const int sx = border_interpolate(
                    x + kx - anchor_x, cols, border_type);
                if (sx >= 0)
                {
                    sum += kernel_x[static_cast<size_t>(kx)] *
                           read_c1(src, y, sx);
                }
            }
            output[x] = sum;
        }
    });

    dst.create(src.shape(), CV_MAKETYPE(out_depth, 1));
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        int x = 0;
        if (out_depth == CV_8U)
        {
            for (; x + 16 <= cols; x += 16)
            {
                cv::v_float32x4 sum0 =
                    cv::v_setall_f32(static_cast<float>(delta));
                cv::v_float32x4 sum1 = sum0;
                cv::v_float32x4 sum2 = sum0;
                cv::v_float32x4 sum3 = sum0;
                for (int ky = 0; ky < kernel_rows; ++ky)
                {
                    const int sy = border_interpolate(
                        y + ky - anchor_y, rows, border_type);
                    if (sy >= 0)
                    {
                        const float* source =
                            temporary.data() +
                            static_cast<size_t>(sy) *
                                static_cast<size_t>(cols) +
                            static_cast<size_t>(x);
                        const cv::v_float32x4 weight =
                            cv::v_setall_f32(
                                kernel_y[static_cast<size_t>(ky)]);
                        sum0 = cv::v_fma(
                            cv::v_load(source), weight, sum0);
                        sum1 = cv::v_fma(
                            cv::v_load(source + 4), weight, sum1);
                        sum2 = cv::v_fma(
                            cv::v_load(source + 8), weight, sum2);
                        sum3 = cv::v_fma(
                            cv::v_load(source + 12), weight, sum3);
                    }
                }
                store_c1_u8x16(
                    dst, y, x, sum0, sum1, sum2, sum3);
            }
        }
        for (; x + lanes <= cols; x += lanes)
        {
            cv::v_float32x4 sum =
                cv::v_setall_f32(static_cast<float>(delta));
            for (int ky = 0; ky < kernel_rows; ++ky)
            {
                const int sy = border_interpolate(
                    y + ky - anchor_y, rows, border_type);
                if (sy >= 0)
                {
                    sum = cv::v_fma(
                        cv::v_load(
                            temporary.data() +
                            static_cast<size_t>(sy) *
                                static_cast<size_t>(cols) +
                            static_cast<size_t>(x)),
                        cv::v_setall_f32(
                            kernel_y[static_cast<size_t>(ky)]),
                        sum);
                }
            }
            store_c1_f32x4(dst, y, x, sum);
        }
        for (; x < cols; ++x)
        {
            double sum = delta;
            for (int ky = 0; ky < kernel_rows; ++ky)
            {
                const int sy = border_interpolate(
                    y + ky - anchor_y, rows, border_type);
                if (sy >= 0)
                {
                    sum += static_cast<double>(
                               kernel_y[static_cast<size_t>(ky)]) *
                           temporary[
                               static_cast<size_t>(sy) *
                                   static_cast<size_t>(cols) +
                               static_cast<size_t>(x)];
                }
            }
            write_c1(dst, y, x, static_cast<float>(sum));
        }
    });
    cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
    return true;
#else
    (void)src;
    (void)dst;
    (void)out_depth;
    (void)kernel_x;
    (void)kernel_y;
    (void)anchor_x;
    (void)anchor_y;
    (void)delta;
    (void)border_type;
    return false;
#endif
}

inline bool spatial_gradient_u8_c1(const Mat& src,
                                   Mat& dx,
                                   Mat& dy,
                                   int border_type)
{
#if CVH_ENABLE_OPENCV_INTRIN && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    if (!enabled() || src.depth() != CV_8U || src.channels() != 1 ||
        src.dims != 2 || src.size.p[0] <= 0 || src.size.p[1] <= 0)
    {
        return false;
    }
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    dx.create(src.shape(), CV_16SC1);
    dy.create(src.shape(), CV_16SC1);
    const bool do_parallel =
        should_parallelize_filter_rows(rows, cols, 1, 9);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const int y0 = border_interpolate(y - 1, rows, border_type);
        const int y1 = y;
        const int y2 = border_interpolate(y + 1, rows, border_type);
        const uchar* row0 =
            src.data + static_cast<size_t>(y0) * src.step(0);
        const uchar* row1 =
            src.data + static_cast<size_t>(y1) * src.step(0);
        const uchar* row2 =
            src.data + static_cast<size_t>(y2) * src.step(0);
        short* output_x = reinterpret_cast<short*>(
            dx.data + static_cast<size_t>(y) * dx.step(0));
        short* output_y = reinterpret_cast<short*>(
            dy.data + static_cast<size_t>(y) * dy.step(0));

        auto scalar_pixel = [&](int x) {
            const int x0 = border_interpolate(x - 1, cols, border_type);
            const int x1 = x;
            const int x2 = border_interpolate(x + 1, cols, border_type);
            output_x[x] = saturate_cast<short>(
                (static_cast<int>(row0[x2]) -
                 static_cast<int>(row0[x0])) +
                2 * (static_cast<int>(row1[x2]) -
                     static_cast<int>(row1[x0])) +
                (static_cast<int>(row2[x2]) -
                 static_cast<int>(row2[x0])));
            output_y[x] = saturate_cast<short>(
                (static_cast<int>(row2[x0]) +
                 2 * static_cast<int>(row2[x1]) +
                 static_cast<int>(row2[x2])) -
                (static_cast<int>(row0[x0]) +
                 2 * static_cast<int>(row0[x1]) +
                 static_cast<int>(row0[x2])));
        };

        int x = 0;
        if (cols > 0)
        {
            scalar_pixel(x++);
        }
        for (; x + 8 <= cols - 1; x += 8)
        {
            cv::v_int32x4 gx0;
            cv::v_int32x4 gx1;
            cv::v_int32x4 gy0;
            cv::v_int32x4 gy1;
            for (int half = 0; half < 2; ++half)
            {
                const int offset = x + half * 4;
                const cv::v_int32x4 top_left =
                    cv::v_reinterpret_as_s32(
                        cv::v_load_expand_q(row0 + offset - 1));
                const cv::v_int32x4 top_center =
                    cv::v_reinterpret_as_s32(
                        cv::v_load_expand_q(row0 + offset));
                const cv::v_int32x4 top_right =
                    cv::v_reinterpret_as_s32(
                        cv::v_load_expand_q(row0 + offset + 1));
                const cv::v_int32x4 middle_left =
                    cv::v_reinterpret_as_s32(
                        cv::v_load_expand_q(row1 + offset - 1));
                const cv::v_int32x4 middle_right =
                    cv::v_reinterpret_as_s32(
                        cv::v_load_expand_q(row1 + offset + 1));
                const cv::v_int32x4 bottom_left =
                    cv::v_reinterpret_as_s32(
                        cv::v_load_expand_q(row2 + offset - 1));
                const cv::v_int32x4 bottom_center =
                    cv::v_reinterpret_as_s32(
                        cv::v_load_expand_q(row2 + offset));
                const cv::v_int32x4 bottom_right =
                    cv::v_reinterpret_as_s32(
                        cv::v_load_expand_q(row2 + offset + 1));
                const cv::v_int32x4 gradient_x =
                    cv::v_add(
                        cv::v_add(
                            cv::v_sub(top_right, top_left),
                            cv::v_add(
                                cv::v_sub(middle_right, middle_left),
                                cv::v_sub(middle_right, middle_left))),
                        cv::v_sub(bottom_right, bottom_left));
                const cv::v_int32x4 gradient_y =
                    cv::v_sub(
                        cv::v_add(
                            cv::v_add(
                                bottom_left,
                                cv::v_add(bottom_center, bottom_center)),
                            bottom_right),
                        cv::v_add(
                            cv::v_add(
                                top_left,
                                cv::v_add(top_center, top_center)),
                            top_right));
                if (half == 0)
                {
                    gx0 = gradient_x;
                    gy0 = gradient_y;
                }
                else
                {
                    gx1 = gradient_x;
                    gy1 = gradient_y;
                }
            }
            cv::v_store(output_x + x, cv::v_pack(gx0, gx1));
            cv::v_store(output_y + x, cv::v_pack(gy0, gy1));
        }
        for (; x < cols; ++x)
        {
            scalar_pixel(x);
        }
    });
    cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
    return true;
#else
    (void)src;
    (void)dx;
    (void)dy;
    (void)border_type;
    return false;
#endif
}

}  // namespace filter_ui
}  // namespace detail
}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_FILTER_UI_HPP
