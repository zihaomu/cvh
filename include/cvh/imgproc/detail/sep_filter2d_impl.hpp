#ifndef CVH_IMGPROC_DETAIL_SEP_FILTER2D_IMPL_HPP
#define CVH_IMGPROC_DETAIL_SEP_FILTER2D_IMPL_HPP

#include "fastpath_common.hpp"
#include "filter_ui.hpp"

#include <cstdint>

namespace cvh
{
namespace detail
{

namespace sep_filter2d_fastpath
{
inline thread_local const char* g_last_sepfilter2d_algorithm_path = "fallback";

inline bool is_binomial3_kernel(const std::vector<float>& kernel)
{
    return kernel.size() == 3u &&
           kernel[0] == 0.25f && kernel[1] == 0.5f &&
           kernel[2] == 0.25f;
}

inline unsigned round_shift2_even(unsigned value)
{
    return (value + 1u + ((value >> 2) & 1u)) >> 2;
}

inline bool try_sep3_binomial(const Mat& src,
                              Mat& dst,
                              int out_depth,
                              const std::vector<float>& kx,
                              const std::vector<float>& ky,
                              int anchor_x,
                              int anchor_y,
                              double delta,
                              int border_type)
{
    if (!is_binomial3_kernel(kx) || !is_binomial3_kernel(ky) ||
        anchor_x != 1 || anchor_y != 1 || delta != 0.0 ||
        out_depth != src.depth() ||
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

    if (src.depth() == CV_8U)
    {
        std::vector<uchar> temporary(
            static_cast<std::size_t>(rows) *
            static_cast<std::size_t>(row_stride));
        parallel_for_index_if(
            should_parallelize_filter_rows(rows, cols, channels, 3),
            rows,
            [&](int y) {
                const uchar* input =
                    src.data + static_cast<std::size_t>(y) * src_step;
                uchar* output =
                    temporary.data() +
                    static_cast<std::size_t>(y) *
                        static_cast<std::size_t>(row_stride);
                for (int c = 0; c < channels; ++c)
                {
                    unsigned sum = 0;
                    for (int k = 0; k < 3; ++k)
                    {
                        const int sx =
                            border_interpolate(k - 1, cols, border_type);
                        if (sx >= 0)
                        {
                            sum += static_cast<unsigned>(k == 1 ? 2 : 1) *
                                   input[sx * channels + c];
                        }
                    }
                    output[c] = static_cast<uchar>((sum + 2u) >> 2);
                }

                int offset = channels;
                const int interior_end =
                    std::max(channels, row_stride - channels);
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
                if (use_ui)
                {
                    const cv::v_uint16x8 rounding = cv::v_setall_u16(2);
                    for (; offset + 16 <= interior_end; offset += 16)
                    {
                        const auto sum8 = [&](int lane) {
                            const cv::v_uint16x8 center = cv::v_load_expand(
                                input + offset + lane);
                            return cv::v_add(
                                cv::v_add(
                                    cv::v_load_expand(
                                        input + offset + lane - channels),
                                    cv::v_load_expand(
                                        input + offset + lane + channels)),
                                cv::v_shl<1>(center));
                        };
                        cv::v_store(
                            output + offset,
                            cv::v_pack(
                                cv::v_shr<2>(cv::v_add(sum8(0), rounding)),
                                cv::v_shr<2>(cv::v_add(sum8(8), rounding))));
                    }
                    for (; offset + 8 <= interior_end; offset += 8)
                    {
                        const cv::v_uint16x8 center =
                            cv::v_load_expand(input + offset);
                        const cv::v_uint16x8 sum = cv::v_add(
                            cv::v_add(
                                cv::v_load_expand(
                                    input + offset - channels),
                                cv::v_load_expand(
                                    input + offset + channels)),
                            cv::v_shl<1>(center));
                        std::uint16_t lanes[8];
                        cv::v_store(
                            lanes,
                            cv::v_shr<2>(cv::v_add(sum, rounding)));
                        for (int lane = 0; lane < 8; ++lane)
                        {
                            output[offset + lane] =
                                static_cast<uchar>(lanes[lane]);
                        }
                    }
                }
#endif
                for (; offset < interior_end; ++offset)
                {
                    const unsigned sum =
                        input[offset - channels] +
                        2u * input[offset] +
                        input[offset + channels];
                    output[offset] =
                        static_cast<uchar>((sum + 2u) >> 2);
                }
                if (cols > 1)
                {
                    const int x = cols - 1;
                    for (int c = 0; c < channels; ++c)
                    {
                        unsigned sum = 0;
                        for (int k = 0; k < 3; ++k)
                        {
                            const int sx = border_interpolate(
                                x + k - 1, cols, border_type);
                            if (sx >= 0)
                            {
                                sum +=
                                    static_cast<unsigned>(k == 1 ? 2 : 1) *
                                    input[sx * channels + c];
                            }
                        }
                        output[x * channels + c] =
                            static_cast<uchar>((sum + 2u) >> 2);
                    }
                }
            });

        std::vector<uchar> zero_row;
        if (border_type == BORDER_CONSTANT)
        {
            zero_row.resize(static_cast<std::size_t>(row_stride), 0u);
        }
        parallel_for_index_if(
            should_parallelize_filter_rows(rows, cols, channels, 3),
            rows,
            [&](int y) {
                const int sy0 =
                    border_interpolate(y - 1, rows, border_type);
                const int sy1 = y;
                const int sy2 =
                    border_interpolate(y + 1, rows, border_type);
                const uchar* row0 =
                    sy0 >= 0
                        ? temporary.data() +
                              static_cast<std::size_t>(sy0) * row_stride
                        : zero_row.data();
                const uchar* row1 =
                    temporary.data() +
                    static_cast<std::size_t>(sy1) * row_stride;
                const uchar* row2 =
                    sy2 >= 0
                        ? temporary.data() +
                              static_cast<std::size_t>(sy2) * row_stride
                        : zero_row.data();
                uchar* output =
                    dst.data + static_cast<std::size_t>(y) * dst_step;
                int offset = 0;
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
                if (use_ui)
                {
                    const cv::v_uint16x8 one = cv::v_setall_u16(1);
                    const auto rounded8 = [&](int lane) {
                        const cv::v_uint16x8 sum = cv::v_add(
                            cv::v_add(
                                cv::v_load_expand(row0 + offset + lane),
                                cv::v_load_expand(row2 + offset + lane)),
                            cv::v_shl<1>(
                                cv::v_load_expand(row1 + offset + lane)));
                        return cv::v_shr<2>(cv::v_add(
                            cv::v_add(sum, one),
                            cv::v_and(cv::v_shr<2>(sum), one)));
                    };
                    for (; offset + 16 <= row_stride; offset += 16)
                    {
                        cv::v_store(
                            output + offset,
                            cv::v_pack(rounded8(0), rounded8(8)));
                    }
                }
#endif
                for (; offset < row_stride; ++offset)
                {
                    const unsigned sum =
                        row0[offset] + 2u * row1[offset] + row2[offset];
                    output[offset] =
                        static_cast<uchar>(round_shift2_even(sum));
                }
            });
    }
    else
    {
        std::vector<float> temporary(
            static_cast<std::size_t>(rows) *
            static_cast<std::size_t>(row_stride));
        parallel_for_index_if(
            should_parallelize_filter_rows(rows, cols, channels, 3),
            rows,
            [&](int y) {
                const float* input = reinterpret_cast<const float*>(
                    src.data + static_cast<std::size_t>(y) * src_step);
                float* output =
                    temporary.data() +
                    static_cast<std::size_t>(y) * row_stride;
                for (int c = 0; c < channels; ++c)
                {
                    float sum = 0.0f;
                    for (int k = 0; k < 3; ++k)
                    {
                        const int sx =
                            border_interpolate(k - 1, cols, border_type);
                        if (sx >= 0)
                        {
                            sum += static_cast<float>(k == 1 ? 2 : 1) *
                                   input[sx * channels + c];
                        }
                    }
                    output[c] = sum;
                }
                int offset = channels;
                const int interior_end =
                    std::max(channels, row_stride - channels);
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
                if (use_ui)
                {
                    for (; offset + 8 <= interior_end; offset += 8)
                    {
                        const auto sum4 = [&](int lane) {
                            const cv::v_float32x4 center =
                                cv::v_load(input + offset + lane);
                            return cv::v_add(
                                cv::v_add(
                                    cv::v_load(
                                        input + offset + lane - channels),
                                    cv::v_load(
                                        input + offset + lane + channels)),
                                cv::v_add(center, center));
                        };
                        cv::v_store(output + offset, sum4(0));
                        cv::v_store(output + offset + 4, sum4(4));
                    }
                    for (; offset + 4 <= interior_end; offset += 4)
                    {
                        const cv::v_float32x4 center =
                            cv::v_load(input + offset);
                        cv::v_store(
                            output + offset,
                            cv::v_add(
                                cv::v_add(
                                    cv::v_load(input + offset - channels),
                                    cv::v_load(input + offset + channels)),
                                cv::v_add(center, center)));
                    }
                }
#endif
                for (; offset < interior_end; ++offset)
                {
                    output[offset] = input[offset - channels] +
                                     2.0f * input[offset] +
                                     input[offset + channels];
                }
                if (cols > 1)
                {
                    const int x = cols - 1;
                    for (int c = 0; c < channels; ++c)
                    {
                        float sum = 0.0f;
                        for (int k = 0; k < 3; ++k)
                        {
                            const int sx = border_interpolate(
                                x + k - 1, cols, border_type);
                            if (sx >= 0)
                            {
                                sum += static_cast<float>(k == 1 ? 2 : 1) *
                                       input[sx * channels + c];
                            }
                        }
                        output[x * channels + c] = sum;
                    }
                }
            });

        std::vector<float> zero_row;
        if (border_type == BORDER_CONSTANT)
        {
            zero_row.resize(static_cast<std::size_t>(row_stride), 0.0f);
        }
        parallel_for_index_if(
            should_parallelize_filter_rows(rows, cols, channels, 3),
            rows,
            [&](int y) {
                const int sy0 =
                    border_interpolate(y - 1, rows, border_type);
                const int sy2 =
                    border_interpolate(y + 1, rows, border_type);
                const float* row0 =
                    sy0 >= 0
                        ? temporary.data() +
                              static_cast<std::size_t>(sy0) * row_stride
                        : zero_row.data();
                const float* row1 =
                    temporary.data() +
                    static_cast<std::size_t>(y) * row_stride;
                const float* row2 =
                    sy2 >= 0
                        ? temporary.data() +
                              static_cast<std::size_t>(sy2) * row_stride
                        : zero_row.data();
                float* output = reinterpret_cast<float*>(
                    dst.data + static_cast<std::size_t>(y) * dst_step);
                int offset = 0;
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
                if (use_ui)
                {
                    const cv::v_float32x4 scale =
                        cv::v_setall_f32(1.0f / 16.0f);
                    for (; offset + 8 <= row_stride; offset += 8)
                    {
                        const auto result4 = [&](int lane) {
                            const cv::v_float32x4 center =
                                cv::v_load(row1 + offset + lane);
                            return cv::v_mul(
                                cv::v_add(
                                    cv::v_add(
                                        cv::v_load(row0 + offset + lane),
                                        cv::v_load(row2 + offset + lane)),
                                    cv::v_add(center, center)),
                                scale);
                        };
                        cv::v_store(output + offset, result4(0));
                        cv::v_store(output + offset + 4, result4(4));
                    }
                    for (; offset + 4 <= row_stride; offset += 4)
                    {
                        const cv::v_float32x4 center =
                            cv::v_load(row1 + offset);
                        cv::v_store(
                            output + offset,
                            cv::v_mul(
                                cv::v_add(
                                    cv::v_add(
                                        cv::v_load(row0 + offset),
                                        cv::v_load(row2 + offset)),
                                    cv::v_add(center, center)),
                                scale));
                    }
                }
#endif
                for (; offset < row_stride; ++offset)
                {
                    output[offset] =
                        (row0[offset] + 2.0f * row1[offset] +
                         row2[offset]) *
                        (1.0f / 16.0f);
                }
            });
    }

    cpu::set_last_dispatch_tag(
        use_ui && row_stride >= (src.depth() == CV_8U ? 16 : 4)
            ? cpu::DispatchTag::OpenCVUI
            : cpu::DispatchTag::Scalar);
    return true;
}

inline bool try_sep_filter2d_fastpath(const Mat& src,
                               Mat& dst,
                               int ddepth,
                               const Mat& kernelX,
                               const Mat& kernelY,
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

    std::vector<float> kx;
    std::vector<float> ky;
    if (kernelX.empty() || kernelY.empty())
    {
        return false;
    }
    if (kernelX.dims != 2 || kernelY.dims != 2 ||
        kernelX.channels() != 1 || kernelY.channels() != 1 ||
        kernelX.depth() != CV_32F || kernelY.depth() != CV_32F)
    {
        return false;
    }
    sepfilter2d_collect_kernel(kernelX, kx, "kernelX");
    sepfilter2d_collect_kernel(kernelY, ky, "kernelY");

    const int kx_len = static_cast<int>(kx.size());
    const int ky_len = static_cast<int>(ky.size());
    const int ax = anchor.x >= 0 ? anchor.x : (kx_len / 2);
    const int ay = anchor.y >= 0 ? anchor.y : (ky_len / 2);
    if (ax < 0 || ax >= kx_len || ay < 0 || ay >= ky_len)
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

    if (try_sep3_binomial(
            *src_ref,
            dst,
            out_depth,
            kx,
            ky,
            ax,
            ay,
            delta,
            border_type))
    {
        g_last_sepfilter2d_algorithm_path = "binomial3_typed";
        return true;
    }

    g_last_sepfilter2d_algorithm_path = "separable_filter2d";

    if (filter_ui::separable_c1(*src_ref,
                                dst,
                                out_depth,
                                kx,
                                ky,
                                ax,
                                ay,
                                delta,
                                border_type))
    {
        return true;
    }

    const int row_stride = cols * channels;
    std::vector<int> x_offsets(static_cast<std::size_t>(cols) * static_cast<std::size_t>(kx_len), -1);
    for (int x = 0; x < cols; ++x)
    {
        int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx_len);
        for (int i = 0; i < kx_len; ++i)
        {
            const int sx = border_interpolate(x + i - ax, cols, border_type);
            x_ofs[i] = sx >= 0 ? sx * channels : -1;
        }
    }

    std::vector<int> y_offsets(static_cast<std::size_t>(rows) * static_cast<std::size_t>(ky_len), -1);
    for (int y = 0; y < rows; ++y)
    {
        int* y_ofs = y_offsets.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ky_len);
        for (int i = 0; i < ky_len; ++i)
        {
            const int sy = border_interpolate(y + i - ay, rows, border_type);
            y_ofs[i] = sy >= 0 ? sy * row_stride : -1;
        }
    }

    std::vector<float> tmp(static_cast<std::size_t>(rows) * static_cast<std::size_t>(row_stride), 0.0f);
    const std::size_t src_step = src_ref->step(0);
    const bool do_parallel_h = should_parallelize_filter_rows(rows, cols, channels, kx_len);

    if (src_depth == CV_8U)
    {
        parallel_for_index_if(do_parallel_h, rows, [&](int y) {
            const uchar* src_row = src_ref->data + static_cast<std::size_t>(y) * src_step;
            float* tmp_row = tmp.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_stride);

            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx_len);
                const int dx = x * channels;

                if (channels == 1)
                {
                    float acc0 = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc0 += kx[static_cast<std::size_t>(i)] * static_cast<float>(src_row[sx]);
                    }
                    tmp_row[dx] = acc0;
                    continue;
                }

                if (channels == 3)
                {
                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kx[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                    }
                    tmp_row[dx + 0] = acc0;
                    tmp_row[dx + 1] = acc1;
                    tmp_row[dx + 2] = acc2;
                    continue;
                }

                if (channels == 4)
                {
                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    float acc3 = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kx[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                        acc3 += w * static_cast<float>(px[3]);
                    }
                    tmp_row[dx + 0] = acc0;
                    tmp_row[dx + 1] = acc1;
                    tmp_row[dx + 2] = acc2;
                    tmp_row[dx + 3] = acc3;
                    continue;
                }

                for (int c = 0; c < channels; ++c)
                {
                    float acc = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc += kx[static_cast<std::size_t>(i)] * static_cast<float>(src_row[sx + c]);
                    }
                    tmp_row[dx + c] = acc;
                }
            }
        });
    }
    else
    {
        parallel_for_index_if(do_parallel_h, rows, [&](int y) {
            const float* src_row = reinterpret_cast<const float*>(src_ref->data + static_cast<std::size_t>(y) * src_step);
            float* tmp_row = tmp.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_stride);

            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx_len);
                const int dx = x * channels;

                if (channels == 1)
                {
                    float acc0 = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc0 += kx[static_cast<std::size_t>(i)] * src_row[sx];
                    }
                    tmp_row[dx] = acc0;
                    continue;
                }

                if (channels == 3)
                {
                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kx[static_cast<std::size_t>(i)];
                        const float* px = src_row + sx;
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                    tmp_row[dx + 0] = acc0;
                    tmp_row[dx + 1] = acc1;
                    tmp_row[dx + 2] = acc2;
                    continue;
                }

                if (channels == 4)
                {
                    float acc0 = 0.0f;
                    float acc1 = 0.0f;
                    float acc2 = 0.0f;
                    float acc3 = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kx[static_cast<std::size_t>(i)];
                        const float* px = src_row + sx;
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                    tmp_row[dx + 0] = acc0;
                    tmp_row[dx + 1] = acc1;
                    tmp_row[dx + 2] = acc2;
                    tmp_row[dx + 3] = acc3;
                    continue;
                }

                for (int c = 0; c < channels; ++c)
                {
                    float acc = 0.0f;
                    for (int i = 0; i < kx_len; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc += kx[static_cast<std::size_t>(i)] * src_row[sx + c];
                    }
                    tmp_row[dx + c] = acc;
                }
            }
        });
    }

    dst.create(std::vector<int>{rows, cols}, CV_MAKETYPE(out_depth, channels));
    const std::size_t dst_step = dst.step(0);
    const bool do_parallel_v = should_parallelize_filter_rows(rows, cols, channels, ky_len);

    parallel_for_index_if(do_parallel_v, rows, [&](int y) {
        const int* y_ofs = y_offsets.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ky_len);
        uchar* dst_row_u8 = out_depth == CV_8U ? (dst.data + static_cast<std::size_t>(y) * dst_step) : nullptr;
        float* dst_row_f32 =
            out_depth == CV_32F ? reinterpret_cast<float*>(dst.data + static_cast<std::size_t>(y) * dst_step) : nullptr;

        for (int x = 0; x < cols; ++x)
        {
            const int dx = x * channels;
            if (channels == 1)
            {
                float acc0 = static_cast<float>(delta);
                for (int i = 0; i < ky_len; ++i)
                {
                    const int sy = y_ofs[i];
                    if (sy < 0)
                    {
                        continue;
                    }
                    acc0 += ky[static_cast<std::size_t>(i)] * tmp[static_cast<std::size_t>(sy + dx)];
                }
                if (dst_row_f32)
                {
                    dst_row_f32[dx] = acc0;
                }
                else
                {
                    dst_row_u8[dx] = saturate_cast<uchar>(acc0);
                }
                continue;
            }

            if (channels == 3)
            {
                float acc0 = static_cast<float>(delta);
                float acc1 = static_cast<float>(delta);
                float acc2 = static_cast<float>(delta);
                for (int i = 0; i < ky_len; ++i)
                {
                    const int sy = y_ofs[i];
                    if (sy < 0)
                    {
                        continue;
                    }
                    const float w = ky[static_cast<std::size_t>(i)];
                    const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                    acc0 += w * px[0];
                    acc1 += w * px[1];
                    acc2 += w * px[2];
                }
                if (dst_row_f32)
                {
                    dst_row_f32[dx + 0] = acc0;
                    dst_row_f32[dx + 1] = acc1;
                    dst_row_f32[dx + 2] = acc2;
                }
                else
                {
                    dst_row_u8[dx + 0] = saturate_cast<uchar>(acc0);
                    dst_row_u8[dx + 1] = saturate_cast<uchar>(acc1);
                    dst_row_u8[dx + 2] = saturate_cast<uchar>(acc2);
                }
                continue;
            }

            if (channels == 4)
            {
                float acc0 = static_cast<float>(delta);
                float acc1 = static_cast<float>(delta);
                float acc2 = static_cast<float>(delta);
                float acc3 = static_cast<float>(delta);
                for (int i = 0; i < ky_len; ++i)
                {
                    const int sy = y_ofs[i];
                    if (sy < 0)
                    {
                        continue;
                    }
                    const float w = ky[static_cast<std::size_t>(i)];
                    const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                    acc0 += w * px[0];
                    acc1 += w * px[1];
                    acc2 += w * px[2];
                    acc3 += w * px[3];
                }
                if (dst_row_f32)
                {
                    dst_row_f32[dx + 0] = acc0;
                    dst_row_f32[dx + 1] = acc1;
                    dst_row_f32[dx + 2] = acc2;
                    dst_row_f32[dx + 3] = acc3;
                }
                else
                {
                    dst_row_u8[dx + 0] = saturate_cast<uchar>(acc0);
                    dst_row_u8[dx + 1] = saturate_cast<uchar>(acc1);
                    dst_row_u8[dx + 2] = saturate_cast<uchar>(acc2);
                    dst_row_u8[dx + 3] = saturate_cast<uchar>(acc3);
                }
                continue;
            }

            for (int c = 0; c < channels; ++c)
            {
                float acc = static_cast<float>(delta);
                for (int i = 0; i < ky_len; ++i)
                {
                    const int sy = y_ofs[i];
                    if (sy < 0)
                    {
                        continue;
                    }
                    acc += ky[static_cast<std::size_t>(i)] * tmp[static_cast<std::size_t>(sy + dx + c)];
                }
                if (dst_row_f32)
                {
                    dst_row_f32[dx + c] = acc;
                }
                else
                {
                    dst_row_u8[dx + c] = saturate_cast<uchar>(acc);
                }
            }
        }
    });

    return true;
}

inline bool is_morph_rect3x3_kernel(const Mat& kernel, Point anchor)
{
    if (kernel.empty())
    {
        return true;
    }

    if (kernel.dims != 2 || kernel.depth() != CV_8U || kernel.channels() != 1)
    {
        return false;
    }

    if (kernel.size[1] != 3 || kernel.size[0] != 3)
    {
        return false;
    }

    const int anchor_x = anchor.x >= 0 ? anchor.x : 1;
    const int anchor_y = anchor.y >= 0 ? anchor.y : 1;
    if (anchor_x != 1 || anchor_y != 1)
    {
        return false;
    }

    const std::size_t kstep = kernel.step(0);
    for (int ky = 0; ky < 3; ++ky)
    {
        const uchar* row = kernel.data + static_cast<std::size_t>(ky) * kstep;
        for (int kx = 0; kx < 3; ++kx)
        {
            if (row[kx] == 0)
            {
                return false;
            }
        }
    }
    return true;
}


} // namespace sep_filter2d_fastpath

inline const char* last_sepfilter2d_algorithm_path()
{
    return sep_filter2d_fastpath::g_last_sepfilter2d_algorithm_path;
}

inline void sepFilter2D_fast_impl(const Mat& src,
                               Mat& dst,
                               int ddepth,
                               const Mat& kernelX,
                               const Mat& kernelY,
                               Point anchor,
                               double delta,
                               int borderType)
{
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    sep_filter2d_fastpath::g_last_sepfilter2d_algorithm_path = "fallback";
    if (sep_filter2d_fastpath::try_sep_filter2d_fastpath(
            src, dst, ddepth, kernelX, kernelY, anchor, delta, borderType))
    {
        return;
    }

    sepFilter2D_fallback(src, dst, ddepth, kernelX, kernelY, anchor, delta, borderType);
}

} // namespace detail
} // namespace cvh

#endif // CVH_IMGPROC_DETAIL_SEP_FILTER2D_IMPL_HPP
