#ifndef CVH_IMGPROC_DETAIL_MORPHOLOGY_IMPL_HPP
#define CVH_IMGPROC_DETAIL_MORPHOLOGY_IMPL_HPP

#include "fastpath_common.hpp"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/simd/opencv_ui.h"

namespace cvh
{
namespace detail
{

namespace morphology_fastpath
{
inline bool try_morph_rect3x3_fastpath(const Mat& src,
                                Mat& dst,
                                const Mat& kernel,
                                Point anchor,
                                int iterations,
                                int borderType,
                                const Scalar& borderValue,
                                bool is_erode)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        return false;
    }

    if (iterations != 1)
    {
        return false;
    }

    if (!is_morph_rect3x3_kernel(kernel, anchor))
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

    const int row_stride = cols * channels;
    const std::size_t src_step = src_ref->step(0);

    std::vector<int> x_offsets(static_cast<std::size_t>(cols) * 3u, -1);
    for (int x = 0; x < cols; ++x)
    {
        int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * 3u;
        for (int kx = 0; kx < 3; ++kx)
        {
            const int sx = border_interpolate(x + kx - 1, cols, border_type);
            x_ofs[kx] = sx >= 0 ? sx * channels : -1;
        }
    }

    std::vector<int> y_indices(static_cast<std::size_t>(rows) * 3u, -1);
    for (int y = 0; y < rows; ++y)
    {
        int* y_idx = y_indices.data() + static_cast<std::size_t>(y) * 3u;
        for (int ky = 0; ky < 3; ++ky)
        {
            y_idx[ky] = border_interpolate(y + ky - 1, rows, border_type);
        }
    }

    std::vector<uchar> border_vals(static_cast<std::size_t>(channels), 0);
    for (int c = 0; c < channels; ++c)
    {
        const int sc = c < 4 ? c : 0;
        border_vals[static_cast<std::size_t>(c)] = saturate_cast<uchar>(borderValue.val[sc]);
    }

    std::vector<uchar> tmp(static_cast<std::size_t>(rows) * static_cast<std::size_t>(row_stride), 0);
    const bool do_parallel_h = should_parallelize_filter_rows(rows, cols, channels, 3);

#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const bool use_ui =
        cpu::opencv_ui_allowed() &&
        row_stride >= 16 + 2 * channels;
    if (use_ui)
    {
        std::vector<uchar> vertical_border;
        if (border_type == BORDER_CONSTANT)
        {
            vertical_border.resize(static_cast<std::size_t>(row_stride));
            for (int offset = 0; offset < row_stride; ++offset)
            {
                vertical_border[static_cast<std::size_t>(offset)] =
                    border_vals[static_cast<std::size_t>(offset % channels)];
            }
        }

        parallel_for_index_if(do_parallel_h, rows, [&](int y) {
            const uchar* src_row =
                src_ref->data +
                static_cast<std::size_t>(y) * src_step;
            uchar* tmp_row =
                tmp.data() +
                static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(row_stride);
            const int* first_offsets = x_offsets.data();
            for (int c = 0; c < channels; ++c)
            {
                const uchar border =
                    border_vals[static_cast<std::size_t>(c)];
                const uchar first =
                    first_offsets[0] >= 0
                        ? src_row[first_offsets[0] + c]
                        : border;
                const uchar second =
                    first_offsets[1] >= 0
                        ? src_row[first_offsets[1] + c]
                        : border;
                const uchar third =
                    first_offsets[2] >= 0
                        ? src_row[first_offsets[2] + c]
                        : border;
                tmp_row[c] =
                    is_erode
                        ? std::min(first, std::min(second, third))
                        : std::max(first, std::max(second, third));
            }

            int offset = channels;
            const int interior_end = row_stride - channels;
            for (; offset + 16 <= interior_end; offset += 16)
            {
                const cv::v_uint8x16 first =
                    cv::v_load(src_row + offset - channels);
                const cv::v_uint8x16 second =
                    cv::v_load(src_row + offset);
                const cv::v_uint8x16 third =
                    cv::v_load(src_row + offset + channels);
                cv::v_store(
                    tmp_row + offset,
                    is_erode
                        ? cv::v_min(first, cv::v_min(second, third))
                        : cv::v_max(first, cv::v_max(second, third)));
            }
            for (; offset < interior_end; ++offset)
            {
                tmp_row[offset] =
                    is_erode
                        ? std::min(
                              src_row[offset - channels],
                              std::min(
                                  src_row[offset],
                                  src_row[offset + channels]))
                        : std::max(
                              src_row[offset - channels],
                              std::max(
                                  src_row[offset],
                                  src_row[offset + channels]));
            }

            const int last = cols - 1;
            const int* last_offsets =
                x_offsets.data() +
                static_cast<std::size_t>(last) * 3u;
            const int last_base = last * channels;
            for (int c = 0; c < channels; ++c)
            {
                const uchar border =
                    border_vals[static_cast<std::size_t>(c)];
                const uchar first =
                    last_offsets[0] >= 0
                        ? src_row[last_offsets[0] + c]
                        : border;
                const uchar second =
                    last_offsets[1] >= 0
                        ? src_row[last_offsets[1] + c]
                        : border;
                const uchar third =
                    last_offsets[2] >= 0
                        ? src_row[last_offsets[2] + c]
                        : border;
                tmp_row[last_base + c] =
                    is_erode
                        ? std::min(first, std::min(second, third))
                        : std::max(first, std::max(second, third));
            }
        });

        dst.create(
            std::vector<int>{rows, cols}, src_ref->type());
        const std::size_t dst_step = dst.step(0);
        const bool do_parallel_v =
            should_parallelize_filter_rows(rows, cols, channels, 3);
        parallel_for_index_if(do_parallel_v, rows, [&](int y) {
            uchar* dst_row =
                dst.data +
                static_cast<std::size_t>(y) * dst_step;
            const int* y_idx =
                y_indices.data() +
                static_cast<std::size_t>(y) * 3u;
            const uchar* row0 =
                y_idx[0] >= 0
                    ? tmp.data() +
                          static_cast<std::size_t>(y_idx[0]) *
                              static_cast<std::size_t>(row_stride)
                    : vertical_border.data();
            const uchar* row1 =
                y_idx[1] >= 0
                    ? tmp.data() +
                          static_cast<std::size_t>(y_idx[1]) *
                              static_cast<std::size_t>(row_stride)
                    : vertical_border.data();
            const uchar* row2 =
                y_idx[2] >= 0
                    ? tmp.data() +
                          static_cast<std::size_t>(y_idx[2]) *
                              static_cast<std::size_t>(row_stride)
                    : vertical_border.data();
            int offset = 0;
            for (; offset + 16 <= row_stride; offset += 16)
            {
                const cv::v_uint8x16 first =
                    cv::v_load(row0 + offset);
                const cv::v_uint8x16 second =
                    cv::v_load(row1 + offset);
                const cv::v_uint8x16 third =
                    cv::v_load(row2 + offset);
                cv::v_store(
                    dst_row + offset,
                    is_erode
                        ? cv::v_min(first, cv::v_min(second, third))
                        : cv::v_max(first, cv::v_max(second, third)));
            }
            for (; offset < row_stride; ++offset)
            {
                dst_row[offset] =
                    is_erode
                        ? std::min(
                              row0[offset],
                              std::min(row1[offset], row2[offset]))
                        : std::max(
                              row0[offset],
                              std::max(row1[offset], row2[offset]));
            }
        });
        cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
        return true;
    }
#endif

    parallel_for_index_if(do_parallel_h, rows, [&](int y) {
        const uchar* src_row = src_ref->data + static_cast<std::size_t>(y) * src_step;
        uchar* tmp_row = tmp.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_stride);

        for (int x = 0; x < cols; ++x)
        {
            const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * 3u;
            const int dx = x * channels;

            for (int c = 0; c < channels; ++c)
            {
                const uchar b = border_vals[static_cast<std::size_t>(c)];
                const int sx0 = x_ofs[0];
                const int sx1 = x_ofs[1];
                const int sx2 = x_ofs[2];
                const uchar v0 = sx0 >= 0 ? src_row[sx0 + c] : b;
                const uchar v1 = sx1 >= 0 ? src_row[sx1 + c] : b;
                const uchar v2 = sx2 >= 0 ? src_row[sx2 + c] : b;

                if (is_erode)
                {
                    tmp_row[dx + c] = std::min(v0, std::min(v1, v2));
                }
                else
                {
                    tmp_row[dx + c] = std::max(v0, std::max(v1, v2));
                }
            }
        }
    });

    dst.create(std::vector<int>{rows, cols}, src_ref->type());
    const std::size_t dst_step = dst.step(0);
    const bool do_parallel_v = should_parallelize_filter_rows(rows, cols, channels, 3);
    parallel_for_index_if(do_parallel_v, rows, [&](int y) {
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst_step;
        const int* y_idx = y_indices.data() + static_cast<std::size_t>(y) * 3u;

        const int sy0 = y_idx[0];
        const int sy1 = y_idx[1];
        const int sy2 = y_idx[2];
        const uchar* row0 = sy0 >= 0 ? (tmp.data() + static_cast<std::size_t>(sy0) * static_cast<std::size_t>(row_stride)) : nullptr;
        const uchar* row1 = sy1 >= 0 ? (tmp.data() + static_cast<std::size_t>(sy1) * static_cast<std::size_t>(row_stride)) : nullptr;
        const uchar* row2 = sy2 >= 0 ? (tmp.data() + static_cast<std::size_t>(sy2) * static_cast<std::size_t>(row_stride)) : nullptr;

        for (int x = 0; x < cols; ++x)
        {
            const int dx = x * channels;
            for (int c = 0; c < channels; ++c)
            {
                const uchar b = border_vals[static_cast<std::size_t>(c)];
                const uchar v0 = row0 ? row0[dx + c] : b;
                const uchar v1 = row1 ? row1[dx + c] : b;
                const uchar v2 = row2 ? row2[dx + c] : b;

                if (is_erode)
                {
                    dst_row[dx + c] = std::min(v0, std::min(v1, v2));
                }
                else
                {
                    dst_row[dx + c] = std::max(v0, std::max(v1, v2));
                }
            }
        }
    });

    return true;
}


} // namespace morphology_fastpath

inline void erode_fast_impl(const Mat& src,
                        Mat& dst,
                        const Mat& kernel,
                        Point anchor,
                        int iterations,
                        int borderType,
                        const Scalar& borderValue)
{
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    if (morphology_fastpath::try_morph_rect3x3_fastpath(
            src, dst, kernel, anchor, iterations, borderType, borderValue, true))
    {
        return;
    }

    erode_fallback(src, dst, kernel, anchor, iterations, borderType, borderValue);
}

inline void dilate_fast_impl(const Mat& src,
                             Mat& dst,
                             const Mat& kernel,
                             Point anchor,
                             int iterations,
                             int borderType,
                             const Scalar& borderValue)
{
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    if (morphology_fastpath::try_morph_rect3x3_fastpath(
            src, dst, kernel, anchor, iterations, borderType, borderValue, false))
    {
        return;
    }

    dilate_fallback(src, dst, kernel, anchor, iterations, borderType, borderValue);
}

} // namespace detail
} // namespace cvh

#endif // CVH_IMGPROC_DETAIL_MORPHOLOGY_IMPL_HPP
