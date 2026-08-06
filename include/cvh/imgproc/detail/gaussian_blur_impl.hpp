#ifndef CVH_IMGPROC_DETAIL_GAUSSIAN_BLUR_IMPL_HPP
#define CVH_IMGPROC_DETAIL_GAUSSIAN_BLUR_IMPL_HPP

#include "fastpath_common.hpp"
#include "filter_ui.hpp"

#include <array>
#include <cstdint>
#include <limits>

namespace cvh
{
namespace detail
{

namespace gaussian_blur_fastpath
{
inline thread_local const char* g_last_gaussianblur_dispatch_path = "fallback";

inline void set_last_gaussianblur_dispatch_path(const char* path)
{
    g_last_gaussianblur_dispatch_path = path ? path : "fallback";
}

inline bool try_gaussian5x5_fixed_u8(const Mat& src,
                                     Mat& dst,
                                     int border_type)
{
    if (src.depth() != CV_8U || src.dims != 2 ||
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

    std::vector<std::uint16_t> row_ring(
        static_cast<std::size_t>(row_stride) * 5u);
    std::vector<std::uint16_t> zero_row;
    if (border_type == BORDER_CONSTANT)
    {
        zero_row.resize(static_cast<std::size_t>(row_stride), 0u);
    }
    std::array<int, 5> ring_keys;
    ring_keys.fill(std::numeric_limits<int>::min());
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const bool use_ui = cpu::opencv_ui_allowed();
#else
    const bool use_ui = false;
#endif

    const auto horizontal = [&](int sy, std::uint16_t* output) {
        const uchar* input =
            src.data + static_cast<std::size_t>(sy) * src_step;
        const int interior_begin = std::min(2, cols);
        const int interior_end = std::max(interior_begin, cols - 2);

        for (int x = 0; x < interior_begin; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                unsigned sum = 0;
                constexpr unsigned weights[5] = {1, 4, 6, 4, 1};
                for (int k = 0; k < 5; ++k)
                {
                    const int sx = border_interpolate(
                        x + k - 2, cols, border_type);
                    if (sx >= 0)
                    {
                        sum += weights[k] * input[sx * channels + c];
                    }
                }
                output[x * channels + c] =
                    static_cast<std::uint16_t>(sum);
            }
        }

        int offset = interior_begin * channels;
        const int byte_end = interior_end * channels;
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
        if (use_ui)
        {
            const auto accumulate = [&](int lane_offset) {
                const int source_offset = offset + lane_offset;
                const cv::v_uint16x8 first = cv::v_load_expand(
                    input + source_offset - 2 * channels);
                const cv::v_uint16x8 second = cv::v_load_expand(
                    input + source_offset - channels);
                const cv::v_uint16x8 center =
                    cv::v_load_expand(input + source_offset);
                const cv::v_uint16x8 fourth = cv::v_load_expand(
                    input + source_offset + channels);
                const cv::v_uint16x8 fifth = cv::v_load_expand(
                    input + source_offset + 2 * channels);
                return cv::v_add(
                    cv::v_add(first, fifth),
                    cv::v_add(
                        cv::v_shl<2>(cv::v_add(second, fourth)),
                        cv::v_add(
                            cv::v_shl<2>(center),
                            cv::v_shl<1>(center))));
            };
            for (; offset + 16 <= byte_end; offset += 16)
            {
                cv::v_store(output + offset, accumulate(0));
                cv::v_store(output + offset + 8, accumulate(8));
            }
            for (; offset + 8 <= byte_end; offset += 8)
            {
                cv::v_store(output + offset, accumulate(0));
            }
        }
#endif
        for (; offset < byte_end; ++offset)
        {
            output[offset] = static_cast<std::uint16_t>(
                input[offset - 2 * channels] +
                4u * input[offset - channels] +
                6u * input[offset] +
                4u * input[offset + channels] +
                input[offset + 2 * channels]);
        }

        for (int x = interior_end; x < cols; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                unsigned sum = 0;
                constexpr unsigned weights[5] = {1, 4, 6, 4, 1};
                for (int k = 0; k < 5; ++k)
                {
                    const int sx = border_interpolate(
                        x + k - 2, cols, border_type);
                    if (sx >= 0)
                    {
                        sum += weights[k] * input[sx * channels + c];
                    }
                }
                output[x * channels + c] =
                    static_cast<std::uint16_t>(sum);
            }
        }
    };

    for (int y = 0; y < rows; ++y)
    {
        std::array<int, 5> requested;
        for (int k = 0; k < 5; ++k)
        {
            requested[static_cast<std::size_t>(k)] =
                border_interpolate(y + k - 2, rows, border_type);
        }

        for (const int sy : requested)
        {
            if (sy < 0 ||
                std::find(ring_keys.begin(), ring_keys.end(), sy) !=
                    ring_keys.end())
            {
                continue;
            }

            int slot = -1;
            for (int candidate = 0; candidate < 5; ++candidate)
            {
                if (std::find(
                        requested.begin(),
                        requested.end(),
                        ring_keys[static_cast<std::size_t>(candidate)]) ==
                    requested.end())
                {
                    slot = candidate;
                    break;
                }
            }
            CV_Assert(slot >= 0);
            horizontal(
                sy,
                row_ring.data() +
                    static_cast<std::size_t>(slot) *
                        static_cast<std::size_t>(row_stride));
            ring_keys[static_cast<std::size_t>(slot)] = sy;
        }

        std::array<const std::uint16_t*, 5> source_rows;
        for (int k = 0; k < 5; ++k)
        {
            const int sy = requested[static_cast<std::size_t>(k)];
            if (sy < 0)
            {
                source_rows[static_cast<std::size_t>(k)] =
                    zero_row.data();
                continue;
            }
            const auto it =
                std::find(ring_keys.begin(), ring_keys.end(), sy);
            CV_Assert(it != ring_keys.end());
            const int slot = static_cast<int>(it - ring_keys.begin());
            source_rows[static_cast<std::size_t>(k)] =
                row_ring.data() +
                static_cast<std::size_t>(slot) *
                    static_cast<std::size_t>(row_stride);
        }

        uchar* output =
            dst.data + static_cast<std::size_t>(y) * dst_step;
        int offset = 0;
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
        if (use_ui)
        {
            const cv::v_uint16x8 rounding = cv::v_setall_u16(128);
            for (; offset + 16 <= row_stride; offset += 16)
            {
                const auto accumulate = [&](int lane_offset) {
                    const cv::v_uint16x8 first = cv::v_load(
                        source_rows[0] + offset + lane_offset);
                    const cv::v_uint16x8 second = cv::v_load(
                        source_rows[1] + offset + lane_offset);
                    const cv::v_uint16x8 center = cv::v_load(
                        source_rows[2] + offset + lane_offset);
                    const cv::v_uint16x8 fourth = cv::v_load(
                        source_rows[3] + offset + lane_offset);
                    const cv::v_uint16x8 fifth = cv::v_load(
                        source_rows[4] + offset + lane_offset);
                    return cv::v_shr<8>(cv::v_add(
                        cv::v_add(
                            cv::v_add(first, fifth),
                            cv::v_add(
                                cv::v_shl<2>(
                                    cv::v_add(second, fourth)),
                                cv::v_add(
                                    cv::v_shl<2>(center),
                                    cv::v_shl<1>(center)))),
                        rounding));
                };
                cv::v_store(
                    output + offset,
                    cv::v_pack(accumulate(0), accumulate(8)));
            }
        }
#endif
        for (; offset < row_stride; ++offset)
        {
            const unsigned sum =
                source_rows[0][offset] +
                4u * source_rows[1][offset] +
                6u * source_rows[2][offset] +
                4u * source_rows[3][offset] +
                source_rows[4][offset];
            output[offset] = static_cast<uchar>((sum + 128u) >> 8);
        }
    }

    cpu::set_last_dispatch_tag(
        use_ui && row_stride >= 16
            ? cpu::DispatchTag::OpenCVUI
            : cpu::DispatchTag::Scalar);
    return true;
}

inline bool try_gaussian5x5_fixed_f32(const Mat& src,
                                      Mat& dst,
                                      int border_type)
{
    if (src.depth() != CV_32F || src.dims != 2 ||
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
    std::vector<float> row_ring(
        static_cast<std::size_t>(row_stride) * 5u);
    std::vector<float> zero_row;
    if (border_type == BORDER_CONSTANT)
    {
        zero_row.resize(static_cast<std::size_t>(row_stride), 0.0f);
    }
    std::array<int, 5> ring_keys;
    ring_keys.fill(std::numeric_limits<int>::min());
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const bool use_ui = cpu::opencv_ui_allowed();
#else
    const bool use_ui = false;
#endif

    const auto horizontal = [&](int sy, float* output) {
        const float* input = reinterpret_cast<const float*>(
            src.data + static_cast<std::size_t>(sy) * src_step);
        const int interior_begin = std::min(2, cols);
        const int interior_end = std::max(interior_begin, cols - 2);
        constexpr float weights[5] = {1.0f, 4.0f, 6.0f, 4.0f, 1.0f};

        for (int x = 0; x < interior_begin; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                float sum = 0.0f;
                for (int k = 0; k < 5; ++k)
                {
                    const int sx = border_interpolate(
                        x + k - 2, cols, border_type);
                    if (sx >= 0)
                    {
                        sum += weights[k] * input[sx * channels + c];
                    }
                }
                output[x * channels + c] = sum * (1.0f / 16.0f);
            }
        }

        int offset = interior_begin * channels;
        const int element_end = interior_end * channels;
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
        if (use_ui)
        {
            const cv::v_float32x4 scale = cv::v_setall_f32(1.0f / 16.0f);
            const auto accumulate = [&](int lane) {
                const int source_offset = offset + lane;
                const cv::v_float32x4 center =
                    cv::v_load(input + source_offset);
                return cv::v_mul(
                    cv::v_add(
                        cv::v_add(
                            cv::v_load(
                                input + source_offset - 2 * channels),
                            cv::v_load(
                                input + source_offset + 2 * channels)),
                        cv::v_add(
                            cv::v_mul(
                                cv::v_add(
                                    cv::v_load(
                                        input + source_offset - channels),
                                    cv::v_load(
                                        input + source_offset + channels)),
                                cv::v_setall_f32(4.0f)),
                            cv::v_mul(center, cv::v_setall_f32(6.0f)))),
                    scale);
            };
            for (; offset + 8 <= element_end; offset += 8)
            {
                cv::v_store(output + offset, accumulate(0));
                cv::v_store(output + offset + 4, accumulate(4));
            }
            for (; offset + 4 <= element_end; offset += 4)
            {
                cv::v_store(output + offset, accumulate(0));
            }
        }
#endif
        for (; offset < element_end; ++offset)
        {
            output[offset] =
                (input[offset - 2 * channels] +
                 4.0f * input[offset - channels] +
                 6.0f * input[offset] +
                 4.0f * input[offset + channels] +
                 input[offset + 2 * channels]) *
                (1.0f / 16.0f);
        }

        for (int x = interior_end; x < cols; ++x)
        {
            for (int c = 0; c < channels; ++c)
            {
                float sum = 0.0f;
                for (int k = 0; k < 5; ++k)
                {
                    const int sx = border_interpolate(
                        x + k - 2, cols, border_type);
                    if (sx >= 0)
                    {
                        sum += weights[k] * input[sx * channels + c];
                    }
                }
                output[x * channels + c] = sum * (1.0f / 16.0f);
            }
        }
    };

    for (int y = 0; y < rows; ++y)
    {
        std::array<int, 5> requested;
        for (int k = 0; k < 5; ++k)
        {
            requested[static_cast<std::size_t>(k)] =
                border_interpolate(y + k - 2, rows, border_type);
        }
        for (const int sy : requested)
        {
            if (sy < 0 ||
                std::find(ring_keys.begin(), ring_keys.end(), sy) !=
                    ring_keys.end())
            {
                continue;
            }
            int slot = -1;
            for (int candidate = 0; candidate < 5; ++candidate)
            {
                if (std::find(
                        requested.begin(),
                        requested.end(),
                        ring_keys[static_cast<std::size_t>(candidate)]) ==
                    requested.end())
                {
                    slot = candidate;
                    break;
                }
            }
            CV_Assert(slot >= 0);
            horizontal(
                sy,
                row_ring.data() +
                    static_cast<std::size_t>(slot) * row_stride);
            ring_keys[static_cast<std::size_t>(slot)] = sy;
        }

        std::array<const float*, 5> source_rows;
        for (int k = 0; k < 5; ++k)
        {
            const int sy = requested[static_cast<std::size_t>(k)];
            if (sy < 0)
            {
                source_rows[static_cast<std::size_t>(k)] = zero_row.data();
                continue;
            }
            const auto it = std::find(ring_keys.begin(), ring_keys.end(), sy);
            CV_Assert(it != ring_keys.end());
            source_rows[static_cast<std::size_t>(k)] =
                row_ring.data() +
                static_cast<std::size_t>(it - ring_keys.begin()) * row_stride;
        }

        float* output = reinterpret_cast<float*>(
            dst.data + static_cast<std::size_t>(y) * dst_step);
        int offset = 0;
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
        if (use_ui)
        {
            const cv::v_float32x4 scale = cv::v_setall_f32(1.0f / 16.0f);
            const auto accumulate = [&](int lane) {
                const cv::v_float32x4 center =
                    cv::v_load(source_rows[2] + offset + lane);
                return cv::v_mul(
                    cv::v_add(
                        cv::v_add(
                            cv::v_load(source_rows[0] + offset + lane),
                            cv::v_load(source_rows[4] + offset + lane)),
                        cv::v_add(
                            cv::v_mul(
                                cv::v_add(
                                    cv::v_load(
                                        source_rows[1] + offset + lane),
                                    cv::v_load(
                                        source_rows[3] + offset + lane)),
                                cv::v_setall_f32(4.0f)),
                            cv::v_mul(center, cv::v_setall_f32(6.0f)))),
                    scale);
            };
            for (; offset + 8 <= row_stride; offset += 8)
            {
                cv::v_store(output + offset, accumulate(0));
                cv::v_store(output + offset + 4, accumulate(4));
            }
            for (; offset + 4 <= row_stride; offset += 4)
            {
                cv::v_store(output + offset, accumulate(0));
            }
        }
#endif
        for (; offset < row_stride; ++offset)
        {
            output[offset] =
                (source_rows[0][offset] +
                 4.0f * source_rows[1][offset] +
                 6.0f * source_rows[2][offset] +
                 4.0f * source_rows[3][offset] +
                 source_rows[4][offset]) *
                (1.0f / 16.0f);
        }
    }

    cpu::set_last_dispatch_tag(
        use_ui && row_stride >= 4
            ? cpu::DispatchTag::OpenCVUI
            : cpu::DispatchTag::Scalar);
    return true;
}

inline bool try_gaussian_blur_fastpath_u8(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY, int borderType)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        return false;
    }

    if (!is_u8_fastpath_channels(src.channels()))
    {
        return false;
    }

    int kx = ksize.width;
    int ky = ksize.height;

    if (kx <= 0 && sigmaX > 0.0)
    {
        kx = auto_gaussian_ksize(sigmaX);
    }
    if (ky <= 0 && sigmaY > 0.0)
    {
        ky = auto_gaussian_ksize(sigmaY);
    }
    if (kx <= 0 && ky > 0)
    {
        kx = ky;
    }
    if (ky <= 0 && kx > 0)
    {
        ky = kx;
    }

    if (kx <= 0 || ky <= 0 || (kx & 1) == 0 || (ky & 1) == 0)
    {
        return false;
    }

    const bool fixed5x5 =
        kx == 5 && ky == 5 && sigmaX <= 0.0 && sigmaY <= 0.0;

    if (sigmaX <= 0.0)
    {
        sigmaX = default_gaussian_sigma_for_ksize(kx);
    }
    if (sigmaY <= 0.0)
    {
        sigmaY = sigmaX;
    }
    if (sigmaX <= 0.0 || sigmaY <= 0.0)
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
    const std::size_t src_step = src_ref->step(0);
    const int row_stride = cols * channels;

    dst.create(std::vector<int>{rows, cols}, src_ref->type());
    const std::size_t dst_step = dst.step(0);

    if (fixed5x5 &&
        try_gaussian5x5_fixed_u8(*src_ref, dst, border_type))
    {
        return true;
    }

    const std::vector<float> kernel_x = build_gaussian_kernel_1d(kx, sigmaX);
    const std::vector<float> kernel_y = build_gaussian_kernel_1d(ky, sigmaY);
    const int rx = kx / 2;
    const int ry = ky / 2;

    if (filter_ui::separable_c1(*src_ref,
                                dst,
                                CV_8U,
                                kernel_x,
                                kernel_y,
                                rx,
                                ry,
                                0.0,
                                border_type))
    {
        return true;
    }

    const bool has_constant_border = border_type == BORDER_CONSTANT;

    std::vector<int> x_offsets(static_cast<std::size_t>(cols) * static_cast<std::size_t>(kx), -1);
    for (int x = 0; x < cols; ++x)
    {
        int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
        for (int i = 0; i < kx; ++i)
        {
            const int sx = border_interpolate(x + i - rx, cols, border_type);
            x_ofs[i] = sx >= 0 ? sx * channels : -1;
        }
    }

    std::vector<int> y_offsets(static_cast<std::size_t>(rows) * static_cast<std::size_t>(ky), -1);
    for (int y = 0; y < rows; ++y)
    {
        int* y_ofs = y_offsets.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ky);
        for (int i = 0; i < ky; ++i)
        {
            const int sy = border_interpolate(y + i - ry, rows, border_type);
            y_ofs[i] = sy >= 0 ? sy * row_stride : -1;
        }
    }

    std::vector<float> tmp(static_cast<std::size_t>(rows) * static_cast<std::size_t>(row_stride), 0.0f);

    const bool do_parallel_h = should_parallelize_filter_rows(rows, cols, channels, kx);
    parallel_for_index_if(do_parallel_h, rows, [&](int y) {
        const uchar* src_row = src_ref->data + static_cast<std::size_t>(y) * src_step;
        float* tmp_row = tmp.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_stride);

        if (channels == 1)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                float acc0 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc0 += kernel_x[static_cast<std::size_t>(i)] * static_cast<float>(src_row[sx]);
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        acc0 += kernel_x[static_cast<std::size_t>(i)] * static_cast<float>(src_row[sx]);
                    }
                }
                tmp_row[x] = acc0;
            }
            return;
        }

        if (channels == 3)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                const int dx = x * 3;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                    }
                }
                tmp_row[dx + 0] = acc0;
                tmp_row[dx + 1] = acc1;
                tmp_row[dx + 2] = acc2;
            }
            return;
        }

        if (channels == 4)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                const int dx = x * 4;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                        acc3 += w * static_cast<float>(px[3]);
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const uchar* px = src_row + sx;
                        acc0 += w * static_cast<float>(px[0]);
                        acc1 += w * static_cast<float>(px[1]);
                        acc2 += w * static_cast<float>(px[2]);
                        acc3 += w * static_cast<float>(px[3]);
                    }
                }
                tmp_row[dx + 0] = acc0;
                tmp_row[dx + 1] = acc1;
                tmp_row[dx + 2] = acc2;
                tmp_row[dx + 3] = acc3;
            }
        }
    });

    const bool do_parallel_v = should_parallelize_filter_rows(rows, cols, channels, ky);
    parallel_for_index_if(do_parallel_v, rows, [&](int y) {
        uchar* dst_row = dst.data + static_cast<std::size_t>(y) * dst_step;
        const int* y_ofs = y_offsets.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ky);

        if (channels == 1)
        {
            for (int x = 0; x < cols; ++x)
            {
                float acc0 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        acc0 += kernel_y[static_cast<std::size_t>(i)] *
                                tmp[static_cast<std::size_t>(sy + x)];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        acc0 += kernel_y[static_cast<std::size_t>(i)] *
                                tmp[static_cast<std::size_t>(sy + x)];
                    }
                }
                dst_row[x] = saturate_cast<uchar>(acc0);
            }
            return;
        }

        if (channels == 3)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int dx = x * 3;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                }
                dst_row[dx + 0] = saturate_cast<uchar>(acc0);
                dst_row[dx + 1] = saturate_cast<uchar>(acc1);
                dst_row[dx + 2] = saturate_cast<uchar>(acc2);
            }
            return;
        }

        if (channels == 4)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int dx = x * 4;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                }
                dst_row[dx + 0] = saturate_cast<uchar>(acc0);
                dst_row[dx + 1] = saturate_cast<uchar>(acc1);
                dst_row[dx + 2] = saturate_cast<uchar>(acc2);
                dst_row[dx + 3] = saturate_cast<uchar>(acc3);
            }
        }
    });

    return true;
}

inline bool try_gaussian_blur_fastpath_f32(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY, int borderType)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_32F)
    {
        return false;
    }

    if (!is_u8_fastpath_channels(src.channels()))
    {
        return false;
    }

    int kx = ksize.width;
    int ky = ksize.height;

    if (kx <= 0 && sigmaX > 0.0)
    {
        kx = auto_gaussian_ksize(sigmaX);
    }
    if (ky <= 0 && sigmaY > 0.0)
    {
        ky = auto_gaussian_ksize(sigmaY);
    }
    if (kx <= 0 && ky > 0)
    {
        kx = ky;
    }
    if (ky <= 0 && kx > 0)
    {
        ky = kx;
    }

    if (kx <= 0 || ky <= 0 || (kx & 1) == 0 || (ky & 1) == 0)
    {
        return false;
    }

    const bool fixed5x5 =
        kx == 5 && ky == 5 && sigmaX <= 0.0 && sigmaY <= 0.0;

    if (sigmaX <= 0.0)
    {
        sigmaX = default_gaussian_sigma_for_ksize(kx);
    }
    if (sigmaY <= 0.0)
    {
        sigmaY = sigmaX;
    }
    if (sigmaX <= 0.0 || sigmaY <= 0.0)
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
    const std::size_t src_step = src_ref->step(0);
    const int row_stride = cols * channels;

    dst.create(std::vector<int>{rows, cols}, src_ref->type());
    const std::size_t dst_step = dst.step(0);

    if (fixed5x5 &&
        try_gaussian5x5_fixed_f32(*src_ref, dst, border_type))
    {
        return true;
    }

    const std::vector<float> kernel_x = build_gaussian_kernel_1d(kx, sigmaX);
    const std::vector<float> kernel_y = build_gaussian_kernel_1d(ky, sigmaY);
    const int rx = kx / 2;
    const int ry = ky / 2;

    if (filter_ui::separable_c1(*src_ref,
                                dst,
                                CV_32F,
                                kernel_x,
                                kernel_y,
                                rx,
                                ry,
                                0.0,
                                border_type))
    {
        return true;
    }

    const bool has_constant_border = border_type == BORDER_CONSTANT;

    std::vector<int> x_offsets(static_cast<std::size_t>(cols) * static_cast<std::size_t>(kx), -1);
    for (int x = 0; x < cols; ++x)
    {
        int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
        for (int i = 0; i < kx; ++i)
        {
            const int sx = border_interpolate(x + i - rx, cols, border_type);
            x_ofs[i] = sx >= 0 ? sx * channels : -1;
        }
    }

    std::vector<int> y_offsets(static_cast<std::size_t>(rows) * static_cast<std::size_t>(ky), -1);
    for (int y = 0; y < rows; ++y)
    {
        int* y_ofs = y_offsets.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ky);
        for (int i = 0; i < ky; ++i)
        {
            const int sy = border_interpolate(y + i - ry, rows, border_type);
            y_ofs[i] = sy >= 0 ? sy * row_stride : -1;
        }
    }

    std::vector<float> tmp(static_cast<std::size_t>(rows) * static_cast<std::size_t>(row_stride), 0.0f);

    const bool do_parallel_h = should_parallelize_filter_rows(rows, cols, channels, kx);
    parallel_for_index_if(do_parallel_h, rows, [&](int y) {
        const float* src_row = reinterpret_cast<const float*>(src_ref->data + static_cast<std::size_t>(y) * src_step);
        float* tmp_row = tmp.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(row_stride);

        if (channels == 1)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                float acc0 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        acc0 += kernel_x[static_cast<std::size_t>(i)] * src_row[sx];
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        acc0 += kernel_x[static_cast<std::size_t>(i)] * src_row[sx];
                    }
                }
                tmp_row[x] = acc0;
            }
            return;
        }

        if (channels == 3)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                const int dx = x * 3;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const float* px = src_row + sx;
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const float* px = src_row + sx;
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                }
                tmp_row[dx + 0] = acc0;
                tmp_row[dx + 1] = acc1;
                tmp_row[dx + 2] = acc2;
            }
            return;
        }

        if (channels == 4)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int* x_ofs = x_offsets.data() + static_cast<std::size_t>(x) * static_cast<std::size_t>(kx);
                const int dx = x * 4;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        if (sx < 0)
                        {
                            continue;
                        }
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const float* px = src_row + sx;
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                }
                else
                {
                    for (int i = 0; i < kx; ++i)
                    {
                        const int sx = x_ofs[i];
                        const float w = kernel_x[static_cast<std::size_t>(i)];
                        const float* px = src_row + sx;
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                }
                tmp_row[dx + 0] = acc0;
                tmp_row[dx + 1] = acc1;
                tmp_row[dx + 2] = acc2;
                tmp_row[dx + 3] = acc3;
            }
        }
    });

    const bool do_parallel_v = should_parallelize_filter_rows(rows, cols, channels, ky);
    parallel_for_index_if(do_parallel_v, rows, [&](int y) {
        float* dst_row = reinterpret_cast<float*>(dst.data + static_cast<std::size_t>(y) * dst_step);
        const int* y_ofs = y_offsets.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(ky);

        if (channels == 1)
        {
            for (int x = 0; x < cols; ++x)
            {
                float acc0 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        acc0 += kernel_y[static_cast<std::size_t>(i)] *
                                tmp[static_cast<std::size_t>(sy + x)];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        acc0 += kernel_y[static_cast<std::size_t>(i)] *
                                tmp[static_cast<std::size_t>(sy + x)];
                    }
                }
                dst_row[x] = acc0;
            }
            return;
        }

        if (channels == 3)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int dx = x * 3;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                    }
                }
                dst_row[dx + 0] = acc0;
                dst_row[dx + 1] = acc1;
                dst_row[dx + 2] = acc2;
            }
            return;
        }

        if (channels == 4)
        {
            for (int x = 0; x < cols; ++x)
            {
                const int dx = x * 4;
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                if (has_constant_border)
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        if (sy < 0)
                        {
                            continue;
                        }
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                }
                else
                {
                    for (int i = 0; i < ky; ++i)
                    {
                        const int sy = y_ofs[i];
                        const float w = kernel_y[static_cast<std::size_t>(i)];
                        const float* px = tmp.data() + static_cast<std::size_t>(sy + dx);
                        acc0 += w * px[0];
                        acc1 += w * px[1];
                        acc2 += w * px[2];
                        acc3 += w * px[3];
                    }
                }
                dst_row[dx + 0] = acc0;
                dst_row[dx + 1] = acc1;
                dst_row[dx + 2] = acc2;
                dst_row[dx + 3] = acc3;
            }
        }
    });

    return true;
}


} // namespace gaussian_blur_fastpath

inline const char* last_gaussianblur_dispatch_path()
{
    return gaussian_blur_fastpath::g_last_gaussianblur_dispatch_path;
}

inline void gaussianBlur_fast_impl(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY, int borderType)
{
    using namespace gaussian_blur_fastpath;
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    set_last_gaussianblur_dispatch_path("fallback");

    if (try_gaussian_blur_fastpath_u8(src, dst, ksize, sigmaX, sigmaY, borderType))
    {
        int kx = 0;
        int ky = 0;
        if (resolve_gaussian_kernel_size(ksize, sigmaX, sigmaY, kx, ky) && kx == 3 && ky == 3)
        {
            set_last_gaussianblur_dispatch_path("gauss3x3");
        }
        else if (kx == 5 && ky == 5 && sigmaX <= 0.0 && sigmaY <= 0.0)
        {
            set_last_gaussianblur_dispatch_path("gauss5x5_fixedpoint");
        }
        else
        {
            set_last_gaussianblur_dispatch_path("gauss_separable");
        }
        return;
    }

    if (try_gaussian_blur_fastpath_f32(src, dst, ksize, sigmaX, sigmaY, borderType))
    {
        int kx = 0;
        int ky = 0;
        if (resolve_gaussian_kernel_size(ksize, sigmaX, sigmaY, kx, ky) && kx == 3 && ky == 3)
        {
            set_last_gaussianblur_dispatch_path("gauss3x3");
        }
        else if (kx == 5 && ky == 5 && sigmaX <= 0.0 && sigmaY <= 0.0)
        {
            set_last_gaussianblur_dispatch_path("gauss5x5_ring_f32");
        }
        else
        {
            set_last_gaussianblur_dispatch_path("gauss_separable");
        }
        return;
    }

    gaussian_blur_fallback(src, dst, ksize, sigmaX, sigmaY, borderType);
}

} // namespace detail
} // namespace cvh

#endif // CVH_IMGPROC_DETAIL_GAUSSIAN_BLUR_IMPL_HPP
