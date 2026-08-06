#ifndef CVH_IMGPROC_PYRAMID_H
#define CVH_IMGPROC_PYRAMID_H

#include "detail/common.h"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/simd/opencv_ui.h"

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace cvh
{
namespace pyramid_detail
{

inline void validate_source(const Mat& src, const char* name)
{
    if (src.empty() || src.dims != 2 ||
        (src.depth() != CV_8U && src.depth() != CV_32F) ||
        (src.channels() != 1 && src.channels() != 3 &&
         src.channels() != 4))
    {
        CV_Error_(Error::StsBadArg, ("%s unsupported source", name));
    }
}

inline int pyramid_border(int border_type, bool upsample)
{
    const int normalized = detail::normalize_border_type(border_type);
    if (upsample)
    {
        if (normalized != BORDER_REFLECT_101)
        {
            CV_Error(
                Error::StsBadArg,
                "pyrUp currently supports BORDER_DEFAULT only");
        }
    }
    else if (normalized == BORDER_CONSTANT ||
             (normalized != BORDER_REPLICATE &&
              normalized != BORDER_REFLECT &&
              normalized != BORDER_REFLECT_101 &&
              normalized != BORDER_WRAP))
    {
        CV_Error(Error::StsBadArg, "pyrDown unsupported border");
    }
    return normalized;
}

template<typename T>
inline T cast_value(double value)
{
    if constexpr (std::is_same<T, uchar>::value)
    {
        return saturate_cast<uchar>(
            static_cast<int>(std::lrint(value)));
    }
    return static_cast<float>(value);
}

struct PyramidIndexTable
{
    std::vector<int> x;
    std::vector<int> y;
};

template<typename T>
using PyramidWorkType =
    typename std::conditional<std::is_same<T, uchar>::value,
                              int,
                              float>::type;

inline bool pyramid_ui_enabled()
{
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return cpu::opencv_ui_allowed();
#else
    return false;
#endif
}

inline PyramidIndexTable make_downsample_indices(
    int rows,
    int cols,
    int output_rows,
    int output_cols,
    int border_type)
{
    PyramidIndexTable indices;
    indices.x.resize(static_cast<size_t>(output_cols) * 5u);
    for (int x = 0; x < output_cols; ++x)
    {
        for (int kx = -2; kx <= 2; ++kx)
        {
            indices.x[static_cast<size_t>(x) * 5u +
                      static_cast<size_t>(kx + 2)] =
                detail::border_interpolate(
                    2 * x + kx, cols, border_type);
        }
    }
    indices.y.resize(static_cast<size_t>(output_rows) * 5u);
    for (int y = 0; y < output_rows; ++y)
    {
        for (int ky = -2; ky <= 2; ++ky)
        {
            indices.y[static_cast<size_t>(y) * 5u +
                      static_cast<size_t>(ky + 2)] =
                detail::border_interpolate(
                    2 * y + ky, rows, border_type);
        }
    }
    return indices;
}

inline PyramidIndexTable make_upsample_indices(
    int rows,
    int cols,
    int output_rows,
    int output_cols,
    int border_type)
{
    PyramidIndexTable indices;
    indices.x.assign(static_cast<size_t>(output_cols) * 5u, -1);
    for (int x = 0; x < output_cols; ++x)
    {
        for (int kx = -2; kx <= 2; ++kx)
        {
            const int expanded_x = detail::border_interpolate(
                x + kx, cols * 2, border_type);
            if ((expanded_x & 1) == 0)
            {
                indices.x[static_cast<size_t>(x) * 5u +
                          static_cast<size_t>(kx + 2)] =
                    expanded_x / 2;
            }
        }
    }
    indices.y.assign(static_cast<size_t>(output_rows) * 5u, -1);
    for (int y = 0; y < output_rows; ++y)
    {
        for (int ky = -2; ky <= 2; ++ky)
        {
            const int expanded_y = detail::border_interpolate(
                y + ky, rows * 2, border_type);
            if ((expanded_y & 1) == 0)
            {
                indices.y[static_cast<size_t>(y) * 5u +
                          static_cast<size_t>(ky + 2)] =
                    expanded_y / 2;
            }
        }
    }
    return indices;
}

inline size_t pyramid_temporary_elements(
    const Mat& src,
    const Mat& dst)
{
    (void)src;
    return 5u *
           static_cast<size_t>(dst.size.p[1]) *
           static_cast<size_t>(dst.channels());
}

inline int horizontal_downsample_u8_c1_ui(
    const uchar* input,
    int* output,
    int begin,
    int end)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    if (!pyramid_ui_enabled())
    {
        return begin;
    }
    const int output_lanes =
        cv::VTraits<cv::v_int32>::vlanes();
    const int input_lanes =
        cv::VTraits<cv::v_int16>::vlanes();
    const cv::v_int16 weights_1_4 =
        cv::v_reinterpret_as_s16(
            cv::vx_setall_u32(0x00040001));
    const cv::v_int16 weights_6_4 =
        cv::v_reinterpret_as_s16(
            cv::vx_setall_u32(0x00040006));
    const uchar* input_01 = input + begin * 2 - 2;
    const uchar* input_23 = input + begin * 2;
    const uchar* input_4 = input + begin * 2 + 1;
    int x = begin;
    for (; x <= end - output_lanes;
         x += output_lanes,
         input_01 += input_lanes,
         input_23 += input_lanes,
         input_4 += input_lanes)
    {
        cv::vx_store(
            output + x,
            cv::v_add(
                cv::v_add(
                    cv::v_dotprod(
                        cv::v_reinterpret_as_s16(
                            cv::vx_load_expand(input_01)),
                        weights_1_4),
                    cv::v_dotprod(
                        cv::v_reinterpret_as_s16(
                            cv::vx_load_expand(input_23)),
                        weights_6_4)),
                cv::v_shr<16>(
                    cv::v_reinterpret_as_s32(
                        cv::vx_load_expand(input_4)))));
    }
    if (x != begin)
    {
        cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
    }
    return x;
#else
    (void)input;
    (void)output;
    (void)end;
    return begin;
#endif
}

inline int horizontal_downsample_u8_c3_ui(
    const uchar* input,
    int* output,
    int begin,
    int end)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    if (!pyramid_ui_enabled())
    {
        return begin;
    }
    const int input_lanes =
        cv::VTraits<cv::v_int8>::vlanes();
    const int output_lanes =
        cv::VTraits<cv::v_int32>::vlanes();
    alignas(64) int indices[
        cv::VTraits<cv::v_int8>::max_nlanes / 2 + 4];
    for (int index = 0; index < input_lanes / 4 + 2; ++index)
    {
        indices[index] = 6 * index;
        indices[index + input_lanes / 4 + 2] =
            6 * index + 3;
    }

    const cv::v_int16 weights_6_4 =
        cv::v_reinterpret_as_s16(
            cv::vx_setall_u32(0x00040006));
    const int width = (end - begin) * 3;
    const int step = 3 * input_lanes / 4;
    const uchar* source = input + begin * 6 - 6;
    int* destination = output + begin * 3;
    int processed = 0;
    for (; processed <= width - input_lanes;
         processed += step,
         source += 6 * input_lanes / 4,
         destination += step)
    {
        cv::v_uint16 value_0_low;
        cv::v_uint16 value_0_high;
        cv::v_uint16 value_1_low;
        cv::v_uint16 value_1_high;
        cv::v_uint16 value_2_low;
        cv::v_uint16 value_2_high;
        cv::v_uint16 value_3_low;
        cv::v_uint16 value_3_high;
        cv::v_uint16 value_4_low;
        cv::v_uint16 value_4_high;
        cv::v_expand(
            cv::vx_lut_quads(source, indices),
            value_0_low,
            value_0_high);
        cv::v_expand(
            cv::vx_lut_quads(
                source, indices + input_lanes / 4 + 2),
            value_1_low,
            value_1_high);
        cv::v_expand(
            cv::vx_lut_quads(source, indices + 1),
            value_2_low,
            value_2_high);
        cv::v_expand(
            cv::vx_lut_quads(
                source, indices + input_lanes / 4 + 3),
            value_3_low,
            value_3_high);
        cv::v_expand(
            cv::vx_lut_quads(source, indices + 2),
            value_4_low,
            value_4_high);

        cv::v_zip(
            value_2_low,
            cv::v_add(value_1_low, value_3_low),
            value_1_low,
            value_3_low);
        cv::v_zip(
            value_2_high,
            cv::v_add(value_1_high, value_3_high),
            value_1_high,
            value_3_high);
        value_0_low = cv::v_add(value_0_low, value_4_low);
        value_0_high = cv::v_add(value_0_high, value_4_high);

        cv::vx_store(
            destination,
            cv::v_pack_triplets(
                cv::v_add(
                    cv::v_dotprod(
                        cv::v_reinterpret_as_s16(value_1_low),
                        weights_6_4),
                    cv::v_reinterpret_as_s32(
                        cv::v_expand_low(value_0_low)))));
        cv::vx_store(
            destination + 3 * output_lanes / 4,
            cv::v_pack_triplets(
                cv::v_add(
                    cv::v_dotprod(
                        cv::v_reinterpret_as_s16(value_3_low),
                        weights_6_4),
                    cv::v_reinterpret_as_s32(
                        cv::v_expand_high(value_0_low)))));
        cv::vx_store(
            destination + 6 * output_lanes / 4,
            cv::v_pack_triplets(
                cv::v_add(
                    cv::v_dotprod(
                        cv::v_reinterpret_as_s16(value_1_high),
                        weights_6_4),
                    cv::v_reinterpret_as_s32(
                        cv::v_expand_low(value_0_high)))));
        cv::vx_store(
            destination + 9 * output_lanes / 4,
            cv::v_pack_triplets(
                cv::v_add(
                    cv::v_dotprod(
                        cv::v_reinterpret_as_s16(value_3_high),
                        weights_6_4),
                    cv::v_reinterpret_as_s32(
                        cv::v_expand_high(value_0_high)))));
    }
    if (processed != 0)
    {
        cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
    }
    return begin + processed / 3;
#else
    (void)input;
    (void)output;
    (void)end;
    return begin;
#endif
}

inline int horizontal_downsample_u8_c4_ui(
    const uchar* input,
    int* output,
    int begin,
    int end)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    if (!pyramid_ui_enabled())
    {
        return begin;
    }
    const int output_lanes =
        cv::VTraits<cv::v_int32>::vlanes();
    const int input_lanes =
        cv::VTraits<cv::v_int16>::vlanes();
    const cv::v_int16 weights_1_4 =
        cv::v_reinterpret_as_s16(
            cv::vx_setall_u32(0x00040001));
    const cv::v_int16 weights_6_4 =
        cv::v_reinterpret_as_s16(
            cv::vx_setall_u32(0x00040006));
    const int width = (end - begin) * 4;
    const uchar* input_01 = input + begin * 8 - 8;
    const uchar* input_23 = input + begin * 8;
    const uchar* input_4 = input + begin * 8 + 4;
    int* destination = output + begin * 4;
    int processed = 0;
    for (; processed <= width - output_lanes;
         processed += output_lanes,
         input_01 += input_lanes,
         input_23 += input_lanes,
         input_4 += input_lanes,
         destination += output_lanes)
    {
        cv::vx_store(
            destination,
            cv::v_add(
                cv::v_add(
                    cv::v_dotprod(
                        cv::v_interleave_quads(
                            cv::v_reinterpret_as_s16(
                                cv::vx_load_expand(input_01))),
                        weights_1_4),
                    cv::v_dotprod(
                        cv::v_interleave_quads(
                            cv::v_reinterpret_as_s16(
                                cv::vx_load_expand(input_23))),
                        weights_6_4)),
                cv::v_shr<16>(
                    cv::v_reinterpret_as_s32(
                        cv::v_interleave_quads(
                            cv::vx_load_expand(input_4))))));
    }
    if (processed != 0)
    {
        cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
    }
    return begin + processed / 4;
#else
    (void)input;
    (void)output;
    (void)end;
    return begin;
#endif
}

inline int horizontal_downsample_f32_c1_ui(
    const float* input,
    float* output,
    int begin,
    int end)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    if (!pyramid_ui_enabled())
    {
        return begin;
    }
    const int lanes = cv::VTraits<cv::v_float32>::vlanes();
    const cv::v_float32 weight_4 = cv::vx_setall_f32(4.0f);
    const cv::v_float32 weight_6 = cv::vx_setall_f32(6.0f);
    const float* input_01 = input + begin * 2 - 2;
    const float* input_23 = input + begin * 2;
    const float* input_4 = input + begin * 2 + 1;
    int x = begin;
    for (; x <= end - lanes;
         x += lanes,
         input_01 += 2 * lanes,
         input_23 += 2 * lanes,
         input_4 += 2 * lanes)
    {
        cv::v_float32 value_0;
        cv::v_float32 value_1;
        cv::v_float32 value_2;
        cv::v_float32 value_3;
        cv::v_float32 value_4;
        cv::v_float32 unused;
        cv::v_load_deinterleave(
            input_01, value_0, value_1);
        cv::v_load_deinterleave(
            input_23, value_2, value_3);
        cv::v_load_deinterleave(
            input_4, unused, value_4);
        cv::vx_store(
            output + x,
            cv::v_fma(
                value_2,
                weight_6,
                cv::v_fma(
                    cv::v_add(value_1, value_3),
                    weight_4,
                    cv::v_add(value_0, value_4))));
    }
    if (x != begin)
    {
        cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
    }
    return x;
#else
    (void)input;
    (void)output;
    (void)end;
    return begin;
#endif
}

template<typename T>
inline void horizontal_downsample_row(
    const Mat& src,
    const PyramidIndexTable& table,
    int source_y,
    PyramidWorkType<T>* output,
    int output_cols)
{
    static constexpr int weights[5] = {1, 4, 6, 4, 1};
    const int channels = src.channels();
    using WorkType = PyramidWorkType<T>;
    const T* input = reinterpret_cast<const T*>(
        src.data + static_cast<size_t>(source_y) * src.step(0));
    const auto compute_scalar = [&](int x) {
        const int* indices =
            table.x.data() + static_cast<size_t>(x) * 5u;
        for (int ch = 0; ch < channels; ++ch)
        {
            const size_t channel = static_cast<size_t>(ch);
            const size_t channel_count =
                static_cast<size_t>(channels);
            output[
                static_cast<size_t>(x) * channel_count + channel] =
                static_cast<WorkType>(
                    input[static_cast<size_t>(indices[0]) *
                              channel_count +
                          channel]) *
                    static_cast<WorkType>(weights[0]) +
                static_cast<WorkType>(
                    input[static_cast<size_t>(indices[1]) *
                              channel_count +
                          channel]) *
                    static_cast<WorkType>(weights[1]) +
                static_cast<WorkType>(
                    input[static_cast<size_t>(indices[2]) *
                              channel_count +
                          channel]) *
                    static_cast<WorkType>(weights[2]) +
                static_cast<WorkType>(
                    input[static_cast<size_t>(indices[3]) *
                              channel_count +
                          channel]) *
                    static_cast<WorkType>(weights[3]) +
                static_cast<WorkType>(
                    input[static_cast<size_t>(indices[4]) *
                              channel_count +
                          channel]) *
                    static_cast<WorkType>(weights[4]);
        }
    };
    int x = 0;
    if (output_cols > 2)
    {
        const int interior_begin = 1;
        const int interior_end = std::min(
            output_cols,
            (src.size.p[1] - 3) / 2 + 1);
        for (; x < interior_begin; ++x)
        {
            compute_scalar(x);
        }
        if constexpr (std::is_same<T, uchar>::value)
        {
            if (channels == 1)
            {
                x = horizontal_downsample_u8_c1_ui(
                    input,
                    reinterpret_cast<int*>(output),
                    interior_begin,
                    interior_end);
            }
            else if (channels == 3)
            {
                x = horizontal_downsample_u8_c3_ui(
                    input,
                    reinterpret_cast<int*>(output),
                    interior_begin,
                    interior_end);
            }
            else if (channels == 4)
            {
                x = horizontal_downsample_u8_c4_ui(
                    input,
                    reinterpret_cast<int*>(output),
                    interior_begin,
                    interior_end);
            }
        }
        else if (channels == 1)
        {
            x = horizontal_downsample_f32_c1_ui(
                input,
                reinterpret_cast<float*>(output),
                interior_begin,
                interior_end);
        }
    }
    for (; x < output_cols; ++x)
    {
        compute_scalar(x);
    }
}

template<typename T>
inline int vertical_pyramid_row_ui(
    const PyramidWorkType<T>* const rows[5],
    T* output,
    int width,
    int shift)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    if (!pyramid_ui_enabled())
    {
        return 0;
    }
    static constexpr int weights[5] = {1, 4, 6, 4, 1};
    const int lanes = cv::VTraits<cv::v_int32>::vlanes();
    int x = 0;
    if constexpr (std::is_same<T, uchar>::value)
    {
        const int byte_lanes = cv::VTraits<cv::v_uint8>::vlanes();
        const int half_lanes = cv::VTraits<cv::v_uint16>::vlanes();
        const int int_lanes = cv::VTraits<cv::v_int32>::vlanes();
        const auto packed_row = [&](int tap, int offset) {
            if (rows[tap] == nullptr)
            {
                return cv::vx_setzero_u16();
            }
            return cv::v_reinterpret_as_u16(
                cv::v_pack(
                    cv::vx_load(rows[tap] + offset),
                    cv::vx_load(rows[tap] + offset + int_lanes)));
        };
        const auto accumulate = [&](int offset) {
            const cv::v_uint16 row0 = packed_row(0, offset);
            const cv::v_uint16 row1 = packed_row(1, offset);
            const cv::v_uint16 row2 = packed_row(2, offset);
            const cv::v_uint16 row3 = packed_row(3, offset);
            const cv::v_uint16 row4 = packed_row(4, offset);
            return cv::v_add(
                cv::v_add(
                    cv::v_add(row0, row4),
                    cv::v_add(row2, row2)),
                cv::v_shl<2>(
                    cv::v_add(cv::v_add(row1, row3), row2)));
        };
        for (; x <= width - byte_lanes; x += byte_lanes)
        {
            const cv::v_uint16 first = accumulate(x);
            const cv::v_uint16 second = accumulate(x + half_lanes);
            if (shift == 8)
            {
                cv::vx_store(output + x, cv::v_rshr_pack<8>(first, second));
            }
            else
            {
                cv::vx_store(output + x, cv::v_rshr_pack<6>(first, second));
            }
        }
        if (x <= width - half_lanes)
        {
            const cv::v_uint16 values = accumulate(x);
            if (shift == 8)
            {
                cv::v_rshr_pack_store<8>(output + x, values);
            }
            else
            {
                cv::v_rshr_pack_store<6>(output + x, values);
            }
            x += half_lanes;
        }
    }
    else
    {
        const cv::v_float32 scale =
            cv::vx_setall_f32(
                shift == 8 ? 1.0f / 256.0f : 1.0f / 64.0f);
        for (; x <= width - lanes; x += lanes)
        {
            cv::v_float32 sum = cv::vx_setzero_f32();
            for (int tap = 0; tap < 5; ++tap)
            {
                if (rows[tap] != nullptr)
                {
                    sum = cv::v_fma(
                        cv::vx_load(rows[tap] + x),
                        cv::vx_setall_f32(
                            static_cast<float>(weights[tap])),
                        sum);
                }
            }
            cv::vx_store(output + x, cv::v_mul(sum, scale));
        }
    }
    if (x != 0)
    {
        cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
    }
    return x;
#else
    (void)rows;
    (void)output;
    (void)width;
    (void)shift;
    return 0;
#endif
}

template<typename T>
inline void vertical_pyramid_row(
    const PyramidWorkType<T>* const rows[5],
    T* output,
    int width,
    int shift)
{
    static constexpr int weights[5] = {1, 4, 6, 4, 1};
    using WorkType = PyramidWorkType<T>;
    int x = vertical_pyramid_row_ui<T>(
        rows, output, width, shift);
    for (; x < width; ++x)
    {
        WorkType sum = WorkType();
        for (int tap = 0; tap < 5; ++tap)
        {
            if (rows[tap] != nullptr)
            {
                sum += static_cast<WorkType>(weights[tap]) *
                       rows[tap][x];
            }
        }
        if constexpr (std::is_same<T, uchar>::value)
        {
            output[x] = saturate_cast<uchar>(
                (sum + (1 << (shift - 1))) >> shift);
        }
        else
        {
            output[x] = sum *
                        (shift == 8 ? 1.0f / 256.0f
                                    : 1.0f / 64.0f);
        }
    }
    if (!pyramid_ui_enabled())
    {
        cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    }
}

inline int ring_slot(int logical_row)
{
    const int slot = logical_row % 5;
    return slot < 0 ? slot + 5 : slot;
}

template<typename T>
inline void downsample_kernel(
    const Mat& src,
    Mat& dst,
    const PyramidIndexTable& table,
    PyramidWorkType<T>* temporary)
{
    const int channels = src.channels();
    const int output_rows = dst.size.p[0];
    const int output_cols = dst.size.p[1];
    const int temporary_stride = output_cols * channels;
    int next_logical_row = -2;

    for (int y = 0; y < output_rows; ++y)
    {
        const int first_logical_row = y * 2 - 2;
        const int last_logical_row = y * 2 + 2;
        for (; next_logical_row <= last_logical_row;
             ++next_logical_row)
        {
            const int tap =
                next_logical_row - first_logical_row;
            const int source_y =
                table.y[static_cast<size_t>(y) * 5u +
                        static_cast<size_t>(tap)];
            horizontal_downsample_row<T>(
                src,
                table,
                source_y,
                temporary +
                    static_cast<size_t>(
                        ring_slot(next_logical_row)) *
                        static_cast<size_t>(temporary_stride),
                output_cols);
        }
        const PyramidWorkType<T>* rows[5];
        for (int tap = 0; tap < 5; ++tap)
        {
            rows[tap] =
                temporary +
                static_cast<size_t>(
                    ring_slot(first_logical_row + tap)) *
                    static_cast<size_t>(temporary_stride);
        }
        T* output = reinterpret_cast<T*>(
            dst.data + static_cast<size_t>(y) * dst.step(0));
        vertical_pyramid_row<T>(
            rows, output, temporary_stride, 8);
    }
}

template<typename T>
inline void downsample_with_workspace(
    const Mat& src,
    Mat& dst,
    int border_type,
    std::vector<PyramidWorkType<T>>& temporary)
{
    const PyramidIndexTable indices = make_downsample_indices(
        src.size.p[0],
        src.size.p[1],
        dst.size.p[0],
        dst.size.p[1],
        border_type);
    temporary.resize(pyramid_temporary_elements(src, dst));
    downsample_kernel<T>(
        src, dst, indices, temporary.data());
}

template<typename T>
inline void downsample(const Mat& src, Mat& dst, int border_type)
{
    std::vector<PyramidWorkType<T>> temporary;
    downsample_with_workspace<T>(
        src, dst, border_type, temporary);
}

template<typename T>
inline void horizontal_upsample_row(
    const Mat& src,
    const PyramidIndexTable& table,
    int source_y,
    PyramidWorkType<T>* output,
    int output_cols)
{
    static constexpr int weights[5] = {1, 4, 6, 4, 1};
    const int channels = src.channels();
    using WorkType = PyramidWorkType<T>;
    const T* input = reinterpret_cast<const T*>(
        src.data + static_cast<size_t>(source_y) * src.step(0));
    const auto compute_scalar = [&](int x) {
        const int* indices =
            table.x.data() + static_cast<size_t>(x) * 5u;
        for (int ch = 0; ch < channels; ++ch)
        {
            WorkType sum = WorkType();
            for (int tap = 0; tap < 5; ++tap)
            {
                if (indices[tap] >= 0)
                {
                    sum += static_cast<WorkType>(weights[tap]) *
                           static_cast<WorkType>(
                               input[
                                   static_cast<size_t>(indices[tap]) *
                                       static_cast<size_t>(channels) +
                                   static_cast<size_t>(ch)]);
                }
            }
            output[
                static_cast<size_t>(x) *
                    static_cast<size_t>(channels) +
                static_cast<size_t>(ch)] = sum;
        }
    };

    int x = 0;
    for (; x < std::min(2, output_cols); ++x)
    {
        compute_scalar(x);
    }
    const int interior_end = std::max(
        x,
        std::min(output_cols - 2, 2 * (src.size.p[1] - 1)));
    for (; x + 1 < interior_end; x += 2)
    {
        const int source_x = x / 2;
        for (int ch = 0; ch < channels; ++ch)
        {
            const size_t center =
                static_cast<size_t>(source_x) * channels + ch;
            const WorkType previous = static_cast<WorkType>(
                input[center - static_cast<size_t>(channels)]);
            const WorkType current =
                static_cast<WorkType>(input[center]);
            const WorkType next = static_cast<WorkType>(
                input[center + static_cast<size_t>(channels)]);
            output[static_cast<size_t>(x) * channels + ch] =
                previous + static_cast<WorkType>(6) * current + next;
            output[static_cast<size_t>(x + 1) * channels + ch] =
                static_cast<WorkType>(4) * (current + next);
        }
    }
    for (; x < output_cols; ++x)
    {
        compute_scalar(x);
    }
}

template<typename T>
inline void upsample_kernel(
    const Mat& src,
    Mat& dst,
    const PyramidIndexTable& table,
    PyramidWorkType<T>* temporary)
{
    const int channels = src.channels();
    const int output_rows = dst.size.p[0];
    const int output_cols = dst.size.p[1];
    const int temporary_stride = output_cols * channels;
    std::array<int, 5> cached_source_rows;
    cached_source_rows.fill(std::numeric_limits<int>::min());

    for (int y = 0; y < output_rows; ++y)
    {
        T* output = reinterpret_cast<T*>(
            dst.data + static_cast<size_t>(y) * dst.step(0));
        const int* indices =
            table.y.data() + static_cast<size_t>(y) * 5u;
        const PyramidWorkType<T>* rows[5] = {
            nullptr, nullptr, nullptr, nullptr, nullptr};
        for (int tap = 0; tap < 5; ++tap)
        {
            const int source_y = indices[tap];
            if (source_y < 0)
            {
                continue;
            }
            int slot = 0;
            for (; slot < 5; ++slot)
            {
                if (cached_source_rows[slot] == source_y)
                {
                    break;
                }
            }
            if (slot == 5)
            {
                for (slot = 0; slot < 5; ++slot)
                {
                    bool needed = false;
                    for (int current_tap = 0;
                         current_tap < 5;
                         ++current_tap)
                    {
                        needed =
                            needed ||
                            (indices[current_tap] >= 0 &&
                             cached_source_rows[slot] ==
                                 indices[current_tap]);
                    }
                    if (!needed)
                    {
                        break;
                    }
                }
                horizontal_upsample_row<T>(
                    src,
                    table,
                    source_y,
                    temporary +
                        static_cast<size_t>(slot) *
                            static_cast<size_t>(temporary_stride),
                    output_cols);
                cached_source_rows[slot] = source_y;
            }
            rows[tap] =
                temporary +
                static_cast<size_t>(slot) *
                    static_cast<size_t>(temporary_stride);
        }
        vertical_pyramid_row<T>(
            rows, output, temporary_stride, 6);
    }
}

template<typename T>
inline void upsample(const Mat& src, Mat& dst, int border_type)
{
    const PyramidIndexTable indices = make_upsample_indices(
        src.size.p[0],
        src.size.p[1],
        dst.size.p[0],
        dst.size.p[1],
        border_type);
    std::vector<PyramidWorkType<T>> temporary(
        pyramid_temporary_elements(src, dst));
    upsample_kernel<T>(
        src, dst, indices, temporary.data());
}

}  // namespace pyramid_detail

inline void pyrDown(const Mat& src,
                    Mat& dst,
                    const Size& dstsize = Size(),
                    int borderType = BORDER_DEFAULT)
{
    pyramid_detail::validate_source(src, "pyrDown");
    const int border_type =
        pyramid_detail::pyramid_border(borderType, false);
    const int output_cols =
        dstsize.width > 0 ? dstsize.width : (src.size.p[1] + 1) / 2;
    const int output_rows =
        dstsize.height > 0 ? dstsize.height : (src.size.p[0] + 1) / 2;
    if ((dstsize.width == 0) != (dstsize.height == 0) ||
        output_cols <= 0 || output_rows <= 0 ||
        std::abs(output_cols * 2 - src.size.p[1]) > 2 ||
        std::abs(output_rows * 2 - src.size.p[0]) > 2)
    {
        CV_Error(Error::StsBadSize, "pyrDown invalid destination size");
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    const Mat source = src.data == dst.data ? src.clone() : src;
    dst.create(
        {output_rows, output_cols}, source.type());
    if (source.depth() == CV_8U)
    {
        pyramid_detail::downsample<uchar>(
            source, dst, border_type);
    }
    else
    {
        pyramid_detail::downsample<float>(
            source, dst, border_type);
    }
}

inline void pyrUp(const Mat& src,
                  Mat& dst,
                  const Size& dstsize = Size(),
                  int borderType = BORDER_DEFAULT)
{
    pyramid_detail::validate_source(src, "pyrUp");
    const int border_type =
        pyramid_detail::pyramid_border(borderType, true);
    const int output_cols =
        dstsize.width > 0 ? dstsize.width : src.size.p[1] * 2;
    const int output_rows =
        dstsize.height > 0 ? dstsize.height : src.size.p[0] * 2;
    if ((dstsize.width == 0) != (dstsize.height == 0) ||
        output_cols <= 0 || output_rows <= 0 ||
        std::abs(output_cols - src.size.p[1] * 2) >
            (output_cols & 1) ||
        std::abs(output_rows - src.size.p[0] * 2) >
            (output_rows & 1))
    {
        CV_Error(Error::StsBadSize, "pyrUp invalid destination size");
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    const Mat source = src.data == dst.data ? src.clone() : src;
    dst.create(
        {output_rows, output_cols}, source.type());
    if (source.depth() == CV_8U)
    {
        pyramid_detail::upsample<uchar>(
            source, dst, border_type);
    }
    else
    {
        pyramid_detail::upsample<float>(
            source, dst, border_type);
    }
}

inline void buildPyramid(const Mat& src,
                         std::vector<Mat>& dst,
                         int maxlevel,
                         int borderType = BORDER_DEFAULT)
{
    pyramid_detail::validate_source(src, "buildPyramid");
    if (maxlevel < 0)
    {
        CV_Error(
            Error::StsOutOfRange,
            "buildPyramid maxlevel must be non-negative");
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    dst.clear();
    dst.reserve(static_cast<size_t>(maxlevel + 1));
    // OpenCV keeps level zero as a Mat header over the input; the generated
    // levels own their storage. Preserve that aliasing contract and avoid an
    // unnecessary full-image copy.
    dst.push_back(src);
    const int border_type =
        pyramid_detail::pyramid_border(borderType, false);
    if (src.depth() == CV_8U)
    {
        std::vector<pyramid_detail::PyramidWorkType<uchar>>
            temporary;
        for (int level = 1; level <= maxlevel; ++level)
        {
            const Mat& previous = dst.back();
            Mat next(
                {(previous.size.p[0] + 1) / 2,
                 (previous.size.p[1] + 1) / 2},
                previous.type());
            pyramid_detail::downsample_with_workspace<uchar>(
                previous, next, border_type, temporary);
            dst.push_back(std::move(next));
        }
    }
    else
    {
        std::vector<pyramid_detail::PyramidWorkType<float>>
            temporary;
        for (int level = 1; level <= maxlevel; ++level)
        {
            const Mat& previous = dst.back();
            Mat next(
                {(previous.size.p[0] + 1) / 2,
                 (previous.size.p[1] + 1) / 2},
                previous.type());
            pyramid_detail::downsample_with_workspace<float>(
                previous, next, border_type, temporary);
            dst.push_back(std::move(next));
        }
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_PYRAMID_H
