#ifndef CVH_IMGPROC_WARP_AFFINE_H
#define CVH_IMGPROC_WARP_AFFINE_H

#include "../core/detail/dispatch_control.h"
#include "detail/geometric_sampling.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace cvh {
namespace detail {

inline thread_local const char* g_last_warp_affine_algorithm_path =
    "affine_generic";

inline const char* last_warp_affine_algorithm_path()
{
    return g_last_warp_affine_algorithm_path;
}

inline void warp_affine_read_matrix(const Mat& M,
                                    double& m00, double& m01, double& m02,
                                    double& m10, double& m11, double& m12)
{
    CV_Assert(M.dims == 2 && "warpAffine: transform matrix must be 2D");
    CV_Assert(M.channels() == 1 && "warpAffine: transform matrix must be single-channel");
    CV_Assert(M.size[0] == 2 && M.size[1] == 3 && "warpAffine: transform matrix must be 2x3");

    if (M.depth() == CV_32F)
    {
        m00 = static_cast<double>(M.at<float>(0, 0));
        m01 = static_cast<double>(M.at<float>(0, 1));
        m02 = static_cast<double>(M.at<float>(0, 2));
        m10 = static_cast<double>(M.at<float>(1, 0));
        m11 = static_cast<double>(M.at<float>(1, 1));
        m12 = static_cast<double>(M.at<float>(1, 2));
        return;
    }

    if (M.depth() == CV_64F)
    {
        m00 = M.at<double>(0, 0);
        m01 = M.at<double>(0, 1);
        m02 = M.at<double>(0, 2);
        m10 = M.at<double>(1, 0);
        m11 = M.at<double>(1, 1);
        m12 = M.at<double>(1, 2);
        return;
    }

    CV_Error_(Error::StsBadArg, ("warpAffine: unsupported transform depth=%d", M.depth()));
}

inline void warp_affine_resolve_inverse_map(double& m00, double& m01, double& m02,
                                            double& m10, double& m11, double& m12,
                                            bool inverse_map)
{
    if (inverse_map)
    {
        return;
    }

    const double det = m00 * m11 - m01 * m10;
    if (std::abs(det) < 1e-12)
    {
        CV_Error(Error::StsBadArg, "warpAffine: singular transform matrix");
    }

    const double inv_det = 1.0 / det;
    const double a00 = m11 * inv_det;
    const double a01 = -m01 * inv_det;
    const double a10 = -m10 * inv_det;
    const double a11 = m00 * inv_det;
    const double a02 = -(a00 * m02 + a01 * m12);
    const double a12 = -(a10 * m02 + a11 * m12);

    m00 = a00;
    m01 = a01;
    m02 = a02;
    m10 = a10;
    m11 = a11;
    m12 = a12;
}

template<int Channels>
inline void warp_affine_linear_f32_translation_row(
    const Mat& src,
    float* destination,
    int dst_cols,
    int source_x,
    int source_y,
    double fraction_x,
    double fraction_y,
    int border_type,
    const Scalar& border_value)
{
    const bool source_rows_are_interior =
        static_cast<unsigned>(source_y) <
        static_cast<unsigned>(src.size[0] - 1);
    const int interior_begin = source_rows_are_interior
        ? std::max(0, -source_x)
        : 0;
    const int interior_end = source_rows_are_interior
        ? std::min(dst_cols, src.size[1] - 1 - source_x)
        : 0;
    int x = 0;
    for (; x < interior_begin; ++x)
    {
        float* destination_pixel =
            destination + static_cast<size_t>(x) * Channels;
        const int current_source_x = source_x + x;
        geometric_write_linear_f32(
            src,
            destination_pixel,
            current_source_x,
            source_y,
            fraction_x,
            fraction_y,
            border_type,
            border_value);
    }
    for (; x < interior_end; ++x)
    {
        geometric_write_linear_f32_interior<Channels>(
            src,
            destination + static_cast<size_t>(x) * Channels,
            source_x + x,
            source_y,
            fraction_x,
            fraction_y);
    }
    for (; x < dst_cols; ++x)
    {
        geometric_write_linear_f32(
            src,
            destination + static_cast<size_t>(x) * Channels,
            source_x + x,
            source_y,
            fraction_x,
            fraction_y,
            border_type,
            border_value);
    }
}

inline bool warp_affine_try_linear_f32_translation_row(
    const Mat& src,
    float* destination,
    int dst_cols,
    int y,
    double m00,
    double m01,
    double m02,
    double m10,
    double m11,
    double m12,
    int border_type,
    const Scalar& border_value)
{
    if (m00 != 1.0 || m01 != 0.0 || m10 != 0.0 || m11 != 1.0)
    {
        return false;
    }
    const double source_x_coordinate = m02;
    const double source_y_coordinate =
        static_cast<double>(y) + m12;
    const int source_x =
        static_cast<int>(std::floor(source_x_coordinate));
    const int source_y =
        static_cast<int>(std::floor(source_y_coordinate));
    const double fraction_x = source_x_coordinate - source_x;
    const double fraction_y = source_y_coordinate - source_y;
    switch (src.channels())
    {
    case 1:
        warp_affine_linear_f32_translation_row<1>(
            src, destination, dst_cols, source_x, source_y,
            fraction_x, fraction_y, border_type, border_value);
        return true;
    case 3:
        warp_affine_linear_f32_translation_row<3>(
            src, destination, dst_cols, source_x, source_y,
            fraction_x, fraction_y, border_type, border_value);
        return true;
    case 4:
        warp_affine_linear_f32_translation_row<4>(
            src, destination, dst_cols, source_x, source_y,
            fraction_x, fraction_y, border_type, border_value);
        return true;
    default:
        return false;
    }
}

template<int Channels>
inline void warp_affine_linear_u8_translation_row(
    const Mat& src,
    uchar* destination,
    int dst_cols,
    int source_x,
    int source_y,
    int fraction_x,
    int fraction_y,
    int border_type,
    const Scalar& border_value)
{
    const bool source_rows_are_interior =
        static_cast<unsigned>(source_y) <
        static_cast<unsigned>(src.size[0] - 1);
    const int interior_begin = source_rows_are_interior
        ? std::max(0, -source_x)
        : 0;
    const int interior_end = source_rows_are_interior
        ? std::min(dst_cols, src.size[1] - 1 - source_x)
        : 0;
    int x = 0;
    for (; x < interior_begin; ++x)
    {
        geometric_write_linear_u8_fixed(
            src,
            destination + static_cast<size_t>(x) * Channels,
            source_x + x,
            source_y,
            fraction_x,
            fraction_y,
            border_type,
            border_value);
    }
    const int inverse_x = INTER_TAB_SIZE - fraction_x;
    const int inverse_y = INTER_TAB_SIZE - fraction_y;
    const int weight00 = inverse_x * inverse_y;
    const int weight01 = fraction_x * inverse_y;
    const int weight10 = inverse_x * fraction_y;
    const int weight11 = fraction_x * fraction_y;
    const uchar* interior_top = source_rows_are_interior
        ? src.data + static_cast<size_t>(source_y) * src.step(0) +
              static_cast<size_t>(source_x + interior_begin) * Channels
        : nullptr;
    const uchar* interior_bottom = source_rows_are_interior
        ? interior_top + src.step(0)
        : nullptr;
    uchar* interior_destination =
        destination + static_cast<size_t>(interior_begin) * Channels;
    for (; x < interior_end; ++x)
    {
        for (int channel = 0; channel < Channels; ++channel)
        {
            interior_destination[channel] = static_cast<uchar>(
                (interior_top[channel] * weight00 +
                 interior_top[Channels + channel] * weight01 +
                 interior_bottom[channel] * weight10 +
                 interior_bottom[Channels + channel] * weight11 +
                 INTER_TAB_SIZE2 / 2) /
                INTER_TAB_SIZE2);
        }
        interior_top += Channels;
        interior_bottom += Channels;
        interior_destination += Channels;
    }
    for (; x < dst_cols; ++x)
    {
        geometric_write_linear_u8_fixed(
            src,
            destination + static_cast<size_t>(x) * Channels,
            source_x + x,
            source_y,
            fraction_x,
            fraction_y,
            border_type,
            border_value);
    }
}

inline bool warp_affine_try_linear_u8_translation_row(
    const Mat& src,
    uchar* destination,
    int dst_cols,
    int y,
    double m00,
    double m01,
    double m02,
    double m10,
    double m11,
    double m12,
    int border_type,
    const Scalar& border_value)
{
    if (m00 != 1.0 || m01 != 0.0 || m10 != 0.0 || m11 != 1.0)
    {
        return false;
    }
    int source_x = 0;
    int source_y = 0;
    int fraction_x = 0;
    int fraction_y = 0;
    geometric_fixed_linear_coordinate(m02, source_x, fraction_x);
    geometric_fixed_linear_coordinate(
        static_cast<double>(y) + m12,
        source_y,
        fraction_y);
    switch (src.channels())
    {
    case 1:
        warp_affine_linear_u8_translation_row<1>(
            src, destination, dst_cols, source_x, source_y,
            fraction_x, fraction_y, border_type, border_value);
        return true;
    case 3:
        warp_affine_linear_u8_translation_row<3>(
            src, destination, dst_cols, source_x, source_y,
            fraction_x, fraction_y, border_type, border_value);
        return true;
    case 4:
        warp_affine_linear_u8_translation_row<4>(
            src, destination, dst_cols, source_x, source_y,
            fraction_x, fraction_y, border_type, border_value);
        return true;
    default:
        return false;
    }
}

template <typename T>
inline void warpAffine_fallback_impl_typed(const Mat& src,
                                           Mat& dst,
                                           const Mat& M,
                                           Size dsize,
                                           int flags,
                                           int borderMode,
                                           const Scalar& borderValue)
{
    Mat src_local;
    const Mat* src_ref = &src;
    if (src.data == dst.data)
    {
        src_local = src.clone();
        src_ref = &src_local;
    }

    const int channels = src_ref->channels();

    const int dst_rows =
        dsize.height > 0 ? dsize.height : src_ref->size[0];
    const int dst_cols =
        dsize.width > 0 ? dsize.width : src_ref->size[1];
    CV_Assert(dst_rows > 0 && dst_cols > 0 && "warpAffine: invalid output size");

    dst.create(std::vector<int>{dst_rows, dst_cols}, src_ref->type());
    const size_t dst_step = dst.step(0);

    double m00 = 0.0;
    double m01 = 0.0;
    double m02 = 0.0;
    double m10 = 0.0;
    double m11 = 0.0;
    double m12 = 0.0;
    warp_affine_read_matrix(M, m00, m01, m02, m10, m11, m12);

    const bool inverse_map = (flags & WARP_INVERSE_MAP) != 0;
    warp_affine_resolve_inverse_map(m00, m01, m02, m10, m11, m12, inverse_map);

    const int interpolation = flags & 7;
    const int border_type = normalize_border_type(borderMode);
    g_last_warp_affine_algorithm_path = "affine_generic";

    for (int y = 0; y < dst_rows; ++y)
    {
        T* dst_row = reinterpret_cast<T*>(dst.data + static_cast<size_t>(y) * dst_step);
        if constexpr (std::is_same<T, float>::value)
        {
            if (interpolation == INTER_LINEAR &&
                warp_affine_try_linear_f32_translation_row(
                    *src_ref,
                    dst_row,
                    dst_cols,
                    y,
                    m00,
                    m01,
                    m02,
                    m10,
                    m11,
                    m12,
                    border_type,
                    borderValue))
            {
                g_last_warp_affine_algorithm_path =
                    "affine_translation_f32_scanline";
                continue;
            }
        }
        if constexpr (std::is_same<T, uchar>::value)
        {
            if (interpolation == INTER_LINEAR)
            {
                if (warp_affine_try_linear_u8_translation_row(
                        *src_ref,
                        dst_row,
                        dst_cols,
                        y,
                        m00,
                        m01,
                        m02,
                        m10,
                        m11,
                        m12,
                        border_type,
                        borderValue))
                {
                    g_last_warp_affine_algorithm_path =
                        "affine_translation_u8_fixed_scanline";
                    continue;
                }
                g_last_warp_affine_algorithm_path =
                    "affine_u8_fixed_coordinate_block";
                constexpr int block_size = 64;
                int coordinates[block_size * 2];
                ushort fractions[block_size];
                for (int block_start = 0;
                     block_start < dst_cols;
                     block_start += block_size)
                {
                    const int count =
                        std::min(block_size, dst_cols - block_start);
                    for (int index = 0; index < count; ++index)
                    {
                        const int x = block_start + index;
                        const double sx =
                            m00 * static_cast<double>(x) +
                            m01 * static_cast<double>(y) + m02;
                        const double sy =
                            m10 * static_cast<double>(x) +
                            m11 * static_cast<double>(y) + m12;
                        int fraction_x = 0;
                        int fraction_y = 0;
                        geometric_fixed_linear_coordinate(
                            sx,
                            coordinates[index * 2],
                            fraction_x);
                        geometric_fixed_linear_coordinate(
                            sy,
                            coordinates[index * 2 + 1],
                            fraction_y);
                        fractions[index] =
                            geometric_pack_linear_fraction(
                                fraction_x,
                                fraction_y);
                    }
                    geometric_write_linear_u8_fixed_row(
                        *src_ref,
                        dst_row +
                            static_cast<size_t>(block_start) *
                                channels,
                        coordinates,
                        fractions,
                        count,
                        border_type,
                        borderValue);
                }
                continue;
            }
        }
        for (int x = 0; x < dst_cols; ++x)
        {
            const double sx = m00 * static_cast<double>(x) + m01 * static_cast<double>(y) + m02;
            const double sy = m10 * static_cast<double>(x) + m11 * static_cast<double>(y) + m12;
            T* dst_px = dst_row + static_cast<size_t>(x) * channels;

            if constexpr (std::is_same<T, uchar>::value)
            {
                if (interpolation == INTER_LINEAR)
                {
                    g_last_warp_affine_algorithm_path =
                        "affine_f32_typed_linear";
                    const int integer_x =
                        static_cast<int>(std::floor(sx));
                    const int integer_y =
                        static_cast<int>(std::floor(sy));
                    geometric_write_linear_u8(
                        *src_ref,
                        dst_px,
                        integer_x,
                        integer_y,
                        sx - integer_x,
                        sy - integer_y,
                        border_type,
                        borderValue);
                    continue;
                }
            }
            if constexpr (std::is_same<T, float>::value)
            {
                if (interpolation == INTER_LINEAR)
                {
                    const int integer_x =
                        static_cast<int>(std::floor(sx));
                    const int integer_y =
                        static_cast<int>(std::floor(sy));
                    geometric_write_linear_f32(
                        *src_ref,
                        dst_px,
                        integer_x,
                        integer_y,
                        sx - integer_x,
                        sy - integer_y,
                        border_type,
                        borderValue);
                    continue;
                }
            }
            geometric_write_coordinate(
                *src_ref,
                dst_px,
                sx,
                sy,
                interpolation,
                border_type,
                borderValue,
                false,
                false);
        }
    }
}

inline void warpAffine_fallback(const Mat& src,
                                Mat& dst,
                                const Mat& M,
                                Size dsize,
                                int flags,
                                int borderMode,
                                const Scalar& borderValue)
{
    CV_Assert(!src.empty() && "warpAffine: source image can not be empty");
    CV_Assert(src.dims == 2 && "warpAffine: only 2D Mat is supported");

    const int interpolation = flags & 7;
    const bool interpolation_ok = interpolation == INTER_NEAREST || interpolation == INTER_LINEAR;
    if (!interpolation_ok)
    {
        CV_Error_(Error::StsBadArg, ("warpAffine: unsupported interpolation flags=%d", flags));
    }

    const int supported_flag_mask = 7 | WARP_INVERSE_MAP;
    if ((flags & ~supported_flag_mask) != 0)
    {
        CV_Error_(Error::StsBadArg, ("warpAffine: unsupported flags=%d", flags));
    }

    const int border_type = normalize_border_type(borderMode);
    if (!is_supported_filter_border(border_type))
    {
        CV_Error_(Error::StsBadArg, ("warpAffine: unsupported borderMode=%d", borderMode));
    }

    const int src_depth = src.depth();
    if (src_depth != CV_8U && src_depth != CV_32F)
    {
        CV_Error_(Error::StsBadArg, ("warpAffine: unsupported src depth=%d", src_depth));
    }

    if (src_depth == CV_8U)
    {
        warpAffine_fallback_impl_typed<uchar>(src, dst, M, dsize, flags, borderMode, borderValue);
        return;
    }

    warpAffine_fallback_impl_typed<float>(src, dst, M, dsize, flags, borderMode, borderValue);
}

}  // namespace detail

inline void warpAffine(const Mat& src,
                       Mat& dst,
                       const Mat& M,
                       Size dsize,
                       int flags = INTER_LINEAR,
                       int borderMode = BORDER_CONSTANT,
                       const Scalar& borderValue = Scalar())
{
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    detail::warpAffine_fallback(src, dst, M, dsize, flags, borderMode, borderValue);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_WARP_AFFINE_H
