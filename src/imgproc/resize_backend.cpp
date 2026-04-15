#include "cvh/imgproc/imgproc.h"
#include "cvh/core/detail/openmp_utils.h"

#include <array>
#include <cstdint>
#include <cmath>
#include <vector>

namespace cvh
{
namespace detail
{

namespace {

inline void build_nearest_index_map(int src_len, int dst_len, bool exact, std::vector<int>& map)
{
    map.resize(static_cast<size_t>(dst_len));
    if (exact)
    {
        const std::int64_t scale = ((static_cast<std::int64_t>(src_len) << 16) + dst_len / 2) / dst_len;
        const std::int64_t offset = scale / 2 - (src_len % 2);
        for (int i = 0; i < dst_len; ++i)
        {
            const int src_i = static_cast<int>((scale * i + offset) >> 16);
            map[static_cast<size_t>(i)] = std::clamp(src_i, 0, src_len - 1);
        }
        return;
    }

    for (int i = 0; i < dst_len; ++i)
    {
        map[static_cast<size_t>(i)] = std::min(src_len - 1, (i * src_len) / dst_len);
    }
}

inline void build_linear_coeffs(int src_len,
                                int dst_len,
                                std::vector<int>& idx0,
                                std::vector<int>& idx1,
                                std::vector<float>& w)
{
    idx0.resize(static_cast<size_t>(dst_len));
    idx1.resize(static_cast<size_t>(dst_len));
    w.resize(static_cast<size_t>(dst_len));
    const float scale = static_cast<float>(src_len) / static_cast<float>(dst_len);
    for (int i = 0; i < dst_len; ++i)
    {
        const float src_f = (static_cast<float>(i) + 0.5f) * scale - 0.5f;
        const int i0 = std::clamp(static_cast<int>(std::floor(src_f)), 0, src_len - 1);
        const int i1 = std::min(i0 + 1, src_len - 1);
        idx0[static_cast<size_t>(i)] = i0;
        idx1[static_cast<size_t>(i)] = i1;
        w[static_cast<size_t>(i)] = src_f - static_cast<float>(i0);
    }
}

inline void resize_nearest_u8_c1(const Mat& src,
                                 Mat& dst,
                                 const std::vector<int>& x_map,
                                 const std::vector<int>& y_map)
{
    const int dst_rows = dst.size[0];
    const int dst_cols = dst.size[1];
    const size_t src_step = src.step(0);
    const size_t dst_step = dst.step(0);
    const long long rows_ll = static_cast<long long>(dst_rows);
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(dst_rows), static_cast<size_t>(dst_cols), 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t yd = static_cast<size_t>(y);
        const uchar* src_row = src.data + static_cast<size_t>(y_map[yd]) * src_step;
        uchar* dst_row = dst.data + yd * dst_step;
        for (int x = 0; x < dst_cols; ++x)
        {
            dst_row[x] = src_row[x_map[static_cast<size_t>(x)]];
        }
    }
}

inline void resize_nearest_u8_c3(const Mat& src,
                                 Mat& dst,
                                 const std::vector<int>& x_map,
                                 const std::vector<int>& y_map)
{
    const int dst_rows = dst.size[0];
    const int dst_cols = dst.size[1];
    const size_t src_step = src.step(0);
    const size_t dst_step = dst.step(0);
    const long long rows_ll = static_cast<long long>(dst_rows);
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(dst_rows), static_cast<size_t>(dst_cols) * 3u, 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t yd = static_cast<size_t>(y);
        const uchar* src_row = src.data + static_cast<size_t>(y_map[yd]) * src_step;
        uchar* dst_row = dst.data + yd * dst_step;
        for (int x = 0; x < dst_cols; ++x)
        {
            const uchar* src_px = src_row + static_cast<size_t>(x_map[static_cast<size_t>(x)]) * 3;
            uchar* dst_px = dst_row + static_cast<size_t>(x) * 3;
            dst_px[0] = src_px[0];
            dst_px[1] = src_px[1];
            dst_px[2] = src_px[2];
        }
    }
}

inline void resize_nearest_u8_c4(const Mat& src,
                                 Mat& dst,
                                 const std::vector<int>& x_map,
                                 const std::vector<int>& y_map)
{
    const int dst_rows = dst.size[0];
    const int dst_cols = dst.size[1];
    const size_t src_step = src.step(0);
    const size_t dst_step = dst.step(0);
    const long long rows_ll = static_cast<long long>(dst_rows);
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(dst_rows), static_cast<size_t>(dst_cols) * 4u, 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t yd = static_cast<size_t>(y);
        const uchar* src_row = src.data + static_cast<size_t>(y_map[yd]) * src_step;
        uchar* dst_row = dst.data + yd * dst_step;
        for (int x = 0; x < dst_cols; ++x)
        {
            const uchar* src_px = src_row + static_cast<size_t>(x_map[static_cast<size_t>(x)]) * 4;
            uchar* dst_px = dst_row + static_cast<size_t>(x) * 4;
            dst_px[0] = src_px[0];
            dst_px[1] = src_px[1];
            dst_px[2] = src_px[2];
            dst_px[3] = src_px[3];
        }
    }
}

inline void resize_linear_u8_c1(const Mat& src,
                                Mat& dst,
                                const std::vector<int>& x0,
                                const std::vector<int>& x1,
                                const std::vector<float>& wx,
                                const std::vector<int>& y0,
                                const std::vector<int>& y1,
                                const std::vector<float>& wy)
{
    const int dst_rows = dst.size[0];
    const int dst_cols = dst.size[1];
    const size_t src_step = src.step(0);
    const size_t dst_step = dst.step(0);
    const long long rows_ll = static_cast<long long>(dst_rows);
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(dst_rows), static_cast<size_t>(dst_cols), 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t yd = static_cast<size_t>(y);
        const uchar* src_row0 = src.data + static_cast<size_t>(y0[yd]) * src_step;
        const uchar* src_row1 = src.data + static_cast<size_t>(y1[yd]) * src_step;
        uchar* dst_row = dst.data + yd * dst_step;
        const float wy_v = wy[yd];
        for (int x = 0; x < dst_cols; ++x)
        {
            const size_t xd = static_cast<size_t>(x);
            const int sx0 = x0[xd];
            const int sx1 = x1[xd];
            const float wx_v = wx[xd];
            const float top = lerp(static_cast<float>(src_row0[sx0]), static_cast<float>(src_row0[sx1]), wx_v);
            const float bot = lerp(static_cast<float>(src_row1[sx0]), static_cast<float>(src_row1[sx1]), wx_v);
            dst_row[x] = saturate_cast<uchar>(lerp(top, bot, wy_v));
        }
    }
}

inline void resize_linear_u8_c3(const Mat& src,
                                Mat& dst,
                                const std::vector<int>& x0,
                                const std::vector<int>& x1,
                                const std::vector<float>& wx,
                                const std::vector<int>& y0,
                                const std::vector<int>& y1,
                                const std::vector<float>& wy)
{
    const int dst_rows = dst.size[0];
    const int dst_cols = dst.size[1];
    const size_t src_step = src.step(0);
    const size_t dst_step = dst.step(0);
    const long long rows_ll = static_cast<long long>(dst_rows);
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(dst_rows), static_cast<size_t>(dst_cols) * 3u, 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t yd = static_cast<size_t>(y);
        const uchar* src_row0 = src.data + static_cast<size_t>(y0[yd]) * src_step;
        const uchar* src_row1 = src.data + static_cast<size_t>(y1[yd]) * src_step;
        uchar* dst_row = dst.data + yd * dst_step;
        const float wy_v = wy[yd];
        for (int x = 0; x < dst_cols; ++x)
        {
            const size_t xd = static_cast<size_t>(x);
            const int sx0 = x0[xd];
            const int sx1 = x1[xd];
            const float wx_v = wx[xd];
            const uchar* p00 = src_row0 + static_cast<size_t>(sx0) * 3;
            const uchar* p01 = src_row0 + static_cast<size_t>(sx1) * 3;
            const uchar* p10 = src_row1 + static_cast<size_t>(sx0) * 3;
            const uchar* p11 = src_row1 + static_cast<size_t>(sx1) * 3;
            uchar* out = dst_row + static_cast<size_t>(x) * 3;

            const float top0 = lerp(static_cast<float>(p00[0]), static_cast<float>(p01[0]), wx_v);
            const float bot0 = lerp(static_cast<float>(p10[0]), static_cast<float>(p11[0]), wx_v);
            const float top1 = lerp(static_cast<float>(p00[1]), static_cast<float>(p01[1]), wx_v);
            const float bot1 = lerp(static_cast<float>(p10[1]), static_cast<float>(p11[1]), wx_v);
            const float top2 = lerp(static_cast<float>(p00[2]), static_cast<float>(p01[2]), wx_v);
            const float bot2 = lerp(static_cast<float>(p10[2]), static_cast<float>(p11[2]), wx_v);
            out[0] = saturate_cast<uchar>(lerp(top0, bot0, wy_v));
            out[1] = saturate_cast<uchar>(lerp(top1, bot1, wy_v));
            out[2] = saturate_cast<uchar>(lerp(top2, bot2, wy_v));
        }
    }
}

inline void resize_linear_u8_c4(const Mat& src,
                                Mat& dst,
                                const std::vector<int>& x0,
                                const std::vector<int>& x1,
                                const std::vector<float>& wx,
                                const std::vector<int>& y0,
                                const std::vector<int>& y1,
                                const std::vector<float>& wy)
{
    const int dst_rows = dst.size[0];
    const int dst_cols = dst.size[1];
    const size_t src_step = src.step(0);
    const size_t dst_step = dst.step(0);
    const long long rows_ll = static_cast<long long>(dst_rows);
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(dst_rows), static_cast<size_t>(dst_cols) * 4u, 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t yd = static_cast<size_t>(y);
        const uchar* src_row0 = src.data + static_cast<size_t>(y0[yd]) * src_step;
        const uchar* src_row1 = src.data + static_cast<size_t>(y1[yd]) * src_step;
        uchar* dst_row = dst.data + yd * dst_step;
        const float wy_v = wy[yd];
        for (int x = 0; x < dst_cols; ++x)
        {
            const size_t xd = static_cast<size_t>(x);
            const int sx0 = x0[xd];
            const int sx1 = x1[xd];
            const float wx_v = wx[xd];
            const uchar* p00 = src_row0 + static_cast<size_t>(sx0) * 4;
            const uchar* p01 = src_row0 + static_cast<size_t>(sx1) * 4;
            const uchar* p10 = src_row1 + static_cast<size_t>(sx0) * 4;
            const uchar* p11 = src_row1 + static_cast<size_t>(sx1) * 4;
            uchar* out = dst_row + static_cast<size_t>(x) * 4;

            const float top0 = lerp(static_cast<float>(p00[0]), static_cast<float>(p01[0]), wx_v);
            const float bot0 = lerp(static_cast<float>(p10[0]), static_cast<float>(p11[0]), wx_v);
            const float top1 = lerp(static_cast<float>(p00[1]), static_cast<float>(p01[1]), wx_v);
            const float bot1 = lerp(static_cast<float>(p10[1]), static_cast<float>(p11[1]), wx_v);
            const float top2 = lerp(static_cast<float>(p00[2]), static_cast<float>(p01[2]), wx_v);
            const float bot2 = lerp(static_cast<float>(p10[2]), static_cast<float>(p11[2]), wx_v);
            const float top3 = lerp(static_cast<float>(p00[3]), static_cast<float>(p01[3]), wx_v);
            const float bot3 = lerp(static_cast<float>(p10[3]), static_cast<float>(p11[3]), wx_v);
            out[0] = saturate_cast<uchar>(lerp(top0, bot0, wy_v));
            out[1] = saturate_cast<uchar>(lerp(top1, bot1, wy_v));
            out[2] = saturate_cast<uchar>(lerp(top2, bot2, wy_v));
            out[3] = saturate_cast<uchar>(lerp(top3, bot3, wy_v));
        }
    }
}

inline bool is_manual_threshold_type(int thresh_type)
{
    return thresh_type == THRESH_BINARY ||
           thresh_type == THRESH_BINARY_INV ||
           thresh_type == THRESH_TRUNC ||
           thresh_type == THRESH_TOZERO ||
           thresh_type == THRESH_TOZERO_INV;
}

inline std::array<uchar, 256> build_threshold_lut(int thresh_type,
                                                  double effective_thresh,
                                                  uchar max_u8,
                                                  uchar trunc_u8)
{
    std::array<uchar, 256> lut {};
    for (int i = 0; i < 256; ++i)
    {
        const uchar s = static_cast<uchar>(i);
        const bool cond = static_cast<double>(s) > effective_thresh;
        switch (thresh_type)
        {
            case THRESH_BINARY:
                lut[static_cast<size_t>(i)] = cond ? max_u8 : 0;
                break;
            case THRESH_BINARY_INV:
                lut[static_cast<size_t>(i)] = cond ? 0 : max_u8;
                break;
            case THRESH_TRUNC:
                lut[static_cast<size_t>(i)] = cond ? trunc_u8 : s;
                break;
            case THRESH_TOZERO:
                lut[static_cast<size_t>(i)] = cond ? s : 0;
                break;
            case THRESH_TOZERO_INV:
                lut[static_cast<size_t>(i)] = cond ? 0 : s;
                break;
            default:
                lut[static_cast<size_t>(i)] = 0;
                break;
        }
    }
    return lut;
}

}  // namespace

namespace {

thread_local const char* g_last_boxfilter_dispatch_path = "fallback";
thread_local const char* g_last_gaussianblur_dispatch_path = "fallback";

inline void set_last_boxfilter_dispatch_path(const char* path)
{
    g_last_boxfilter_dispatch_path = path ? path : "fallback";
}

inline void set_last_gaussianblur_dispatch_path(const char* path)
{
    g_last_gaussianblur_dispatch_path = path ? path : "fallback";
}

inline bool is_backend_filter_src_supported(const Mat& src)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        return false;
    }
    const int channels = src.channels();
    return channels == 1 || channels == 3 || channels == 4;
}

inline bool is_boxfilter_params_supported(int ddepth, Size ksize, Point anchor, int borderType)
{
    if (ddepth != -1 && ddepth != CV_8U)
    {
        return false;
    }
    if (ksize.width <= 0 || ksize.height <= 0)
    {
        return false;
    }

    const int anchor_x = anchor.x >= 0 ? anchor.x : (ksize.width / 2);
    const int anchor_y = anchor.y >= 0 ? anchor.y : (ksize.height / 2);
    if (anchor_x < 0 || anchor_x >= ksize.width || anchor_y < 0 || anchor_y >= ksize.height)
    {
        return false;
    }

    const int border_type = normalize_border_type(borderType);
    return is_supported_filter_border(border_type);
}

inline bool is_boxfilter_3x3_candidate(Size ksize, Point anchor, bool normalize)
{
    if (!normalize || ksize.width != 3 || ksize.height != 3)
    {
        return false;
    }

    const int anchor_x = anchor.x >= 0 ? anchor.x : (ksize.width / 2);
    const int anchor_y = anchor.y >= 0 ? anchor.y : (ksize.height / 2);
    return anchor_x == 1 && anchor_y == 1;
}

inline bool resolve_gaussian_params(Size ksize,
                                    double sigmaX,
                                    double sigmaY,
                                    int& kx,
                                    int& ky,
                                    double& resolved_sigma_x,
                                    double& resolved_sigma_y)
{
    kx = ksize.width;
    ky = ksize.height;

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

    resolved_sigma_x = sigmaX;
    resolved_sigma_y = sigmaY;
    return true;
}

inline void build_neighbor_index_maps(int len, int border_type, std::vector<int>& prev, std::vector<int>& next)
{
    prev.resize(static_cast<size_t>(len));
    next.resize(static_cast<size_t>(len));
    for (int i = 0; i < len; ++i)
    {
        const size_t idx = static_cast<size_t>(i);
        prev[idx] = border_interpolate(i - 1, len, border_type);
        next[idx] = border_interpolate(i + 1, len, border_type);
    }
}

inline void box3x3_u8(const Mat& src, Mat& dst, int border_type)
{
    const int rows = src.size[0];
    const int cols = src.size[1];
    const int channels = src.channels();
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows, cols}, src.type());
    const size_t dst_step = dst.step(0);

    std::vector<int> x_prev;
    std::vector<int> x_next;
    std::vector<int> y_prev;
    std::vector<int> y_next;
    build_neighbor_index_maps(cols, border_type, x_prev, x_next);
    build_neighbor_index_maps(rows, border_type, y_prev, y_next);

    const float inv_kernel_area = 1.0f / 9.0f;
    const long long rows_ll = static_cast<long long>(rows);
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(rows), static_cast<size_t>(cols) * static_cast<size_t>(channels), 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t yd = static_cast<size_t>(y);
        const int sy0 = y_prev[yd];
        const int sy1 = static_cast<int>(y);
        const int sy2 = y_next[yd];

        const uchar* row0 = sy0 >= 0 ? (src.data + static_cast<size_t>(sy0) * src_step) : nullptr;
        const uchar* row1 = src.data + static_cast<size_t>(sy1) * src_step;
        const uchar* row2 = sy2 >= 0 ? (src.data + static_cast<size_t>(sy2) * src_step) : nullptr;
        uchar* dst_row = dst.data + yd * dst_step;

        auto sample = [channels](const uchar* row, int sx, int c) -> int {
            if (!row || sx < 0)
            {
                return 0;
            }
            return static_cast<int>(row[static_cast<size_t>(sx) * channels + c]);
        };

        for (int x = 0; x < cols; ++x)
        {
            const size_t xd = static_cast<size_t>(x);
            const int sx0 = x_prev[xd];
            const int sx1 = x;
            const int sx2 = x_next[xd];
            uchar* out = dst_row + xd * static_cast<size_t>(channels);

            for (int c = 0; c < channels; ++c)
            {
                const int sum =
                    sample(row0, sx0, c) + sample(row0, sx1, c) + sample(row0, sx2, c) +
                    sample(row1, sx0, c) + sample(row1, sx1, c) + sample(row1, sx2, c) +
                    sample(row2, sx0, c) + sample(row2, sx1, c) + sample(row2, sx2, c);
                out[c] = saturate_cast<uchar>(static_cast<float>(sum) * inv_kernel_area);
            }
        }
    }
}

inline void gaussian3x3_u8(const Mat& src,
                           Mat& dst,
                           int border_type,
                           const std::array<float, 3>& kernel_x,
                           const std::array<float, 3>& kernel_y)
{
    const int rows = src.size[0];
    const int cols = src.size[1];
    const int channels = src.channels();
    const size_t src_step = src.step(0);

    dst.create(std::vector<int>{rows, cols}, src.type());
    const size_t dst_step = dst.step(0);

    std::vector<int> x_prev;
    std::vector<int> x_next;
    std::vector<int> y_prev;
    std::vector<int> y_next;
    build_neighbor_index_maps(cols, border_type, x_prev, x_next);
    build_neighbor_index_maps(rows, border_type, y_prev, y_next);

    std::vector<float> tmp(static_cast<size_t>(rows) * static_cast<size_t>(cols) * static_cast<size_t>(channels), 0.0f);

    const long long rows_ll = static_cast<long long>(rows);
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(rows), static_cast<size_t>(cols) * static_cast<size_t>(channels), 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t yd = static_cast<size_t>(y);
        const uchar* src_row = src.data + yd * src_step;
        for (int x = 0; x < cols; ++x)
        {
            const size_t xd = static_cast<size_t>(x);
            const int sx0 = x_prev[xd];
            const int sx1 = x;
            const int sx2 = x_next[xd];

            for (int c = 0; c < channels; ++c)
            {
                const float v0 = sx0 >= 0 ? static_cast<float>(src_row[static_cast<size_t>(sx0) * channels + c]) : 0.0f;
                const float v1 = static_cast<float>(src_row[static_cast<size_t>(sx1) * channels + c]);
                const float v2 = sx2 >= 0 ? static_cast<float>(src_row[static_cast<size_t>(sx2) * channels + c]) : 0.0f;
                const size_t tmp_idx = (yd * static_cast<size_t>(cols) + xd) * static_cast<size_t>(channels) + static_cast<size_t>(c);
                tmp[tmp_idx] = kernel_x[0] * v0 + kernel_x[1] * v1 + kernel_x[2] * v2;
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(rows), static_cast<size_t>(cols) * static_cast<size_t>(channels), 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t yd = static_cast<size_t>(y);
        const int sy0 = y_prev[yd];
        const int sy1 = static_cast<int>(y);
        const int sy2 = y_next[yd];
        uchar* dst_row = dst.data + yd * dst_step;

        for (int x = 0; x < cols; ++x)
        {
            const size_t xd = static_cast<size_t>(x);
            for (int c = 0; c < channels; ++c)
            {
                const size_t idx1 = (static_cast<size_t>(sy1) * static_cast<size_t>(cols) + xd) * static_cast<size_t>(channels) + static_cast<size_t>(c);
                const float t0 = sy0 >= 0 ? tmp[(static_cast<size_t>(sy0) * static_cast<size_t>(cols) + xd) * static_cast<size_t>(channels) + static_cast<size_t>(c)] : 0.0f;
                const float t1 = tmp[idx1];
                const float t2 = sy2 >= 0 ? tmp[(static_cast<size_t>(sy2) * static_cast<size_t>(cols) + xd) * static_cast<size_t>(channels) + static_cast<size_t>(c)] : 0.0f;
                const float out = kernel_y[0] * t0 + kernel_y[1] * t1 + kernel_y[2] * t2;
                dst_row[xd * static_cast<size_t>(channels) + static_cast<size_t>(c)] = saturate_cast<uchar>(out);
            }
        }
    }
}

inline void build_filter_index_table(int len, int ksize, int anchor, int border_type, std::vector<int>& table)
{
    table.resize(static_cast<size_t>(len) * static_cast<size_t>(ksize));
    for (int i = 0; i < len; ++i)
    {
        const size_t base = static_cast<size_t>(i) * static_cast<size_t>(ksize);
        for (int k = 0; k < ksize; ++k)
        {
            table[base + static_cast<size_t>(k)] = border_interpolate(i + k - anchor, len, border_type);
        }
    }
}

inline void box_filter_separable_u8(const Mat& src,
                                    Mat& dst,
                                    Size ksize,
                                    Point anchor,
                                    bool normalize,
                                    int border_type)
{
    const int rows = src.size[0];
    const int cols = src.size[1];
    const int channels = src.channels();
    const size_t src_step = src.step(0);
    const int anchor_x = anchor.x >= 0 ? anchor.x : (ksize.width / 2);
    const int anchor_y = anchor.y >= 0 ? anchor.y : (ksize.height / 2);

    dst.create(std::vector<int>{rows, cols}, src.type());
    const size_t dst_step = dst.step(0);

    std::vector<int> x_table;
    std::vector<int> y_table;
    build_filter_index_table(cols, ksize.width, anchor_x, border_type, x_table);
    build_filter_index_table(rows, ksize.height, anchor_y, border_type, y_table);

    std::vector<int> tmp(static_cast<size_t>(rows) * static_cast<size_t>(cols) * static_cast<size_t>(channels), 0);

    const long long rows_ll = static_cast<long long>(rows);
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(rows), static_cast<size_t>(cols) * static_cast<size_t>(channels), 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t yd = static_cast<size_t>(y);
        const uchar* src_row = src.data + yd * src_step;
        for (int x = 0; x < cols; ++x)
        {
            const size_t xd = static_cast<size_t>(x);
            const int* x_idx = x_table.data() + xd * static_cast<size_t>(ksize.width);
            for (int c = 0; c < channels; ++c)
            {
                int sum = 0;
                for (int k = 0; k < ksize.width; ++k)
                {
                    const int sx = x_idx[k];
                    if (sx >= 0)
                    {
                        sum += static_cast<int>(src_row[static_cast<size_t>(sx) * channels + c]);
                    }
                }
                const size_t tmp_idx = (yd * static_cast<size_t>(cols) + xd) * static_cast<size_t>(channels) + static_cast<size_t>(c);
                tmp[tmp_idx] = sum;
            }
        }
    }

    const int kernel_area = ksize.width * ksize.height;
    const float inv_kernel_area = kernel_area > 0 ? (1.0f / static_cast<float>(kernel_area)) : 0.0f;
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(rows), static_cast<size_t>(cols) * static_cast<size_t>(channels), 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t yd = static_cast<size_t>(y);
        const int* y_idx = y_table.data() + yd * static_cast<size_t>(ksize.height);
        uchar* dst_row = dst.data + yd * dst_step;

        for (int x = 0; x < cols; ++x)
        {
            const size_t xd = static_cast<size_t>(x);
            uchar* out = dst_row + xd * static_cast<size_t>(channels);
            for (int c = 0; c < channels; ++c)
            {
                int sum = 0;
                for (int k = 0; k < ksize.height; ++k)
                {
                    const int sy = y_idx[k];
                    if (sy >= 0)
                    {
                        const size_t tmp_idx =
                            (static_cast<size_t>(sy) * static_cast<size_t>(cols) + xd) * static_cast<size_t>(channels) +
                            static_cast<size_t>(c);
                        sum += tmp[tmp_idx];
                    }
                }
                if (normalize)
                {
                    out[c] = saturate_cast<uchar>(static_cast<float>(sum) * inv_kernel_area);
                }
                else
                {
                    out[c] = saturate_cast<uchar>(sum);
                }
            }
        }
    }
}

inline void gaussian_separable_u8(const Mat& src,
                                  Mat& dst,
                                  int kx,
                                  int ky,
                                  double sigmaX,
                                  double sigmaY,
                                  int border_type)
{
    const int rows = src.size[0];
    const int cols = src.size[1];
    const int channels = src.channels();
    const size_t src_step = src.step(0);
    const int anchor_x = kx / 2;
    const int anchor_y = ky / 2;

    dst.create(std::vector<int>{rows, cols}, src.type());
    const size_t dst_step = dst.step(0);

    const std::vector<float> kernel_x = build_gaussian_kernel_1d(kx, sigmaX);
    const std::vector<float> kernel_y = build_gaussian_kernel_1d(ky, sigmaY);

    std::vector<int> x_table;
    std::vector<int> y_table;
    build_filter_index_table(cols, kx, anchor_x, border_type, x_table);
    build_filter_index_table(rows, ky, anchor_y, border_type, y_table);

    std::vector<float> tmp(static_cast<size_t>(rows) * static_cast<size_t>(cols) * static_cast<size_t>(channels), 0.0f);

    const long long rows_ll = static_cast<long long>(rows);
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(rows), static_cast<size_t>(cols) * static_cast<size_t>(channels), 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t yd = static_cast<size_t>(y);
        const uchar* src_row = src.data + yd * src_step;
        for (int x = 0; x < cols; ++x)
        {
            const size_t xd = static_cast<size_t>(x);
            const int* x_idx = x_table.data() + xd * static_cast<size_t>(kx);
            for (int c = 0; c < channels; ++c)
            {
                float acc = 0.0f;
                for (int k = 0; k < kx; ++k)
                {
                    const int sx = x_idx[k];
                    if (sx >= 0)
                    {
                        acc += kernel_x[static_cast<size_t>(k)] *
                               static_cast<float>(src_row[static_cast<size_t>(sx) * channels + c]);
                    }
                }
                const size_t tmp_idx = (yd * static_cast<size_t>(cols) + xd) * static_cast<size_t>(channels) + static_cast<size_t>(c);
                tmp[tmp_idx] = acc;
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(rows), static_cast<size_t>(cols) * static_cast<size_t>(channels), 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t yd = static_cast<size_t>(y);
        const int* y_idx = y_table.data() + yd * static_cast<size_t>(ky);
        uchar* dst_row = dst.data + yd * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            const size_t xd = static_cast<size_t>(x);
            uchar* out = dst_row + xd * static_cast<size_t>(channels);
            for (int c = 0; c < channels; ++c)
            {
                float acc = 0.0f;
                for (int k = 0; k < ky; ++k)
                {
                    const int sy = y_idx[k];
                    if (sy >= 0)
                    {
                        const size_t tmp_idx =
                            (static_cast<size_t>(sy) * static_cast<size_t>(cols) + xd) * static_cast<size_t>(channels) +
                            static_cast<size_t>(c);
                        acc += kernel_y[static_cast<size_t>(k)] * tmp[tmp_idx];
                    }
                }
                out[c] = saturate_cast<uchar>(acc);
            }
        }
    }
}

}  // namespace

const char* last_boxfilter_dispatch_path()
{
    return g_last_boxfilter_dispatch_path;
}

const char* last_gaussianblur_dispatch_path()
{
    return g_last_gaussianblur_dispatch_path;
}

void resize_backend_impl(const Mat& src, Mat& dst, Size dsize, double fx, double fy, int interpolation)
{
    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        resize_fallback(src, dst, dsize, fx, fy, interpolation);
        return;
    }

    const int channels = src.channels();
    if (channels != 1 && channels != 3 && channels != 4)
    {
        resize_fallback(src, dst, dsize, fx, fy, interpolation);
        return;
    }

    const int src_rows = src.size[0];
    const int src_cols = src.size[1];
    const int dst_cols = detail::resolve_resize_dim(src_cols, dsize.width, fx);
    const int dst_rows = detail::resolve_resize_dim(src_rows, dsize.height, fy);
    if (dst_cols <= 0 || dst_rows <= 0)
    {
        resize_fallback(src, dst, dsize, fx, fy, interpolation);
        return;
    }

    if (interpolation != INTER_NEAREST &&
        interpolation != INTER_NEAREST_EXACT &&
        interpolation != INTER_LINEAR)
    {
        resize_fallback(src, dst, dsize, fx, fy, interpolation);
        return;
    }

    Mat src_local;
    const Mat* src_ref = &src;
    if (src.data == dst.data)
    {
        src_local = src.clone();
        src_ref = &src_local;
    }

    dst.create(std::vector<int>{dst_rows, dst_cols}, src_ref->type());

    if (interpolation == INTER_NEAREST || interpolation == INTER_NEAREST_EXACT)
    {
        std::vector<int> x_map;
        std::vector<int> y_map;
        const bool exact = interpolation == INTER_NEAREST_EXACT;
        build_nearest_index_map(src_ref->size[1], dst_cols, exact, x_map);
        build_nearest_index_map(src_ref->size[0], dst_rows, exact, y_map);

        if (channels == 1)
        {
            resize_nearest_u8_c1(*src_ref, dst, x_map, y_map);
            return;
        }
        if (channels == 3)
        {
            resize_nearest_u8_c3(*src_ref, dst, x_map, y_map);
            return;
        }

        resize_nearest_u8_c4(*src_ref, dst, x_map, y_map);
        return;
    }

    std::vector<int> x0, x1, y0, y1;
    std::vector<float> wx, wy;
    build_linear_coeffs(src_ref->size[1], dst_cols, x0, x1, wx);
    build_linear_coeffs(src_ref->size[0], dst_rows, y0, y1, wy);

    if (channels == 1)
    {
        resize_linear_u8_c1(*src_ref, dst, x0, x1, wx, y0, y1, wy);
        return;
    }
    if (channels == 3)
    {
        resize_linear_u8_c3(*src_ref, dst, x0, x1, wx, y0, y1, wy);
        return;
    }

    resize_linear_u8_c4(*src_ref, dst, x0, x1, wx, y0, y1, wy);
}

void cvtColor_backend_impl(const Mat& src, Mat& dst, int code)
{
    if (code != COLOR_BGR2GRAY && code != COLOR_GRAY2BGR)
    {
        cvtColor_fallback(src, dst, code);
        return;
    }

    if (src.empty() || src.dims != 2 || src.depth() != CV_8U)
    {
        cvtColor_fallback(src, dst, code);
        return;
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
    const size_t src_step = src_ref->step(0);

    if (code == COLOR_BGR2GRAY)
    {
        if (src_ref->channels() != 3)
        {
            cvtColor_fallback(src, dst, code);
            return;
        }

        dst.create(src_ref->dims, src_ref->size.p, CV_8UC1);
        const size_t dst_step = dst.step(0);

        const long long rows_ll = static_cast<long long>(rows);
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(rows), static_cast<size_t>(cols), 1LL << 16, 2))
#endif
        for (long long y = 0; y < rows_ll; ++y)
        {
            const size_t row_idx = static_cast<size_t>(y);
            const uchar* src_row = src_ref->data + row_idx * src_step;
            uchar* dst_row = dst.data + row_idx * dst_step;
            for (int x = 0; x < cols; ++x)
            {
                const uchar* px = src_row + static_cast<size_t>(x) * 3;
                const float gray = 0.114f * static_cast<float>(px[0]) +
                                   0.587f * static_cast<float>(px[1]) +
                                   0.299f * static_cast<float>(px[2]);
                dst_row[x] = saturate_cast<uchar>(gray);
            }
        }
        return;
    }

    if (src_ref->channels() != 1)
    {
        cvtColor_fallback(src, dst, code);
        return;
    }

    dst.create(src_ref->dims, src_ref->size.p, CV_8UC3);
    const size_t dst_step = dst.step(0);

    const long long rows_ll = static_cast<long long>(rows);
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(rows), static_cast<size_t>(cols), 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t row_idx = static_cast<size_t>(y);
        const uchar* src_row = src_ref->data + row_idx * src_step;
        uchar* dst_row = dst.data + row_idx * dst_step;
        for (int x = 0; x < cols; ++x)
        {
            const uchar g = src_row[x];
            uchar* out = dst_row + static_cast<size_t>(x) * 3;
            out[0] = g;
            out[1] = g;
            out[2] = g;
        }
    }
}

double threshold_backend_impl(const Mat& src, Mat& dst, double thresh, double maxval, int type)
{
    if (src.empty() || src.depth() != CV_8U)
    {
        return threshold_fallback(src, dst, thresh, maxval, type);
    }

    const bool is_dryrun = (type & THRESH_DRYRUN) != 0;
    const int normalized_type = type & (~THRESH_DRYRUN);
    const int automatic_thresh = normalized_type & (~THRESH_MASK);
    const int thresh_type = normalized_type & THRESH_MASK;
    if (automatic_thresh != 0 || !is_manual_threshold_type(thresh_type))
    {
        return threshold_fallback(src, dst, thresh, maxval, type);
    }

    const double effective_thresh = std::floor(thresh);
    if (is_dryrun)
    {
        return effective_thresh;
    }

    dst.create(src.dims, src.size.p, src.type());

    const uchar max_u8 = saturate_cast<uchar>(maxval);
    const uchar trunc_u8 = saturate_cast<uchar>(effective_thresh);
    const std::array<uchar, 256> lut = build_threshold_lut(thresh_type, effective_thresh, max_u8, trunc_u8);

    if (src.isContinuous() && dst.isContinuous())
    {
        const size_t scalar_count = src.total() * static_cast<size_t>(src.channels());
        const long long scalar_count_ll = static_cast<long long>(scalar_count);
        const uchar* src_ptr = src.data;
        uchar* dst_ptr = dst.data;
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(1, scalar_count, 1LL << 18, 2))
#endif
        for (long long i = 0; i < scalar_count_ll; ++i)
        {
            const size_t idx = static_cast<size_t>(i);
            dst_ptr[idx] = lut[src_ptr[idx]];
        }
        return effective_thresh;
    }

    if (src.dims != 2)
    {
        return threshold_fallback(src, dst, thresh, maxval, type);
    }

    const int rows = src.size[0];
    const int cols_scalar = src.size[1] * src.channels();
    const size_t src_step = src.step(0);
    const size_t dst_step = dst.step(0);

    const long long rows_ll = static_cast<long long>(rows);
#ifdef _OPENMP
#pragma omp parallel for if(cvh::cpu::should_parallelize_1d_loop(static_cast<size_t>(rows), static_cast<size_t>(cols_scalar), 1LL << 16, 2))
#endif
    for (long long y = 0; y < rows_ll; ++y)
    {
        const size_t row_idx = static_cast<size_t>(y);
        const uchar* src_row = src.data + row_idx * src_step;
        uchar* dst_row = dst.data + row_idx * dst_step;
        for (int x = 0; x < cols_scalar; ++x)
        {
            dst_row[x] = lut[src_row[x]];
        }
    }

    return effective_thresh;
}

void boxFilter_backend_impl(const Mat& src,
                            Mat& dst,
                            int ddepth,
                            Size ksize,
                            Point anchor,
                            bool normalize,
                            int borderType)
{
    set_last_boxfilter_dispatch_path("fallback");
    if (!is_backend_filter_src_supported(src))
    {
        boxFilter_fallback(src, dst, ddepth, ksize, anchor, normalize, borderType);
        return;
    }

    if (!is_boxfilter_params_supported(ddepth, ksize, anchor, borderType))
    {
        boxFilter_fallback(src, dst, ddepth, ksize, anchor, normalize, borderType);
        return;
    }

    Mat src_local;
    const Mat* src_ref = &src;
    if (src.data == dst.data)
    {
        src_local = src.clone();
        src_ref = &src_local;
    }

    const int border_type = normalize_border_type(borderType);

    if (is_boxfilter_3x3_candidate(ksize, anchor, normalize))
    {
        set_last_boxfilter_dispatch_path("box3x3");
        box3x3_u8(*src_ref, dst, border_type);
        return;
    }

    set_last_boxfilter_dispatch_path("box_generic");
    box_filter_separable_u8(*src_ref, dst, ksize, anchor, normalize, border_type);
}

void gaussianBlur_backend_impl(const Mat& src, Mat& dst, Size ksize, double sigmaX, double sigmaY, int borderType)
{
    set_last_gaussianblur_dispatch_path("fallback");
    if (!is_backend_filter_src_supported(src))
    {
        gaussian_blur_fallback(src, dst, ksize, sigmaX, sigmaY, borderType);
        return;
    }

    int kx = 0;
    int ky = 0;
    double resolved_sigma_x = 0.0;
    double resolved_sigma_y = 0.0;
    const int border_type = normalize_border_type(borderType);
    if (!is_supported_filter_border(border_type) ||
        !resolve_gaussian_params(ksize, sigmaX, sigmaY, kx, ky, resolved_sigma_x, resolved_sigma_y))
    {
        gaussian_blur_fallback(src, dst, ksize, sigmaX, sigmaY, borderType);
        return;
    }

    Mat src_local;
    const Mat* src_ref = &src;
    if (src.data == dst.data)
    {
        src_local = src.clone();
        src_ref = &src_local;
    }

    if (kx == 3 && ky == 3)
    {
        set_last_gaussianblur_dispatch_path("gauss3x3");
        const std::vector<float> kernel_x = build_gaussian_kernel_1d(3, resolved_sigma_x);
        const std::vector<float> kernel_y = build_gaussian_kernel_1d(3, resolved_sigma_y);
        const std::array<float, 3> kernel_x3 {
            kernel_x[0], kernel_x[1], kernel_x[2]
        };
        const std::array<float, 3> kernel_y3 {
            kernel_y[0], kernel_y[1], kernel_y[2]
        };
        gaussian3x3_u8(*src_ref, dst, border_type, kernel_x3, kernel_y3);
        return;
    }

    set_last_gaussianblur_dispatch_path("gauss_separable");
    gaussian_separable_u8(*src_ref, dst, kx, ky, resolved_sigma_x, resolved_sigma_y, border_type);
}

} // namespace detail

void register_all_backends()
{
    static bool initialized = []() {
        detail::register_resize_backend(&detail::resize_backend_impl);
        detail::register_cvtcolor_backend(&detail::cvtColor_backend_impl);
        detail::register_threshold_backend(&detail::threshold_backend_impl);
        detail::register_boxfilter_backend(&detail::boxFilter_backend_impl);
        detail::register_gaussianblur_backend(&detail::gaussianBlur_backend_impl);
        return true;
    }();
    (void)initialized;
}

} // namespace cvh
