#ifndef CVH_CORE_DETAIL_REDUCE_UI_HPP
#define CVH_CORE_DETAIL_REDUCE_UI_HPP

#include "dispatch_control.h"
#include "../mat.h"
#include "../simd/opencv_ui.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace cvh {
namespace detail {
namespace reduce_ui {

inline bool enabled()
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return cpu::opencv_ui_allowed();
#else
    return false;
#endif
}

struct SourceRows
{
    const uchar* data = nullptr;
    size_t step = 0;
    size_t row_scalars = 0;
    size_t rows = 0;
};

inline SourceRows source_rows(const Mat& src)
{
    SourceRows result;
    result.data = src.data;
    if (src.isContinuous())
    {
        result.step = src.total() * src.elemSize();
        result.row_scalars = src.total();
        result.rows = 1;
        return result;
    }

    result.rows = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    result.row_scalars =
        src.dims > 1 ? src.total(1, src.dims) : src.total();
    result.step = src.dims > 1
        ? src.step(0)
        : result.row_scalars * src.elemSize();
    return result;
}

struct PixelRows
{
    const uchar* data = nullptr;
    const uchar* mask = nullptr;
    size_t step = 0;
    size_t mask_step = 0;
    size_t row_pixels = 0;
    size_t rows = 0;
    int channels = 0;
};

inline PixelRows pixel_rows(const Mat& src, const Mat& mask)
{
    PixelRows result;
    result.data = src.data;
    result.mask = mask.empty() ? nullptr : mask.data;
    result.channels = src.channels();
    if (src.isContinuous() && (mask.empty() || mask.isContinuous()))
    {
        result.step = src.total() * src.elemSize();
        result.mask_step = mask.empty() ? 0 : mask.total();
        result.row_pixels = src.total();
        result.rows = 1;
        return result;
    }

    result.rows = src.dims > 1 ? static_cast<size_t>(src.size.p[0]) : 1;
    result.row_pixels =
        src.dims > 1 ? src.total(1, src.dims) : src.total();
    result.step = src.dims > 1
        ? src.step(0)
        : result.row_pixels * src.elemSize();
    result.mask_step = mask.empty()
        ? 0
        : (mask.dims > 1 ? mask.step(0) : result.row_pixels);
    return result;
}

struct PixelPairRows
{
    const uchar* first = nullptr;
    const uchar* second = nullptr;
    const uchar* mask = nullptr;
    size_t first_step = 0;
    size_t second_step = 0;
    size_t mask_step = 0;
    size_t row_pixels = 0;
    size_t rows = 0;
    int channels = 0;
};

inline PixelPairRows pixel_pair_rows(const Mat& first,
                                     const Mat& second,
                                     const Mat& mask)
{
    PixelPairRows result;
    result.first = first.data;
    result.second = second.data;
    result.mask = mask.empty() ? nullptr : mask.data;
    result.channels = first.channels();
    if (first.isContinuous() && second.isContinuous() &&
        (mask.empty() || mask.isContinuous()))
    {
        result.first_step = first.total() * first.elemSize();
        result.second_step = second.total() * second.elemSize();
        result.mask_step = mask.empty() ? 0 : mask.total();
        result.row_pixels = first.total();
        result.rows = 1;
        return result;
    }

    result.rows =
        first.dims > 1 ? static_cast<size_t>(first.size.p[0]) : 1;
    result.row_pixels =
        first.dims > 1 ? first.total(1, first.dims) : first.total();
    result.first_step = first.dims > 1
        ? first.step(0)
        : result.row_pixels * first.elemSize();
    result.second_step = second.dims > 1
        ? second.step(0)
        : result.row_pixels * second.elemSize();
    result.mask_step = mask.empty()
        ? 0
        : (mask.dims > 1 ? mask.step(0) : result.row_pixels);
    return result;
}

struct SumCount
{
    long double sums[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    size_t count = 0;
};

struct StableStatistics
{
    long double means[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    long double m2[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    size_t count = 0;
};

struct ExtremaResult
{
    bool found = false;
    double min_value = 0.0;
    double max_value = 0.0;
    size_t min_linear = 0;
    size_t max_linear = 0;
};

struct NormResult
{
    long double accumulator = 0.0L;
    double maximum = 0.0;
    bool has_nan = false;
};

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)

template<typename T>
struct VectorType;

template<>
struct VectorType<uchar>
{
    using type = cv::v_uint8;
};

template<>
struct VectorType<schar>
{
    using type = cv::v_int8;
};

template<>
struct VectorType<ushort>
{
    using type = cv::v_uint16;
};

template<>
struct VectorType<short>
{
    using type = cv::v_int16;
};

template<>
struct VectorType<uint>
{
    using type = cv::v_uint32;
};

template<>
struct VectorType<int>
{
    using type = cv::v_int32;
};

template<>
struct VectorType<float>
{
    using type = cv::v_float32;
};

#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
template<>
struct VectorType<double>
{
    using type = cv::v_float64;
};
#endif

inline long double horizontal_sum(const cv::v_uint8& value)
{
    return static_cast<long double>(cv::v_reduce_sum(value));
}

inline long double horizontal_sum(const cv::v_int8& value)
{
    return static_cast<long double>(cv::v_reduce_sum(value));
}

inline long double horizontal_sum(const cv::v_uint16& value)
{
    return static_cast<long double>(cv::v_reduce_sum(value));
}

inline long double horizontal_sum(const cv::v_int16& value)
{
    return static_cast<long double>(cv::v_reduce_sum(value));
}

inline long double horizontal_sum(const cv::v_uint32& value)
{
    cv::v_uint64 low;
    cv::v_uint64 high;
    cv::v_expand(value, low, high);
    return static_cast<long double>(cv::v_reduce_sum(low)) +
           static_cast<long double>(cv::v_reduce_sum(high));
}

inline long double horizontal_sum(const cv::v_int32& value)
{
    cv::v_int64 low;
    cv::v_int64 high;
    cv::v_expand(value, low, high);
    return static_cast<long double>(cv::v_reduce_sum(low)) +
           static_cast<long double>(cv::v_reduce_sum(high));
}

inline long double horizontal_sum(const cv::v_float32& value)
{
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
    return static_cast<long double>(
               cv::v_reduce_sum(cv::v_cvt_f64(value))) +
           static_cast<long double>(
               cv::v_reduce_sum(cv::v_cvt_f64_high(value)));
#else
    return static_cast<long double>(cv::v_reduce_sum(value));
#endif
}

#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
inline long double horizontal_sum(const cv::v_float64& value)
{
    return static_cast<long double>(cv::v_reduce_sum(value));
}
#endif

inline cv::v_uint8 broadcast_value(uchar value)
{
    return cv::vx_setall_u8(value);
}

inline cv::v_int8 broadcast_value(schar value)
{
    return cv::vx_setall_s8(value);
}

inline cv::v_uint16 broadcast_value(ushort value)
{
    return cv::vx_setall_u16(value);
}

inline cv::v_int16 broadcast_value(short value)
{
    return cv::vx_setall_s16(value);
}

inline cv::v_uint32 broadcast_value(uint value)
{
    return cv::vx_setall_u32(value);
}

inline cv::v_int32 broadcast_value(int value)
{
    return cv::vx_setall_s32(value);
}

inline cv::v_float32 broadcast_value(float value)
{
    return cv::vx_setall_f32(value);
}

#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
inline cv::v_float64 broadcast_value(double value)
{
    return cv::vx_setall_f64(value);
}
#endif

template<typename T>
inline bool extrema_value_is_valid(T value)
{
    return value == value;
}

template<typename T>
inline void update_extrema(ExtremaResult& result,
                           T value,
                           size_t linear_index)
{
    if (!extrema_value_is_valid(value))
    {
        return;
    }
    if (!result.found)
    {
        result.found = true;
        result.min_value = static_cast<double>(value);
        result.max_value = static_cast<double>(value);
        result.min_linear = linear_index;
        result.max_linear = linear_index;
        return;
    }
    if (value < static_cast<T>(result.min_value))
    {
        result.min_value = static_cast<double>(value);
        result.min_linear = linear_index;
    }
    if (value > static_cast<T>(result.max_value))
    {
        result.max_value = static_cast<double>(value);
        result.max_linear = linear_index;
    }
}

template<typename T, typename Vector>
inline bool extrema_pixel_range(const T* src,
                                const uchar* mask,
                                size_t pixels,
                                size_t linear_start,
                                ExtremaResult& result)
{
    const size_t lanes =
        static_cast<size_t>(cv::VTraits<Vector>::vlanes());
    size_t pixel = 0;
    bool used_vector_block = false;
    for (; pixel + lanes <= pixels; pixel += lanes)
    {
        if (mask != nullptr)
        {
            size_t selected = 0;
            for (size_t lane = 0; lane < lanes; ++lane)
            {
                selected += mask[pixel + lane] != 0 ? 1u : 0u;
            }
            used_vector_block = true;
            if (selected == 0)
            {
                continue;
            }
            if (selected != lanes)
            {
                for (size_t lane = 0; lane < lanes; ++lane)
                {
                    if (mask[pixel + lane] != 0)
                    {
                        update_extrema(
                            result,
                            src[pixel + lane],
                            linear_start + pixel + lane);
                    }
                }
                continue;
            }
        }

        const Vector values = cv::vx_load(src + pixel);
        const Vector min_values =
            broadcast_value(static_cast<T>(result.min_value));
        const Vector max_values =
            broadcast_value(static_cast<T>(result.max_value));
        const Vector lower = cv::v_lt(values, min_values);
        const Vector higher = cv::v_gt(values, max_values);
        used_vector_block = true;
        if (cv::v_check_any(lower) || cv::v_check_any(higher))
        {
            T lanes_data[cv::VTraits<Vector>::max_nlanes];
            cv::v_store(lanes_data, values);
            for (size_t lane = 0; lane < lanes; ++lane)
            {
                update_extrema(
                    result,
                    lanes_data[lane],
                    linear_start + pixel + lane);
            }
        }
    }
    for (; pixel < pixels; ++pixel)
    {
        if (mask == nullptr || mask[pixel] != 0)
        {
            update_extrema(
                result, src[pixel], linear_start + pixel);
        }
    }
    return used_vector_block;
}

template<typename T, typename Vector>
inline bool extrema_rows(const PixelRows& rows, ExtremaResult& result)
{
    size_t seed_row = 0;
    size_t seed_pixel = 0;
    bool seeded = false;
    for (; seed_row < rows.rows && !seeded; ++seed_row)
    {
        const T* src_row = reinterpret_cast<const T*>(
            rows.data + seed_row * rows.step);
        const uchar* mask_row =
            rows.mask == nullptr ? nullptr : rows.mask + seed_row * rows.mask_step;
        for (seed_pixel = 0; seed_pixel < rows.row_pixels; ++seed_pixel)
        {
            if ((mask_row == nullptr || mask_row[seed_pixel] != 0) &&
                extrema_value_is_valid(src_row[seed_pixel]))
            {
                update_extrema(
                    result,
                    src_row[seed_pixel],
                    seed_row * rows.row_pixels + seed_pixel);
                seeded = true;
                break;
            }
        }
    }
    if (!seeded)
    {
        return false;
    }
    --seed_row;

    bool used_vector_block = false;
    for (size_t row = seed_row; row < rows.rows; ++row)
    {
        const size_t start = row == seed_row ? seed_pixel + 1 : 0;
        if (start >= rows.row_pixels)
        {
            continue;
        }
        const T* src_row = reinterpret_cast<const T*>(
            rows.data + row * rows.step);
        const uchar* mask_row =
            rows.mask == nullptr ? nullptr : rows.mask + row * rows.mask_step;
        used_vector_block =
            extrema_pixel_range<T, Vector>(
                src_row + start,
                mask_row == nullptr ? nullptr : mask_row + start,
                rows.row_pixels - start,
                row * rows.row_pixels + start,
                result) ||
            used_vector_block;
    }
    cv::v_cleanup();
    return used_vector_block;
}

template<bool FindMax, typename T>
inline bool arg_better(T value, T best, bool last_index)
{
    if (FindMax)
    {
        return last_index ? value >= best : value > best;
    }
    return last_index ? value <= best : value < best;
}

template<bool FindMax, typename Vector>
inline Vector arg_compare(const Vector& value,
                          const Vector& best,
                          bool last_index)
{
    if (FindMax)
    {
        return last_index ? cv::v_ge(value, best) : cv::v_gt(value, best);
    }
    return last_index ? cv::v_le(value, best) : cv::v_lt(value, best);
}

template<typename Vector>
inline bool mask_lane_is_set(
    const typename cv::VTraits<Vector>::lane_type& value)
{
    return value !=
           static_cast<typename cv::VTraits<Vector>::lane_type>(0);
}

template<bool FindMax, typename T, typename Vector>
inline bool reduce_arg_axis1(const Mat& src,
                             Mat& dst,
                             bool last_index)
{
    const int lanes = cv::VTraits<Vector>::vlanes();
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    if (cols < lanes)
    {
        return false;
    }

    for (int row = 0; row < rows; ++row)
    {
        const T* src_row = reinterpret_cast<const T*>(
            src.data + static_cast<size_t>(row) * src.step(0));
        Vector best_values = cv::vx_load(src_row);
        int best_indices[cv::VTraits<Vector>::max_nlanes];
        for (int lane = 0; lane < lanes; ++lane)
        {
            best_indices[lane] = lane;
        }

        int column = lanes;
        for (; column + lanes <= cols; column += lanes)
        {
            const Vector values = cv::vx_load(src_row + column);
            const Vector better =
                arg_compare<FindMax>(values, best_values, last_index);
            if (cv::v_check_any(better))
            {
                best_values = cv::v_select(
                    better, values, best_values);
                typename cv::VTraits<Vector>::lane_type
                    mask_data[cv::VTraits<Vector>::max_nlanes];
                cv::v_store(mask_data, better);
                for (int lane = 0; lane < lanes; ++lane)
                {
                    if (mask_lane_is_set<Vector>(mask_data[lane]))
                    {
                        best_indices[lane] = column + lane;
                    }
                }
            }
        }

        T values_data[cv::VTraits<Vector>::max_nlanes];
        cv::v_store(values_data, best_values);
        T best_value = values_data[0];
        int best_index = best_indices[0];
        for (int lane = 1; lane < lanes; ++lane)
        {
            const bool better =
                FindMax ? values_data[lane] > best_value
                        : values_data[lane] < best_value;
            const bool equal_tie =
                values_data[lane] == best_value &&
                (last_index ? best_indices[lane] > best_index
                            : best_indices[lane] < best_index);
            if (better || equal_tie)
            {
                best_value = values_data[lane];
                best_index = best_indices[lane];
            }
        }
        for (; column < cols; ++column)
        {
            if (arg_better<FindMax>(
                    src_row[column], best_value, last_index))
            {
                best_value = src_row[column];
                best_index = column;
            }
        }
        reinterpret_cast<int*>(
            dst.data + static_cast<size_t>(row) * dst.step(0))[0] =
            best_index;
    }
    cv::v_cleanup();
    return true;
}

template<bool FindMax, typename T, typename Vector>
inline bool reduce_arg_axis0(const Mat& src,
                             Mat& dst,
                             bool last_index)
{
    const int lanes = cv::VTraits<Vector>::vlanes();
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    if (cols < lanes)
    {
        return false;
    }

    int* dst_data = reinterpret_cast<int*>(dst.data);
    std::fill(dst_data, dst_data + cols, 0);
    const T* first_row = reinterpret_cast<const T*>(src.data);
    int column = 0;
    for (; column + lanes <= cols; column += lanes)
    {
        Vector best_values = cv::vx_load(first_row + column);
        for (int row = 1; row < rows; ++row)
        {
            const T* src_row = reinterpret_cast<const T*>(
                src.data + static_cast<size_t>(row) * src.step(0));
            const Vector values = cv::vx_load(src_row + column);
            const Vector better =
                arg_compare<FindMax>(values, best_values, last_index);
            if (cv::v_check_any(better))
            {
                best_values = cv::v_select(
                    better, values, best_values);
                typename cv::VTraits<Vector>::lane_type
                    mask_data[cv::VTraits<Vector>::max_nlanes];
                cv::v_store(mask_data, better);
                for (int lane = 0; lane < lanes; ++lane)
                {
                    if (mask_lane_is_set<Vector>(mask_data[lane]))
                    {
                        dst_data[column + lane] = row;
                    }
                }
            }
        }
    }
    for (; column < cols; ++column)
    {
        T best_value = first_row[column];
        int best_index = 0;
        for (int row = 1; row < rows; ++row)
        {
            const T* src_row = reinterpret_cast<const T*>(
                src.data + static_cast<size_t>(row) * src.step(0));
            if (arg_better<FindMax>(
                    src_row[column], best_value, last_index))
            {
                best_value = src_row[column];
                best_index = row;
            }
        }
        dst_data[column] = best_index;
    }
    cv::v_cleanup();
    return true;
}

template<bool FindMax, typename T, typename Vector>
inline bool reduce_arg_typed(const Mat& src,
                             Mat& dst,
                             int axis,
                             bool last_index)
{
    return axis == 0
        ? reduce_arg_axis0<FindMax, T, Vector>(
              src, dst, last_index)
        : reduce_arg_axis1<FindMax, T, Vector>(
              src, dst, last_index);
}

template<typename Vector, typename Fn>
inline void for_each_channel_vector(
    const typename cv::VTraits<Vector>::lane_type* src,
    int channels,
    Fn&& fn)
{
    if (channels == 1)
    {
        fn(0, cv::vx_load(src));
        return;
    }

    Vector channel0;
    Vector channel1;
    if (channels == 2)
    {
        cv::v_load_deinterleave(src, channel0, channel1);
        fn(0, channel0);
        fn(1, channel1);
        return;
    }

    Vector channel2;
    if (channels == 3)
    {
        cv::v_load_deinterleave(
            src, channel0, channel1, channel2);
        fn(0, channel0);
        fn(1, channel1);
        fn(2, channel2);
        return;
    }

    Vector channel3;
    cv::v_load_deinterleave(
        src, channel0, channel1, channel2, channel3);
    fn(0, channel0);
    fn(1, channel1);
    fn(2, channel2);
    fn(3, channel3);
}

inline long double horizontal_square_sum(const cv::v_uint8& value)
{
    return static_cast<long double>(
        cv::v_reduce_sum(cv::v_dotprod_expand_fast(
            value, value, cv::vx_setzero_u32())));
}

inline long double horizontal_square_sum(const cv::v_float32& value)
{
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
    const cv::v_float64 low = cv::v_cvt_f64(value);
    const cv::v_float64 high = cv::v_cvt_f64_high(value);
    return static_cast<long double>(
        cv::v_reduce_sum(cv::v_mul(low, low))) +
           static_cast<long double>(
               cv::v_reduce_sum(cv::v_mul(high, high)));
#else
    return static_cast<long double>(
        cv::v_reduce_sum(cv::v_mul(value, value)));
#endif
}

inline bool row_has_nan(const uchar*, size_t)
{
    return false;
}

inline bool row_has_nan(const float* src, size_t length)
{
    const size_t lanes =
        static_cast<size_t>(cv::VTraits<cv::v_float32>::vlanes());
    size_t x = 0;
    for (; x + lanes <= length; x += lanes)
    {
        if (!cv::v_check_all(cv::v_not_nan(cv::vx_load(src + x))))
        {
            return true;
        }
    }
    for (; x < length; ++x)
    {
        if (std::isnan(src[x]))
        {
            return true;
        }
    }
    return false;
}

template<typename T, typename WriteFn>
inline void reduce_axis1_scalar_row(const T* src,
                                    int row,
                                    int cols,
                                    int channels,
                                    int rtype,
                                    WriteFn& write)
{
    long double accumulators[4] = {0.0L, 0.0L, 0.0L, 0.0L};
    T extrema[4] = {};
    if (rtype == REDUCE_MAX || rtype == REDUCE_MIN)
    {
        for (int channel = 0; channel < channels; ++channel)
        {
            extrema[channel] = src[channel];
        }
    }

    for (int pixel = 0; pixel < cols; ++pixel)
    {
        const size_t offset =
            static_cast<size_t>(pixel) * static_cast<size_t>(channels);
        for (int channel = 0; channel < channels; ++channel)
        {
            const T value =
                src[offset + static_cast<size_t>(channel)];
            if (rtype == REDUCE_MAX)
            {
                if (value > extrema[channel])
                {
                    extrema[channel] = value;
                }
            }
            else if (rtype == REDUCE_MIN)
            {
                if (value < extrema[channel])
                {
                    extrema[channel] = value;
                }
            }
            else if (rtype == REDUCE_SUM2)
            {
                const long double wide = static_cast<long double>(value);
                accumulators[channel] += wide * wide;
            }
            else
            {
                accumulators[channel] +=
                    static_cast<long double>(value);
            }
        }
    }

    for (int channel = 0; channel < channels; ++channel)
    {
        long double value =
            rtype == REDUCE_MAX || rtype == REDUCE_MIN
                ? static_cast<long double>(extrema[channel])
                : accumulators[channel];
        if (rtype == REDUCE_AVG)
        {
            value /= static_cast<long double>(cols);
        }
        write(row, channel, value);
    }
}

#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
template<typename WriteFn>
inline bool reduce_axis1_f32_accumulation(const Mat& src,
                                          int rtype,
                                          WriteFn& write)
{
    const int lanes = cv::VTraits<cv::v_float32>::vlanes();
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int channels = src.channels();
    if (cols < lanes)
    {
        return false;
    }

    for (int row = 0; row < rows; ++row)
    {
        const float* src_row = reinterpret_cast<const float*>(
            src.data + static_cast<size_t>(row) * src.step(0));
        cv::v_float64 accumulator_low[4];
        cv::v_float64 accumulator_high[4];
        for (int channel = 0; channel < channels; ++channel)
        {
            accumulator_low[channel] = cv::vx_setzero_f64();
            accumulator_high[channel] = cv::vx_setzero_f64();
        }

        int pixel = 0;
        for (; pixel + lanes <= cols; pixel += lanes)
        {
            const float* block =
                src_row +
                static_cast<size_t>(pixel) *
                    static_cast<size_t>(channels);
            for_each_channel_vector<cv::v_float32>(
                block,
                channels,
                [&](int channel,
                    const cv::v_float32& values) {
                    const cv::v_float64 low =
                        cv::v_cvt_f64(values);
                    const cv::v_float64 high =
                        cv::v_cvt_f64_high(values);
                    if (rtype == REDUCE_SUM2)
                    {
                        accumulator_low[channel] = cv::v_fma(
                            low,
                            low,
                            accumulator_low[channel]);
                        accumulator_high[channel] = cv::v_fma(
                            high,
                            high,
                            accumulator_high[channel]);
                    }
                    else
                    {
                        accumulator_low[channel] = cv::v_add(
                            accumulator_low[channel], low);
                        accumulator_high[channel] = cv::v_add(
                            accumulator_high[channel], high);
                    }
                });
        }

        long double totals[4] = {0.0L, 0.0L, 0.0L, 0.0L};
        for (int channel = 0; channel < channels; ++channel)
        {
            totals[channel] = static_cast<long double>(
                cv::v_reduce_sum(cv::v_add(
                    accumulator_low[channel],
                    accumulator_high[channel])));
        }
        for (; pixel < cols; ++pixel)
        {
            const size_t offset =
                static_cast<size_t>(pixel) *
                static_cast<size_t>(channels);
            for (int channel = 0; channel < channels; ++channel)
            {
                const long double value =
                    static_cast<long double>(
                        src_row[
                            offset +
                            static_cast<size_t>(channel)]);
                totals[channel] += rtype == REDUCE_SUM2
                    ? value * value
                    : value;
            }
        }

        for (int channel = 0; channel < channels; ++channel)
        {
            long double value = totals[channel];
            if (rtype == REDUCE_AVG)
            {
                value /= static_cast<long double>(cols);
            }
            write(row, channel, value);
        }
    }
    cv::v_cleanup();
    return true;
}
#endif

template<typename T, typename Vector, typename WriteFn>
inline bool reduce_axis1(const Mat& src,
                         int rtype,
                         WriteFn& write)
{
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
    if constexpr (std::is_same<T, float>::value)
    {
        if (rtype != REDUCE_MAX && rtype != REDUCE_MIN)
        {
            return reduce_axis1_f32_accumulation(
                src, rtype, write);
        }
    }
#endif

    const int lanes = cv::VTraits<Vector>::vlanes();
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int channels = src.channels();
    if (cols < lanes)
    {
        return false;
    }

    if (channels == 1 &&
        (rtype == REDUCE_MAX || rtype == REDUCE_MIN))
    {
        for (int row = 0; row < rows; ++row)
        {
            const T* src_row = reinterpret_cast<const T*>(
                src.data + static_cast<size_t>(row) * src.step(0));
            if (row_has_nan(src_row, static_cast<size_t>(cols)))
            {
                reduce_axis1_scalar_row(
                    src_row, row, cols, channels, rtype, write);
                continue;
            }
            Vector best = cv::vx_load(src_row);
            int pixel = lanes;
            for (; pixel + lanes <= cols; pixel += lanes)
            {
                const Vector values =
                    cv::vx_load(src_row + pixel);
                best = rtype == REDUCE_MAX
                    ? cv::v_max(best, values)
                    : cv::v_min(best, values);
            }
            T extrema = rtype == REDUCE_MAX
                ? cv::v_reduce_max(best)
                : cv::v_reduce_min(best);
            for (; pixel < cols; ++pixel)
            {
                if ((rtype == REDUCE_MAX &&
                     src_row[pixel] > extrema) ||
                    (rtype == REDUCE_MIN &&
                     src_row[pixel] < extrema))
                {
                    extrema = src_row[pixel];
                }
            }
            write(row, 0, static_cast<long double>(extrema));
        }
        cv::v_cleanup();
        return true;
    }

    bool used_vector_block = false;
    for (int row = 0; row < rows; ++row)
    {
        const T* src_row = reinterpret_cast<const T*>(
            src.data + static_cast<size_t>(row) * src.step(0));
        if ((rtype == REDUCE_MAX || rtype == REDUCE_MIN) &&
            row_has_nan(
                src_row,
                static_cast<size_t>(cols) *
                    static_cast<size_t>(channels)))
        {
            reduce_axis1_scalar_row(
                src_row, row, cols, channels, rtype, write);
            continue;
        }

        long double accumulators[4] =
            {0.0L, 0.0L, 0.0L, 0.0L};
        T extrema[4] = {};
        if (rtype == REDUCE_MAX || rtype == REDUCE_MIN)
        {
            for (int channel = 0; channel < channels; ++channel)
            {
                extrema[channel] = src_row[channel];
            }
        }

        int pixel = 0;
        for (; pixel + lanes <= cols; pixel += lanes)
        {
            const T* block =
                src_row +
                static_cast<size_t>(pixel) *
                    static_cast<size_t>(channels);
            for_each_channel_vector<Vector>(
                block,
                channels,
                [&](int channel, const Vector& values) {
                    if (rtype == REDUCE_MAX)
                    {
                        const T candidate = cv::v_reduce_max(values);
                        if (candidate > extrema[channel])
                        {
                            extrema[channel] = candidate;
                        }
                    }
                    else if (rtype == REDUCE_MIN)
                    {
                        const T candidate = cv::v_reduce_min(values);
                        if (candidate < extrema[channel])
                        {
                            extrema[channel] = candidate;
                        }
                    }
                    else if (rtype == REDUCE_SUM2)
                    {
                        accumulators[channel] +=
                            horizontal_square_sum(values);
                    }
                    else
                    {
                        accumulators[channel] +=
                            horizontal_sum(values);
                    }
                });
            used_vector_block = true;
        }

        for (; pixel < cols; ++pixel)
        {
            const size_t offset =
                static_cast<size_t>(pixel) *
                static_cast<size_t>(channels);
            for (int channel = 0; channel < channels; ++channel)
            {
                const T value =
                    src_row[offset + static_cast<size_t>(channel)];
                if (rtype == REDUCE_MAX)
                {
                    if (value > extrema[channel])
                    {
                        extrema[channel] = value;
                    }
                }
                else if (rtype == REDUCE_MIN)
                {
                    if (value < extrema[channel])
                    {
                        extrema[channel] = value;
                    }
                }
                else if (rtype == REDUCE_SUM2)
                {
                    const long double wide =
                        static_cast<long double>(value);
                    accumulators[channel] += wide * wide;
                }
                else
                {
                    accumulators[channel] +=
                        static_cast<long double>(value);
                }
            }
        }

        for (int channel = 0; channel < channels; ++channel)
        {
            long double value =
                rtype == REDUCE_MAX || rtype == REDUCE_MIN
                    ? static_cast<long double>(extrema[channel])
                    : accumulators[channel];
            if (rtype == REDUCE_AVG)
            {
                value /= static_cast<long double>(cols);
            }
            write(row, channel, value);
        }
    }
    cv::v_cleanup();
    return used_vector_block;
}

template<typename WriteFn>
inline bool reduce_axis0_u8(const Mat& src,
                            int rtype,
                            WriteFn& write)
{
    using Vector = cv::v_uint8;
    const int lanes = cv::VTraits<Vector>::vlanes();
    const int rows = src.size.p[0];
    const int channels = src.channels();
    const int width = src.size.p[1] * channels;
    if (width < lanes)
    {
        return false;
    }

    int scalar = 0;
    if (rtype == REDUCE_MAX || rtype == REDUCE_MIN)
    {
        const uchar* first_row =
            reinterpret_cast<const uchar*>(src.data);
        for (; scalar + lanes <= width; scalar += lanes)
        {
            Vector best = cv::vx_load(first_row + scalar);
            for (int row = 1; row < rows; ++row)
            {
                const uchar* src_row =
                    reinterpret_cast<const uchar*>(
                        src.data +
                        static_cast<size_t>(row) * src.step(0));
                const Vector values =
                    cv::vx_load(src_row + scalar);
                const Vector better = rtype == REDUCE_MAX
                    ? cv::v_gt(values, best)
                    : cv::v_lt(values, best);
                best = cv::v_select(better, values, best);
            }
            uchar values[cv::VTraits<Vector>::max_nlanes];
            cv::v_store(values, best);
            for (int lane = 0; lane < lanes; ++lane)
            {
                const int output_scalar = scalar + lane;
                write(
                    output_scalar / channels,
                    output_scalar % channels,
                    static_cast<long double>(values[lane]));
            }
        }
    }
    else
    {
        const int wide_lanes =
            cv::VTraits<cv::v_uint32>::vlanes();
        const int rows_per_chunk = 65535;
        for (; scalar + lanes <= width; scalar += lanes)
        {
            long double totals[cv::VTraits<Vector>::max_nlanes] = {};
            for (int row_begin = 0;
                 row_begin < rows;
                 row_begin += rows_per_chunk)
            {
                const int row_end =
                    std::min(rows, row_begin + rows_per_chunk);
                cv::v_uint32 sum0 = cv::vx_setzero_u32();
                cv::v_uint32 sum1 = cv::vx_setzero_u32();
                cv::v_uint32 sum2 = cv::vx_setzero_u32();
                cv::v_uint32 sum3 = cv::vx_setzero_u32();
                for (int row = row_begin; row < row_end; ++row)
                {
                    const uchar* src_row =
                        reinterpret_cast<const uchar*>(
                            src.data +
                            static_cast<size_t>(row) * src.step(0));
                    cv::v_uint16 low16;
                    cv::v_uint16 high16;
                    cv::v_expand(
                        cv::vx_load(src_row + scalar),
                        low16,
                        high16);
                    if (rtype == REDUCE_SUM2)
                    {
                        low16 = cv::v_mul(low16, low16);
                        high16 = cv::v_mul(high16, high16);
                    }
                    cv::v_uint32 part0;
                    cv::v_uint32 part1;
                    cv::v_uint32 part2;
                    cv::v_uint32 part3;
                    cv::v_expand(low16, part0, part1);
                    cv::v_expand(high16, part2, part3);
                    sum0 = cv::v_add(sum0, part0);
                    sum1 = cv::v_add(sum1, part1);
                    sum2 = cv::v_add(sum2, part2);
                    sum3 = cv::v_add(sum3, part3);
                }

                uint values0[cv::VTraits<cv::v_uint32>::max_nlanes];
                uint values1[cv::VTraits<cv::v_uint32>::max_nlanes];
                uint values2[cv::VTraits<cv::v_uint32>::max_nlanes];
                uint values3[cv::VTraits<cv::v_uint32>::max_nlanes];
                cv::v_store(values0, sum0);
                cv::v_store(values1, sum1);
                cv::v_store(values2, sum2);
                cv::v_store(values3, sum3);
                for (int lane = 0; lane < wide_lanes; ++lane)
                {
                    totals[lane] += values0[lane];
                    totals[wide_lanes + lane] += values1[lane];
                    totals[2 * wide_lanes + lane] += values2[lane];
                    totals[3 * wide_lanes + lane] += values3[lane];
                }
            }

            for (int lane = 0; lane < lanes; ++lane)
            {
                long double value = totals[lane];
                if (rtype == REDUCE_AVG)
                {
                    value /= static_cast<long double>(rows);
                }
                const int output_scalar = scalar + lane;
                write(
                    output_scalar / channels,
                    output_scalar % channels,
                    value);
            }
        }
    }

    for (; scalar < width; ++scalar)
    {
        long double accumulator = 0.0L;
        uchar extrema =
            reinterpret_cast<const uchar*>(src.data)[scalar];
        for (int row = 0; row < rows; ++row)
        {
            const uchar* src_row =
                reinterpret_cast<const uchar*>(
                    src.data +
                    static_cast<size_t>(row) * src.step(0));
            const uchar value = src_row[scalar];
            if (rtype == REDUCE_MAX)
            {
                if (value > extrema)
                {
                    extrema = value;
                }
            }
            else if (rtype == REDUCE_MIN)
            {
                if (value < extrema)
                {
                    extrema = value;
                }
            }
            else if (rtype == REDUCE_SUM2)
            {
                accumulator +=
                    static_cast<long double>(value) * value;
            }
            else
            {
                accumulator += value;
            }
        }
        long double result =
            rtype == REDUCE_MAX || rtype == REDUCE_MIN
                ? static_cast<long double>(extrema)
                : accumulator;
        if (rtype == REDUCE_AVG)
        {
            result /= static_cast<long double>(rows);
        }
        write(scalar / channels, scalar % channels, result);
    }
    cv::v_cleanup();
    return true;
}

template<typename WriteFn>
inline bool reduce_axis0_f32(const Mat& src,
                             int rtype,
                             WriteFn& write)
{
    using Vector = cv::v_float32;
    const int lanes = cv::VTraits<Vector>::vlanes();
    const int rows = src.size.p[0];
    const int channels = src.channels();
    const int width = src.size.p[1] * channels;
    if (width < lanes)
    {
        return false;
    }

    int scalar = 0;
    if (rtype == REDUCE_MAX || rtype == REDUCE_MIN)
    {
        const float* first_row =
            reinterpret_cast<const float*>(src.data);
        const int unroll = 8;
        for (; scalar + unroll * lanes <= width;
             scalar += unroll * lanes)
        {
            Vector best[unroll];
            for (int block = 0; block < unroll; ++block)
            {
                best[block] = cv::vx_load(
                    first_row + scalar + block * lanes);
            }
            for (int row = 1; row < rows; ++row)
            {
                const float* src_row =
                    reinterpret_cast<const float*>(
                        src.data +
                        static_cast<size_t>(row) * src.step(0));
                for (int block = 0; block < unroll; ++block)
                {
                    const Vector values = cv::vx_load(
                        src_row + scalar + block * lanes);
                    const Vector better = rtype == REDUCE_MAX
                        ? cv::v_gt(values, best[block])
                        : cv::v_lt(values, best[block]);
                    best[block] = cv::v_select(
                        better, values, best[block]);
                }
            }
            for (int block = 0; block < unroll; ++block)
            {
                float values[cv::VTraits<Vector>::max_nlanes];
                cv::v_store(values, best[block]);
                for (int lane = 0; lane < lanes; ++lane)
                {
                    const int output_scalar =
                        scalar + block * lanes + lane;
                    write(
                        output_scalar / channels,
                        output_scalar % channels,
                        static_cast<long double>(values[lane]));
                }
            }
        }
        for (; scalar + lanes <= width; scalar += lanes)
        {
            Vector best = cv::vx_load(first_row + scalar);
            for (int row = 1; row < rows; ++row)
            {
                const float* src_row =
                    reinterpret_cast<const float*>(
                        src.data +
                        static_cast<size_t>(row) * src.step(0));
                const Vector values =
                    cv::vx_load(src_row + scalar);
                const Vector better = rtype == REDUCE_MAX
                    ? cv::v_gt(values, best)
                    : cv::v_lt(values, best);
                best = cv::v_select(better, values, best);
            }
            float values[cv::VTraits<Vector>::max_nlanes];
            cv::v_store(values, best);
            for (int lane = 0; lane < lanes; ++lane)
            {
                const int output_scalar = scalar + lane;
                write(
                    output_scalar / channels,
                    output_scalar % channels,
                    static_cast<long double>(values[lane]));
            }
        }
    }
    else
    {
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        const int wide_lanes =
            cv::VTraits<cv::v_float64>::vlanes();
        const int unroll = 4;
        for (; scalar + unroll * lanes <= width;
             scalar += unroll * lanes)
        {
            cv::v_float64 accumulator_low[unroll];
            cv::v_float64 accumulator_high[unroll];
            for (int block = 0; block < unroll; ++block)
            {
                accumulator_low[block] = cv::vx_setzero_f64();
                accumulator_high[block] = cv::vx_setzero_f64();
            }
            for (int row = 0; row < rows; ++row)
            {
                const float* src_row =
                    reinterpret_cast<const float*>(
                        src.data +
                        static_cast<size_t>(row) * src.step(0));
                for (int block = 0; block < unroll; ++block)
                {
                    const Vector values = cv::vx_load(
                        src_row + scalar + block * lanes);
                    const cv::v_float64 low =
                        cv::v_cvt_f64(values);
                    const cv::v_float64 high =
                        cv::v_cvt_f64_high(values);
                    if (rtype == REDUCE_SUM2)
                    {
                        accumulator_low[block] = cv::v_fma(
                            low,
                            low,
                            accumulator_low[block]);
                        accumulator_high[block] = cv::v_fma(
                            high,
                            high,
                            accumulator_high[block]);
                    }
                    else
                    {
                        accumulator_low[block] = cv::v_add(
                            accumulator_low[block], low);
                        accumulator_high[block] = cv::v_add(
                            accumulator_high[block], high);
                    }
                }
            }
            for (int block = 0; block < unroll; ++block)
            {
                double low_values[
                    cv::VTraits<cv::v_float64>::max_nlanes];
                double high_values[
                    cv::VTraits<cv::v_float64>::max_nlanes];
                cv::v_store(
                    low_values, accumulator_low[block]);
                cv::v_store(
                    high_values, accumulator_high[block]);
                for (int lane = 0; lane < lanes; ++lane)
                {
                    long double value = lane < wide_lanes
                        ? static_cast<long double>(
                              low_values[lane])
                        : static_cast<long double>(
                              high_values[lane - wide_lanes]);
                    if (rtype == REDUCE_AVG)
                    {
                        value /= static_cast<long double>(rows);
                    }
                    const int output_scalar =
                        scalar + block * lanes + lane;
                    write(
                        output_scalar / channels,
                        output_scalar % channels,
                        value);
                }
            }
        }
        for (; scalar + lanes <= width; scalar += lanes)
        {
            cv::v_float64 accumulator_low =
                cv::vx_setzero_f64();
            cv::v_float64 accumulator_high =
                cv::vx_setzero_f64();
            for (int row = 0; row < rows; ++row)
            {
                const float* src_row =
                    reinterpret_cast<const float*>(
                        src.data +
                        static_cast<size_t>(row) * src.step(0));
                const Vector values =
                    cv::vx_load(src_row + scalar);
                const cv::v_float64 low =
                    cv::v_cvt_f64(values);
                const cv::v_float64 high =
                    cv::v_cvt_f64_high(values);
                if (rtype == REDUCE_SUM2)
                {
                    accumulator_low = cv::v_fma(
                        low, low, accumulator_low);
                    accumulator_high = cv::v_fma(
                        high, high, accumulator_high);
                }
                else
                {
                    accumulator_low =
                        cv::v_add(accumulator_low, low);
                    accumulator_high =
                        cv::v_add(accumulator_high, high);
                }
            }

            double low_values[
                cv::VTraits<cv::v_float64>::max_nlanes];
            double high_values[
                cv::VTraits<cv::v_float64>::max_nlanes];
            cv::v_store(low_values, accumulator_low);
            cv::v_store(high_values, accumulator_high);
            for (int lane = 0; lane < lanes; ++lane)
            {
                long double value = lane < wide_lanes
                    ? static_cast<long double>(low_values[lane])
                    : static_cast<long double>(
                          high_values[lane - wide_lanes]);
                if (rtype == REDUCE_AVG)
                {
                    value /= static_cast<long double>(rows);
                }
                const int output_scalar = scalar + lane;
                write(
                    output_scalar / channels,
                    output_scalar % channels,
                    value);
            }
        }
#else
        return false;
#endif
    }

    for (; scalar < width; ++scalar)
    {
        long double accumulator = 0.0L;
        float extrema =
            reinterpret_cast<const float*>(src.data)[scalar];
        for (int row = 0; row < rows; ++row)
        {
            const float* src_row =
                reinterpret_cast<const float*>(
                    src.data +
                    static_cast<size_t>(row) * src.step(0));
            const float value = src_row[scalar];
            if (rtype == REDUCE_MAX)
            {
                if (value > extrema)
                {
                    extrema = value;
                }
            }
            else if (rtype == REDUCE_MIN)
            {
                if (value < extrema)
                {
                    extrema = value;
                }
            }
            else if (rtype == REDUCE_SUM2)
            {
                const long double wide =
                    static_cast<long double>(value);
                accumulator += wide * wide;
            }
            else
            {
                accumulator +=
                    static_cast<long double>(value);
            }
        }
        long double result =
            rtype == REDUCE_MAX || rtype == REDUCE_MIN
                ? static_cast<long double>(extrema)
                : accumulator;
        if (rtype == REDUCE_AVG)
        {
            result /= static_cast<long double>(rows);
        }
        write(scalar / channels, scalar % channels, result);
    }
    cv::v_cleanup();
    return true;
}

template<typename T, typename Vector, typename WriteFn>
inline bool reduce_typed(const Mat& src,
                         int axis,
                         int rtype,
                         WriteFn& write)
{
    if (axis == 1)
    {
        return reduce_axis1<T, Vector>(src, rtype, write);
    }
    if constexpr (std::is_same<T, uchar>::value)
    {
        return reduce_axis0_u8(src, rtype, write);
    }
    else
    {
        return reduce_axis0_f32(src, rtype, write);
    }
}

template<typename T, typename Vector>
inline void add_full_vector(const T* src,
                            int channels,
                            long double sums[4])
{
    Vector channel0;
    Vector channel1;
    Vector channel2;
    Vector channel3;
    switch (channels)
    {
        case 1:
            channel0 = cv::vx_load(src);
            sums[0] += horizontal_sum(channel0);
            return;
        case 2:
            cv::v_load_deinterleave(src, channel0, channel1);
            sums[0] += horizontal_sum(channel0);
            sums[1] += horizontal_sum(channel1);
            return;
        case 3:
            cv::v_load_deinterleave(src, channel0, channel1, channel2);
            sums[0] += horizontal_sum(channel0);
            sums[1] += horizontal_sum(channel1);
            sums[2] += horizontal_sum(channel2);
            return;
        default:
            cv::v_load_deinterleave(
                src, channel0, channel1, channel2, channel3);
            sums[0] += horizontal_sum(channel0);
            sums[1] += horizontal_sum(channel1);
            sums[2] += horizontal_sum(channel2);
            sums[3] += horizontal_sum(channel3);
            return;
    }
}

template<typename T>
inline void add_scalar_pixels(const T* src,
                              const uchar* mask,
                              size_t pixels,
                              int channels,
                              SumCount& result)
{
    for (size_t pixel = 0; pixel < pixels; ++pixel)
    {
        if (mask != nullptr && mask[pixel] == 0)
        {
            continue;
        }
        const size_t offset = pixel * static_cast<size_t>(channels);
        for (int channel = 0; channel < channels; ++channel)
        {
            result.sums[channel] += static_cast<long double>(
                src[offset + static_cast<size_t>(channel)]);
        }
        ++result.count;
    }
}

template<typename T, typename Vector>
inline bool add_pixel_range(const T* src,
                            const uchar* mask,
                            size_t pixels,
                            int channels,
                            SumCount& result)
{
    const size_t lanes =
        static_cast<size_t>(cv::VTraits<Vector>::vlanes());
    size_t pixel = 0;
    bool used_vector_block = false;
    for (; pixel + lanes <= pixels; pixel += lanes)
    {
        const T* block_src =
            src + pixel * static_cast<size_t>(channels);
        if (mask == nullptr)
        {
            add_full_vector<T, Vector>(
                block_src, channels, result.sums);
            result.count += lanes;
            used_vector_block = true;
            continue;
        }

        size_t selected = 0;
        for (size_t lane = 0; lane < lanes; ++lane)
        {
            selected += mask[pixel + lane] != 0 ? 1u : 0u;
        }
        used_vector_block = true;
        if (selected == 0)
        {
            continue;
        }
        if (selected == lanes)
        {
            add_full_vector<T, Vector>(
                block_src, channels, result.sums);
            result.count += lanes;
            continue;
        }
        add_scalar_pixels(
            block_src, mask + pixel, lanes, channels, result);
    }

    add_scalar_pixels(
        src + pixel * static_cast<size_t>(channels),
        mask == nullptr ? nullptr : mask + pixel,
        pixels - pixel,
        channels,
        result);
    return used_vector_block;
}

template<typename T, typename Vector>
inline bool sum_count_rows(const PixelRows& rows, SumCount& result)
{
    bool used_vector_block = false;
    for (size_t row = 0; row < rows.rows; ++row)
    {
        const T* src_row = reinterpret_cast<const T*>(
            rows.data + row * rows.step);
        const uchar* mask_row =
            rows.mask == nullptr ? nullptr : rows.mask + row * rows.mask_step;
        used_vector_block =
            add_pixel_range<T, Vector>(
                src_row,
                mask_row,
                rows.row_pixels,
                rows.channels,
                result) ||
            used_vector_block;
    }
    cv::v_cleanup();
    return used_vector_block;
}

inline void merge_statistics(StableStatistics& total,
                             const StableStatistics& block,
                             int channels)
{
    if (block.count == 0)
    {
        return;
    }
    if (total.count == 0)
    {
        total = block;
        return;
    }

    const size_t merged_count = total.count + block.count;
    const long double block_weight =
        static_cast<long double>(block.count) /
        static_cast<long double>(merged_count);
    for (int channel = 0; channel < channels; ++channel)
    {
        const long double delta =
            block.means[channel] - total.means[channel];
        total.m2[channel] +=
            block.m2[channel] +
            delta * delta * static_cast<long double>(total.count) *
                block_weight;
        total.means[channel] += delta * block_weight;
    }
    total.count = merged_count;
}

template<typename T, typename Vector>
inline bool stable_statistics_rows(const PixelRows& rows,
                                   StableStatistics& result)
{
    constexpr size_t block_pixels = 2048;
    bool used_vector_block = false;
    for (size_t row = 0; row < rows.rows; ++row)
    {
        const T* src_row = reinterpret_cast<const T*>(
            rows.data + row * rows.step);
        const uchar* mask_row =
            rows.mask == nullptr ? nullptr : rows.mask + row * rows.mask_step;
        for (size_t start = 0; start < rows.row_pixels;
             start += block_pixels)
        {
            const size_t length =
                std::min(block_pixels, rows.row_pixels - start);
            const T* block_src =
                src_row + start * static_cast<size_t>(rows.channels);
            const uchar* block_mask =
                mask_row == nullptr ? nullptr : mask_row + start;
            SumCount sums;
            used_vector_block =
                add_pixel_range<T, Vector>(
                    block_src,
                    block_mask,
                    length,
                    rows.channels,
                    sums) ||
                used_vector_block;
            if (sums.count == 0)
            {
                continue;
            }

            StableStatistics block;
            block.count = sums.count;
            for (int channel = 0; channel < rows.channels; ++channel)
            {
                block.means[channel] =
                    sums.sums[channel] /
                    static_cast<long double>(sums.count);
            }
            // Center each block before accumulating M2, then merge blocks with
            // Chan's formula. This avoids subtracting large raw moments.
            for (size_t pixel = 0; pixel < length; ++pixel)
            {
                if (block_mask != nullptr && block_mask[pixel] == 0)
                {
                    continue;
                }
                const size_t offset =
                    pixel * static_cast<size_t>(rows.channels);
                for (int channel = 0; channel < rows.channels; ++channel)
                {
                    const long double delta =
                        static_cast<long double>(
                            block_src[
                                offset + static_cast<size_t>(channel)]) -
                        block.means[channel];
                    block.m2[channel] += delta * delta;
                }
            }
            merge_statistics(result, block, rows.channels);
        }
    }
    cv::v_cleanup();
    return used_vector_block;
}

inline void accumulate_norm_scalar(double first,
                                   double second,
                                   bool difference,
                                   int norm_type,
                                   NormResult& result)
{
    const double value = difference ? first - second : first;
    const double magnitude = std::fabs(value);
    result.has_nan = result.has_nan || std::isnan(magnitude);
    if (norm_type == NORM_INF)
    {
        result.maximum = std::max(result.maximum, magnitude);
    }
    else if (norm_type == NORM_L1)
    {
        result.accumulator += magnitude;
    }
    else
    {
        result.accumulator +=
            static_cast<long double>(value) * value;
    }
}

inline bool norm_u8_range(const uchar* first,
                          const uchar* second,
                          size_t length,
                          int norm_type,
                          NormResult& result)
{
    const size_t lanes =
        static_cast<size_t>(cv::VTraits<cv::v_uint8>::vlanes());
    const bool used_vector_block = length >= lanes;
    size_t x = 0;
    if (norm_type == NORM_INF)
    {
        cv::v_uint8 maximum = cv::vx_setzero_u8();
        for (; x + lanes <= length; x += lanes)
        {
            const cv::v_uint8 first_values = cv::vx_load(first + x);
            const cv::v_uint8 magnitudes = second == nullptr
                ? first_values
                : cv::v_absdiff(
                      first_values, cv::vx_load(second + x));
            maximum = cv::v_max(maximum, magnitudes);
        }
        if (used_vector_block)
        {
            result.maximum = std::max(
                result.maximum,
                static_cast<double>(cv::v_reduce_max(maximum)));
        }
    }
    else
    {
        const cv::v_uint8 one = cv::vx_setall_u8(1);
        // Keep each uint32 lane below overflow for both L1 and L2.
        const size_t vectors_per_chunk = 2048;
        while (x + lanes <= length)
        {
            const size_t vector_count = std::min(
                vectors_per_chunk, (length - x) / lanes);
            const size_t end = x + vector_count * lanes;
            cv::v_uint32 accumulator = cv::vx_setzero_u32();
            for (; x < end; x += lanes)
            {
                const cv::v_uint8 first_values =
                    cv::vx_load(first + x);
                const cv::v_uint8 magnitudes = second == nullptr
                    ? first_values
                    : cv::v_absdiff(
                          first_values, cv::vx_load(second + x));
                accumulator = norm_type == NORM_L1
                    ? cv::v_dotprod_expand_fast(
                          magnitudes, one, accumulator)
                    : cv::v_dotprod_expand_fast(
                          magnitudes, magnitudes, accumulator);
            }
            result.accumulator +=
                static_cast<long double>(
                    cv::v_reduce_sum(accumulator));
        }
    }

    for (; x < length; ++x)
    {
        accumulate_norm_scalar(
            first[x],
            second == nullptr ? 0.0 : second[x],
            second != nullptr,
            norm_type,
            result);
    }
    return used_vector_block;
}

#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
inline bool norm_f32_range(const float* first,
                           const float* second,
                           size_t length,
                           int norm_type,
                           NormResult& result)
{
    const size_t lanes =
        static_cast<size_t>(cv::VTraits<cv::v_float32>::vlanes());
    if (length < lanes)
    {
        for (size_t x = 0; x < length; ++x)
        {
            accumulate_norm_scalar(
                first[x],
                second == nullptr ? 0.0 : second[x],
                second != nullptr,
                norm_type,
                result);
        }
        return false;
    }

    if (norm_type == NORM_INF && second == nullptr)
    {
        cv::v_float32 maximum = cv::vx_setzero_f32();
        size_t x = 0;
        for (; x + lanes <= length; x += lanes)
        {
            const cv::v_float32 values =
                cv::vx_load(first + x);
            result.has_nan =
                result.has_nan ||
                !cv::v_check_all(cv::v_not_nan(values));
            maximum = cv::v_max(maximum, cv::v_abs(values));
        }
        result.maximum = std::max(
            result.maximum,
            static_cast<double>(cv::v_reduce_max(maximum)));
        for (; x < length; ++x)
        {
            accumulate_norm_scalar(
                first[x],
                0.0,
                false,
                norm_type,
                result);
        }
        return true;
    }

    cv::v_float64 accumulator_low = cv::vx_setzero_f64();
    cv::v_float64 accumulator_high = cv::vx_setzero_f64();
    cv::v_float64 maximum_low = cv::vx_setzero_f64();
    cv::v_float64 maximum_high = cv::vx_setzero_f64();
    size_t x = 0;
    for (; x + lanes <= length; x += lanes)
    {
        const cv::v_float32 first_values = cv::vx_load(first + x);
        cv::v_float64 low = cv::v_cvt_f64(first_values);
        cv::v_float64 high = cv::v_cvt_f64_high(first_values);
        if (second != nullptr)
        {
            const cv::v_float32 second_values =
                cv::vx_load(second + x);
            low = cv::v_sub(low, cv::v_cvt_f64(second_values));
            high = cv::v_sub(
                high, cv::v_cvt_f64_high(second_values));
        }
        result.has_nan =
            result.has_nan ||
            !cv::v_check_all(cv::v_not_nan(low)) ||
            !cv::v_check_all(cv::v_not_nan(high));
        if (norm_type == NORM_INF)
        {
            maximum_low = cv::v_max(maximum_low, cv::v_abs(low));
            maximum_high = cv::v_max(maximum_high, cv::v_abs(high));
        }
        else if (norm_type == NORM_L1)
        {
            accumulator_low =
                cv::v_add(accumulator_low, cv::v_abs(low));
            accumulator_high =
                cv::v_add(accumulator_high, cv::v_abs(high));
        }
        else
        {
            accumulator_low =
                cv::v_fma(low, low, accumulator_low);
            accumulator_high =
                cv::v_fma(high, high, accumulator_high);
        }
    }

    if (norm_type == NORM_INF)
    {
        double low_lanes[cv::VTraits<cv::v_float64>::max_nlanes];
        double high_lanes[cv::VTraits<cv::v_float64>::max_nlanes];
        cv::v_store(low_lanes, maximum_low);
        cv::v_store(high_lanes, maximum_high);
        for (int lane = 0;
             lane < cv::VTraits<cv::v_float64>::vlanes();
             ++lane)
        {
            result.maximum =
                std::max(result.maximum, low_lanes[lane]);
            result.maximum =
                std::max(result.maximum, high_lanes[lane]);
        }
    }
    else
    {
        result.accumulator += static_cast<long double>(
            cv::v_reduce_sum(
                cv::v_add(accumulator_low, accumulator_high)));
    }

    for (; x < length; ++x)
    {
        accumulate_norm_scalar(
            first[x],
            second == nullptr ? 0.0 : second[x],
            second != nullptr,
            norm_type,
            result);
    }
    return true;
}
#endif

#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
inline bool stable_statistics_f32_c1_unmasked(
    const PixelRows& rows,
    StableStatistics& result)
{
    if (rows.mask != nullptr || rows.channels != 1)
    {
        return false;
    }
    SumCount sums;
    if (!sum_count_rows<
            float,
            typename VectorType<float>::type>(rows, sums) ||
        sums.count == 0)
    {
        return false;
    }

    const long double mean =
        sums.sums[0] / static_cast<long double>(sums.count);
    const cv::v_float64 mean_vector =
        cv::vx_setall_f64(static_cast<double>(mean));
    long double m2 = 0.0L;
    const size_t lanes = static_cast<size_t>(
        cv::VTraits<cv::v_float32>::vlanes());
    for (size_t row = 0; row < rows.rows; ++row)
    {
        const float* source = reinterpret_cast<const float*>(
            rows.data + row * rows.step);
        cv::v_float64 accumulator_low = cv::vx_setzero_f64();
        cv::v_float64 accumulator_high = cv::vx_setzero_f64();
        size_t pixel = 0;
        for (; pixel + lanes <= rows.row_pixels;
             pixel += lanes)
        {
            const cv::v_float32 values =
                cv::vx_load(source + pixel);
            const cv::v_float64 low =
                cv::v_sub(cv::v_cvt_f64(values), mean_vector);
            const cv::v_float64 high = cv::v_sub(
                cv::v_cvt_f64_high(values),
                mean_vector);
            accumulator_low = cv::v_fma(
                low, low, accumulator_low);
            accumulator_high = cv::v_fma(
                high, high, accumulator_high);
        }
        m2 += static_cast<long double>(
            cv::v_reduce_sum(
                cv::v_add(accumulator_low, accumulator_high)));
        for (; pixel < rows.row_pixels; ++pixel)
        {
            const long double delta =
                static_cast<long double>(source[pixel]) - mean;
            m2 += delta * delta;
        }
    }
    cv::v_cleanup();
    result.count = sums.count;
    result.means[0] = mean;
    result.m2[0] = m2;
    return true;
}
#endif

template<typename T>
using SingleNormKernel =
    bool (*)(const T*, const T*, size_t, int, NormResult&);

template<typename T>
inline bool norm_rows(const PixelRows& rows,
                      int norm_type,
                      NormResult& result,
                      SingleNormKernel<T> kernel)
{
    bool used_vector_block = false;
    for (size_t row = 0; row < rows.rows; ++row)
    {
        const T* src_row = reinterpret_cast<const T*>(
            rows.data + row * rows.step);
        const uchar* mask_row =
            rows.mask == nullptr
                ? nullptr
                : rows.mask + row * rows.mask_step;
        if (mask_row == nullptr)
        {
            used_vector_block =
                kernel(
                    src_row,
                    nullptr,
                    rows.row_pixels *
                        static_cast<size_t>(rows.channels),
                    norm_type,
                    result) ||
                used_vector_block;
            continue;
        }

        size_t pixel = 0;
        while (pixel < rows.row_pixels)
        {
            while (pixel < rows.row_pixels && mask_row[pixel] == 0)
            {
                ++pixel;
            }
            const size_t start = pixel;
            while (pixel < rows.row_pixels && mask_row[pixel] != 0)
            {
                ++pixel;
            }
            const size_t scalar_start =
                start * static_cast<size_t>(rows.channels);
            const size_t scalar_count =
                (pixel - start) *
                static_cast<size_t>(rows.channels);
            used_vector_block =
                kernel(
                    src_row + scalar_start,
                    nullptr,
                    scalar_count,
                    norm_type,
                    result) ||
                used_vector_block;
        }
    }
    cv::v_cleanup();
    return used_vector_block;
}

template<typename T>
inline bool norm_diff_rows(const PixelPairRows& rows,
                           int norm_type,
                           NormResult& result,
                           SingleNormKernel<T> kernel)
{
    bool used_vector_block = false;
    for (size_t row = 0; row < rows.rows; ++row)
    {
        const T* first_row = reinterpret_cast<const T*>(
            rows.first + row * rows.first_step);
        const T* second_row = reinterpret_cast<const T*>(
            rows.second + row * rows.second_step);
        const uchar* mask_row =
            rows.mask == nullptr
                ? nullptr
                : rows.mask + row * rows.mask_step;
        if (mask_row == nullptr)
        {
            used_vector_block =
                kernel(
                    first_row,
                    second_row,
                    rows.row_pixels *
                        static_cast<size_t>(rows.channels),
                    norm_type,
                    result) ||
                used_vector_block;
            continue;
        }

        size_t pixel = 0;
        while (pixel < rows.row_pixels)
        {
            while (pixel < rows.row_pixels && mask_row[pixel] == 0)
            {
                ++pixel;
            }
            const size_t start = pixel;
            while (pixel < rows.row_pixels && mask_row[pixel] != 0)
            {
                ++pixel;
            }
            const size_t scalar_start =
                start * static_cast<size_t>(rows.channels);
            const size_t scalar_count =
                (pixel - start) *
                static_cast<size_t>(rows.channels);
            used_vector_block =
                kernel(
                    first_row + scalar_start,
                    second_row + scalar_start,
                    scalar_count,
                    norm_type,
                    result) ||
                used_vector_block;
        }
    }
    cv::v_cleanup();
    return used_vector_block;
}

#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
inline void normalize_f32_range(const float* src,
                                float* dst,
                                size_t length,
                                double scale,
                                double shift)
{
    const size_t lanes =
        static_cast<size_t>(cv::VTraits<cv::v_float32>::vlanes());
    const cv::v_float64 scale_vector = cv::vx_setall_f64(scale);
    const cv::v_float64 shift_vector = cv::vx_setall_f64(shift);
    size_t x = 0;
    for (; x + lanes <= length; x += lanes)
    {
        const cv::v_float32 values = cv::vx_load(src + x);
        const cv::v_float64 low = cv::v_fma(
            cv::v_cvt_f64(values), scale_vector, shift_vector);
        const cv::v_float64 high = cv::v_fma(
            cv::v_cvt_f64_high(values), scale_vector, shift_vector);
        cv::v_store(dst + x, cv::v_cvt_f32(low, high));
    }
    for (; x < length; ++x)
    {
        dst[x] = static_cast<float>(
            static_cast<double>(src[x]) * scale + shift);
    }
}

inline bool normalize_rows_have_vector_run(const Mat& src,
                                           const Mat& dst,
                                           const Mat& mask)
{
    const size_t lanes =
        static_cast<size_t>(cv::VTraits<cv::v_float32>::vlanes());
    const bool flatten =
        src.isContinuous() && dst.isContinuous() &&
        (mask.empty() || mask.isContinuous());
    const size_t rows =
        flatten ? 1 : (src.dims > 1
                            ? static_cast<size_t>(src.size.p[0])
                            : 1);
    const size_t row_pixels =
        flatten ? src.total()
                : (src.dims > 1
                       ? src.total(1, src.dims)
                       : src.total());
    const size_t mask_step =
        mask.empty()
            ? 0
            : (flatten
                   ? mask.total()
                   : (mask.dims > 1 ? mask.step(0) : row_pixels));
    const int channels = src.channels();
    if (mask.empty())
    {
        return row_pixels * static_cast<size_t>(channels) >= lanes;
    }
    for (size_t row = 0; row < rows; ++row)
    {
        const uchar* mask_row = mask.data + row * mask_step;
        size_t pixel = 0;
        while (pixel < row_pixels)
        {
            while (pixel < row_pixels && mask_row[pixel] == 0)
            {
                ++pixel;
            }
            const size_t start = pixel;
            while (pixel < row_pixels && mask_row[pixel] != 0)
            {
                ++pixel;
            }
            if ((pixel - start) * static_cast<size_t>(channels) >= lanes)
            {
                return true;
            }
        }
    }
    return false;
}

inline void normalize_f32_rows(const Mat& src,
                               Mat& dst,
                               const Mat& mask,
                               double scale,
                               double shift)
{
    const bool flatten =
        src.isContinuous() && dst.isContinuous() &&
        (mask.empty() || mask.isContinuous());
    const size_t rows =
        flatten ? 1 : (src.dims > 1
                            ? static_cast<size_t>(src.size.p[0])
                            : 1);
    const size_t row_pixels =
        flatten ? src.total()
                : (src.dims > 1
                       ? src.total(1, src.dims)
                       : src.total());
    const size_t src_step =
        flatten ? src.total() * src.elemSize()
                : (src.dims > 1
                       ? src.step(0)
                       : row_pixels * src.elemSize());
    const size_t dst_step =
        flatten ? dst.total() * dst.elemSize()
                : (dst.dims > 1
                       ? dst.step(0)
                       : row_pixels * dst.elemSize());
    const size_t mask_step =
        mask.empty()
            ? 0
            : (flatten
                   ? mask.total()
                   : (mask.dims > 1 ? mask.step(0) : row_pixels));
    const int channels = src.channels();

    for (size_t row = 0; row < rows; ++row)
    {
        const float* src_row = reinterpret_cast<const float*>(
            src.data + row * src_step);
        float* dst_row = reinterpret_cast<float*>(
            dst.data + row * dst_step);
        const uchar* mask_row =
            mask.empty() ? nullptr : mask.data + row * mask_step;
        if (mask_row == nullptr)
        {
            normalize_f32_range(
                src_row,
                dst_row,
                row_pixels * static_cast<size_t>(channels),
                scale,
                shift);
            continue;
        }

        size_t pixel = 0;
        while (pixel < row_pixels)
        {
            while (pixel < row_pixels && mask_row[pixel] == 0)
            {
                ++pixel;
            }
            const size_t start = pixel;
            while (pixel < row_pixels && mask_row[pixel] != 0)
            {
                ++pixel;
            }
            const size_t scalar_start =
                start * static_cast<size_t>(channels);
            normalize_f32_range(
                src_row + scalar_start,
                dst_row + scalar_start,
                (pixel - start) * static_cast<size_t>(channels),
                scale,
                shift);
        }
    }
    cv::v_cleanup();
}
#endif

// Packing limits and block widths follow OpenCV's count/has_non_zero SIMD kernels.
template<typename T>
inline int count_nonzero_scalar(const T* src, int length)
{
    int count = 0;
    int x = 0;
    for (; x + 4 <= length; x += 4)
    {
        count += (src[x] != 0);
        count += (src[x + 1] != 0);
        count += (src[x + 2] != 0);
        count += (src[x + 3] != 0);
    }
    for (; x < length; ++x)
    {
        count += (src[x] != 0);
    }
    return count;
}

struct CountPack16U
{
    static inline cv::v_int8 eq(const ushort* src, int x, cv::v_uint16 zero)
    {
        const int lanes = cv::VTraits<cv::v_uint16>::vlanes();
        return cv::v_pack(
            cv::v_reinterpret_as_s16(cv::v_eq(cv::vx_load(src + x), zero)),
            cv::v_reinterpret_as_s16(
                cv::v_eq(cv::vx_load(src + x + lanes), zero)));
    }
};

struct CountPack16S
{
    static inline cv::v_int8 eq(const short* src, int x, cv::v_int16 zero)
    {
        const int lanes = cv::VTraits<cv::v_int16>::vlanes();
        return cv::v_pack(
            cv::v_eq(cv::vx_load(src + x), zero),
            cv::v_eq(cv::vx_load(src + x + lanes), zero));
    }
};

struct CountPack32S
{
    static inline cv::v_int8 eq(const int* src, int x, cv::v_int32 zero)
    {
        const int lanes = cv::VTraits<cv::v_int32>::vlanes();
        return cv::v_pack(
            cv::v_pack(
                cv::v_eq(cv::vx_load(src + x), zero),
                cv::v_eq(cv::vx_load(src + x + lanes), zero)),
            cv::v_pack(
                cv::v_eq(cv::vx_load(src + x + 2 * lanes), zero),
                cv::v_eq(cv::vx_load(src + x + 3 * lanes), zero)));
    }
};

struct CountPack32U
{
    static inline cv::v_int8 eq(const uint* src, int x, cv::v_uint32 zero)
    {
        const int lanes = cv::VTraits<cv::v_uint32>::vlanes();
        return cv::v_pack(
            cv::v_pack(
                cv::v_reinterpret_as_s32(
                    cv::v_eq(cv::vx_load(src + x), zero)),
                cv::v_reinterpret_as_s32(
                    cv::v_eq(cv::vx_load(src + x + lanes), zero))),
            cv::v_pack(
                cv::v_reinterpret_as_s32(
                    cv::v_eq(cv::vx_load(src + x + 2 * lanes), zero)),
                cv::v_reinterpret_as_s32(
                    cv::v_eq(cv::vx_load(src + x + 3 * lanes), zero))));
    }
};

struct CountPack32F
{
    static inline cv::v_int8 eq(const float* src, int x, cv::v_float32 zero)
    {
        const int lanes = cv::VTraits<cv::v_float32>::vlanes();
        return cv::v_pack(
            cv::v_pack(
                cv::v_reinterpret_as_s32(
                    cv::v_eq(cv::vx_load(src + x), zero)),
                cv::v_reinterpret_as_s32(
                    cv::v_eq(cv::vx_load(src + x + lanes), zero))),
            cv::v_pack(
                cv::v_reinterpret_as_s32(
                    cv::v_eq(cv::vx_load(src + x + 2 * lanes), zero)),
                cv::v_reinterpret_as_s32(
                    cv::v_eq(cv::vx_load(src + x + 3 * lanes), zero))));
    }
};

template<typename PackOp, typename T, typename ZeroVector>
inline int count_nonzero_batched(const T* src,
                                 int length,
                                 const ZeroVector& zero)
{
    int x = 0;
    const int lanes8 = cv::VTraits<cv::v_int8>::vlanes();
    const int vector_end = length & -lanes8;
    const cv::v_int8 one = cv::vx_setall_s8(1);
    cv::v_int32 sum32 = cv::vx_setzero_s32();

    while (x < vector_end)
    {
        cv::v_int16 sum16 = cv::vx_setzero_s16();
        int middle = x;
        while (middle < std::min(
                   vector_end,
                   x + 32766 * cv::VTraits<cv::v_int16>::vlanes()))
        {
            cv::v_int8 sum8 = cv::vx_setzero_s8();
            int inner = middle;
            for (; inner < std::min(vector_end, middle + 127 * lanes8);
                 inner += lanes8)
            {
                sum8 = cv::v_add(
                    sum8,
                    cv::v_and(one, PackOp::eq(src, inner, zero)));
            }
            cv::v_int16 low;
            cv::v_int16 high;
            cv::v_expand(sum8, low, high);
            sum16 = cv::v_add(sum16, cv::v_add(low, high));
            middle = inner;
        }
        cv::v_int32 low;
        cv::v_int32 high;
        cv::v_expand(sum16, low, high);
        sum32 = cv::v_add(sum32, cv::v_add(low, high));
        x = middle;
    }

    const int count = x - cv::v_reduce_sum(sum32);
    cv::v_cleanup();
    return count + count_nonzero_scalar(src + x, length - x);
}

inline int count_nonzero_u8(const uchar* src, int length)
{
    int x = 0;
    const int lanes = cv::VTraits<cv::v_uint8>::vlanes();
    const int vector_end = length & -lanes;
    const cv::v_uint8 zero = cv::vx_setzero_u8();
    const cv::v_uint8 one = cv::vx_setall_u8(1);
    cv::v_uint32 sum32 = cv::vx_setzero_u32();

    while (x < vector_end)
    {
        cv::v_uint16 sum16 = cv::vx_setzero_u16();
        int middle = x;
        while (middle < std::min(
                   vector_end,
                   x + 65280 * cv::VTraits<cv::v_uint16>::vlanes()))
        {
            cv::v_uint8 sum8 = cv::vx_setzero_u8();
            int inner = middle;
            for (; inner < std::min(vector_end, middle + 255 * lanes);
                 inner += lanes)
            {
                sum8 = cv::v_add(
                    sum8,
                    cv::v_and(
                        one,
                        cv::v_eq(cv::vx_load(src + inner), zero)));
            }
            cv::v_uint16 low;
            cv::v_uint16 high;
            cv::v_expand(sum8, low, high);
            sum16 = cv::v_add(sum16, cv::v_add(low, high));
            middle = inner;
        }
        cv::v_uint32 low;
        cv::v_uint32 high;
        cv::v_expand(sum16, low, high);
        sum32 = cv::v_add(sum32, cv::v_add(low, high));
        x = middle;
    }

    const int count = x - cv::v_reduce_sum(sum32);
    cv::v_cleanup();
    return count + count_nonzero_scalar(src + x, length - x);
}

#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
inline int count_nonzero_f64(const double* src, int length)
{
    int x = 0;
    cv::v_int64 sum0 = cv::vx_setzero_s64();
    cv::v_int64 sum1 = cv::vx_setzero_s64();
    const cv::v_float64 zero = cv::vx_setzero_f64();
    const int step = cv::VTraits<cv::v_float64>::vlanes() * 2;
    const int vector_end = length & -step;

    for (; x < vector_end; x += step)
    {
        sum0 = cv::v_add(
            sum0,
            cv::v_reinterpret_as_s64(
                cv::v_eq(cv::vx_load(src + x), zero)));
        sum1 = cv::v_add(
            sum1,
            cv::v_reinterpret_as_s64(
                cv::v_eq(cv::vx_load(src + x + step / 2), zero)));
    }

    const int count =
        x + static_cast<int>(cv::v_reduce_sum(cv::v_add(sum0, sum1)));
    cv::v_cleanup();
    return count + count_nonzero_scalar(src + x, length - x);
}
#endif

template<typename T, typename CountFn>
inline size_t count_nonzero_rows(const SourceRows& rows, CountFn count_fn)
{
    constexpr size_t max_chunk =
        static_cast<size_t>(std::numeric_limits<int>::max() / 2);
    size_t total = 0;
    for (size_t row = 0; row < rows.rows; ++row)
    {
        const T* row_data = reinterpret_cast<const T*>(
            rows.data + row * rows.step);
        size_t offset = 0;
        while (offset < rows.row_scalars)
        {
            const int chunk = static_cast<int>(
                std::min(max_chunk, rows.row_scalars - offset));
            total += static_cast<size_t>(count_fn(row_data + offset, chunk));
            offset += static_cast<size_t>(chunk);
        }
    }
    return total;
}

template<typename Vector>
inline bool vector_has_nonzero(const Vector& values, const Vector& zero)
{
    return !cv::v_check_all(cv::v_eq(values, zero));
}

template<typename T, typename Vector, size_t Unroll>
inline bool has_nonzero_row(const T* src,
                            size_t length,
                            const Vector& zero)
{
    const size_t lanes =
        static_cast<size_t>(cv::VTraits<Vector>::vlanes());
    const size_t block = lanes * Unroll;
    size_t x = 0;
    for (; x + block <= length; x += block)
    {
        Vector combined = cv::vx_load(src + x);
        for (size_t part = 1; part < Unroll; ++part)
        {
            combined = cv::v_or(
                combined,
                cv::vx_load(src + x + part * lanes));
        }
        if (vector_has_nonzero(combined, zero))
        {
            cv::v_cleanup();
            return true;
        }
    }
    for (; x + lanes <= length; x += lanes)
    {
        if (vector_has_nonzero(cv::vx_load(src + x), zero))
        {
            cv::v_cleanup();
            return true;
        }
    }
    cv::v_cleanup();
    for (; x < length; ++x)
    {
        if (src[x] != 0)
        {
            return true;
        }
    }
    return false;
}

template<typename T, typename Vector, size_t Unroll>
inline bool has_nonzero_rows(const SourceRows& rows, const Vector& zero)
{
    for (size_t row = 0; row < rows.rows; ++row)
    {
        const T* row_data = reinterpret_cast<const T*>(
            rows.data + row * rows.step);
        if (has_nonzero_row<T, Vector, Unroll>(
                row_data,
                rows.row_scalars,
                zero))
        {
            return true;
        }
    }
    return false;
}

template<typename T, typename Vector, typename EmitFn>
inline bool find_nonzero_rows(const Mat& src,
                              const Vector& zero,
                              EmitFn emit)
{
    const size_t rows = static_cast<size_t>(src.size.p[0]);
    const size_t cols = static_cast<size_t>(src.size.p[1]);
    const size_t lanes =
        static_cast<size_t>(cv::VTraits<Vector>::vlanes());
    if (cols < lanes)
    {
        return false;
    }

    for (size_t row = 0; row < rows; ++row)
    {
        const T* row_data = reinterpret_cast<const T*>(
            src.data + row * src.step(0));
        size_t column = 0;
        size_t consecutive_hit_blocks = 0;
        for (; column + lanes <= cols; column += lanes)
        {
            const Vector values = cv::vx_load(row_data + column);
            if (!vector_has_nonzero(values, zero))
            {
                consecutive_hit_blocks = 0;
                continue;
            }

            if (cv::v_check_all(cv::v_ne(values, zero)))
            {
                for (size_t lane = 0; lane < lanes; ++lane)
                {
                    emit(column + lane, row);
                }
                column += lanes;
                break;
            }

            T lane_values[cv::VTraits<Vector>::max_nlanes];
            cv::v_store(lane_values, values);
            for (size_t lane = 0; lane < lanes; ++lane)
            {
                if (lane_values[lane] != 0)
                {
                    emit(column + lane, row);
                }
            }
            ++consecutive_hit_blocks;
            if (consecutive_hit_blocks == 4)
            {
                column += lanes;
                break;
            }
        }
        for (; column < cols; ++column)
        {
            if (row_data[column] != 0)
            {
                emit(column, row);
            }
        }
    }
    cv::v_cleanup();
    return true;
}

#endif

inline bool try_norm(const Mat& src,
                     const Mat& mask,
                     int norm_type,
                     NormResult& result)
{
    if (!enabled())
    {
        return false;
    }

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const PixelRows rows = pixel_rows(src, mask);
    NormResult candidate;
    bool used_vector_block = false;
    switch (src.depth())
    {
        case CV_8U:
            used_vector_block = norm_rows<uchar>(
                rows, norm_type, candidate, norm_u8_range);
            break;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        case CV_32F:
            used_vector_block = norm_rows<float>(
                rows, norm_type, candidate, norm_f32_range);
            break;
#endif
        default:
            return false;
    }
    if (!used_vector_block)
    {
        return false;
    }
    result = candidate;
    return true;
#else
    (void)src;
    (void)mask;
    (void)norm_type;
    (void)result;
    return false;
#endif
}

inline bool try_norm_diff(const Mat& first,
                          const Mat& second,
                          const Mat& mask,
                          int norm_type,
                          NormResult& result)
{
    if (!enabled())
    {
        return false;
    }

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const PixelPairRows rows =
        pixel_pair_rows(first, second, mask);
    NormResult candidate;
    bool used_vector_block = false;
    switch (first.depth())
    {
        case CV_8U:
            used_vector_block = norm_diff_rows<uchar>(
                rows, norm_type, candidate, norm_u8_range);
            break;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        case CV_32F:
            used_vector_block = norm_diff_rows<float>(
                rows, norm_type, candidate, norm_f32_range);
            break;
#endif
        default:
            return false;
    }
    if (!used_vector_block)
    {
        return false;
    }
    result = candidate;
    return true;
#else
    (void)first;
    (void)second;
    (void)mask;
    (void)norm_type;
    (void)result;
    return false;
#endif
}

inline bool try_apply_normalize(const Mat& src,
                                Mat& dst,
                                const Mat& mask,
                                double scale,
                                double shift)
{
    if (!enabled() || src.depth() != CV_32F ||
        dst.depth() != CV_32F)
    {
        return false;
    }

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX) && \
    (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
    if (!normalize_rows_have_vector_run(src, dst, mask))
    {
        return false;
    }
    normalize_f32_rows(src, dst, mask, scale, shift);
    return true;
#else
    (void)src;
    (void)dst;
    (void)mask;
    (void)scale;
    (void)shift;
    return false;
#endif
}

inline bool try_count_nonzero(const Mat& src, size_t& result)
{
    if (!enabled())
    {
        return false;
    }

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const SourceRows rows = source_rows(src);
    switch (src.depth())
    {
        case CV_8U:
        case CV_8S:
            if (rows.row_scalars <
                static_cast<size_t>(cv::VTraits<cv::v_uint8>::vlanes()))
            {
                return false;
            }
            result = count_nonzero_rows<uchar>(
                rows,
                [](const uchar* data, int length) {
                    return count_nonzero_u8(data, length);
                });
            return true;
        case CV_16U:
            if (rows.row_scalars <
                static_cast<size_t>(cv::VTraits<cv::v_int8>::vlanes()))
            {
                return false;
            }
            result = count_nonzero_rows<ushort>(
                rows,
                [](const ushort* data, int length) {
                    return count_nonzero_batched<CountPack16U>(
                        data,
                        length,
                        cv::vx_setzero_u16());
                });
            return true;
        case CV_16S:
            if (rows.row_scalars <
                static_cast<size_t>(cv::VTraits<cv::v_int8>::vlanes()))
            {
                return false;
            }
            result = count_nonzero_rows<short>(
                rows,
                [](const short* data, int length) {
                    return count_nonzero_batched<CountPack16S>(
                        data,
                        length,
                        cv::vx_setzero_s16());
                });
            return true;
        case CV_32S:
            if (rows.row_scalars <
                static_cast<size_t>(cv::VTraits<cv::v_int8>::vlanes()))
            {
                return false;
            }
            result = count_nonzero_rows<int>(
                rows,
                [](const int* data, int length) {
                    return count_nonzero_batched<CountPack32S>(
                        data,
                        length,
                        cv::vx_setzero_s32());
                });
            return true;
        case CV_32U:
            if (rows.row_scalars <
                static_cast<size_t>(cv::VTraits<cv::v_int8>::vlanes()))
            {
                return false;
            }
            result = count_nonzero_rows<uint>(
                rows,
                [](const uint* data, int length) {
                    return count_nonzero_batched<CountPack32U>(
                        data,
                        length,
                        cv::vx_setzero_u32());
                });
            return true;
        case CV_32F:
            if (rows.row_scalars <
                static_cast<size_t>(cv::VTraits<cv::v_int8>::vlanes()))
            {
                return false;
            }
            result = count_nonzero_rows<float>(
                rows,
                [](const float* data, int length) {
                    return count_nonzero_batched<CountPack32F>(
                        data,
                        length,
                        cv::vx_setzero_f32());
                });
            return true;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        case CV_64F:
            if (rows.row_scalars <
                static_cast<size_t>(
                    cv::VTraits<cv::v_float64>::vlanes() * 2))
            {
                return false;
            }
            result = count_nonzero_rows<double>(
                rows,
                [](const double* data, int length) {
                    return count_nonzero_f64(data, length);
                });
            return true;
#endif
        default:
            return false;
    }
#else
    (void)src;
    (void)result;
    return false;
#endif
}

inline bool try_has_nonzero(const Mat& src, bool& result)
{
    if (!enabled())
    {
        return false;
    }

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const SourceRows rows = source_rows(src);
    switch (src.depth())
    {
        case CV_8U:
        case CV_8S:
            if (rows.row_scalars <
                static_cast<size_t>(
                    cv::VTraits<cv::v_uint8>::vlanes() * 2))
            {
                return false;
            }
            result = has_nonzero_rows<uchar, cv::v_uint8, 2>(
                rows,
                cv::vx_setzero_u8());
            return true;
        case CV_16U:
            if (rows.row_scalars <
                static_cast<size_t>(
                    cv::VTraits<cv::v_uint16>::vlanes() * 4))
            {
                return false;
            }
            result = has_nonzero_rows<ushort, cv::v_uint16, 4>(
                rows,
                cv::vx_setzero_u16());
            return true;
        case CV_16S:
            if (rows.row_scalars <
                static_cast<size_t>(
                    cv::VTraits<cv::v_int16>::vlanes() * 4))
            {
                return false;
            }
            result = has_nonzero_rows<short, cv::v_int16, 4>(
                rows,
                cv::vx_setzero_s16());
            return true;
        case CV_32S:
            if (rows.row_scalars <
                static_cast<size_t>(
                    cv::VTraits<cv::v_int32>::vlanes() * 8))
            {
                return false;
            }
            result = has_nonzero_rows<int, cv::v_int32, 8>(
                rows,
                cv::vx_setzero_s32());
            return true;
        case CV_32U:
            if (rows.row_scalars <
                static_cast<size_t>(
                    cv::VTraits<cv::v_uint32>::vlanes() * 8))
            {
                return false;
            }
            result = has_nonzero_rows<uint, cv::v_uint32, 8>(
                rows,
                cv::vx_setzero_u32());
            return true;
        case CV_32F:
            if (rows.row_scalars <
                static_cast<size_t>(
                    cv::VTraits<cv::v_float32>::vlanes() * 8))
            {
                return false;
            }
            result = has_nonzero_rows<float, cv::v_float32, 8>(
                rows,
                cv::vx_setzero_f32());
            return true;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        case CV_64F:
            if (rows.row_scalars <
                static_cast<size_t>(
                    cv::VTraits<cv::v_float64>::vlanes() * 16))
            {
                return false;
            }
            result = has_nonzero_rows<double, cv::v_float64, 16>(
                rows,
                cv::vx_setzero_f64());
            return true;
#endif
        default:
            return false;
    }
#else
    (void)src;
    (void)result;
    return false;
#endif
}

template<typename EmitFn>
inline bool try_find_nonzero(const Mat& src, EmitFn emit)
{
    if (!enabled())
    {
        return false;
    }

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    switch (src.depth())
    {
        case CV_8U:
            return find_nonzero_rows<uchar, cv::v_uint8>(
                src, cv::vx_setzero_u8(), emit);
        case CV_8S:
            return find_nonzero_rows<schar, cv::v_int8>(
                src, cv::vx_setzero_s8(), emit);
        case CV_16U:
            return find_nonzero_rows<ushort, cv::v_uint16>(
                src, cv::vx_setzero_u16(), emit);
        case CV_16S:
            return find_nonzero_rows<short, cv::v_int16>(
                src, cv::vx_setzero_s16(), emit);
        case CV_32S:
            return find_nonzero_rows<int, cv::v_int32>(
                src, cv::vx_setzero_s32(), emit);
        case CV_32U:
            return find_nonzero_rows<uint, cv::v_uint32>(
                src, cv::vx_setzero_u32(), emit);
        case CV_32F:
            return find_nonzero_rows<float, cv::v_float32>(
                src, cv::vx_setzero_f32(), emit);
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        case CV_64F:
            return find_nonzero_rows<double, cv::v_float64>(
                src, cv::vx_setzero_f64(), emit);
#endif
        default:
            return false;
    }
#else
    (void)src;
    (void)emit;
    return false;
#endif
}

inline bool try_sum_count(const Mat& src,
                          const Mat& mask,
                          SumCount& result)
{
    if (!enabled())
    {
        return false;
    }

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const PixelRows rows = pixel_rows(src, mask);
    SumCount candidate;
    bool used_vector_block = false;
    switch (src.depth())
    {
        case CV_8U:
            used_vector_block =
                sum_count_rows<uchar, typename VectorType<uchar>::type>(
                    rows, candidate);
            break;
        case CV_8S:
            used_vector_block =
                sum_count_rows<schar, typename VectorType<schar>::type>(
                    rows, candidate);
            break;
        case CV_16U:
            used_vector_block =
                sum_count_rows<ushort, typename VectorType<ushort>::type>(
                    rows, candidate);
            break;
        case CV_16S:
            used_vector_block =
                sum_count_rows<short, typename VectorType<short>::type>(
                    rows, candidate);
            break;
        case CV_32S:
            used_vector_block =
                sum_count_rows<int, typename VectorType<int>::type>(
                    rows, candidate);
            break;
        case CV_32U:
            used_vector_block =
                sum_count_rows<uint, typename VectorType<uint>::type>(
                    rows, candidate);
            break;
        case CV_32F:
            used_vector_block =
                sum_count_rows<float, typename VectorType<float>::type>(
                    rows, candidate);
            break;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        case CV_64F:
            used_vector_block =
                sum_count_rows<double, typename VectorType<double>::type>(
                    rows, candidate);
            break;
#endif
        default:
            return false;
    }
    if (!used_vector_block)
    {
        return false;
    }
    result = candidate;
    return true;
#else
    (void)src;
    (void)mask;
    (void)result;
    return false;
#endif
}

inline bool try_stable_statistics(const Mat& src,
                                  const Mat& mask,
                                  StableStatistics& result)
{
    if (!enabled())
    {
        return false;
    }

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const PixelRows rows = pixel_rows(src, mask);
    StableStatistics candidate;
    bool used_vector_block = false;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
    if (src.depth() == CV_32F &&
        stable_statistics_f32_c1_unmasked(rows, candidate))
    {
        result = candidate;
        return true;
    }
#endif
    switch (src.depth())
    {
        case CV_8U:
            used_vector_block =
                stable_statistics_rows<
                    uchar,
                    typename VectorType<uchar>::type>(rows, candidate);
            break;
        case CV_8S:
            used_vector_block =
                stable_statistics_rows<
                    schar,
                    typename VectorType<schar>::type>(rows, candidate);
            break;
        case CV_16U:
            used_vector_block =
                stable_statistics_rows<
                    ushort,
                    typename VectorType<ushort>::type>(rows, candidate);
            break;
        case CV_16S:
            used_vector_block =
                stable_statistics_rows<
                    short,
                    typename VectorType<short>::type>(rows, candidate);
            break;
        case CV_32S:
            used_vector_block =
                stable_statistics_rows<
                    int,
                    typename VectorType<int>::type>(rows, candidate);
            break;
        case CV_32U:
            used_vector_block =
                stable_statistics_rows<
                    uint,
                    typename VectorType<uint>::type>(rows, candidate);
            break;
        case CV_32F:
            used_vector_block =
                stable_statistics_rows<
                    float,
                    typename VectorType<float>::type>(rows, candidate);
            break;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        case CV_64F:
            used_vector_block =
                stable_statistics_rows<
                    double,
                    typename VectorType<double>::type>(rows, candidate);
            break;
#endif
        default:
            return false;
    }
    if (!used_vector_block)
    {
        return false;
    }
    result = candidate;
    return true;
#else
    (void)src;
    (void)mask;
    (void)result;
    return false;
#endif
}

inline bool try_extrema(const Mat& src,
                        const Mat& mask,
                        ExtremaResult& result)
{
    if (!enabled() || src.channels() != 1)
    {
        return false;
    }

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    const PixelRows rows = pixel_rows(src, mask);
    ExtremaResult candidate;
    bool used_vector_block = false;
    switch (src.depth())
    {
        case CV_8U:
            used_vector_block =
                extrema_rows<uchar, typename VectorType<uchar>::type>(
                    rows, candidate);
            break;
        case CV_8S:
            used_vector_block =
                extrema_rows<schar, typename VectorType<schar>::type>(
                    rows, candidate);
            break;
        case CV_16U:
            used_vector_block =
                extrema_rows<ushort, typename VectorType<ushort>::type>(
                    rows, candidate);
            break;
        case CV_16S:
            used_vector_block =
                extrema_rows<short, typename VectorType<short>::type>(
                    rows, candidate);
            break;
        case CV_32S:
            used_vector_block =
                extrema_rows<int, typename VectorType<int>::type>(
                    rows, candidate);
            break;
        case CV_32U:
            used_vector_block =
                extrema_rows<uint, typename VectorType<uint>::type>(
                    rows, candidate);
            break;
        case CV_32F:
            used_vector_block =
                extrema_rows<float, typename VectorType<float>::type>(
                    rows, candidate);
            break;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        case CV_64F:
            used_vector_block =
                extrema_rows<double, typename VectorType<double>::type>(
                    rows, candidate);
            break;
#endif
        default:
            return false;
    }
    if (!used_vector_block)
    {
        return false;
    }
    result = candidate;
    return true;
#else
    (void)src;
    (void)mask;
    (void)result;
    return false;
#endif
}

template<typename WriteFn>
inline bool try_reduce(const Mat& src,
                       int axis,
                       int rtype,
                       WriteFn&& write)
{
    if (!enabled() || src.channels() < 1 || src.channels() > 4)
    {
        return false;
    }

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    switch (src.depth())
    {
        case CV_8U:
            return reduce_typed<
                uchar,
                typename VectorType<uchar>::type>(
                    src, axis, rtype, write);
        case CV_32F:
            return reduce_typed<
                float,
                typename VectorType<float>::type>(
                    src, axis, rtype, write);
        default:
            return false;
    }
#else
    (void)src;
    (void)axis;
    (void)rtype;
    (void)write;
    return false;
#endif
}

template<bool FindMax>
inline bool try_reduce_arg(const Mat& src,
                           Mat& dst,
                           int axis,
                           bool last_index)
{
    if (!enabled())
    {
        return false;
    }

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    switch (src.depth())
    {
        case CV_8U:
            return reduce_arg_typed<
                FindMax,
                uchar,
                typename VectorType<uchar>::type>(
                    src, dst, axis, last_index);
        case CV_8S:
            return reduce_arg_typed<
                FindMax,
                schar,
                typename VectorType<schar>::type>(
                    src, dst, axis, last_index);
        case CV_16U:
            return reduce_arg_typed<
                FindMax,
                ushort,
                typename VectorType<ushort>::type>(
                    src, dst, axis, last_index);
        case CV_16S:
            return reduce_arg_typed<
                FindMax,
                short,
                typename VectorType<short>::type>(
                    src, dst, axis, last_index);
        case CV_32S:
            return reduce_arg_typed<
                FindMax,
                int,
                typename VectorType<int>::type>(
                    src, dst, axis, last_index);
        case CV_32U:
            return reduce_arg_typed<
                FindMax,
                uint,
                typename VectorType<uint>::type>(
                    src, dst, axis, last_index);
        case CV_32F:
            return reduce_arg_typed<
                FindMax,
                float,
                typename VectorType<float>::type>(
                    src, dst, axis, last_index);
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
        case CV_64F:
            return reduce_arg_typed<
                FindMax,
                double,
                typename VectorType<double>::type>(
                    src, dst, axis, last_index);
#endif
        default:
            return false;
    }
#else
    (void)src;
    (void)dst;
    (void)axis;
    (void)last_index;
    return false;
#endif
}

}  // namespace reduce_ui
}  // namespace detail
}  // namespace cvh

#endif  // CVH_CORE_DETAIL_REDUCE_UI_HPP
