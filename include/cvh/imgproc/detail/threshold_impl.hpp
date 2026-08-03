#ifndef CVH_IMGPROC_DETAIL_THRESHOLD_IMPL_HPP
#define CVH_IMGPROC_DETAIL_THRESHOLD_IMPL_HPP

#include "fastpath_common.hpp"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/simd/opencv_ui.h"

namespace cvh
{
namespace detail
{

namespace threshold_fastpath
{
inline bool ui_enabled()
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    return cpu::opencv_ui_allowed();
#else
    return false;
#endif
}

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)

template<int ThresholdType, typename Vector>
inline Vector apply_threshold_vector(const Vector& value,
                                     const Vector& threshold,
                                     const Vector& maximum,
                                     const Vector& zero)
{
    const Vector above = cv::v_lt(threshold, value);
    if constexpr (ThresholdType == THRESH_BINARY)
    {
        return cv::v_select(above, maximum, zero);
    }
    else if constexpr (ThresholdType == THRESH_BINARY_INV)
    {
        return cv::v_select(above, zero, maximum);
    }
    else if constexpr (ThresholdType == THRESH_TRUNC)
    {
        return cv::v_select(above, threshold, value);
    }
    else if constexpr (ThresholdType == THRESH_TOZERO)
    {
        return cv::v_select(above, value, zero);
    }
    else
    {
        return cv::v_select(above, zero, value);
    }
}

template<int ThresholdType, typename T>
inline T apply_threshold_scalar(T value, T threshold, T maximum)
{
    const bool above = value > threshold;
    if constexpr (ThresholdType == THRESH_BINARY)
    {
        return above ? maximum : T(0);
    }
    else if constexpr (ThresholdType == THRESH_BINARY_INV)
    {
        return above ? T(0) : maximum;
    }
    else if constexpr (ThresholdType == THRESH_TRUNC)
    {
        return above ? threshold : value;
    }
    else if constexpr (ThresholdType == THRESH_TOZERO)
    {
        return above ? value : T(0);
    }
    else
    {
        return above ? T(0) : value;
    }
}

template<int ThresholdType>
inline void threshold_rows_u8(const Mat& src,
                              Mat& dst,
                              uchar threshold,
                              uchar maximum)
{
    const cv::v_uint8 threshold_vector = cv::vx_setall_u8(threshold);
    const cv::v_uint8 maximum_vector = cv::vx_setall_u8(maximum);
    const cv::v_uint8 zero_vector = cv::vx_setzero_u8();
    const size_t lanes =
        static_cast<size_t>(cv::VTraits<cv::v_uint8>::vlanes());

    const bool continuous = src.isContinuous() && dst.isContinuous();
    const int rows = continuous ? 1 : src.size.p[0];
    const size_t row_scalars =
        continuous
            ? src.total() * static_cast<size_t>(src.channels())
            : static_cast<size_t>(src.size.p[1]) *
                  static_cast<size_t>(src.channels());

    for (int y = 0; y < rows; ++y)
    {
        const uchar* input =
            src.data + static_cast<size_t>(y) * src.step(0);
        uchar* output =
            dst.data + static_cast<size_t>(y) * dst.step(0);
        size_t x = 0;
        for (; x + lanes <= row_scalars; x += lanes)
        {
            cv::vx_store(
                output + x,
                apply_threshold_vector<ThresholdType>(
                    cv::vx_load(input + x),
                    threshold_vector,
                    maximum_vector,
                    zero_vector));
        }
        for (; x < row_scalars; ++x)
        {
            output[x] = apply_threshold_scalar<ThresholdType>(
                input[x], threshold, maximum);
        }
    }
}

template<int ThresholdType>
inline void threshold_rows_f32(const Mat& src,
                               Mat& dst,
                               float threshold,
                               float maximum)
{
    const cv::v_float32 threshold_vector =
        cv::vx_setall_f32(threshold);
    const cv::v_float32 maximum_vector =
        cv::vx_setall_f32(maximum);
    const cv::v_float32 zero_vector = cv::vx_setzero_f32();
    const size_t lanes =
        static_cast<size_t>(cv::VTraits<cv::v_float32>::vlanes());

    const bool continuous = src.isContinuous() && dst.isContinuous();
    const int rows = continuous ? 1 : src.size.p[0];
    const size_t row_scalars =
        continuous
            ? src.total() * static_cast<size_t>(src.channels())
            : static_cast<size_t>(src.size.p[1]) *
                  static_cast<size_t>(src.channels());

    for (int y = 0; y < rows; ++y)
    {
        const float* input = reinterpret_cast<const float*>(
            src.data + static_cast<size_t>(y) * src.step(0));
        float* output = reinterpret_cast<float*>(
            dst.data + static_cast<size_t>(y) * dst.step(0));
        size_t x = 0;
        for (; x + lanes <= row_scalars; x += lanes)
        {
            cv::vx_store(
                output + x,
                apply_threshold_vector<ThresholdType>(
                    cv::vx_load(input + x),
                    threshold_vector,
                    maximum_vector,
                    zero_vector));
        }
        for (; x < row_scalars; ++x)
        {
            output[x] = apply_threshold_scalar<ThresholdType>(
                input[x], threshold, maximum);
        }
    }
}

template<typename Run>
inline bool dispatch_threshold_type(int threshold_type, Run&& run)
{
    switch (threshold_type)
    {
        case THRESH_BINARY:
            run(std::integral_constant<int, THRESH_BINARY>());
            return true;
        case THRESH_BINARY_INV:
            run(std::integral_constant<int, THRESH_BINARY_INV>());
            return true;
        case THRESH_TRUNC:
            run(std::integral_constant<int, THRESH_TRUNC>());
            return true;
        case THRESH_TOZERO:
            run(std::integral_constant<int, THRESH_TOZERO>());
            return true;
        case THRESH_TOZERO_INV:
            run(std::integral_constant<int, THRESH_TOZERO_INV>());
            return true;
        default:
            return false;
    }
}

#endif

inline bool try_threshold_fastpath_u8(const Mat& src,
                                      Mat& dst,
                                      double thresh,
                                      double maxval,
                                      int type,
                                      double* out_ret)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    if (!ui_enabled() || out_ret == nullptr || src.empty() ||
        src.depth() != CV_8U)
    {
        return false;
    }

    const bool is_dryrun = (type & THRESH_DRYRUN) != 0;
    type &= ~THRESH_DRYRUN;
    const int automatic_thresh = type & (~THRESH_MASK);
    const int threshold_type = type & THRESH_MASK;

    if (automatic_thresh == THRESH_OTSU && src.type() == CV_8UC1)
    {
        thresh = threshold_otsu_u8(src);
    }
    else if (automatic_thresh == THRESH_TRIANGLE &&
             src.type() == CV_8UC1)
    {
        thresh = threshold_triangle_u8(src);
    }
    else if (automatic_thresh != 0)
    {
        return false;
    }

    const double effective_threshold = std::floor(thresh);
    *out_ret = effective_threshold;
    if (is_dryrun)
    {
        return true;
    }
    if (effective_threshold < 0.0 || effective_threshold > 255.0)
    {
        return false;
    }

    dst.create(src.dims, src.size.p, src.type());
    const uchar threshold_u8 =
        static_cast<uchar>(effective_threshold);
    const uchar maximum_u8 = saturate_cast<uchar>(maxval);
    const bool dispatched = dispatch_threshold_type(
        threshold_type,
        [&](auto threshold_tag) {
            threshold_rows_u8<decltype(threshold_tag)::value>(
                src, dst, threshold_u8, maximum_u8);
        });
    if (dispatched)
    {
        cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
    }
    return dispatched;
#else
    (void)src;
    (void)dst;
    (void)thresh;
    (void)maxval;
    (void)type;
    (void)out_ret;
    return false;
#endif
}

inline bool try_threshold_fastpath_f32(const Mat& src, Mat& dst, double thresh, double maxval, int type, double* out_ret)
{
    if (out_ret == nullptr)
    {
        return false;
    }

    if (src.empty() || src.depth() != CV_32F)
    {
        return false;
    }

    const bool is_dryrun = (type & THRESH_DRYRUN) != 0;
    type &= ~THRESH_DRYRUN;

    const int automatic_thresh = type & (~THRESH_MASK);
    const int thresh_type = type & THRESH_MASK;

    if (automatic_thresh == THRESH_OTSU || automatic_thresh == THRESH_TRIANGLE)
    {
        CV_Error(Error::StsBadArg, "threshold: OTSU/TRIANGLE requires CV_8UC1 source");
    }

    if (automatic_thresh != 0)
    {
        CV_Error_(Error::StsBadArg, ("threshold: unsupported automatic threshold flag=%d", automatic_thresh));
    }

    if (thresh_type != THRESH_BINARY &&
        thresh_type != THRESH_BINARY_INV &&
        thresh_type != THRESH_TRUNC &&
        thresh_type != THRESH_TOZERO &&
        thresh_type != THRESH_TOZERO_INV)
    {
        CV_Error_(Error::StsBadArg, ("threshold: unsupported threshold type=%d", thresh_type));
    }

    const float thresh_f = static_cast<float>(thresh);
    const float max_f = static_cast<float>(maxval);
    *out_ret = thresh;

    if (is_dryrun)
    {
        return true;
    }

    dst.create(src.dims, src.size.p, src.type());

#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    if (ui_enabled())
    {
        const bool dispatched = dispatch_threshold_type(
            thresh_type,
            [&](auto threshold_tag) {
                threshold_rows_f32<decltype(threshold_tag)::value>(
                    src, dst, thresh_f, max_f);
            });
        if (dispatched)
        {
            cpu::set_last_dispatch_tag(
                cpu::DispatchTag::OpenCVUI);
            return true;
        }
    }
#endif

    const std::size_t scalar_count = src.total() * static_cast<std::size_t>(src.channels());
    const float* src_ptr = reinterpret_cast<const float*>(src.data);
    float* dst_ptr = reinterpret_cast<float*>(dst.data);

    if (src.isContinuous() && dst.isContinuous())
    {
        const bool do_parallel = should_parallelize_threshold_contiguous(scalar_count);
        parallel_for_count_if(do_parallel, scalar_count, [&](std::size_t i) {
            const float s = src_ptr[i];
            const bool cond = s > thresh_f;
            switch (thresh_type)
            {
            case THRESH_BINARY:
                dst_ptr[i] = cond ? max_f : 0.0f;
                break;
            case THRESH_BINARY_INV:
                dst_ptr[i] = cond ? 0.0f : max_f;
                break;
            case THRESH_TRUNC:
                dst_ptr[i] = cond ? thresh_f : s;
                break;
            case THRESH_TOZERO:
                dst_ptr[i] = cond ? s : 0.0f;
                break;
            case THRESH_TOZERO_INV:
                dst_ptr[i] = cond ? 0.0f : s;
                break;
            default:
                CV_Error_(Error::StsBadArg, ("threshold: unsupported threshold type=%d", thresh_type));
            }
        });
        return true;
    }

    CV_Assert(src.dims == 2 && "threshold: non-contiguous path supports 2D Mat only");
    const int rows = src.size[0];
    const int cols_scalar = src.size[1] * src.channels();
    const std::size_t src_step = src.step(0);
    const std::size_t dst_step = dst.step(0);
    const bool do_parallel = should_parallelize_threshold_rows(rows, cols_scalar);
    parallel_for_index_if(do_parallel, rows, [&](int y) {
        const float* src_row = reinterpret_cast<const float*>(src.data + static_cast<std::size_t>(y) * src_step);
        float* dst_row = reinterpret_cast<float*>(dst.data + static_cast<std::size_t>(y) * dst_step);
        for (int x = 0; x < cols_scalar; ++x)
        {
            const float s = src_row[x];
            const bool cond = s > thresh_f;
            switch (thresh_type)
            {
            case THRESH_BINARY:
                dst_row[x] = cond ? max_f : 0.0f;
                break;
            case THRESH_BINARY_INV:
                dst_row[x] = cond ? 0.0f : max_f;
                break;
            case THRESH_TRUNC:
                dst_row[x] = cond ? thresh_f : s;
                break;
            case THRESH_TOZERO:
                dst_row[x] = cond ? s : 0.0f;
                break;
            case THRESH_TOZERO_INV:
                dst_row[x] = cond ? 0.0f : s;
                break;
            default:
                CV_Error_(Error::StsBadArg, ("threshold: unsupported threshold type=%d", thresh_type));
            }
        }
    });

    return true;
}


} // namespace threshold_fastpath

inline double threshold_fast_impl(const Mat& src, Mat& dst, double thresh, double maxval, int type)
{
    double ret_value = 0.0;
    if (threshold_fastpath::try_threshold_fastpath_u8(
            src, dst, thresh, maxval, type, &ret_value))
    {
        return ret_value;
    }
    if (threshold_fastpath::try_threshold_fastpath_f32(src, dst, thresh, maxval, type, &ret_value))
    {
        return ret_value;
    }

    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    return threshold_fallback(src, dst, thresh, maxval, type);
}

} // namespace detail
} // namespace cvh

#endif // CVH_IMGPROC_DETAIL_THRESHOLD_IMPL_HPP
