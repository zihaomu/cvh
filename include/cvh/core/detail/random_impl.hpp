#ifndef CVH_CORE_DETAIL_RANDOM_IMPL_HPP
#define CVH_CORE_DETAIL_RANDOM_IMPL_HPP

#include "../random.h"
#include "../saturate.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace cvh
{
namespace detail
{

inline thread_local std::mt19937_64 random_engine{0xFFFFFFFFULL};

inline void validate_random_destination(const Mat& dst, const char* function_name)
{
    if (dst.empty())
    {
        CV_Error_(Error::StsBadArg, ("%s expects an allocated, non-empty dst", function_name));
    }
    if (dst.channels() < 1 || dst.channels() > 4)
    {
        CV_Error_(Error::StsBadArg, ("%s supports C1-C4", function_name));
    }
    switch (dst.depth())
    {
        case CV_8U:
        case CV_8S:
        case CV_16U:
        case CV_16S:
        case CV_32S:
        case CV_32F:
        case CV_64F:
            break;
        default:
            CV_Error_(Error::StsUnsupportedFormat,
                      ("%s unsupported dst depth=%d", function_name, dst.depth()));
    }
    if (!dst.isContinuous() && dst.dims != 2)
    {
        CV_Error_(Error::StsNotImplemented,
                  ("%s supports non-contiguous layout only for 2D Mat", function_name));
    }
}

template<typename T, typename Generator>
inline void fill_random_typed(Mat& dst, Generator&& generator)
{
    const int channels = dst.channels();
    if (dst.isContinuous())
    {
        T* values = reinterpret_cast<T*>(dst.data);
        const size_t count = dst.total() * static_cast<size_t>(channels);
        for (size_t index = 0; index < count; ++index)
        {
            values[index] = generator(static_cast<int>(index % channels));
        }
        return;
    }

    for (int row = 0; row < dst.size[0]; ++row)
    {
        T* values = reinterpret_cast<T*>(dst.data + static_cast<size_t>(row) * dst.step(0));
        const size_t count = static_cast<size_t>(dst.size[1]) * channels;
        for (size_t index = 0; index < count; ++index)
        {
            values[index] = generator(static_cast<int>(index % channels));
        }
    }
}

template<typename T>
inline T uniform_integer_value(double low, double high)
{
    double lower = std::ceil(std::min(low, high));
    double upper = std::floor(std::max(low, high)) - 1.0;
    lower = std::max(lower, static_cast<double>(std::numeric_limits<T>::lowest()));
    upper = std::min(upper, static_cast<double>(std::numeric_limits<T>::max()));
    if (upper < lower)
    {
        upper = lower;
    }
    std::uniform_int_distribution<std::int64_t> distribution(
        static_cast<std::int64_t>(lower), static_cast<std::int64_t>(upper));
    return static_cast<T>(distribution(random_engine));
}

template<>
inline int uniform_integer_value<int>(double low, double high)
{
    double lower = std::ceil(std::min(low, high));
    double upper = std::floor(std::max(low, high)) - 1.0;
    lower = std::max(lower, static_cast<double>(std::numeric_limits<int>::lowest()));
    upper = std::min(upper, static_cast<double>(std::numeric_limits<int>::max()));
    if (upper < lower)
    {
        upper = lower;
    }
    std::uniform_int_distribution<std::int64_t> distribution(
        static_cast<std::int64_t>(lower), static_cast<std::int64_t>(upper));
    return static_cast<int>(distribution(random_engine));
}

template<typename T>
inline T uniform_floating_value(double low, double high)
{
    const double lower = std::min(low, high);
    const double upper = std::max(low, high);
    if (lower == upper)
    {
        return static_cast<T>(lower);
    }
    std::uniform_real_distribution<double> distribution(lower, upper);
    return static_cast<T>(distribution(random_engine));
}

template<typename T>
inline T normal_value(double mean, double stddev)
{
    std::normal_distribution<double> distribution(mean, std::fabs(stddev));
    return saturate_cast<T>(distribution(random_engine));
}

template<typename Generator>
inline void dispatch_random_fill(Mat& dst, Generator&& generator)
{
    switch (dst.depth())
    {
        case CV_8U: fill_random_typed<uchar>(dst, generator); break;
        case CV_8S: fill_random_typed<schar>(dst, generator); break;
        case CV_16U: fill_random_typed<ushort>(dst, generator); break;
        case CV_16S: fill_random_typed<short>(dst, generator); break;
        case CV_32S: fill_random_typed<int>(dst, generator); break;
        case CV_32F: fill_random_typed<float>(dst, generator); break;
        case CV_64F: fill_random_typed<double>(dst, generator); break;
        default: CV_Error(Error::StsUnsupportedFormat, "unsupported random destination");
    }
}

}  // namespace detail

inline void randu(Mat& dst, const Scalar& low, const Scalar& high)
{
    detail::validate_random_destination(dst, "randu");
    const int depth = dst.depth();
    detail::dispatch_random_fill(dst, [&](int channel) {
        switch (depth)
        {
            case CV_8U: return static_cast<double>(detail::uniform_integer_value<uchar>(low[channel], high[channel]));
            case CV_8S: return static_cast<double>(detail::uniform_integer_value<schar>(low[channel], high[channel]));
            case CV_16U: return static_cast<double>(detail::uniform_integer_value<ushort>(low[channel], high[channel]));
            case CV_16S: return static_cast<double>(detail::uniform_integer_value<short>(low[channel], high[channel]));
            case CV_32S: return static_cast<double>(detail::uniform_integer_value<int>(low[channel], high[channel]));
            case CV_32F: return static_cast<double>(detail::uniform_floating_value<float>(low[channel], high[channel]));
            default: return detail::uniform_floating_value<double>(low[channel], high[channel]);
        }
    });
}

inline void randn(Mat& dst, const Scalar& mean, const Scalar& stddev)
{
    detail::validate_random_destination(dst, "randn");
    const int depth = dst.depth();
    detail::dispatch_random_fill(dst, [&](int channel) {
        switch (depth)
        {
            case CV_8U: return static_cast<double>(detail::normal_value<uchar>(mean[channel], stddev[channel]));
            case CV_8S: return static_cast<double>(detail::normal_value<schar>(mean[channel], stddev[channel]));
            case CV_16U: return static_cast<double>(detail::normal_value<ushort>(mean[channel], stddev[channel]));
            case CV_16S: return static_cast<double>(detail::normal_value<short>(mean[channel], stddev[channel]));
            case CV_32S: return static_cast<double>(detail::normal_value<int>(mean[channel], stddev[channel]));
            case CV_32F: return static_cast<double>(detail::normal_value<float>(mean[channel], stddev[channel]));
            default: return detail::normal_value<double>(mean[channel], stddev[channel]);
        }
    });
}

}  // namespace cvh

#endif  // CVH_CORE_DETAIL_RANDOM_IMPL_HPP
