#ifndef CVH_CORE_DETAIL_RANDOM_IMPL_HPP
#define CVH_CORE_DETAIL_RANDOM_IMPL_HPP

#include "../random.h"
#include "../saturate.h"
#include "dispatch_control.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

namespace cvh
{
namespace detail
{

class RandomEngine64
{
public:
    using result_type = std::uint64_t;

    static constexpr result_type min() { return 0; }
    static constexpr result_type max()
    {
        return std::numeric_limits<result_type>::max();
    }

    result_type operator()()
    {
        result_type value = state_;
        value ^= value >> 12;
        value ^= value << 25;
        value ^= value >> 27;
        state_ = value;
        return value * 0x2545F4914F6CDD1DULL;
    }

private:
    result_type state_ = 0x9e3779b97f4a7c15ULL;
};

inline thread_local RandomEngine64 random_engine;

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
inline void fill_random_span(T* values,
                             std::size_t pixel_count,
                             int channels,
                             Generator& generator)
{
    switch (channels)
    {
        case 1:
            for (std::size_t pixel = 0; pixel < pixel_count; ++pixel)
                values[pixel] = generator(0);
            break;
        case 2:
            for (std::size_t pixel = 0; pixel < pixel_count; ++pixel)
            {
                values[2 * pixel] = generator(0);
                values[2 * pixel + 1] = generator(1);
            }
            break;
        case 3:
            for (std::size_t pixel = 0; pixel < pixel_count; ++pixel)
            {
                values[3 * pixel] = generator(0);
                values[3 * pixel + 1] = generator(1);
                values[3 * pixel + 2] = generator(2);
            }
            break;
        default:
            for (std::size_t pixel = 0; pixel < pixel_count; ++pixel)
            {
                values[4 * pixel] = generator(0);
                values[4 * pixel + 1] = generator(1);
                values[4 * pixel + 2] = generator(2);
                values[4 * pixel + 3] = generator(3);
            }
            break;
    }
}

template<typename T, typename Generator>
inline void fill_random_typed(Mat& dst, Generator&& generator)
{
    const int channels = dst.channels();
    if (dst.isContinuous())
    {
        fill_random_span(
            reinterpret_cast<T*>(dst.data), dst.total(), channels, generator);
        return;
    }

    for (int row = 0; row < dst.size[0]; ++row)
    {
        T* values = reinterpret_cast<T*>(
            dst.data + static_cast<std::size_t>(row) * dst.step(0));
        fill_random_span(
            values, static_cast<std::size_t>(dst.size[1]), channels, generator);
    }
}

template<typename T>
inline void fill_uniform_typed(Mat& dst, const Scalar& low, const Scalar& high)
{
    if constexpr (std::is_integral<T>::value)
    {
        std::array<std::int64_t, 4> lower {};
        std::array<std::int64_t, 4> upper {};
        std::array<std::uint32_t, 4> spans {};
        std::array<std::uint32_t, 4> rejection_thresholds {};
        bool all_constant = true;
        bool all_small_ranges = true;
        bool all_values_in_type = true;
        for (int channel = 0; channel < 4; ++channel)
        {
            double lower_value = std::ceil(std::min(low[channel], high[channel]));
            double upper_value = std::floor(std::max(low[channel], high[channel])) - 1.0;
            if (upper_value < lower_value)
                upper_value = lower_value;
            const auto clamp_to_int64 = [](double value) {
                if (value <= static_cast<double>(
                        std::numeric_limits<std::int64_t>::lowest()))
                    return std::numeric_limits<std::int64_t>::lowest();
                if (value >= static_cast<double>(
                        std::numeric_limits<std::int64_t>::max()))
                    return std::numeric_limits<std::int64_t>::max();
                return static_cast<std::int64_t>(value);
            };
            lower[static_cast<std::size_t>(channel)] =
                clamp_to_int64(lower_value);
            upper[static_cast<std::size_t>(channel)] =
                clamp_to_int64(upper_value);
            all_constant = all_constant &&
                lower[static_cast<std::size_t>(channel)] ==
                upper[static_cast<std::size_t>(channel)];
            all_values_in_type = all_values_in_type &&
                lower[static_cast<std::size_t>(channel)] >=
                    static_cast<std::int64_t>(std::numeric_limits<T>::lowest()) &&
                upper[static_cast<std::size_t>(channel)] <=
                    static_cast<std::int64_t>(std::numeric_limits<T>::max());
            const std::uint64_t span =
                static_cast<std::uint64_t>(
                    upper[static_cast<std::size_t>(channel)]) -
                static_cast<std::uint64_t>(
                    lower[static_cast<std::size_t>(channel)]) + 1;
            all_small_ranges = all_small_ranges && span != 0 &&
                span <= std::numeric_limits<std::uint32_t>::max();
            if (span != 0 && span <= std::numeric_limits<std::uint32_t>::max())
            {
                spans[static_cast<std::size_t>(channel)] =
                    static_cast<std::uint32_t>(span);
                rejection_thresholds[static_cast<std::size_t>(channel)] =
                    static_cast<std::uint32_t>(-spans[static_cast<std::size_t>(channel)]) %
                    spans[static_cast<std::size_t>(channel)];
            }
        }

        if (all_constant)
        {
            auto constant = [&](int channel) {
                return saturate_cast<T>(
                    lower[static_cast<std::size_t>(channel)]);
            };
            fill_random_typed<T>(dst, constant);
            return;
        }

        if (all_small_ranges)
        {
            if (all_values_in_type)
            {
                auto sample = [&](int channel) {
                    const std::size_t index = static_cast<std::size_t>(channel);
                    std::uint64_t product =
                        static_cast<std::uint64_t>(
                            static_cast<std::uint32_t>(random_engine() >> 32)) *
                        spans[index];
                    std::uint32_t product_low =
                        static_cast<std::uint32_t>(product);
                    while (product_low < rejection_thresholds[index])
                    {
                        product = static_cast<std::uint64_t>(
                            static_cast<std::uint32_t>(random_engine() >> 32)) *
                            spans[index];
                        product_low = static_cast<std::uint32_t>(product);
                    }
                    return static_cast<T>(lower[index] +
                        static_cast<std::int64_t>(product >> 32));
                };
                fill_random_typed<T>(dst, sample);
                return;
            }
            auto sample = [&](int channel) {
                const std::size_t index = static_cast<std::size_t>(channel);
                std::uint64_t product =
                    static_cast<std::uint64_t>(
                        static_cast<std::uint32_t>(random_engine() >> 32)) *
                    spans[index];
                std::uint32_t product_low =
                    static_cast<std::uint32_t>(product);
                while (product_low < rejection_thresholds[index])
                {
                    product = static_cast<std::uint64_t>(
                        static_cast<std::uint32_t>(random_engine() >> 32)) *
                        spans[index];
                    product_low = static_cast<std::uint32_t>(product);
                }
                const std::int64_t value = lower[index] +
                    static_cast<std::int64_t>(product >> 32);
                return saturate_cast<T>(value);
            };
            fill_random_typed<T>(dst, sample);
            return;
        }

        using Distribution = std::uniform_int_distribution<std::int64_t>;
        std::array<Distribution, 4> distributions = {
            Distribution(lower[0], upper[0]), Distribution(lower[1], upper[1]),
            Distribution(lower[2], upper[2]), Distribution(lower[3], upper[3])};
        auto sample = [&](int channel) {
            return saturate_cast<T>(
                distributions[static_cast<std::size_t>(channel)](random_engine));
        };
        fill_random_typed<T>(dst, sample);
    }
    else
    {
        std::array<double, 4> lower {};
        std::array<double, 4> upper {};
        bool all_constant = true;
        for (int channel = 0; channel < 4; ++channel)
        {
            lower[static_cast<std::size_t>(channel)] =
                std::min(low[channel], high[channel]);
            upper[static_cast<std::size_t>(channel)] =
                std::max(low[channel], high[channel]);
            all_constant = all_constant &&
                lower[static_cast<std::size_t>(channel)] ==
                upper[static_cast<std::size_t>(channel)];
        }

        if (all_constant)
        {
            auto constant = [&](int channel) {
                return static_cast<T>(lower[static_cast<std::size_t>(channel)]);
            };
            fill_random_typed<T>(dst, constant);
            return;
        }

        using Distribution = std::uniform_real_distribution<double>;
        std::array<Distribution, 4> distributions = {
            Distribution(lower[0], upper[0]), Distribution(lower[1], upper[1]),
            Distribution(lower[2], upper[2]), Distribution(lower[3], upper[3])};
        auto sample = [&](int channel) {
            return static_cast<T>(
                distributions[static_cast<std::size_t>(channel)](random_engine));
        };
        fill_random_typed<T>(dst, sample);
    }
}

template<typename T>
inline void fill_normal_typed(Mat& dst,
                              const Scalar& mean,
                              const Scalar& stddev)
{
    std::array<double, 4> sigma {};
    bool all_constant = true;
    for (int channel = 0; channel < 4; ++channel)
    {
        sigma[static_cast<std::size_t>(channel)] = std::fabs(stddev[channel]);
        all_constant = all_constant && sigma[static_cast<std::size_t>(channel)] == 0.0;
    }

    if (all_constant)
    {
        std::array<T, 4> constants = {
            saturate_cast<T>(mean[0]), saturate_cast<T>(mean[1]),
            saturate_cast<T>(mean[2]), saturate_cast<T>(mean[3])};
        auto constant = [&](int channel) {
            return constants[static_cast<std::size_t>(channel)];
        };
        fill_random_typed<T>(dst, constant);
        return;
    }

    using Distribution = std::normal_distribution<double>;
    std::array<Distribution, 4> distributions = {
        Distribution(mean[0], sigma[0] == 0.0 ? 1.0 : sigma[0]),
        Distribution(mean[1], sigma[1] == 0.0 ? 1.0 : sigma[1]),
        Distribution(mean[2], sigma[2] == 0.0 ? 1.0 : sigma[2]),
        Distribution(mean[3], sigma[3] == 0.0 ? 1.0 : sigma[3])};
    auto sample = [&](int channel) {
        const std::size_t index = static_cast<std::size_t>(channel);
        return sigma[index] == 0.0
            ? saturate_cast<T>(mean[channel])
            : saturate_cast<T>(distributions[index](random_engine));
    };
    fill_random_typed<T>(dst, sample);
}

}  // namespace detail

inline void randu(Mat& dst, const Scalar& low, const Scalar& high)
{
    detail::validate_random_destination(dst, "randu");
    switch (dst.depth())
    {
        case CV_8U: detail::fill_uniform_typed<uchar>(dst, low, high); break;
        case CV_8S: detail::fill_uniform_typed<schar>(dst, low, high); break;
        case CV_16U: detail::fill_uniform_typed<ushort>(dst, low, high); break;
        case CV_16S: detail::fill_uniform_typed<short>(dst, low, high); break;
        case CV_32S: detail::fill_uniform_typed<int>(dst, low, high); break;
        case CV_32F: detail::fill_uniform_typed<float>(dst, low, high); break;
        case CV_64F: detail::fill_uniform_typed<double>(dst, low, high); break;
        default: CV_Error(Error::StsUnsupportedFormat, "unsupported random destination");
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
}

inline void randn(Mat& dst, const Scalar& mean, const Scalar& stddev)
{
    detail::validate_random_destination(dst, "randn");
    switch (dst.depth())
    {
        case CV_8U: detail::fill_normal_typed<uchar>(dst, mean, stddev); break;
        case CV_8S: detail::fill_normal_typed<schar>(dst, mean, stddev); break;
        case CV_16U: detail::fill_normal_typed<ushort>(dst, mean, stddev); break;
        case CV_16S: detail::fill_normal_typed<short>(dst, mean, stddev); break;
        case CV_32S: detail::fill_normal_typed<int>(dst, mean, stddev); break;
        case CV_32F: detail::fill_normal_typed<float>(dst, mean, stddev); break;
        case CV_64F: detail::fill_normal_typed<double>(dst, mean, stddev); break;
        default: CV_Error(Error::StsUnsupportedFormat, "unsupported random destination");
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
}

}  // namespace cvh

#endif  // CVH_CORE_DETAIL_RANDOM_IMPL_HPP
