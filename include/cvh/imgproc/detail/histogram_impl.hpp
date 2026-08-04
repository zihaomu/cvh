#ifndef CVH_IMGPROC_DETAIL_HISTOGRAM_IMPL_HPP
#define CVH_IMGPROC_DETAIL_HISTOGRAM_IMPL_HPP

#include "../histogram.h"
#include "cvh/core/detail/dispatch_control.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <vector>

namespace cvh
{
namespace detail
{

inline float histogram_value(const Mat& histogram, std::size_t index)
{
    if (histogram.isContinuous())
    {
        return reinterpret_cast<const float*>(histogram.data)[index];
    }
    std::size_t offset = 0;
    for (int dimension = histogram.dims - 1; dimension >= 0; --dimension)
    {
        const std::size_t extent =
            static_cast<std::size_t>(histogram.size[dimension]);
        offset += (index % extent) * histogram.step(dimension);
        index /= extent;
    }
    return *reinterpret_cast<const float*>(histogram.data + offset);
}

inline void add_histogram_counts(Mat& histogram,
                                 const std::vector<float>& counts)
{
    if (histogram.isContinuous())
    {
        float* destination = reinterpret_cast<float*>(histogram.data);
        for (std::size_t index = 0; index < counts.size(); ++index)
        {
            destination[index] += counts[index];
        }
        return;
    }
    for (std::size_t index = 0; index < counts.size(); ++index)
    {
        std::size_t remaining = index;
        std::size_t offset = 0;
        for (int dimension = histogram.dims - 1; dimension >= 0; --dimension)
        {
            const std::size_t extent =
                static_cast<std::size_t>(histogram.size[dimension]);
            offset += (remaining % extent) * histogram.step(dimension);
            remaining /= extent;
        }
        *reinterpret_cast<float*>(histogram.data + offset) += counts[index];
    }
}

inline void calc_hist_u8(const Mat& image,
                         int channel,
                         const Mat& mask,
                         std::vector<float>& counts,
                         float lower,
                         float upper)
{
    std::array<int, 256> bins {};
    bins.fill(-1);
    const double scale = static_cast<double>(counts.size()) /
        (static_cast<double>(upper) - lower);
    for (int value = 0; value < 256; ++value)
    {
        if (value >= lower && value < upper)
        {
            bins[static_cast<std::size_t>(value)] = std::min(
                static_cast<int>((static_cast<double>(value) - lower) * scale),
                static_cast<int>(counts.size()) - 1);
        }
    }

    const int channels = image.channels();
    for (int row = 0; row < image.size[0]; ++row)
    {
        const uchar* source = image.data +
            static_cast<std::size_t>(row) * image.step(0) + channel;
        const uchar* mask_row = mask.empty()
            ? nullptr
            : mask.data + static_cast<std::size_t>(row) * mask.step(0);
        for (int column = 0; column < image.size[1]; ++column)
        {
            if (mask_row == nullptr || mask_row[column] != 0)
            {
                const int bin = bins[source[
                    static_cast<std::size_t>(column) * channels]];
                if (bin >= 0)
                {
                    counts[static_cast<std::size_t>(bin)] += 1.0f;
                }
            }
        }
    }
}

inline void calc_hist_f32(const Mat& image,
                          int channel,
                          const Mat& mask,
                          std::vector<float>& counts,
                          float lower,
                          float upper)
{
    const int channels = image.channels();
    const double scale = static_cast<double>(counts.size()) /
        (static_cast<double>(upper) - lower);
    for (int row = 0; row < image.size[0]; ++row)
    {
        const float* source = reinterpret_cast<const float*>(
            image.data + static_cast<std::size_t>(row) * image.step(0)) +
            channel;
        const uchar* mask_row = mask.empty()
            ? nullptr
            : mask.data + static_cast<std::size_t>(row) * mask.step(0);
        for (int column = 0; column < image.size[1]; ++column)
        {
            if (mask_row != nullptr && mask_row[column] == 0)
            {
                continue;
            }
            const double value = source[
                static_cast<std::size_t>(column) * channels];
            if (value < lower || value >= upper)
            {
                continue;
            }
            const int bin = std::min(
                static_cast<int>((value - lower) * scale),
                static_cast<int>(counts.size()) - 1);
            counts[static_cast<std::size_t>(bin)] += 1.0f;
        }
    }
}

inline double compare_hist_correlation(const Mat& left_histogram,
                                       const Mat& right_histogram)
{
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum11 = 0.0;
    double sum12 = 0.0;
    double sum22 = 0.0;
    const std::size_t count = left_histogram.total();
    if (left_histogram.isContinuous() && right_histogram.isContinuous())
    {
        const float* left_values =
            reinterpret_cast<const float*>(left_histogram.data);
        const float* right_values =
            reinterpret_cast<const float*>(right_histogram.data);
        for (std::size_t index = 0; index < count; ++index)
        {
            const double left = left_values[index];
            const double right = right_values[index];
            sum1 += left;
            sum2 += right;
            sum11 += left * left;
            sum12 += left * right;
            sum22 += right * right;
        }
    }
    else
    {
        for (std::size_t index = 0; index < count; ++index)
        {
            const double left = histogram_value(left_histogram, index);
            const double right = histogram_value(right_histogram, index);
            sum1 += left;
            sum2 += right;
            sum11 += left * left;
            sum12 += left * right;
            sum22 += right * right;
        }
    }
    const double scale = 1.0 / static_cast<double>(count);
    const double numerator = sum12 - sum1 * sum2 * scale;
    const double denominator_squared =
        (sum11 - sum1 * sum1 * scale) *
        (sum22 - sum2 * sum2 * scale);
    return denominator_squared > DBL_EPSILON
        ? numerator / std::sqrt(denominator_squared)
        : 1.0;
}

inline double compare_hist_chisqr(const Mat& left_histogram,
                                  const Mat& right_histogram)
{
    double result = 0.0;
    const std::size_t count = left_histogram.total();
    if (left_histogram.isContinuous() && right_histogram.isContinuous())
    {
        const float* left_values =
            reinterpret_cast<const float*>(left_histogram.data);
        const float* right_values =
            reinterpret_cast<const float*>(right_histogram.data);
        for (std::size_t index = 0; index < count; ++index)
        {
            const double left = left_values[index];
            if (std::fabs(left) > DBL_EPSILON)
            {
                const double difference = left - right_values[index];
                result += difference * difference / left;
            }
        }
    }
    else
    {
        for (std::size_t index = 0; index < count; ++index)
        {
            const double left = histogram_value(left_histogram, index);
            if (std::fabs(left) > DBL_EPSILON)
            {
                const double difference =
                    left - histogram_value(right_histogram, index);
                result += difference * difference / left;
            }
        }
    }
    return result;
}

inline double compare_hist_intersect(const Mat& left_histogram,
                                     const Mat& right_histogram)
{
    double result = 0.0;
    const std::size_t count = left_histogram.total();
    if (left_histogram.isContinuous() && right_histogram.isContinuous())
    {
        const float* left_values =
            reinterpret_cast<const float*>(left_histogram.data);
        const float* right_values =
            reinterpret_cast<const float*>(right_histogram.data);
        for (std::size_t index = 0; index < count; ++index)
        {
            result += std::min(
                static_cast<double>(left_values[index]),
                static_cast<double>(right_values[index]));
        }
    }
    else
    {
        for (std::size_t index = 0; index < count; ++index)
        {
            result += std::min(
                static_cast<double>(histogram_value(left_histogram, index)),
                static_cast<double>(histogram_value(right_histogram, index)));
        }
    }
    return result;
}

inline double compare_hist_bhattacharyya(const Mat& left_histogram,
                                         const Mat& right_histogram)
{
    double coefficient_sum = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    const std::size_t count = left_histogram.total();
    if (left_histogram.isContinuous() && right_histogram.isContinuous())
    {
        const float* left_values =
            reinterpret_cast<const float*>(left_histogram.data);
        const float* right_values =
            reinterpret_cast<const float*>(right_histogram.data);
        for (std::size_t index = 0; index < count; ++index)
        {
            const double left = left_values[index];
            const double right = right_values[index];
            coefficient_sum += std::sqrt(left * right);
            sum1 += left;
            sum2 += right;
        }
    }
    else
    {
        for (std::size_t index = 0; index < count; ++index)
        {
            const double left = histogram_value(left_histogram, index);
            const double right = histogram_value(right_histogram, index);
            coefficient_sum += std::sqrt(left * right);
            sum1 += left;
            sum2 += right;
        }
    }
    const double product = sum1 * sum2;
    const double coefficient = product > DBL_EPSILON
        ? coefficient_sum / std::sqrt(product)
        : 0.0;
    return std::sqrt(std::max(1.0 - coefficient, 0.0));
}

}  // namespace detail

inline void calcHist(const Mat& image, int channel, const Mat& mask, Mat& histogram,
                     int histogram_size, float lower, float upper, bool accumulate)
{
    if (image.empty() || image.dims != 2 ||
        (image.depth() != CV_8U && image.depth() != CV_32F) ||
        (image.channels() != 1 && image.channels() != 3 && image.channels() != 4))
    {
        CV_Error(Error::StsUnsupportedFormat, "calcHist supports non-empty 2D U8/F32 C1/C3/C4 image");
    }
    if (channel < 0 || channel >= image.channels())
    {
        CV_Error(Error::StsOutOfRange, "calcHist channel is out of range");
    }
    if (histogram_size <= 0 || !(lower < upper))
    {
        CV_Error(Error::StsBadArg, "calcHist expects positive bins and an increasing range");
    }
    if (!mask.empty() &&
        (mask.dims != 2 || mask.type() != CV_8UC1 ||
         mask.size[0] != image.size[0] || mask.size[1] != image.size[1]))
    {
        CV_Error(Error::StsBadMask, "calcHist mask must be CV_8UC1 and match image geometry");
    }

    const bool preserve = accumulate && !histogram.empty() &&
        histogram.type() == CV_32FC1 && histogram.dims == 2 &&
        histogram.size[0] == histogram_size && histogram.size[1] == 1;
    if (!preserve)
    {
        histogram.create({histogram_size, 1}, CV_32FC1);
        histogram = 0.0f;
    }

    std::vector<float> counts(static_cast<std::size_t>(histogram_size), 0.0f);
    if (image.depth() == CV_8U)
    {
        detail::calc_hist_u8(
            image, channel, mask, counts, lower, upper);
    }
    else
    {
        detail::calc_hist_f32(
            image, channel, mask, counts, lower, upper);
    }
    detail::add_histogram_counts(histogram, counts);
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
}

inline void calcHist(const Mat* images, int nimages, const int* channels,
                     const Mat& mask, Mat& histogram, int dims,
                     const int* histogram_sizes, const float** ranges,
                     bool uniform, bool accumulate)
{
    if (images == nullptr || nimages != 1 || channels == nullptr ||
        dims != 1 || histogram_sizes == nullptr || ranges == nullptr ||
        ranges[0] == nullptr)
    {
        CV_Error(Error::StsBadArg, "calcHist P2-P0 supports one image and one histogram dimension");
    }
    if (!uniform)
    {
        CV_Error(Error::StsBadFlag, "calcHist P2-P0 supports uniform bins only");
    }
    calcHist(images[0], channels[0], mask, histogram, histogram_sizes[0],
             ranges[0][0], ranges[0][1], accumulate);
}

inline double compareHist(const Mat& histogram1, const Mat& histogram2, int method)
{
    if (histogram1.empty() || histogram2.empty() ||
        histogram1.type() != CV_32FC1 || histogram2.type() != CV_32FC1 ||
        histogram1.total() != histogram2.total())
    {
        CV_Error(Error::StsUnmatchedFormats, "compareHist expects equal-size dense F32 C1 histograms");
    }

    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    switch (method)
    {
        case HISTCMP_CORREL:
            return detail::compare_hist_correlation(histogram1, histogram2);
        case HISTCMP_CHISQR:
            return detail::compare_hist_chisqr(histogram1, histogram2);
        case HISTCMP_INTERSECT:
            return detail::compare_hist_intersect(histogram1, histogram2);
        case HISTCMP_BHATTACHARYYA:
            return detail::compare_hist_bhattacharyya(histogram1, histogram2);
        default:
            CV_Error(Error::StsBadFlag, "compareHist method is unsupported in P2-P0");
    }
    return 0.0;
}

}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_HISTOGRAM_IMPL_HPP
