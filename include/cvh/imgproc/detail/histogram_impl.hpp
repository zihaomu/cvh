#ifndef CVH_IMGPROC_DETAIL_HISTOGRAM_IMPL_HPP
#define CVH_IMGPROC_DETAIL_HISTOGRAM_IMPL_HPP

#include "../histogram.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace cvh
{

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
    else if (!accumulate)
    {
        histogram = 0.0f;
    }

    const double scale = static_cast<double>(histogram_size) /
                         (static_cast<double>(upper) - lower);
    for (int row = 0; row < image.size[0]; ++row)
    {
        for (int column = 0; column < image.size[1]; ++column)
        {
            if (!mask.empty() && mask.at<uchar>(row, column) == 0)
            {
                continue;
            }
            const double value = image.depth() == CV_8U
                ? static_cast<double>(image.at<uchar>(row, column, channel))
                : static_cast<double>(image.at<float>(row, column, channel));
            if (value < lower || value >= upper)
            {
                continue;
            }
            int bin = static_cast<int>((value - lower) * scale);
            bin = std::min(bin, histogram_size - 1);
            histogram.at<float>(bin, 0) += 1.0f;
        }
    }
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
    if (method != HISTCMP_CORREL && method != HISTCMP_CHISQR &&
        method != HISTCMP_INTERSECT && method != HISTCMP_BHATTACHARYYA)
    {
        CV_Error(Error::StsBadFlag, "compareHist method is unsupported in P2-P0");
    }

    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum11 = 0.0;
    double sum12 = 0.0;
    double sum22 = 0.0;
    double result = 0.0;
    const size_t count = histogram1.total();
    for (size_t index = 0; index < count; ++index)
    {
        const double left = histogram1.at<float>(static_cast<int>(index));
        const double right = histogram2.at<float>(static_cast<int>(index));
        if (method == HISTCMP_CHISQR)
        {
            if (std::fabs(left) > DBL_EPSILON)
            {
                const double difference = left - right;
                result += difference * difference / left;
            }
        }
        else if (method == HISTCMP_INTERSECT)
        {
            result += std::min(left, right);
        }
        else if (method == HISTCMP_BHATTACHARYYA)
        {
            result += std::sqrt(left * right);
            sum1 += left;
            sum2 += right;
        }
        else
        {
            sum1 += left;
            sum2 += right;
            sum11 += left * left;
            sum12 += left * right;
            sum22 += right * right;
        }
    }

    if (method == HISTCMP_CORREL)
    {
        const double scale = 1.0 / static_cast<double>(count);
        const double numerator = sum12 - sum1 * sum2 * scale;
        const double denominator_squared =
            (sum11 - sum1 * sum1 * scale) * (sum22 - sum2 * sum2 * scale);
        return denominator_squared > DBL_EPSILON
            ? numerator / std::sqrt(denominator_squared)
            : 1.0;
    }
    if (method == HISTCMP_BHATTACHARYYA)
    {
        const double product = sum1 * sum2;
        const double coefficient = product > DBL_EPSILON ? result / std::sqrt(product) : 0.0;
        return std::sqrt(std::max(1.0 - coefficient, 0.0));
    }
    return result;
}

}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_HISTOGRAM_IMPL_HPP
