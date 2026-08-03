#ifndef CVH_IMGPROC_HISTOGRAM_H
#define CVH_IMGPROC_HISTOGRAM_H

#include "../core/mat.h"

namespace cvh
{

enum HistCompMethods
{
    HISTCMP_CORREL = 0,
    HISTCMP_CHISQR = 1,
    HISTCMP_INTERSECT = 2,
    HISTCMP_BHATTACHARYYA = 3,
    HISTCMP_HELLINGER = HISTCMP_BHATTACHARYYA,
    HISTCMP_CHISQR_ALT = 4,
    HISTCMP_KL_DIV = 5,
};

void calcHist(const Mat* images, int nimages, const int* channels,
              const Mat& mask, Mat& histogram, int dims,
              const int* histogram_sizes, const float** ranges,
              bool uniform = true, bool accumulate = false);

void calcHist(const Mat& image, int channel, const Mat& mask, Mat& histogram,
              int histogram_size, float lower, float upper,
              bool accumulate = false);

double compareHist(const Mat& histogram1, const Mat& histogram2, int method);

}  // namespace cvh

#include "detail/histogram_impl.hpp"

#endif  // CVH_IMGPROC_HISTOGRAM_H
