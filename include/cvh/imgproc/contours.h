#ifndef CVH_IMGPROC_CONTOURS_H
#define CVH_IMGPROC_CONTOURS_H

#include "../core/mat.h"

#include <vector>

namespace cvh
{

enum RetrievalModes
{
    RETR_EXTERNAL = 0,
    RETR_LIST = 1,
    RETR_CCOMP = 2,
    RETR_TREE = 3,
    RETR_FLOODFILL = 4,
};

enum ContourApproximationModes
{
    CHAIN_APPROX_NONE = 1,
    CHAIN_APPROX_SIMPLE = 2,
    CHAIN_APPROX_TC89_L1 = 3,
    CHAIN_APPROX_TC89_KCOS = 4,
};

void findContours(const Mat& image, std::vector<std::vector<Point>>& contours,
                  int mode, int method, Point offset = Point());

}  // namespace cvh

#include "detail/contours_impl.hpp"

#endif  // CVH_IMGPROC_CONTOURS_H
