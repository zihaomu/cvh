#ifndef CVH_IMGPROC_CONNECTED_COMPONENTS_H
#define CVH_IMGPROC_CONNECTED_COMPONENTS_H

#include "../core/mat.h"

namespace cvh
{

enum ConnectedComponentsTypes
{
    CC_STAT_LEFT = 0,
    CC_STAT_TOP = 1,
    CC_STAT_WIDTH = 2,
    CC_STAT_HEIGHT = 3,
    CC_STAT_AREA = 4,
    CC_STAT_MAX = 5,
};

int connectedComponents(const Mat& image, Mat& labels, int connectivity = 8,
                        int ltype = CV_32S);
int connectedComponentsWithStats(const Mat& image, Mat& labels, Mat& stats,
                                 Mat& centroids, int connectivity = 8,
                                 int ltype = CV_32S);

}  // namespace cvh

#include "detail/connected_components_impl.hpp"

#endif  // CVH_IMGPROC_CONNECTED_COMPONENTS_H
