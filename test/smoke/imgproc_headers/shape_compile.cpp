#include "cvh/imgproc/shape.h"

int cvh_imgproc_shape_header_compile()
{
    const std::vector<cvh::Point> points = {cvh::Point(0, 0), cvh::Point(1, 1)};
    return cvh::boundingRect(points).area();
}
