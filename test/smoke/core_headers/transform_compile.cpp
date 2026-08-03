#include "cvh/core/transform.h"

int cvh_core_transform_header_compile()
{
    cvh::Mat source({1, 1}, CV_32FC2);
    cvh::Mat matrix({2, 3}, CV_32FC1);
    cvh::Mat destination;
    cvh::transform(source, destination, matrix);
    return destination.channels();
}
