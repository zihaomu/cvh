#include "cvh/core/random.h"

int cvh_core_random_header_compile()
{
    cvh::Mat matrix({1, 2}, CV_32FC1);
    cvh::randu(matrix, cvh::Scalar(0.0), cvh::Scalar(1.0));
    return static_cast<int>(matrix.total());
}
