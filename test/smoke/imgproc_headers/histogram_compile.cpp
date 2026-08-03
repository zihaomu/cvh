#include "cvh/imgproc/histogram.h"

int cvh_imgproc_histogram_header_compile()
{
    cvh::Mat image({1, 1}, CV_8UC1);
    image = 0;
    cvh::Mat histogram;
    cvh::calcHist(image, 0, cvh::Mat(), histogram, 2, 0.0f, 256.0f);
    return static_cast<int>(histogram.total());
}
