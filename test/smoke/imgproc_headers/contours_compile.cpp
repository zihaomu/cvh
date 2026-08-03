#include "cvh/imgproc/contours.h"

int cvh_imgproc_contours_header_compile()
{
    cvh::Mat image({1, 1}, CV_8UC1);
    image = 0;
    std::vector<std::vector<cvh::Point>> contours;
    cvh::findContours(image, contours, cvh::RETR_LIST, cvh::CHAIN_APPROX_SIMPLE);
    return static_cast<int>(contours.size());
}
