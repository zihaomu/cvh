#include "cvh/imgproc/template_match.h"

int cvh_imgproc_template_match_header_compile()
{
    cvh::Mat image({1, 1}, CV_8UC1);
    image = 0;
    cvh::Mat result;
    cvh::matchTemplate(image, image, result, cvh::TM_SQDIFF);
    return static_cast<int>(result.total());
}
