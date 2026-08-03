#ifndef CVH_IMGPROC_TEMPLATE_MATCH_H
#define CVH_IMGPROC_TEMPLATE_MATCH_H

#include "../core/mat.h"

namespace cvh
{

enum TemplateMatchModes
{
    TM_SQDIFF = 0,
    TM_SQDIFF_NORMED = 1,
    TM_CCORR = 2,
    TM_CCORR_NORMED = 3,
    TM_CCOEFF = 4,
    TM_CCOEFF_NORMED = 5,
};

void matchTemplate(const Mat& image, const Mat& templ, Mat& result, int method);

}  // namespace cvh

#include "detail/template_match_impl.hpp"

#endif  // CVH_IMGPROC_TEMPLATE_MATCH_H
