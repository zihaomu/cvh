#include "cvh/cvh.h"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/simd/opencv_ui.h"

#include <cstring>

#if CVH_DETAIL_HAVE_OPENCV_UI
#error "Optimization-disabled smoke requires CVH_DETAIL_HAVE_OPENCV_UI=0"
#endif

int main()
{
    if (std::strcmp(
            cvh::detail::opencv_ui_backend_name(),
            "scalar") != 0)
    {
        return 1;
    }

    cvh::Mat source({7, 67}, CV_8UC1);
    for (int y = 0; y < source.size.p[0]; ++y)
    {
        for (int x = 0; x < source.size.p[1]; ++x)
        {
            source.at<uchar>(y, x) =
                static_cast<uchar>((y * 17 + x * 23) & 255);
        }
    }

    cvh::Mat output;
    cvh::cpu::reset_last_dispatch_tag();
    cvh::medianBlur(source, output, 3);
    if (output.empty() ||
        cvh::cpu::last_dispatch_tag() !=
            cvh::cpu::DispatchTag::Scalar)
    {
        return 2;
    }
    return 0;
}
