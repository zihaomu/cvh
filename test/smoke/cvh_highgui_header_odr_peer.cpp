#include "cvh/highgui/highgui.hpp"

void cvh_highgui_odr_peer()
{
    cvh::namedWindow(
        "cvh_highgui_odr",
        cvh::WINDOW_AUTOSIZE);
    cvh::Mat image({2, 3}, CV_8UC3);
    image = 17;
    cvh::imshow("cvh_highgui_odr", image);
}
