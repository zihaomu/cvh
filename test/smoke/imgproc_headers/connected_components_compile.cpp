#include "cvh/imgproc/connected_components.h"

int cvh_imgproc_connected_components_header_compile()
{
    cvh::Mat image({1, 1}, CV_8UC1);
    image = 0;
    cvh::Mat labels;
    return cvh::connectedComponents(image, labels);
}
