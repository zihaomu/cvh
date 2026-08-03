#include "cvh/highgui/highgui.h"

void cvh_highgui_odr_peer();

int main()
{
    cvh_highgui_odr_peer();
    {
        cvh::detail::HighguiRegistry& registry =
            cvh::detail::highgui_registry();
        std::lock_guard<std::mutex> lock(registry.mutex);
        if (registry.windows.count("cvh_highgui_odr") != 1)
        {
            return 2;
        }
    }
    const int key = cvh::waitKey(1);
    cvh::destroyWindow("cvh_highgui_odr");
    cvh::destroyAllWindows();
    return key == -1 ? 0 : 1;
}
