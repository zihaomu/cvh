#include "cvh/pipeline/pipeline.h"

#include <cstdint>

std::uint64_t cvh_pipeline_header_odr_peer();

int main()
{
    cvh::Mat input({2, 2}, CV_8UC3);
    input = cvh::Scalar(1, 2, 3);
    cvh::Mat output;
    cvh::pipe(input, output)
        .color(cvh::Color::RGB)
        .resize(1, 1)
        .run();

    return output.type() == CV_8UC3 &&
                   cvh_pipeline_header_odr_peer() != 0
               ? 0
               : 1;
}
