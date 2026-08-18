#include "cvh/pipeline/views.h"

int cvh_pipeline_views_header_compile()
{
    uchar pixels[3]{};
    uchar y_plane[4]{};
    uchar vu_plane[2]{};
    float tensor_data[3]{};
    const cvh::ConstImageView image =
        cvh::bgr(pixels, sizeof(pixels), 1, 1, 3);
    const cvh::TensorView tensor =
        cvh::nhwc(tensor_data, sizeof(tensor_data), 1, 1, 1, 3);
    const cvh::ImageView yuv = cvh::nv21(
        y_plane,
        2,
        sizeof(y_plane),
        vu_plane,
        2,
        sizeof(vu_plane),
        2,
        2,
        cvh::ColorSpec{});
    return image.descriptor.color == cvh::Color::BGR &&
                   tensor.descriptor.layout == cvh::Layout::NHWC &&
                   yuv.descriptor.pixel_format == cvh::PixelFormat::NV21
               ? 0
               : 1;
}
