#ifndef CVH_PIPELINE_VIEWS_H
#define CVH_PIPELINE_VIEWS_H

#include "types.h"

#include <array>
#include <cstddef>

namespace cvh {

struct ConstPlaneView
{
    const uchar* data = nullptr;
    std::size_t row_stride = 0;
    std::size_t size_bytes = 0;
};

struct PlaneView
{
    uchar* data = nullptr;
    std::size_t row_stride = 0;
    std::size_t size_bytes = 0;
};

struct ConstImageView
{
    ImageDescriptor descriptor{};
    std::array<ConstPlaneView, 3> planes{};
    int plane_count = 0;
};

struct ImageView
{
    ImageDescriptor descriptor{};
    std::array<PlaneView, 3> planes{};
    int plane_count = 0;

    operator ConstImageView() const
    {
        ConstImageView result;
        result.descriptor = descriptor;
        result.plane_count = plane_count;
        for (std::size_t index = 0; index < planes.size(); ++index)
        {
            result.planes[index] = ConstPlaneView{
                planes[index].data,
                planes[index].row_stride,
                planes[index].size_bytes};
        }
        return result;
    }
};

struct ConstTensorView
{
    const void* data = nullptr;
    std::size_t size_bytes = 0;
    TensorDescriptor descriptor{};
};

struct TensorView
{
    void* data = nullptr;
    std::size_t size_bytes = 0;
    TensorDescriptor descriptor{};
};

inline ConstImageView rgb(const uchar* data,
                          std::size_t size_bytes,
                          int width,
                          int height,
                          std::size_t row_stride)
{
    ConstImageView view;
    view.descriptor = imageDesc(width, height, PixelFormat::RGB8);
    view.planes[0] = ConstPlaneView{data, row_stride, size_bytes};
    view.plane_count = 1;
    return view;
}

inline ImageView rgb(uchar* data,
                     std::size_t size_bytes,
                     int width,
                     int height,
                     std::size_t row_stride)
{
    ImageView view;
    view.descriptor = imageDesc(width, height, PixelFormat::RGB8);
    view.planes[0] = PlaneView{data, row_stride, size_bytes};
    view.plane_count = 1;
    return view;
}

inline ConstImageView bgr(const uchar* data,
                          std::size_t size_bytes,
                          int width,
                          int height,
                          std::size_t row_stride)
{
    ConstImageView view;
    view.descriptor = imageDesc(width, height, PixelFormat::BGR8);
    view.planes[0] = ConstPlaneView{data, row_stride, size_bytes};
    view.plane_count = 1;
    return view;
}

inline ImageView bgr(uchar* data,
                     std::size_t size_bytes,
                     int width,
                     int height,
                     std::size_t row_stride)
{
    ImageView view;
    view.descriptor = imageDesc(width, height, PixelFormat::BGR8);
    view.planes[0] = PlaneView{data, row_stride, size_bytes};
    view.plane_count = 1;
    return view;
}

inline ConstImageView nv12(const uchar* y_data,
                           std::size_t y_stride,
                           std::size_t y_size_bytes,
                           const uchar* uv_data,
                           std::size_t uv_stride,
                           std::size_t uv_size_bytes,
                           int width,
                           int height,
                           ColorSpec color_spec)
{
    ConstImageView view;
    view.descriptor =
        imageDesc(width, height, PixelFormat::NV12, color_spec);
    view.planes[0] = ConstPlaneView{y_data, y_stride, y_size_bytes};
    view.planes[1] = ConstPlaneView{uv_data, uv_stride, uv_size_bytes};
    view.plane_count = 2;
    return view;
}

inline ImageView nv12(uchar* y_data,
                      std::size_t y_stride,
                      std::size_t y_size_bytes,
                      uchar* uv_data,
                      std::size_t uv_stride,
                      std::size_t uv_size_bytes,
                      int width,
                      int height,
                      ColorSpec color_spec)
{
    ImageView view;
    view.descriptor =
        imageDesc(width, height, PixelFormat::NV12, color_spec);
    view.planes[0] = PlaneView{y_data, y_stride, y_size_bytes};
    view.planes[1] = PlaneView{uv_data, uv_stride, uv_size_bytes};
    view.plane_count = 2;
    return view;
}

inline ConstImageView nv21(const uchar* y_data,
                           std::size_t y_stride,
                           std::size_t y_size_bytes,
                           const uchar* vu_data,
                           std::size_t vu_stride,
                           std::size_t vu_size_bytes,
                           int width,
                           int height,
                           ColorSpec color_spec)
{
    ConstImageView view;
    view.descriptor =
        imageDesc(width, height, PixelFormat::NV21, color_spec);
    view.planes[0] = ConstPlaneView{y_data, y_stride, y_size_bytes};
    view.planes[1] =
        ConstPlaneView{vu_data, vu_stride, vu_size_bytes};
    view.plane_count = 2;
    return view;
}

inline ImageView nv21(uchar* y_data,
                      std::size_t y_stride,
                      std::size_t y_size_bytes,
                      uchar* vu_data,
                      std::size_t vu_stride,
                      std::size_t vu_size_bytes,
                      int width,
                      int height,
                      ColorSpec color_spec)
{
    ImageView view;
    view.descriptor =
        imageDesc(width, height, PixelFormat::NV21, color_spec);
    view.planes[0] = PlaneView{y_data, y_stride, y_size_bytes};
    view.planes[1] = PlaneView{vu_data, vu_stride, vu_size_bytes};
    view.plane_count = 2;
    return view;
}

template <typename T>
inline ConstTensorView nchw(const T* data,
                            std::size_t size_bytes,
                            int batch,
                            int channels,
                            int height,
                            int width)
{
    ConstTensorView view;
    view.data = data;
    view.size_bytes = size_bytes;
    view.descriptor =
        tensorDesc<T>({batch, channels, height, width}, Layout::NCHW);
    return view;
}

template <typename T>
inline TensorView nchw(T* data,
                       std::size_t size_bytes,
                       int batch,
                       int channels,
                       int height,
                       int width)
{
    TensorView view;
    view.data = data;
    view.size_bytes = size_bytes;
    view.descriptor =
        tensorDesc<T>({batch, channels, height, width}, Layout::NCHW);
    return view;
}

template <typename T>
inline ConstTensorView nhwc(const T* data,
                            std::size_t size_bytes,
                            int batch,
                            int height,
                            int width,
                            int channels)
{
    ConstTensorView view;
    view.data = data;
    view.size_bytes = size_bytes;
    view.descriptor =
        tensorDesc<T>({batch, height, width, channels}, Layout::NHWC);
    return view;
}

template <typename T>
inline TensorView nhwc(T* data,
                       std::size_t size_bytes,
                       int batch,
                       int height,
                       int width,
                       int channels)
{
    TensorView view;
    view.data = data;
    view.size_bytes = size_bytes;
    view.descriptor =
        tensorDesc<T>({batch, height, width, channels}, Layout::NHWC);
    return view;
}

}  // namespace cvh

#endif  // CVH_PIPELINE_VIEWS_H
