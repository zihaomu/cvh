#ifndef CVH_PIPELINE_TYPES_H
#define CVH_PIPELINE_TYPES_H

#include "../core/define.h"
#include "../core/system.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

namespace cvh {

constexpr int PIPELINE_MAX_DIM = 8;

enum class PipelineDataKind
{
    Invalid,
    Image,
    Tensor,
    Mask,
    Regions,
};

enum class PipelineDataType
{
    Unknown,
    U8,
    S8,
    U16,
    S16,
    S32,
    F32,
    F64,
};

enum class PixelFormat
{
    Unknown,
    Gray8,
    RGB8,
    BGR8,
    RGBA8,
    BGRA8,
    NV12,
    NV21,
};

enum class Color
{
    Unknown,
    Gray,
    RGB,
    BGR,
    RGBA,
    BGRA,
    YUV,
};

enum class Layout
{
    Unknown,
    HWC,
    CHW,
    NHWC,
    NCHW,
};

enum class ColorMatrix
{
    BT601,
    BT709,
    BT2020,
};

enum class ColorRange
{
    Limited,
    Full,
};

enum class ChromaLocation
{
    Center,
    Left,
};

struct ColorSpec
{
    ColorMatrix matrix = ColorMatrix::BT601;
    ColorRange range = ColorRange::Limited;
    ChromaLocation chroma_location = ChromaLocation::Center;
};

struct PipelinePoint
{
    float x = 0.0f;
    float y = 0.0f;
};

struct PipelineTransform
{
    bool valid = false;
    int source_width = 0;
    int source_height = 0;
    int target_width = 0;
    int target_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    float scale = 1.0f;
    float scale_x = 1.0f;
    float scale_y = 1.0f;
    int pad_left = 0;
    int pad_top = 0;
    int pad_right = 0;
    int pad_bottom = 0;

    PipelinePoint sourceToTarget(PipelinePoint point) const
    {
        return PipelinePoint{
            point.x * scale_x + static_cast<float>(pad_left),
            point.y * scale_y + static_cast<float>(pad_top)};
    }

    PipelinePoint targetToSource(PipelinePoint point) const
    {
        return PipelinePoint{
            (point.x - static_cast<float>(pad_left)) / scale_x,
            (point.y - static_cast<float>(pad_top)) / scale_y};
    }

    bool isPadding(PipelinePoint point) const
    {
        return !valid ||
               point.x < static_cast<float>(pad_left) ||
               point.y < static_cast<float>(pad_top) ||
               point.x >= static_cast<float>(pad_left + resized_width) ||
               point.y >= static_cast<float>(pad_top + resized_height);
    }
};

struct ImageDescriptor
{
    int width = 0;
    int height = 0;
    PipelineDataType data_type = PipelineDataType::Unknown;
    Color color = Color::Unknown;
    PixelFormat pixel_format = PixelFormat::Unknown;
    ColorSpec color_spec{};
    int plane_count = 0;

    bool valid() const
    {
        return width > 0 &&
               height > 0 &&
               data_type != PipelineDataType::Unknown &&
               color != Color::Unknown &&
               plane_count > 0;
    }
};

struct TensorDescriptor
{
    PipelineDataType data_type = PipelineDataType::Unknown;
    Layout layout = Layout::Unknown;
    int dims = 0;
    std::array<int, PIPELINE_MAX_DIM> shape{};

    bool valid() const
    {
        if (data_type == PipelineDataType::Unknown ||
            layout == Layout::Unknown ||
            dims <= 0 ||
            dims > PIPELINE_MAX_DIM)
        {
            return false;
        }
        for (int i = 0; i < dims; ++i)
        {
            if (shape[static_cast<std::size_t>(i)] <= 0)
            {
                return false;
            }
        }
        return true;
    }
};

struct PipelineDataDescriptor
{
    PipelineDataKind kind = PipelineDataKind::Invalid;
    ImageDescriptor image{};
    TensorDescriptor tensor{};

    PipelineDataDescriptor() = default;

    PipelineDataDescriptor(const ImageDescriptor& value)
        : kind(PipelineDataKind::Image), image(value)
    {
    }

    PipelineDataDescriptor(const TensorDescriptor& value)
        : kind(PipelineDataKind::Tensor), tensor(value)
    {
    }

    bool valid() const
    {
        if (kind == PipelineDataKind::Image)
        {
            return image.valid();
        }
        if (kind == PipelineDataKind::Tensor)
        {
            return tensor.valid();
        }
        return false;
    }
};

inline std::size_t pipelineDataTypeSize(PipelineDataType type)
{
    switch (type)
    {
    case PipelineDataType::U8:
    case PipelineDataType::S8:
        return 1;
    case PipelineDataType::U16:
    case PipelineDataType::S16:
        return 2;
    case PipelineDataType::S32:
    case PipelineDataType::F32:
        return 4;
    case PipelineDataType::F64:
        return 8;
    default:
        return 0;
    }
}

inline int pipelineColorChannels(Color color)
{
    switch (color)
    {
    case Color::Gray:
        return 1;
    case Color::RGB:
    case Color::BGR:
    case Color::YUV:
        return 3;
    case Color::RGBA:
    case Color::BGRA:
        return 4;
    default:
        return 0;
    }
}

inline ImageDescriptor imageDesc(int width,
                                 int height,
                                 PixelFormat format,
                                 ColorSpec color_spec = {})
{
    ImageDescriptor descriptor;
    descriptor.width = width;
    descriptor.height = height;
    descriptor.data_type = PipelineDataType::U8;
    descriptor.pixel_format = format;
    descriptor.color_spec = color_spec;

    switch (format)
    {
    case PixelFormat::Gray8:
        descriptor.color = Color::Gray;
        descriptor.plane_count = 1;
        break;
    case PixelFormat::RGB8:
        descriptor.color = Color::RGB;
        descriptor.plane_count = 1;
        break;
    case PixelFormat::BGR8:
        descriptor.color = Color::BGR;
        descriptor.plane_count = 1;
        break;
    case PixelFormat::RGBA8:
        descriptor.color = Color::RGBA;
        descriptor.plane_count = 1;
        break;
    case PixelFormat::BGRA8:
        descriptor.color = Color::BGRA;
        descriptor.plane_count = 1;
        break;
    case PixelFormat::NV12:
    case PixelFormat::NV21:
        descriptor.color = Color::YUV;
        descriptor.plane_count = 2;
        break;
    default:
        descriptor.data_type = PipelineDataType::Unknown;
        descriptor.color = Color::Unknown;
        descriptor.plane_count = 0;
        break;
    }
    return descriptor;
}

template <typename T>
struct PipelineDataTypeOf;

template <>
struct PipelineDataTypeOf<uchar>
{
    static constexpr PipelineDataType value = PipelineDataType::U8;
};

template <>
struct PipelineDataTypeOf<signed char>
{
    static constexpr PipelineDataType value = PipelineDataType::S8;
};

template <>
struct PipelineDataTypeOf<unsigned short>
{
    static constexpr PipelineDataType value = PipelineDataType::U16;
};

template <>
struct PipelineDataTypeOf<short>
{
    static constexpr PipelineDataType value = PipelineDataType::S16;
};

template <>
struct PipelineDataTypeOf<int>
{
    static constexpr PipelineDataType value = PipelineDataType::S32;
};

template <>
struct PipelineDataTypeOf<float>
{
    static constexpr PipelineDataType value = PipelineDataType::F32;
};

template <>
struct PipelineDataTypeOf<double>
{
    static constexpr PipelineDataType value = PipelineDataType::F64;
};

template <typename T>
inline TensorDescriptor tensorDesc(std::initializer_list<int> shape, Layout layout)
{
    if (shape.size() == 0 || shape.size() > PIPELINE_MAX_DIM)
    {
        CV_Error(Error::StsBadSize, "tensorDesc expects 1..8 dimensions");
    }

    TensorDescriptor descriptor;
    descriptor.data_type =
        PipelineDataTypeOf<typename std::remove_cv<T>::type>::value;
    descriptor.layout = layout;
    descriptor.dims = static_cast<int>(shape.size());

    int index = 0;
    for (int extent : shape)
    {
        if (extent <= 0)
        {
            CV_Error(Error::StsBadSize, "tensorDesc extents must be positive");
        }
        descriptor.shape[static_cast<std::size_t>(index++)] = extent;
    }
    return descriptor;
}

}  // namespace cvh

#endif  // CVH_PIPELINE_TYPES_H
