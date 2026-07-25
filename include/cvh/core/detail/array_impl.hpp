#ifndef CVH_CORE_DETAIL_ARRAY_IMPL_HPP
#define CVH_CORE_DETAIL_ARRAY_IMPL_HPP

#include "channels_ui.hpp"
#include "copy_ui.hpp"
#include "layout_ui.hpp"
#include "transpose_kernel.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>

namespace cvh
{
namespace array_detail
{

inline std::vector<int> coordinates_from_linear(const Mat& mat, size_t linear)
{
    std::vector<int> coordinates(static_cast<size_t>(mat.dims), 0);
    for (int dim = mat.dims - 1; dim >= 0; --dim)
    {
        const size_t extent = static_cast<size_t>(mat.size.p[dim]);
        coordinates[static_cast<size_t>(dim)] =
            static_cast<int>(linear % extent);
        linear /= extent;
    }
    return coordinates;
}

inline size_t pixel_offset(const Mat& mat, const std::vector<int>& coordinates)
{
    size_t offset = 0;
    for (int dim = 0; dim < mat.dims; ++dim)
    {
        offset += static_cast<size_t>(coordinates[static_cast<size_t>(dim)]) *
                  mat.step(dim);
    }
    return offset;
}

inline size_t pixel_offset_from_linear(const Mat& mat, size_t linear)
{
    size_t offset = 0;
    for (int dim = mat.dims - 1; dim >= 0; --dim)
    {
        const size_t extent = static_cast<size_t>(mat.size.p[dim]);
        const size_t coordinate = linear % extent;
        linear /= extent;
        offset += coordinate * mat.step(dim);
    }
    return offset;
}

inline const uchar* pixel_at(const Mat& mat, size_t linear)
{
    return mat.data + pixel_offset_from_linear(mat, linear);
}

inline uchar* pixel_at(Mat& mat, size_t linear)
{
    return mat.data + pixel_offset_from_linear(mat, linear);
}

inline bool shares_storage(const Mat& a, const Mat& b)
{
    return !a.empty() && !b.empty() && a.u != nullptr && a.u == b.u;
}

inline Mat alias_safe_source(const Mat& src, const Mat& dst)
{
    return shares_storage(src, dst) && src.data != dst.data ? src.clone() : src;
}

inline void validate_copy_mask(const Mat& src, const Mat& mask)
{
    if (!mask.empty() &&
        (mask.type() != CV_8UC1 || mask.shape() != src.shape()))
    {
        CV_Error(
            Error::StsBadArg,
            "copyTo mask must be CV_8UC1 with the same shape as src");
    }
}

inline int total_channels(const Mat* mats, size_t count)
{
    int total = 0;
    for (size_t i = 0; i < count; ++i)
    {
        if (mats[i].channels() > std::numeric_limits<int>::max() - total)
        {
            CV_Error(Error::StsOutOfRange, "mixChannels channel count overflow");
        }
        total += mats[i].channels();
    }
    return total;
}

inline std::pair<size_t, int> resolve_channel(const Mat* mats,
                                              size_t count,
                                              int channel)
{
    for (size_t i = 0; i < count; ++i)
    {
        if (channel < mats[i].channels())
        {
            return {i, channel};
        }
        channel -= mats[i].channels();
    }
    CV_Error(Error::StsOutOfRange, "mixChannels channel index is out of range");
    return {0, 0};
}

inline void validate_same_geometry_and_depth(const Mat& reference,
                                             const Mat& candidate,
                                             const char* fn_name)
{
    if (candidate.empty() || candidate.shape() != reference.shape() ||
        candidate.depth() != reference.depth())
    {
        CV_Error_(Error::StsBadArg,
                  ("%s inputs and outputs must have the same shape and depth",
                   fn_name));
    }
}

inline std::vector<Mat> snapshot_sources(const Mat* src, size_t count)
{
    if (src == nullptr || count == 0)
    {
        CV_Error(Error::StsBadArg, "concat expects at least one input");
    }
    return std::vector<Mat>(src, src + count);
}

inline void repeat_initialized_bytes(uchar* dst,
                                     size_t initialized_bytes,
                                     size_t total_bytes)
{
    while (initialized_bytes < total_bytes)
    {
        const size_t copy_bytes =
            std::min(initialized_bytes, total_bytes - initialized_bytes);
        std::memcpy(
            dst + initialized_bytes,
            dst,
            copy_bytes);
        initialized_bytes += copy_bytes;
    }
}

}  // namespace array_detail

inline int borderInterpolate(int p, int len, int borderType)
{
    if (len <= 0)
    {
        CV_Error(Error::StsBadSize, "borderInterpolate expects len > 0");
    }
    borderType &= ~BORDER_ISOLATED;
    if (static_cast<unsigned>(p) < static_cast<unsigned>(len))
    {
        return p;
    }
    if (borderType == BORDER_REPLICATE)
    {
        return p < 0 ? 0 : len - 1;
    }
    if (borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101)
    {
        if (len == 1)
        {
            return 0;
        }
        const int64_t period =
            borderType == BORDER_REFLECT_101
                ? 2LL * static_cast<int64_t>(len - 1)
                : 2LL * static_cast<int64_t>(len);
        int64_t folded = static_cast<int64_t>(p) % period;
        if (folded < 0)
        {
            folded += period;
        }
        if (folded >= len)
        {
            folded =
                borderType == BORDER_REFLECT_101
                    ? period - folded
                    : period - 1 - folded;
        }
        return static_cast<int>(folded);
    }
    if (borderType == BORDER_WRAP)
    {
        if (p < 0)
        {
            p -= ((p - len + 1) / len) * len;
        }
        if (p >= len)
        {
            p %= len;
        }
        return p;
    }
    if (borderType == BORDER_CONSTANT)
    {
        return -1;
    }
    CV_Error_(Error::StsBadArg,
              ("borderInterpolate unsupported borderType=%d", borderType));
    return -1;
}

inline void copyTo(const Mat& src, Mat& dst, const Mat& mask)
{
    if (src.empty())
    {
        dst.release();
        return;
    }
    array_detail::validate_copy_mask(src, mask);
    if (mask.empty())
    {
        if (src.data == dst.data)
        {
            return;
        }
        const Mat source = array_detail::alias_safe_source(src, dst);
        source.copyTo(dst);
        return;
    }
    if (src.data == dst.data)
    {
        return;
    }

    const Mat source = array_detail::alias_safe_source(src, dst);
    const bool allocate =
        dst.empty() || dst.type() != source.type() ||
        dst.shape() != source.shape();
    if (allocate)
    {
        dst.create(source.dims, source.size.p, source.type());
        dst.setTo(Scalar::all(0.0));
    }

    const size_t rows =
        source.dims > 1 ? static_cast<size_t>(source.size.p[0]) : 1;
    const size_t pixels_per_row =
        source.dims > 1 ? source.total(1, source.dims) : source.total();
    const size_t row_bytes = pixels_per_row * source.elemSize();
    const size_t source_step = source.dims > 1 ? source.step(0) : row_bytes;
    const size_t dst_step = source.dims > 1 ? dst.step(0) : row_bytes;
    const size_t mask_step =
        mask.dims > 1 ? mask.step(0) : pixels_per_row;

    if ((source.depth() == CV_8U || source.depth() == CV_8S) &&
        detail::copy_ui::copy_masked_u8_rows(
            source.data,
            source_step,
            mask.data,
            mask_step,
            dst.data,
            dst_step,
            pixels_per_row,
            rows,
            source.channels()))
    {
        cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
        return;
    }

    const size_t pixel_bytes = source.elemSize();
    for (size_t row = 0; row < rows; ++row)
    {
        const uchar* source_row = source.data + row * source_step;
        const uchar* mask_row = mask.data + row * mask_step;
        uchar* dst_row = dst.data + row * dst_step;
        for (size_t x = 0; x < pixels_per_row; ++x)
        {
            if (mask_row[x] != 0)
            {
                std::memcpy(
                    dst_row + x * pixel_bytes,
                    source_row + x * pixel_bytes,
                    pixel_bytes);
            }
        }
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
}

inline void mixChannels(const Mat* src,
                        size_t nsrcs,
                        Mat* dst,
                        size_t ndsts,
                        const int* fromTo,
                        size_t npairs)
{
    if (src == nullptr || nsrcs == 0 || dst == nullptr || ndsts == 0)
    {
        CV_Error(Error::StsBadArg, "mixChannels expects non-empty arrays");
    }
    if (npairs != 0 && fromTo == nullptr)
    {
        CV_Error(Error::StsBadArg, "mixChannels fromTo must not be null");
    }
    const Mat& reference = src[0];
    if (reference.empty())
    {
        CV_Error(Error::StsBadArg, "mixChannels expects non-empty inputs");
    }
    for (size_t i = 0; i < nsrcs; ++i)
    {
        array_detail::validate_same_geometry_and_depth(
            reference, src[i], "mixChannels");
    }
    for (size_t i = 0; i < ndsts; ++i)
    {
        array_detail::validate_same_geometry_and_depth(
            reference, dst[i], "mixChannels");
    }

    const int source_channels = array_detail::total_channels(src, nsrcs);
    const int destination_channels = array_detail::total_channels(dst, ndsts);
    std::vector<Mat> source_snapshots(src, src + nsrcs);
    for (size_t i = 0; i < nsrcs; ++i)
    {
        for (size_t j = 0; j < ndsts; ++j)
        {
            if (array_detail::shares_storage(src[i], dst[j]))
            {
                source_snapshots[i] = src[i].clone();
                break;
            }
        }
    }

    const size_t scalar_bytes = reference.elemSize1();
    for (size_t pair = 0; pair < npairs; ++pair)
    {
        const int from = fromTo[2 * pair];
        const int to = fromTo[2 * pair + 1];
        if (to < 0 || to >= destination_channels || from >= source_channels)
        {
            CV_Error(Error::StsOutOfRange, "mixChannels route is out of range");
        }
    }

    const size_t rows =
        reference.dims > 1 ? static_cast<size_t>(reference.size.p[0]) : 1;
    const size_t pixels_per_row =
        reference.dims > 1
            ? reference.total(1, reference.dims)
            : reference.total();
    const bool byte_depth =
        reference.depth() == CV_8U || reference.depth() == CV_8S;
    if (byte_depth && nsrcs == 1 && ndsts == 1 && npairs == 1)
    {
        const int from = fromTo[0];
        const int to = fromTo[1];
        const Mat& source = source_snapshots[0];
        if (dst[0].channels() == 1 && to == 0 && from >= 0 &&
            detail::channels_ui::extract_u8(
                source.data,
                source.dims > 1
                    ? source.step(0)
                    : pixels_per_row * source.elemSize(),
                dst[0].data,
                dst[0].dims > 1
                    ? dst[0].step(0)
                    : pixels_per_row * dst[0].elemSize(),
                rows,
                pixels_per_row,
                source.channels(),
                from))
        {
            cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
            return;
        }
        if (source.channels() == 1 && from == 0 && to >= 0 &&
            detail::channels_ui::insert_u8(
                source.data,
                source.dims > 1
                    ? source.step(0)
                    : pixels_per_row * source.elemSize(),
                dst[0].data,
                dst[0].dims > 1
                    ? dst[0].step(0)
                    : pixels_per_row * dst[0].elemSize(),
                rows,
                pixels_per_row,
                dst[0].channels(),
                to))
        {
            cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
            return;
        }
    }

    if (byte_depth && nsrcs == 1 && ndsts == 1 &&
        source_snapshots[0].channels() == dst[0].channels() &&
        (dst[0].channels() == 3 || dst[0].channels() == 4) &&
        npairs == static_cast<size_t>(dst[0].channels()))
    {
        int source_for_destination[4] = {-1, -1, -1, -1};
        bool complete = true;
        for (size_t pair = 0; pair < npairs; ++pair)
        {
            const int from = fromTo[2 * pair];
            const int to = fromTo[2 * pair + 1];
            if (from < 0 || from >= dst[0].channels() ||
                to < 0 || to >= dst[0].channels() ||
                source_for_destination[to] >= 0)
            {
                complete = false;
                break;
            }
            source_for_destination[to] = from;
        }
        for (int channel = 0; channel < dst[0].channels(); ++channel)
        {
            complete =
                complete && source_for_destination[channel] >= 0;
        }
        const Mat& source = source_snapshots[0];
        if (complete && detail::channels_ui::reorder_u8(
                            source.data,
                            source.dims > 1
                                ? source.step(0)
                                : pixels_per_row * source.elemSize(),
                            dst[0].data,
                            dst[0].dims > 1
                                ? dst[0].step(0)
                                : pixels_per_row * dst[0].elemSize(),
                            rows,
                            pixels_per_row,
                            dst[0].channels(),
                            source_for_destination))
        {
            cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
            return;
        }
    }

    for (size_t pair = 0; pair < npairs; ++pair)
    {
        const int from = fromTo[2 * pair];
        const int to = fromTo[2 * pair + 1];
        const auto destination =
            array_detail::resolve_channel(dst, ndsts, to);
        std::pair<size_t, int> source{0, 0};
        if (from >= 0)
        {
            source = array_detail::resolve_channel(
                source_snapshots.data(), nsrcs, from);
        }
        Mat& destination_mat = dst[destination.first];
        const size_t destination_step =
            destination_mat.dims > 1
                ? destination_mat.step(0)
                : pixels_per_row * destination_mat.elemSize();
        const Mat* source_mat =
            from >= 0 ? &source_snapshots[source.first] : nullptr;
        const size_t source_step =
            source_mat == nullptr
                ? 0
                : (source_mat->dims > 1
                       ? source_mat->step(0)
                       : pixels_per_row * source_mat->elemSize());
        for (size_t row = 0; row < rows; ++row)
        {
            uchar* destination_row =
                destination_mat.data + row * destination_step;
            const uchar* source_row =
                source_mat == nullptr
                    ? nullptr
                    : source_mat->data + row * source_step;
            for (size_t pixel = 0; pixel < pixels_per_row; ++pixel)
            {
                uchar* destination_scalar =
                    destination_row +
                    pixel * destination_mat.elemSize() +
                    static_cast<size_t>(destination.second) * scalar_bytes;
                if (source_row == nullptr)
                {
                    std::memset(destination_scalar, 0, scalar_bytes);
                }
                else
                {
                    std::memcpy(
                        destination_scalar,
                        source_row +
                            pixel * source_mat->elemSize() +
                            static_cast<size_t>(source.second) * scalar_bytes,
                        scalar_bytes);
                }
            }
        }
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
}

inline void mixChannels(const std::vector<Mat>& src,
                        std::vector<Mat>& dst,
                        const std::vector<int>& fromTo)
{
    if ((fromTo.size() & 1u) != 0)
    {
        CV_Error(Error::StsBadArg, "mixChannels fromTo must contain pairs");
    }
    mixChannels(
        src.data(),
        src.size(),
        dst.data(),
        dst.size(),
        fromTo.data(),
        fromTo.size() / 2);
}

inline void extractChannel(const Mat& src, Mat& dst, int coi)
{
    if (src.empty() || coi < 0 || coi >= src.channels())
    {
        CV_Error(Error::StsOutOfRange, "extractChannel channel is out of range");
    }
    const Mat source = array_detail::alias_safe_source(src, dst);
    dst.create(
        source.dims,
        source.size.p,
        CV_MAKETYPE(source.depth(), 1));
    const int route[] = {coi, 0};
    mixChannels(&source, 1, &dst, 1, route, 1);
}

inline void insertChannel(const Mat& src, Mat& dst, int coi)
{
    if (src.empty() || src.channels() != 1 || dst.empty() ||
        src.shape() != dst.shape() || src.depth() != dst.depth() ||
        coi < 0 || coi >= dst.channels())
    {
        CV_Error(
            Error::StsBadArg,
            "insertChannel expects single-channel src and compatible dst/coi");
    }
    const Mat source =
        array_detail::shares_storage(src, dst) ? src.clone() : src;
    const int route[] = {0, coi};
    mixChannels(&source, 1, &dst, 1, route, 1);
}

inline void flip(const Mat& src, Mat& dst, int flipCode)
{
    if (src.empty() || src.dims != 2)
    {
        CV_Error(Error::StsBadArg, "flip expects non-empty 2D src");
    }
    const Mat source =
        array_detail::shares_storage(src, dst) ? src.clone() : src;
    dst.create(source.dims, source.size.p, source.type());
    const size_t rows = static_cast<size_t>(source.size.p[0]);
    const size_t cols = static_cast<size_t>(source.size.p[1]);
    const size_t elem_size = source.elemSize();
    const bool flip_vertical = flipCode <= 0;
    const bool flip_horizontal = flipCode != 0;
    if (!flip_horizontal)
    {
        const size_t row_bytes = cols * elem_size;
        for (size_t row = 0; row < rows; ++row)
        {
            std::memcpy(
                dst.data + row * dst.step(0),
                source.data + (rows - 1 - row) * source.step(0),
                row_bytes);
        }
        cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
        return;
    }
    if (detail::layout_ui::flip_horizontal_rows(
            source.data,
            source.step(0),
            dst.data,
            dst.step(0),
            rows,
            cols,
            elem_size,
            flip_vertical))
    {
        cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
        return;
    }

    for (size_t row = 0; row < rows; ++row)
    {
        const size_t source_row = flip_vertical ? rows - 1 - row : row;
        const uchar* source_ptr =
            source.data + source_row * source.step(0);
        uchar* dst_ptr = dst.data + row * dst.step(0);
        for (size_t col = 0; col < cols; ++col)
        {
            std::memcpy(
                dst_ptr + col * elem_size,
                source_ptr + (cols - 1 - col) * elem_size,
                elem_size);
        }
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
}

inline void flipND(const Mat& src, Mat& dst, int axis)
{
    if (src.empty())
    {
        CV_Error(Error::StsBadArg, "flipND expects non-empty src");
    }
    if (axis < -src.dims || axis >= src.dims)
    {
        CV_Error(Error::StsOutOfRange, "flipND axis is out of range");
    }
    if (axis < 0)
    {
        axis += src.dims;
    }
    if (src.dims == 2)
    {
        flip(src, dst, axis == 0 ? 0 : 1);
        return;
    }
    Mat source_storage;
    const Mat* source = &src;
    if (array_detail::shares_storage(src, dst) || !src.isContinuous())
    {
        source_storage = src.clone();
        source = &source_storage;
    }
    dst.create(source->dims, source->size.p, source->type());
    const size_t inner_pixels =
        axis + 1 < source->dims ? source->total(axis + 1, source->dims) : 1;
    const size_t axis_extent = static_cast<size_t>(source->size.p[axis]);
    const size_t outer_blocks =
        source->total() / (axis_extent * inner_pixels);
    const size_t block_bytes = inner_pixels * source->elemSize();
    for (size_t outer = 0; outer < outer_blocks; ++outer)
    {
        for (size_t index = 0; index < axis_extent; ++index)
        {
            const size_t destination_block =
                (outer * axis_extent + index) * block_bytes;
            const size_t source_block =
                (outer * axis_extent + axis_extent - 1 - index) * block_bytes;
            std::memcpy(
                dst.data + destination_block,
                source->data + source_block,
                block_bytes);
        }
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
}

inline void rotate(const Mat& src, Mat& dst, int rotateCode)
{
    if (src.empty() || src.dims != 2)
    {
        CV_Error(Error::StsBadArg, "rotate expects non-empty 2D src");
    }
    if (rotateCode != ROTATE_90_CLOCKWISE && rotateCode != ROTATE_180 &&
        rotateCode != ROTATE_90_COUNTERCLOCKWISE)
    {
        CV_Error(Error::StsBadArg, "rotate code is unsupported");
    }
    const Mat source =
        array_detail::shares_storage(src, dst) ? src.clone() : src;
    if (rotateCode == ROTATE_180)
    {
        flip(source, dst, -1);
        return;
    }
    const Mat transpose_source =
        source.isContinuous() ? source : source.clone();
    Mat transposed(
        {source.size.p[1], source.size.p[0]},
        source.type());
    cpu::transpose2d_kernel_blocked(
        transpose_source.data,
        transposed.data,
        source.size.p[0],
        source.size.p[1],
        source.elemSize1(),
        source.channels());
    flip(
        transposed,
        dst,
        rotateCode == ROTATE_90_CLOCKWISE ? 1 : 0);
}

inline void repeat(const Mat& src, int ny, int nx, Mat& dst)
{
    if (src.empty() || src.dims != 2 || ny <= 0 || nx <= 0)
    {
        CV_Error(Error::StsBadArg, "repeat expects 2D src and positive factors");
    }
    if (src.size.p[0] > std::numeric_limits<int>::max() / ny ||
        src.size.p[1] > std::numeric_limits<int>::max() / nx)
    {
        CV_Error(Error::StsOutOfRange, "repeat output shape overflow");
    }
    const Mat source =
        array_detail::shares_storage(src, dst) ? src.clone() : src;
    dst.create(
        {source.size.p[0] * ny, source.size.p[1] * nx}, source.type());
    const size_t source_row_bytes =
        static_cast<size_t>(source.size.p[1]) * source.elemSize();
    const size_t destination_row_bytes =
        static_cast<size_t>(dst.size.p[1]) * dst.elemSize();
    for (int y = 0; y < source.size.p[0]; ++y)
    {
        uchar* destination_row =
            dst.data + static_cast<size_t>(y) * dst.step(0);
        std::memcpy(
            destination_row,
            source.data + static_cast<size_t>(y) * source.step(0),
            source_row_bytes);
        array_detail::repeat_initialized_bytes(
            destination_row,
            source_row_bytes,
            destination_row_bytes);
    }
    const size_t initialized_bytes =
        static_cast<size_t>(source.size.p[0]) * destination_row_bytes;
    array_detail::repeat_initialized_bytes(
        dst.data,
        initialized_bytes,
        dst.total() * dst.elemSize());
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
}

inline void hconcat(const Mat* src, size_t nsrc, Mat& dst)
{
    std::vector<Mat> sources =
        array_detail::snapshot_sources(src, nsrc);
    for (Mat& source : sources)
    {
        if (array_detail::shares_storage(source, dst))
        {
            source = source.clone();
        }
    }
    const Mat& first = sources[0];
    if (first.empty() || first.dims != 2)
    {
        CV_Error(Error::StsBadArg, "hconcat expects non-empty 2D inputs");
    }
    int output_cols = 0;
    for (const Mat& input : sources)
    {
        if (input.empty() || input.dims != 2 ||
            input.type() != first.type() ||
            input.size.p[0] != first.size.p[0] ||
            input.size.p[1] > std::numeric_limits<int>::max() - output_cols)
        {
            CV_Error(Error::StsBadArg, "hconcat input mismatch");
        }
        output_cols += input.size.p[1];
    }
    dst.create({first.size.p[0], output_cols}, first.type());
    for (int y = 0; y < first.size.p[0]; ++y)
    {
        int destination_x = 0;
        for (const Mat& input : sources)
        {
            const size_t row_bytes =
                static_cast<size_t>(input.size.p[1]) * input.elemSize();
            std::memcpy(
                dst.pixelPtr(y, destination_x),
                input.pixelPtr(y, 0),
                row_bytes);
            destination_x += input.size.p[1];
        }
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
}

inline void hconcat(const Mat& src1, const Mat& src2, Mat& dst)
{
    if (src1.empty() || src2.empty() ||
        src1.dims != 2 || src2.dims != 2 ||
        src1.type() != src2.type() ||
        src1.size.p[0] != src2.size.p[0] ||
        src1.size.p[1] >
            std::numeric_limits<int>::max() - src2.size.p[1])
    {
        CV_Error(Error::StsBadArg, "hconcat input mismatch");
    }
    const Mat first =
        array_detail::shares_storage(src1, dst) ? src1.clone() : src1;
    const Mat second =
        array_detail::shares_storage(src2, dst) ? src2.clone() : src2;
    dst.create(
        {first.size.p[0], first.size.p[1] + second.size.p[1]},
        first.type());
    const size_t first_bytes =
        static_cast<size_t>(first.size.p[1]) * first.elemSize();
    const size_t second_bytes =
        static_cast<size_t>(second.size.p[1]) * second.elemSize();
    for (int row = 0; row < first.size.p[0]; ++row)
    {
        uchar* dst_row =
            dst.data + static_cast<size_t>(row) * dst.step(0);
        std::memcpy(
            dst_row,
            first.data + static_cast<size_t>(row) * first.step(0),
            first_bytes);
        std::memcpy(
            dst_row + first_bytes,
            second.data + static_cast<size_t>(row) * second.step(0),
            second_bytes);
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
}

inline void hconcat(const std::vector<Mat>& src, Mat& dst)
{
    hconcat(src.data(), src.size(), dst);
}

inline void vconcat(const Mat* src, size_t nsrc, Mat& dst)
{
    std::vector<Mat> sources =
        array_detail::snapshot_sources(src, nsrc);
    for (Mat& source : sources)
    {
        if (array_detail::shares_storage(source, dst))
        {
            source = source.clone();
        }
    }
    const Mat& first = sources[0];
    if (first.empty() || first.dims != 2)
    {
        CV_Error(Error::StsBadArg, "vconcat expects non-empty 2D inputs");
    }
    int output_rows = 0;
    for (const Mat& input : sources)
    {
        if (input.empty() || input.dims != 2 ||
            input.type() != first.type() ||
            input.size.p[1] != first.size.p[1] ||
            input.size.p[0] > std::numeric_limits<int>::max() - output_rows)
        {
            CV_Error(Error::StsBadArg, "vconcat input mismatch");
        }
        output_rows += input.size.p[0];
    }
    dst.create({output_rows, first.size.p[1]}, first.type());
    int destination_y = 0;
    const size_t row_bytes =
        static_cast<size_t>(first.size.p[1]) * first.elemSize();
    for (const Mat& input : sources)
    {
        if (input.isContinuous())
        {
            std::memcpy(
                dst.data + static_cast<size_t>(destination_y) * dst.step(0),
                input.data,
                static_cast<size_t>(input.size.p[0]) * row_bytes);
            destination_y += input.size.p[0];
        }
        else
        {
            for (int y = 0; y < input.size.p[0]; ++y)
            {
                std::memcpy(
                    dst.data +
                        static_cast<size_t>(destination_y++) * dst.step(0),
                    input.data + static_cast<size_t>(y) * input.step(0),
                    row_bytes);
            }
        }
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
}

inline void vconcat(const Mat& src1, const Mat& src2, Mat& dst)
{
    if (src1.empty() || src2.empty() ||
        src1.dims != 2 || src2.dims != 2 ||
        src1.type() != src2.type() ||
        src1.size.p[1] != src2.size.p[1] ||
        src1.size.p[0] >
            std::numeric_limits<int>::max() - src2.size.p[0])
    {
        CV_Error(Error::StsBadArg, "vconcat input mismatch");
    }
    const Mat first =
        array_detail::shares_storage(src1, dst) ? src1.clone() : src1;
    const Mat second =
        array_detail::shares_storage(src2, dst) ? src2.clone() : src2;
    dst.create(
        {first.size.p[0] + second.size.p[0], first.size.p[1]},
        first.type());
    const size_t row_bytes =
        static_cast<size_t>(first.size.p[1]) * first.elemSize();
    const size_t first_bytes =
        static_cast<size_t>(first.size.p[0]) * row_bytes;
    if (first.isContinuous())
    {
        std::memcpy(dst.data, first.data, first_bytes);
    }
    else
    {
        for (int row = 0; row < first.size.p[0]; ++row)
        {
            std::memcpy(
                dst.data + static_cast<size_t>(row) * dst.step(0),
                first.data + static_cast<size_t>(row) * first.step(0),
                row_bytes);
        }
    }
    uchar* second_destination = dst.data + first_bytes;
    if (second.isContinuous())
    {
        std::memcpy(
            second_destination,
            second.data,
            static_cast<size_t>(second.size.p[0]) * row_bytes);
    }
    else
    {
        for (int row = 0; row < second.size.p[0]; ++row)
        {
            std::memcpy(
                second_destination +
                    static_cast<size_t>(row) * dst.step(0),
                second.data + static_cast<size_t>(row) * second.step(0),
                row_bytes);
        }
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
}

inline void vconcat(const std::vector<Mat>& src, Mat& dst)
{
    vconcat(src.data(), src.size(), dst);
}

inline void broadcast(const Mat& src,
                      const std::vector<int>& shape,
                      Mat& dst)
{
    if (src.empty() || shape.empty() ||
        shape.size() > static_cast<size_t>(MAT_MAX_DIM) ||
        src.dims > static_cast<int>(shape.size()))
    {
        CV_Error(Error::StsBadArg, "broadcast shape is incompatible");
    }
    for (int extent : shape)
    {
        if (extent <= 0)
        {
            CV_Error(Error::StsBadSize, "broadcast shape must be positive");
        }
    }
    const size_t leading = shape.size() - static_cast<size_t>(src.dims);
    for (int dim = 0; dim < src.dims; ++dim)
    {
        const int source_extent = src.size.p[dim];
        const int destination_extent =
            shape[leading + static_cast<size_t>(dim)];
        if (source_extent != 1 && source_extent != destination_extent)
        {
            CV_Error(Error::StsUnmatchedSizes, "broadcast extent mismatch");
        }
    }

    Mat source_storage;
    const Mat* source = &src;
    if (array_detail::shares_storage(src, dst) || !src.isContinuous())
    {
        source_storage = src.clone();
        source = &source_storage;
    }
    dst.create(shape, source->type());

    bool exact_trailing_shape = true;
    for (int dim = 0; dim < source->dims; ++dim)
    {
        exact_trailing_shape =
            exact_trailing_shape &&
            source->size.p[dim] ==
                shape[leading + static_cast<size_t>(dim)];
    }
    if (exact_trailing_shape)
    {
        const size_t source_bytes = source->total() * source->elemSize();
        std::memcpy(dst.data, source->data, source_bytes);
        array_detail::repeat_initialized_bytes(
            dst.data,
            source_bytes,
            dst.total() * dst.elemSize());
        cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
        return;
    }

    int trailing_start = source->dims;
    size_t block_pixels = 1;
    while (trailing_start > 0)
    {
        const int source_dim = trailing_start - 1;
        const size_t destination_dim =
            leading + static_cast<size_t>(source_dim);
        if (source->size.p[source_dim] != shape[destination_dim])
        {
            break;
        }
        block_pixels *= static_cast<size_t>(source->size.p[source_dim]);
        --trailing_start;
    }
    const size_t block_bytes = block_pixels * source->elemSize();
    if (source->total() == block_pixels)
    {
        std::memcpy(dst.data, source->data, block_bytes);
        array_detail::repeat_initialized_bytes(
            dst.data,
            block_bytes,
            dst.total() * dst.elemSize());
        cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
        return;
    }

    const size_t output_prefix_dims =
        leading + static_cast<size_t>(trailing_start);
    std::vector<size_t> source_strides(output_prefix_dims, 0);
    for (int dim = 0; dim < trailing_start; ++dim)
    {
        if (source->size.p[dim] != 1)
        {
            source_strides[
                leading + static_cast<size_t>(dim)] =
                source->step(dim);
        }
    }
    std::vector<int> coordinates(output_prefix_dims, 0);
    size_t source_offset = 0;
    const size_t output_blocks = dst.total() / block_pixels;
    for (size_t output_block = 0;
         output_block < output_blocks;
         ++output_block)
    {
        std::memcpy(
            dst.data + output_block * block_bytes,
            source->data + source_offset,
            block_bytes);
        for (int dim = static_cast<int>(output_prefix_dims) - 1;
             dim >= 0;
             --dim)
        {
            const size_t index = static_cast<size_t>(dim);
            ++coordinates[index];
            source_offset += source_strides[index];
            if (coordinates[index] < shape[index])
            {
                break;
            }
            coordinates[index] = 0;
            source_offset -=
                source_strides[index] * static_cast<size_t>(shape[index]);
        }
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
}

inline void broadcast(const Mat& src, const Mat& shape, Mat& dst)
{
    if (shape.empty() || shape.type() != CV_32SC1 ||
        !shape.isContinuous())
    {
        CV_Error(
            Error::StsBadArg,
            "broadcast target shape must be a continuous CV_32SC1 Mat");
    }
    const int* values = reinterpret_cast<const int*>(shape.data);
    broadcast(
        src,
        std::vector<int>(values, values + shape.total()),
        dst);
}

inline void swap(Mat& a, Mat& b)
{
    if (&a == &b)
    {
        return;
    }
    using std::swap;
    swap(a.dims, b.dims);
    swap(a.data, b.data);
    swap(a.allocator, b.allocator);
    swap(a.u, b.u);
    swap(a.size, b.size);
    for (int dim = 0; dim < MAT_MAX_DIM; ++dim)
    {
        swap(a.stepBuf[dim], b.stepBuf[dim]);
    }
    swap(a.matType, b.matType);
    swap(a.extrInfo, b.extrInfo);
}

}  // namespace cvh

#endif  // CVH_CORE_DETAIL_ARRAY_IMPL_HPP
