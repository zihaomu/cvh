#ifndef CVH_IMGPROC_MEDIAN_BLUR_H
#define CVH_IMGPROC_MEDIAN_BLUR_H

#include "detail/common.h"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/simd/opencv_ui.h"

#include <algorithm>
#include <array>
#include <vector>

namespace cvh
{
namespace median_blur_detail
{

template<typename Value, typename Compare>
inline void median3_network(Value* values, Compare&& compare)
{
    compare(values[1], values[2]);
    compare(values[4], values[5]);
    compare(values[7], values[8]);
    compare(values[0], values[1]);
    compare(values[3], values[4]);
    compare(values[6], values[7]);
    compare(values[1], values[2]);
    compare(values[4], values[5]);
    compare(values[7], values[8]);
    compare(values[0], values[3]);
    compare(values[5], values[8]);
    compare(values[4], values[7]);
    compare(values[3], values[6]);
    compare(values[1], values[4]);
    compare(values[2], values[5]);
    compare(values[4], values[7]);
    compare(values[4], values[2]);
    compare(values[6], values[4]);
    compare(values[4], values[2]);
}

template<typename Value, typename Compare>
inline void median5_network(Value* p, Compare&& op)
{
    op(p[1], p[2]); op(p[0], p[1]); op(p[1], p[2]); op(p[4], p[5]); op(p[3], p[4]);
    op(p[4], p[5]); op(p[0], p[3]); op(p[2], p[5]); op(p[2], p[3]); op(p[1], p[4]);
    op(p[1], p[2]); op(p[3], p[4]); op(p[7], p[8]); op(p[6], p[7]); op(p[7], p[8]);
    op(p[10], p[11]); op(p[9], p[10]); op(p[10], p[11]); op(p[6], p[9]); op(p[8], p[11]);
    op(p[8], p[9]); op(p[7], p[10]); op(p[7], p[8]); op(p[9], p[10]); op(p[0], p[6]);
    op(p[4], p[10]); op(p[4], p[6]); op(p[2], p[8]); op(p[2], p[4]); op(p[6], p[8]);
    op(p[1], p[7]); op(p[5], p[11]); op(p[5], p[7]); op(p[3], p[9]); op(p[3], p[5]);
    op(p[7], p[9]); op(p[1], p[2]); op(p[3], p[4]); op(p[5], p[6]); op(p[7], p[8]);
    op(p[9], p[10]); op(p[13], p[14]); op(p[12], p[13]); op(p[13], p[14]); op(p[16], p[17]);
    op(p[15], p[16]); op(p[16], p[17]); op(p[12], p[15]); op(p[14], p[17]); op(p[14], p[15]);
    op(p[13], p[16]); op(p[13], p[14]); op(p[15], p[16]); op(p[19], p[20]); op(p[18], p[19]);
    op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[21], p[23]); op(p[22], p[24]);
    op(p[22], p[23]); op(p[18], p[21]); op(p[20], p[23]); op(p[20], p[21]); op(p[19], p[22]);
    op(p[22], p[24]); op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[12], p[18]);
    op(p[16], p[22]); op(p[16], p[18]); op(p[14], p[20]); op(p[20], p[24]); op(p[14], p[16]);
    op(p[18], p[20]); op(p[22], p[24]); op(p[13], p[19]); op(p[17], p[23]); op(p[17], p[19]);
    op(p[15], p[21]); op(p[15], p[17]); op(p[19], p[21]); op(p[13], p[14]); op(p[15], p[16]);
    op(p[17], p[18]); op(p[19], p[20]); op(p[21], p[22]); op(p[23], p[24]); op(p[0], p[12]);
    op(p[8], p[20]); op(p[8], p[12]); op(p[4], p[16]); op(p[16], p[24]); op(p[12], p[16]);
    op(p[2], p[14]); op(p[10], p[22]); op(p[10], p[14]); op(p[6], p[18]); op(p[6], p[10]);
    op(p[10], p[12]); op(p[1], p[13]); op(p[9], p[21]); op(p[9], p[13]); op(p[5], p[17]);
    op(p[13], p[17]); op(p[3], p[15]); op(p[11], p[23]); op(p[11], p[15]); op(p[7], p[19]);
    op(p[7], p[11]); op(p[11], p[13]); op(p[11], p[12]);
}

inline uchar scalar_median_at(const Mat& src,
                              int y,
                              int x,
                              int channel,
                              int ksize)
{
    std::array<uchar, 25> values {};
    const int radius = ksize / 2;
    int count = 0;
    for (int ky = -radius; ky <= radius; ++ky)
    {
        const int source_y =
            std::clamp(y + ky, 0, src.size.p[0] - 1);
        const uchar* input =
            src.data + static_cast<size_t>(source_y) * src.step(0);
        for (int kx = -radius; kx <= radius; ++kx)
        {
            const int source_x =
                std::clamp(x + kx, 0, src.size.p[1] - 1);
            values[static_cast<size_t>(count++)] =
                input[static_cast<size_t>(source_x) *
                          static_cast<size_t>(src.channels()) +
                      static_cast<size_t>(channel)];
        }
    }
    auto middle = values.begin() + count / 2;
    std::nth_element(values.begin(), middle, values.begin() + count);
    return *middle;
}

inline bool run_u8_sorting_network(const Mat& src,
                                   Mat& dst,
                                   int ksize)
{
#if CVH_DETAIL_HAVE_OPENCV_UI && CV_SIMD128 && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
    if (!cpu::opencv_ui_allowed() ||
        (ksize != 3 && ksize != 5))
    {
        return false;
    }
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int channels = src.channels();
    const int radius = ksize / 2;
    const int row_width = cols * channels;
    const int vector_begin = radius * channels;
    const int vector_end = (cols - radius) * channels;
    const int lanes = cv::VTraits<cv::v_uint8>::vlanes();
    if (vector_begin > vector_end - lanes)
    {
        return false;
    }
    auto compare = [](cv::v_uint8& first, cv::v_uint8& second) {
        const cv::v_uint8 original = first;
        first = cv::v_min(first, second);
        second = cv::v_max(second, original);
    };

    for (int y = 0; y < rows; ++y)
    {
        const uchar* source_rows[5] = {};
        for (int ky = -radius; ky <= radius; ++ky)
        {
            const int source_y =
                std::clamp(y + ky, 0, rows - 1);
            source_rows[ky + radius] =
                src.data +
                static_cast<size_t>(source_y) * src.step(0);
        }
        uchar* output =
            dst.data + static_cast<size_t>(y) * dst.step(0);
        int index = 0;
        for (; index < vector_begin; ++index)
        {
            output[index] = scalar_median_at(
                src,
                y,
                index / channels,
                index % channels,
                ksize);
        }
        for (; index <= vector_end - lanes; index += lanes)
        {
            if (ksize == 3)
            {
                cv::v_uint8 values[9];
                int value_index = 0;
                for (int ky = 0; ky < 3; ++ky)
                {
                    for (int kx = -1; kx <= 1; ++kx)
                    {
                        values[value_index++] = cv::vx_load(
                            source_rows[ky] +
                            index + kx * channels);
                    }
                }
                median3_network(values, compare);
                cv::vx_store(output + index, values[4]);
            }
            else
            {
                cv::v_uint8 values[25];
                int value_index = 0;
                for (int ky = 0; ky < 5; ++ky)
                {
                    for (int kx = -2; kx <= 2; ++kx)
                    {
                        values[value_index++] = cv::vx_load(
                            source_rows[ky] +
                            index + kx * channels);
                    }
                }
                median5_network(values, compare);
                cv::vx_store(output + index, values[12]);
            }
        }
        for (; index < row_width; ++index)
        {
            output[index] = scalar_median_at(
                src,
                y,
                index / channels,
                index % channels,
                ksize);
        }
    }
    cpu::set_last_dispatch_tag(cpu::DispatchTag::OpenCVUI);
    return true;
#else
    (void)src;
    (void)dst;
    (void)ksize;
    return false;
#endif
}

inline void run_u8_c1_histogram(const Mat& src,
                                Mat& dst,
                                int ksize)
{
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int radius = ksize / 2;
    const int target = (ksize * ksize) / 2;
    for (int y = 0; y < rows; ++y)
    {
        std::array<int, 16> coarse = {};
        std::array<int, 256> fine = {};
        auto add_value = [&](uchar value, int delta) {
            fine[value] += delta;
            coarse[value >> 4] += delta;
        };
        for (int ky = -radius; ky <= radius; ++ky)
        {
            const int source_y =
                std::clamp(y + ky, 0, rows - 1);
            const uchar* input =
                src.data +
                static_cast<size_t>(source_y) * src.step(0);
            for (int kx = -radius; kx <= radius; ++kx)
            {
                add_value(
                    input[std::clamp(kx, 0, cols - 1)],
                    1);
            }
        }

        uchar* output =
            dst.data + static_cast<size_t>(y) * dst.step(0);
        for (int x = 0; x < cols; ++x)
        {
            int count = 0;
            int coarse_bin = 0;
            for (; coarse_bin < 16; ++coarse_bin)
            {
                if (count + coarse[coarse_bin] > target)
                {
                    break;
                }
                count += coarse[coarse_bin];
            }
            const int fine_begin = coarse_bin << 4;
            int value = fine_begin;
            for (; value < fine_begin + 16; ++value)
            {
                count += fine[value];
                if (count > target)
                {
                    break;
                }
            }
            output[x] = static_cast<uchar>(value);

            if (x + 1 == cols)
            {
                continue;
            }
            const int remove_x =
                std::clamp(x - radius, 0, cols - 1);
            const int add_x =
                std::clamp(x + radius + 1, 0, cols - 1);
            for (int ky = -radius; ky <= radius; ++ky)
            {
                const int source_y =
                    std::clamp(y + ky, 0, rows - 1);
                const uchar* input =
                    src.data +
                    static_cast<size_t>(source_y) * src.step(0);
                add_value(input[remove_x], -1);
                add_value(input[add_x], 1);
            }
        }
    }
}

template<typename T>
inline void run(const Mat& src, Mat& dst, int ksize)
{
    const int rows = src.size.p[0];
    const int cols = src.size.p[1];
    const int channels = src.channels();
    const int radius = ksize / 2;
    std::vector<T> window(static_cast<size_t>(ksize) * ksize);

    for (int y = 0; y < rows; ++y)
    {
        T* output = reinterpret_cast<T*>(
            dst.data + static_cast<size_t>(y) * dst.step(0));
        for (int x = 0; x < cols; ++x)
        {
            for (int ch = 0; ch < channels; ++ch)
            {
                size_t index = 0;
                for (int ky = -radius; ky <= radius; ++ky)
                {
                    const int source_y = std::clamp(y + ky, 0, rows - 1);
                    const T* input = reinterpret_cast<const T*>(
                        src.data +
                        static_cast<size_t>(source_y) * src.step(0));
                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        const int source_x =
                            std::clamp(x + kx, 0, cols - 1);
                        window[index++] =
                            input[static_cast<size_t>(source_x) * channels +
                                  static_cast<size_t>(ch)];
                    }
                }
                auto middle =
                    window.begin() + static_cast<std::ptrdiff_t>(window.size() / 2);
                std::nth_element(window.begin(), middle, window.end());
                output[static_cast<size_t>(x) * channels +
                       static_cast<size_t>(ch)] = *middle;
            }
        }
    }
}

}  // namespace median_blur_detail

inline void medianBlur(const Mat& src, Mat& dst, int ksize)
{
    if (src.empty() || src.dims != 2 ||
        (src.depth() != CV_8U && src.depth() != CV_32F) ||
        (src.channels() != 1 && src.channels() != 3 &&
         src.channels() != 4))
    {
        CV_Error(Error::StsBadArg, "medianBlur unsupported source");
    }
    if (ksize <= 1 || (ksize & 1) == 0)
    {
        CV_Error(Error::StsBadSize, "medianBlur ksize must be odd and greater than 1");
    }
    if (src.depth() == CV_32F && ksize != 3 && ksize != 5)
    {
        CV_Error(Error::StsBadSize, "medianBlur CV_32F supports ksize 3 or 5");
    }

    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    const Mat source = src.data == dst.data ? src.clone() : src;
    dst.create(source.shape(), source.type());
    if (source.depth() == CV_8U &&
        median_blur_detail::run_u8_sorting_network(
            source, dst, ksize))
    {
        return;
    }
    if (source.type() == CV_8UC1)
    {
        median_blur_detail::run_u8_c1_histogram(
            source, dst, ksize);
    }
    else if (source.depth() == CV_8U)
    {
        median_blur_detail::run<uchar>(source, dst, ksize);
    }
    else
    {
        median_blur_detail::run<float>(source, dst, ksize);
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_MEDIAN_BLUR_H
