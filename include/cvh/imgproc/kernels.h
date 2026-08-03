#ifndef CVH_IMGPROC_KERNELS_H
#define CVH_IMGPROC_KERNELS_H

#include "detail/common.h"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/core/simd/opencv_ui.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace cvh
{
namespace kernel_detail
{

inline void validate_float_kernel_type(int type, const char* fn_name)
{
    if (CV_MAT_CN(type) != 1 ||
        (CV_MAT_DEPTH(type) != CV_32F && CV_MAT_DEPTH(type) != CV_64F))
    {
        CV_Error_(Error::StsBadArg,
                  ("%s kernel type must be CV_32F or CV_64F", fn_name));
    }
}

inline void write_kernel_value(Mat& kernel, int y, int x, double value)
{
    if (kernel.depth() == CV_32F)
    {
        kernel.at<float>(y, x) = static_cast<float>(value);
    }
    else
    {
        kernel.at<double>(y, x) = value;
    }
}

inline std::vector<double> sobel_kernel(int order, int requested_size)
{
    int size = requested_size;
    if (size == 1 && order > 0)
    {
        size = 3;
    }
    if (size <= order || size <= 0 || (size & 1) == 0 || size > 31)
    {
        CV_Error(Error::StsBadArg, "getDerivKernels invalid order/ksize");
    }
    std::vector<int> coefficients(static_cast<size_t>(size + 1), 0);
    if (size == 1)
    {
        coefficients[0] = 1;
    }
    else if (size == 3)
    {
        if (order == 0)
        {
            coefficients[0] = 1;
            coefficients[1] = 2;
            coefficients[2] = 1;
        }
        else if (order == 1)
        {
            coefficients[0] = -1;
            coefficients[1] = 0;
            coefficients[2] = 1;
        }
        else
        {
            coefficients[0] = 1;
            coefficients[1] = -2;
            coefficients[2] = 1;
        }
    }
    else
    {
        coefficients[0] = 1;
        for (int i = 0; i < size - order - 1; ++i)
        {
            int old_value = coefficients[0];
            for (int j = 1; j <= size; ++j)
            {
                const int new_value =
                    coefficients[static_cast<size_t>(j)] +
                    coefficients[static_cast<size_t>(j - 1)];
                coefficients[static_cast<size_t>(j - 1)] = old_value;
                old_value = new_value;
            }
        }
        for (int i = 0; i < order; ++i)
        {
            int old_value = -coefficients[0];
            for (int j = 1; j <= size; ++j)
            {
                const int new_value =
                    coefficients[static_cast<size_t>(j - 1)] -
                    coefficients[static_cast<size_t>(j)];
                coefficients[static_cast<size_t>(j - 1)] = old_value;
                old_value = new_value;
            }
        }
    }
    std::vector<double> result(static_cast<size_t>(size));
    for (int i = 0; i < size; ++i)
    {
        result[static_cast<size_t>(i)] =
            static_cast<double>(coefficients[static_cast<size_t>(i)]);
    }
    return result;
}

}  // namespace kernel_detail

inline Mat getStructuringElement(int shape,
                                 Size ksize,
                                 Point anchor = Point(-1, -1))
{
    if (ksize.width <= 0 || ksize.height <= 0)
    {
        CV_Error(Error::StsBadSize, "getStructuringElement invalid size");
    }
    if (anchor.x < 0)
    {
        anchor.x = ksize.width / 2;
    }
    if (anchor.y < 0)
    {
        anchor.y = ksize.height / 2;
    }
    if (anchor.x < 0 || anchor.x >= ksize.width ||
        anchor.y < 0 || anchor.y >= ksize.height)
    {
        CV_Error(Error::StsOutOfRange, "getStructuringElement invalid anchor");
    }
    if (shape != MORPH_RECT && shape != MORPH_CROSS &&
        shape != MORPH_ELLIPSE && shape != MORPH_DIAMOND)
    {
        CV_Error(Error::StsBadArg, "getStructuringElement invalid shape");
    }
    if (ksize == Size(1, 1))
    {
        shape = MORPH_RECT;
    }

    Mat result({ksize.height, ksize.width}, CV_8UC1);
    std::memset(
        result.data,
        0,
        static_cast<size_t>(ksize.height) *
            static_cast<size_t>(ksize.width));
    const int radius_y = ksize.height / 2;
    const int center_x = ksize.width / 2;
    for (int y = 0; y < ksize.height; ++y)
    {
        int start = 0;
        int end = 0;
        if (shape == MORPH_RECT || (shape == MORPH_CROSS && y == anchor.y))
        {
            end = ksize.width;
        }
        else if (shape == MORPH_CROSS)
        {
            start = anchor.x;
            end = anchor.x + 1;
        }
        else if (shape == MORPH_DIAMOND)
        {
            const int dx = radius_y - std::abs(y - radius_y);
            if (dx >= 0)
            {
                start = std::max(0, center_x - dx);
                end = std::min(ksize.width, center_x + dx + 1);
            }
        }
        else
        {
            const int dy = y - radius_y;
            if (std::abs(dy) <= radius_y)
            {
                const double remaining =
                    radius_y == 0
                        ? 0.0
                        : static_cast<double>(
                              radius_y * radius_y - dy * dy) /
                              static_cast<double>(radius_y * radius_y);
                const int dx = static_cast<int>(
                    std::lround(center_x * std::sqrt(remaining)));
                start = std::max(0, center_x - dx);
                end = std::min(ksize.width, center_x + dx + 1);
            }
        }
        if (end > start)
        {
            uchar* row =
                result.data + static_cast<size_t>(y) * result.step(0);
            std::memset(
                row + start,
                1,
                static_cast<size_t>(end - start));
        }
    }
    return result;
}

inline Mat getGaussianKernel(int ksize, double sigma, int ktype = CV_64F)
{
    kernel_detail::validate_float_kernel_type(ktype, "getGaussianKernel");
    if (ksize <= 0 || (ksize & 1) == 0)
    {
        CV_Error(Error::StsBadSize, "getGaussianKernel expects positive odd ksize");
    }
    std::vector<double> values;
    if (sigma <= 0.0)
    {
        if (ksize == 1) values = {1.0};
        else if (ksize == 3) values = {0.25, 0.5, 0.25};
        else if (ksize == 5) values = {0.0625, 0.25, 0.375, 0.25, 0.0625};
        else if (ksize == 7) values = {0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125};
        else if (ksize == 9) values = {4.0 / 256, 13.0 / 256, 30.0 / 256, 51.0 / 256, 60.0 / 256, 51.0 / 256, 30.0 / 256, 13.0 / 256, 4.0 / 256};
    }
    if (values.empty())
    {
        const double sigma_value =
            sigma > 0.0 ? sigma : 0.15 * ksize + 0.35;
        const double scale = -0.5 / (sigma_value * sigma_value);
        values.resize(static_cast<size_t>(ksize));
        double total = 0.0;
        for (int i = 0; i < ksize; ++i)
        {
            const double x = i - (ksize - 1) * 0.5;
            values[static_cast<size_t>(i)] = std::exp(x * x * scale);
            total += values[static_cast<size_t>(i)];
        }
        for (double& value : values)
        {
            value /= total;
        }
    }
    Mat result({ksize, 1}, CV_MAKETYPE(CV_MAT_DEPTH(ktype), 1));
    for (int i = 0; i < ksize; ++i)
    {
        kernel_detail::write_kernel_value(
            result, i, 0, values[static_cast<size_t>(i)]);
    }
    return result;
}

inline void getDerivKernels(Mat& kx,
                            Mat& ky,
                            int dx,
                            int dy,
                            int ksize,
                            bool normalize = false,
                            int ktype = CV_32F)
{
    kernel_detail::validate_float_kernel_type(ktype, "getDerivKernels");
    if (dx < 0 || dy < 0 || dx + dy <= 0)
    {
        CV_Error(Error::StsBadArg, "getDerivKernels invalid derivative order");
    }
    std::vector<double> values_x;
    std::vector<double> values_y;
    if (ksize <= 0)
    {
        if (dx + dy != 1)
        {
            CV_Error(Error::StsBadArg, "Scharr kernels require first derivative");
        }
        values_x = dx == 0
                       ? std::vector<double>{3.0, 10.0, 3.0}
                       : std::vector<double>{-1.0, 0.0, 1.0};
        values_y = dy == 0
                       ? std::vector<double>{3.0, 10.0, 3.0}
                       : std::vector<double>{-1.0, 0.0, 1.0};
        if (normalize)
        {
            if (dx == 0)
            {
                for (double& value : values_x) value /= 32.0;
            }
            if (dy == 0)
            {
                for (double& value : values_y) value /= 32.0;
            }
        }
    }
    else
    {
        values_x = kernel_detail::sobel_kernel(dx, ksize);
        values_y = kernel_detail::sobel_kernel(dy, ksize);
        if (normalize)
        {
            const double scale_x =
                std::ldexp(1.0, -(static_cast<int>(values_x.size()) - dx - 1));
            const double scale_y =
                std::ldexp(1.0, -(static_cast<int>(values_y.size()) - dy - 1));
            for (double& value : values_x) value *= scale_x;
            for (double& value : values_y) value *= scale_y;
        }
    }
    kx.create(
        {static_cast<int>(values_x.size()), 1},
        CV_MAKETYPE(CV_MAT_DEPTH(ktype), 1));
    ky.create(
        {static_cast<int>(values_y.size()), 1},
        CV_MAKETYPE(CV_MAT_DEPTH(ktype), 1));
    if (CV_MAT_DEPTH(ktype) == CV_32F)
    {
        float* output_x = reinterpret_cast<float*>(kx.data);
        float* output_y = reinterpret_cast<float*>(ky.data);
        for (size_t i = 0; i < values_x.size(); ++i)
        {
            output_x[i] = static_cast<float>(values_x[i]);
        }
        for (size_t i = 0; i < values_y.size(); ++i)
        {
            output_y[i] = static_cast<float>(values_y[i]);
        }
    }
    else
    {
        std::copy(
            values_x.begin(),
            values_x.end(),
            reinterpret_cast<double*>(kx.data));
        std::copy(
            values_y.begin(),
            values_y.end(),
            reinterpret_cast<double*>(ky.data));
    }
}

inline Mat getGaborKernel(Size ksize,
                          double sigma,
                          double theta,
                          double lambd,
                          double gamma,
                          double psi = CV_PI * 0.5,
                          int ktype = CV_64F)
{
    kernel_detail::validate_float_kernel_type(ktype, "getGaborKernel");
    if (sigma <= 0.0 || lambd == 0.0 || gamma == 0.0)
    {
        CV_Error(Error::StsBadArg, "getGaborKernel invalid parameters");
    }
    const double sigma_x = sigma;
    const double sigma_y = sigma / gamma;
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    const int xmax =
        ksize.width > 0
            ? ksize.width / 2
            : static_cast<int>(std::lround(std::max(
                  std::fabs(3.0 * sigma_x * cosine),
                  std::fabs(3.0 * sigma_y * sine))));
    const int ymax =
        ksize.height > 0
            ? ksize.height / 2
            : static_cast<int>(std::lround(std::max(
                  std::fabs(3.0 * sigma_x * sine),
                  std::fabs(3.0 * sigma_y * cosine))));
    Mat result(
        {2 * ymax + 1, 2 * xmax + 1},
        CV_MAKETYPE(CV_MAT_DEPTH(ktype), 1));
    const double exponent_x = -0.5 / (sigma_x * sigma_x);
    const double exponent_y = -0.5 / (sigma_y * sigma_y);
    const double cosine_scale = 2.0 * CV_PI / lambd;
    if (CV_MAT_DEPTH(ktype) == CV_32F)
    {
        for (int y = -ymax; y <= ymax; ++y)
        {
            float* output = reinterpret_cast<float*>(
                result.data +
                static_cast<size_t>(ymax - y) * result.step(0));
            for (int x = -xmax; x <= xmax; ++x)
            {
                const double rotated_x = x * cosine + y * sine;
                const double rotated_y = -x * sine + y * cosine;
                output[xmax - x] = static_cast<float>(
                    std::exp(
                        exponent_x * rotated_x * rotated_x +
                        exponent_y * rotated_y * rotated_y) *
                    std::cos(cosine_scale * rotated_x + psi));
            }
        }
    }
    else
    {
        for (int y = -ymax; y <= ymax; ++y)
        {
            double* output = reinterpret_cast<double*>(
                result.data +
                static_cast<size_t>(ymax - y) * result.step(0));
            for (int x = -xmax; x <= xmax; ++x)
            {
                const double rotated_x = x * cosine + y * sine;
                const double rotated_y = -x * sine + y * cosine;
                output[xmax - x] =
                    std::exp(
                        exponent_x * rotated_x * rotated_x +
                        exponent_y * rotated_y * rotated_y) *
                    std::cos(cosine_scale * rotated_x + psi);
            }
        }
    }
    return result;
}

inline void createHanningWindow(Mat& dst, Size winSize, int type)
{
    kernel_detail::validate_float_kernel_type(type, "createHanningWindow");
    if (winSize.width <= 1 || winSize.height <= 1)
    {
        CV_Error(Error::StsBadSize, "createHanningWindow dimensions must exceed one");
    }
    dst.create(
        {winSize.height, winSize.width},
        CV_MAKETYPE(CV_MAT_DEPTH(type), 1));
    const double x_scale = 2.0 * CV_PI / (winSize.width - 1);
    const double y_scale = 2.0 * CV_PI / (winSize.height - 1);
    std::vector<double> x_weights(
        static_cast<size_t>(winSize.width));
    std::vector<double> y_weights(
        static_cast<size_t>(winSize.height));
    for (int x = 0; x <= (winSize.width - 1) / 2; ++x)
    {
        const double weight = std::sqrt(
            0.5 * (1.0 - std::cos(x_scale * x)));
        x_weights[static_cast<size_t>(x)] = weight;
        x_weights[static_cast<size_t>(winSize.width - 1 - x)] =
            weight;
    }
    for (int y = 0; y <= (winSize.height - 1) / 2; ++y)
    {
        const double weight = std::sqrt(
            0.5 * (1.0 - std::cos(y_scale * y)));
        y_weights[static_cast<size_t>(y)] = weight;
        y_weights[static_cast<size_t>(winSize.height - 1 - y)] =
            weight;
    }

    bool used_ui = false;
    if (dst.depth() == CV_32F)
    {
        std::vector<float> x_weights_f32(
            static_cast<size_t>(winSize.width));
        for (int x = 0; x < winSize.width; ++x)
        {
            x_weights_f32[static_cast<size_t>(x)] =
                static_cast<float>(
                    x_weights[static_cast<size_t>(x)]);
        }
        for (int y = 0; y < winSize.height; ++y)
        {
            float* output = reinterpret_cast<float*>(
                dst.data + static_cast<size_t>(y) * dst.step(0));
            const float row_weight = static_cast<float>(
                y_weights[static_cast<size_t>(y)]);
            int x = 0;
#if CVH_DETAIL_HAVE_OPENCV_UI && (CV_SIMD || CV_SIMD_SCALABLE) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
            if (cpu::opencv_ui_allowed())
            {
                const int lanes =
                    cv::VTraits<cv::v_float32>::vlanes();
                const cv::v_float32 row_vector =
                    cv::vx_setall_f32(row_weight);
                for (; x + lanes <= winSize.width; x += lanes)
                {
                    cv::vx_store(
                        output + x,
                        cv::v_mul(
                            cv::vx_load(
                                x_weights_f32.data() + x),
                            row_vector));
                }
                used_ui = used_ui || x > 0;
            }
#endif
            for (; x < winSize.width; ++x)
            {
                output[x] =
                    x_weights_f32[static_cast<size_t>(x)] *
                    row_weight;
            }
        }
    }
    else
    {
        for (int y = 0; y < winSize.height; ++y)
        {
            double* output = reinterpret_cast<double*>(
                dst.data + static_cast<size_t>(y) * dst.step(0));
            const double row_weight =
                y_weights[static_cast<size_t>(y)];
            int x = 0;
#if CVH_DETAIL_HAVE_OPENCV_UI && \
    (CV_SIMD_64F || CV_SIMD_SCALABLE_64F) && \
    (CV_NEON || CV_SSE2 || CV_AVX2 || CV_AVX512_SKX)
            if (cpu::opencv_ui_allowed())
            {
                const int lanes =
                    cv::VTraits<cv::v_float64>::vlanes();
                const cv::v_float64 row_vector =
                    cv::vx_setall_f64(row_weight);
                for (; x + lanes <= winSize.width; x += lanes)
                {
                    cv::vx_store(
                        output + x,
                        cv::v_mul(
                            cv::vx_load(x_weights.data() + x),
                            row_vector));
                }
                used_ui = used_ui || x > 0;
            }
#endif
            for (; x < winSize.width; ++x)
            {
                output[x] =
                    x_weights[static_cast<size_t>(x)] *
                    row_weight;
            }
        }
    }
    cpu::set_last_dispatch_tag(
        used_ui
            ? cpu::DispatchTag::OpenCVUI
            : cpu::DispatchTag::Scalar);
}

}  // namespace cvh

#endif  // CVH_IMGPROC_KERNELS_H
