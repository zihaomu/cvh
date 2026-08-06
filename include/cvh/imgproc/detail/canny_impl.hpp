#ifndef CVH_IMGPROC_DETAIL_CANNY_IMPL_HPP
#define CVH_IMGPROC_DETAIL_CANNY_IMPL_HPP

#include "fastpath_common.hpp"
#include "filter_ui.hpp"

namespace cvh
{
namespace detail
{

namespace canny_fastpath
{
inline thread_local const char* g_last_canny_algorithm_path =
    "canny_fallback";

inline bool try_canny_from_derivatives_fastpath_s16(const Mat& dx,
                                             const Mat& dy,
                                             Mat& edges,
                                             double threshold1,
                                             double threshold2,
                                             bool L2gradient)
{
    if (dx.empty() || dy.empty())
    {
        return false;
    }
    if (dx.dims != 2 || dy.dims != 2)
    {
        return false;
    }
    if (dx.type() != CV_16SC1 || dy.type() != CV_16SC1)
    {
        return false;
    }
    if (dx.size[0] != dy.size[0] || dx.size[1] != dy.size[1])
    {
        return false;
    }

    const int rows = dx.size[0];
    const int cols = dx.size[1];
    if (rows <= 0 || cols <= 0)
    {
        return false;
    }

    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t dx_step = dx.step(0);
    const std::size_t dy_step = dy.step(0);
    const float low_threshold = static_cast<float>(std::min(threshold1, threshold2));
    const float high_threshold = static_cast<float>(std::max(threshold1, threshold2));
    const double tan_pi_8 = std::tan(CV_PI / 8.0);
    const double tan_3pi_8 = std::tan(CV_PI * 3.0 / 8.0);

    std::vector<float> magnitude_ring(
        static_cast<std::size_t>(cols) * 3u,
        0.0f);
    std::vector<float> zero_magnitude(
        static_cast<std::size_t>(cols),
        0.0f);
    const auto compute_magnitude_row = [&](int y) {
        const short* dx_row = reinterpret_cast<const short*>(dx.data + static_cast<std::size_t>(y) * dx_step);
        const short* dy_row = reinterpret_cast<const short*>(dy.data + static_cast<std::size_t>(y) * dy_step);
        float* mag_row =
            magnitude_ring.data() +
            static_cast<std::size_t>(y % 3) * cols;
        for (int x = 0; x < cols; ++x)
        {
            const int gx = dx_row[x];
            const int gy = dy_row[x];
            if (L2gradient)
            {
                mag_row[x] =
                    static_cast<float>(std::sqrt(static_cast<double>(gx) * gx + static_cast<double>(gy) * gy));
            }
            else
            {
                mag_row[x] = static_cast<float>(std::abs(gx) + std::abs(gy));
            }
        }
    };
    compute_magnitude_row(0);

    const int map_cols = cols + 2;
    std::vector<uchar> edge_state(
        static_cast<std::size_t>(rows + 2) *
            static_cast<std::size_t>(map_cols),
        static_cast<uchar>(0));
    for (int y = 0; y < rows; ++y)
    {
        if (y + 1 < rows)
        {
            compute_magnitude_row(y + 1);
        }
        const short* dx_row = reinterpret_cast<const short*>(dx.data + static_cast<std::size_t>(y) * dx_step);
        const short* dy_row = reinterpret_cast<const short*>(dy.data + static_cast<std::size_t>(y) * dy_step);
        const float* previous_magnitude =
            y > 0
                ? magnitude_ring.data() +
                      static_cast<std::size_t>((y - 1) % 3) * cols
                : zero_magnitude.data();
        const float* current_magnitude =
            magnitude_ring.data() +
            static_cast<std::size_t>(y % 3) * cols;
        const float* next_magnitude =
            y + 1 < rows
                ? magnitude_ring.data() +
                      static_cast<std::size_t>((y + 1) % 3) * cols
                : zero_magnitude.data();
        uchar* state_row =
            edge_state.data() +
            static_cast<std::size_t>(y + 1) * map_cols + 1u;
        for (int x = 0; x < cols; ++x)
        {
            const float a = current_magnitude[x];
            if (a <= low_threshold)
            {
                continue;
            }

            const int gx = dx_row[x];
            const int gy = dy_row[x];
            const int ax = std::abs(gx);
            const int ay = std::abs(gy);
            float b = 0.0f;
            float c = 0.0f;
            bool keep = false;
            if (static_cast<double>(ay) < tan_pi_8 * ax)
            {
                b = x + 1 < cols ? current_magnitude[x + 1] : 0.0f;
                c = x > 0 ? current_magnitude[x - 1] : 0.0f;
                keep = a >= b && a > c;
            }
            else if (static_cast<double>(ay) > tan_3pi_8 * ax)
            {
                b = next_magnitude[x];
                c = previous_magnitude[x];
                keep = a >= b && a > c;
            }
            else if ((gx ^ gy) >= 0)
            {
                b = x + 1 < cols ? next_magnitude[x + 1] : 0.0f;
                c = x > 0 ? previous_magnitude[x - 1] : 0.0f;
                keep = a > b && a > c;
            }
            else
            {
                b = x + 1 < cols ? previous_magnitude[x + 1] : 0.0f;
                c = x > 0 ? next_magnitude[x - 1] : 0.0f;
                keep = a > b && a > c;
            }
            if (keep)
            {
                state_row[x] =
                    a > high_threshold ? static_cast<uchar>(2)
                                       : static_cast<uchar>(1);
            }
        }
    }

    std::vector<uchar> edge_map(count, static_cast<uchar>(0));
    const int neighbor_offsets[8] = {
        1,
        1 - map_cols,
        -map_cols,
        -1 - map_cols,
        -1,
        -1 + map_cols,
        map_cols,
        1 + map_cols,
    };

    std::vector<int> stack;
    stack.reserve(count / 8u + 8u);

    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            const int seed_idx = y * cols + x;
            const int seed_map_idx = (y + 1) * map_cols + x + 1;
            if (edge_state[static_cast<std::size_t>(seed_map_idx)] != 2)
            {
                continue;
            }

            edge_state[static_cast<std::size_t>(seed_map_idx)] = 0;
            edge_map[static_cast<std::size_t>(seed_idx)] = 255;
            stack.push_back(seed_map_idx);

            while (!stack.empty())
            {
                const int p = stack.back();
                stack.pop_back();

                for (int k = 0; k < 8; ++k)
                {
                    const int neighbor = p + neighbor_offsets[k];
                    if (edge_state[static_cast<std::size_t>(neighbor)] != 0)
                    {
                        edge_state[static_cast<std::size_t>(neighbor)] = 0;
                        const int ny = neighbor / map_cols - 1;
                        const int nx = neighbor % map_cols - 1;
                        edge_map[static_cast<std::size_t>(ny * cols + nx)] = 255;
                        stack.push_back(neighbor);
                    }
                }
            }
        }
    }

    edges.create(std::vector<int>{rows, cols}, CV_8UC1);
    const std::size_t edge_step = edges.step(0);
    for (int y = 0; y < rows; ++y)
    {
        std::memcpy(edges.data + static_cast<std::size_t>(y) * edge_step,
                    edge_map.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(cols),
                    static_cast<std::size_t>(cols));
    }

    return true;
}

inline bool try_canny_image_fastpath_u8(const Mat& image,
                                 Mat& edges,
                                 double threshold1,
                                 double threshold2,
                                 int apertureSize,
                                 bool L2gradient)
{
    if (image.empty() || image.dims != 2 || image.type() != CV_8UC1)
    {
        return false;
    }
    if (apertureSize != 3 && apertureSize != 5)
    {
        return false;
    }

    Mat src_local;
    const Mat* src_ref = &image;
    if (image.data == edges.data)
    {
        src_local = image.clone();
        src_ref = &src_local;
    }

    Mat dx;
    Mat dy;
    const int sobel_border = BORDER_REPLICATE | BORDER_ISOLATED;
    const bool fused_gradient =
        apertureSize == 3 &&
        filter_ui::spatial_gradient_u8_c1(
            *src_ref, dx, dy, BORDER_REPLICATE);
    if (!fused_gradient)
    {
        Sobel(
            *src_ref,
            dx,
            CV_16S,
            1,
            0,
            apertureSize,
            1.0,
            0.0,
            sobel_border);
        Sobel(
            *src_ref,
            dy,
            CV_16S,
            0,
            1,
            apertureSize,
            1.0,
            0.0,
            sobel_border);
    }
    const bool handled = try_canny_from_derivatives_fastpath_s16(
        dx, dy, edges, threshold1, threshold2, L2gradient);
    if (handled)
    {
        g_last_canny_algorithm_path =
            fused_gradient ? "canny_fused_gradient_ring_nms"
                           : "canny_ring_nms";
    }
    return handled;
}

} // namespace canny_fastpath

inline const char* last_canny_algorithm_path()
{
    return canny_fastpath::g_last_canny_algorithm_path;
}

inline void canny_image_fast_impl(const Mat& image,
                              Mat& edges,
                              double threshold1,
                              double threshold2,
                              int apertureSize,
                              bool L2gradient)
{
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    canny_fastpath::g_last_canny_algorithm_path = "canny_fallback";
    if (canny_fastpath::try_canny_image_fastpath_u8(
            image, edges, threshold1, threshold2, apertureSize, L2gradient))
    {
        return;
    }

    canny_fallback(image, edges, threshold1, threshold2, apertureSize, L2gradient);
}

inline void canny_deriv_fast_impl(const Mat& dx,
                                  const Mat& dy,
                                  Mat& edges,
                                  double threshold1,
                                  double threshold2,
                                  bool L2gradient)
{
    cpu::set_last_dispatch_tag(cpu::DispatchTag::Scalar);
    if (canny_fastpath::try_canny_from_derivatives_fastpath_s16(
            dx, dy, edges, threshold1, threshold2, L2gradient))
    {
        canny_fastpath::g_last_canny_algorithm_path = "canny_ring_nms";
        return;
    }

    canny_from_derivatives_fallback(dx, dy, edges, threshold1, threshold2, L2gradient);
}

} // namespace detail
} // namespace cvh

#endif // CVH_IMGPROC_DETAIL_CANNY_IMPL_HPP
