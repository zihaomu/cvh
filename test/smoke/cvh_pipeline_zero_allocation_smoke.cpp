#include "cvh/pipeline/pipeline.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <new>

namespace {

std::atomic<bool> g_track_allocations{false};
std::atomic<std::size_t> g_allocation_count{0};

void recordAllocation()
{
    if (g_track_allocations.load(std::memory_order_relaxed))
    {
        g_allocation_count.fetch_add(1, std::memory_order_relaxed);
    }
}

}  // namespace

void* operator new(std::size_t size)
{
    recordAllocation();
    if (void* pointer = std::malloc(size == 0 ? 1 : size))
    {
        return pointer;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size)
{
    return ::operator new(size);
}

void* operator new(std::size_t size, std::align_val_t alignment)
{
    recordAllocation();
    void* pointer = nullptr;
    const std::size_t alignment_value =
        static_cast<std::size_t>(alignment);
    if (posix_memalign(
            &pointer, alignment_value, size == 0 ? 1 : size) != 0)
    {
        throw std::bad_alloc();
    }
    return pointer;
}

void* operator new[](std::size_t size, std::align_val_t alignment)
{
    return ::operator new(size, alignment);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept
{
    try
    {
        return ::operator new(size);
    }
    catch (...)
    {
        return nullptr;
    }
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept
{
    return ::operator new(size, std::nothrow);
}

void* operator new(std::size_t size,
                   std::align_val_t alignment,
                   const std::nothrow_t&) noexcept
{
    try
    {
        return ::operator new(size, alignment);
    }
    catch (...)
    {
        return nullptr;
    }
}

void* operator new[](std::size_t size,
                     std::align_val_t alignment,
                     const std::nothrow_t&) noexcept
{
    return ::operator new(size, alignment, std::nothrow);
}

void operator delete(void* pointer) noexcept
{
    std::free(pointer);
}

void operator delete[](void* pointer) noexcept
{
    ::operator delete(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept
{
    std::free(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept
{
    std::free(pointer);
}

void operator delete(void* pointer, std::align_val_t) noexcept
{
    std::free(pointer);
}

void operator delete[](void* pointer, std::align_val_t) noexcept
{
    std::free(pointer);
}

void operator delete(void* pointer,
                     std::size_t,
                     std::align_val_t) noexcept
{
    std::free(pointer);
}

void operator delete[](void* pointer,
                       std::size_t,
                       std::align_val_t) noexcept
{
    std::free(pointer);
}

void operator delete(void* pointer, const std::nothrow_t&) noexcept
{
    std::free(pointer);
}

void operator delete[](void* pointer, const std::nothrow_t&) noexcept
{
    std::free(pointer);
}

void operator delete(void* pointer,
                     std::align_val_t,
                     const std::nothrow_t&) noexcept
{
    std::free(pointer);
}

void operator delete[](void* pointer,
                       std::align_val_t,
                       const std::nothrow_t&) noexcept
{
    std::free(pointer);
}

int main()
{
    cvh::Mat input({9, 13}, CV_8UC3);
    for (int y = 0; y < input.size[0]; ++y)
    {
        for (int x = 0; x < input.size[1]; ++x)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                input.at<uchar>(y, x, channel) =
                    static_cast<uchar>(y * 17 + x * 5 + channel);
            }
        }
    }

    const cvh::PipelinePlan plan =
        cvh::pipe(
            cvh::imageDesc(13, 9, cvh::PixelFormat::BGR8),
            cvh::tensorDesc<float>({1, 3, 5, 7}, cvh::Layout::NCHW))
            .color(cvh::Color::RGB)
            .resize(7, 5, cvh::Interpolation::Linear)
            .normalize({1.0f, 2.0f, 3.0f}, {2.0f, 3.0f, 4.0f})
            .layout(cvh::Layout::NCHW)
            .prepare();
    cvh::PipelineWorkspace workspace(plan);
    cvh::Mat output({1, 3, 5, 7}, CV_32FC1);
    cvh::PipelineRunInfo run_info;
    const cvh::ConstImageView input_view = cvh::bgr(
        input.data,
        static_cast<std::size_t>(input.size[0] - 1) * input.step(0) +
            static_cast<std::size_t>(input.size[1]) * input.elemSize(),
        input.size[1],
        input.size[0],
        input.step(0));
    const cvh::TensorView output_view = cvh::nchw(
        reinterpret_cast<float*>(output.data),
        output.total() * output.elemSize(),
        1,
        3,
        5,
        7);

    const cvh::PipelineStatus warmup =
        plan.tryRun(input, output, workspace.view(), &run_info);
    if (!warmup)
    {
        std::cerr << "pipeline warmup failed: " << warmup.message() << '\n';
        return 1;
    }
    const cvh::PipelineStatus view_warmup =
        plan.tryRun(input_view, output_view, workspace.view(), &run_info);
    if (!view_warmup)
    {
        std::cerr << "borrowed pipeline warmup failed: "
                  << view_warmup.message() << '\n';
        return 1;
    }

    cvh::Mat neon_input({24, 22}, CV_8UC3);
    for (int y = 0; y < neon_input.size[0]; ++y)
    {
        for (int x = 0; x < neon_input.size[1]; ++x)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                neon_input.at<uchar>(y, x, channel) =
                    static_cast<uchar>(y * 19 + x * 7 + channel * 41);
            }
        }
    }
    const cvh::PipelinePlan neon_plan =
        cvh::pipe(
            cvh::imageDesc(22, 24, cvh::PixelFormat::BGR8),
            cvh::tensorDesc<float>({1, 3, 24, 11}, cvh::Layout::NCHW))
            .color(cvh::Color::RGB)
            .resize(11, 24, cvh::Interpolation::Nearest)
            .normalize({1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 8.0f})
            .layout(cvh::Layout::NCHW)
            .prepare();
    cvh::PipelineWorkspace neon_workspace(neon_plan);
    cvh::Mat neon_output({1, 3, 24, 11}, CV_32FC1);
    const cvh::PipelineStatus neon_warmup = neon_plan.tryRun(
        neon_input, neon_output, neon_workspace.view(), &run_info);
    if (!neon_warmup)
    {
        std::cerr << "NEON-candidate pipeline warmup failed: "
                  << neon_warmup.message() << '\n';
        return 1;
    }

    const cvh::ColorSpec yuv_color_spec{
        cvh::ColorMatrix::BT709,
        cvh::ColorRange::Limited,
        cvh::ChromaLocation::Left};
    const cvh::PipelinePlan yuv_plan =
        cvh::pipe(
            cvh::imageDesc(
                4, 4, cvh::PixelFormat::NV12, yuv_color_spec),
            cvh::tensorDesc<signed char>(
                {1, 3, 4, 6}, cvh::Layout::NCHW))
            .color(cvh::Color::RGB)
            .letterbox(6, 4, 114.0f, cvh::Interpolation::Linear)
            .normalize({1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 8.0f})
            .quantize(cvh::PipelineDataType::S8, 0.025f, -3)
            .layout(cvh::Layout::NCHW)
            .prepare();
    std::array<uchar, 24> y_plane{};
    std::array<uchar, 12> uv_plane{};
    std::array<signed char, 72> yuv_output{};
    const cvh::ConstImageView yuv_input_view = cvh::nv12(
        y_plane.data(),
        6,
        y_plane.size(),
        uv_plane.data(),
        6,
        uv_plane.size(),
        4,
        4,
        yuv_color_spec);
    const cvh::TensorView yuv_output_view = cvh::nchw(
        yuv_output.data(),
        yuv_output.size() * sizeof(signed char),
        1,
        3,
        4,
        6);
    cvh::PipelineWorkspace yuv_workspace(yuv_plan);
    const cvh::PipelineStatus yuv_warmup = yuv_plan.tryRun(
        yuv_input_view,
        yuv_output_view,
        yuv_workspace.view(),
        &run_info);
    if (!yuv_warmup)
    {
        std::cerr << "YUV pipeline warmup failed: "
                  << yuv_warmup.message() << '\n';
        return 1;
    }

    g_allocation_count.store(0, std::memory_order_relaxed);
    g_track_allocations.store(true, std::memory_order_relaxed);
    const cvh::PipelineStatus status =
        plan.tryRun(input, output, workspace.view(), &run_info);
    const cvh::PipelineStatus view_status =
        plan.tryRun(input_view, output_view, workspace.view(), &run_info);
    const cvh::PipelineStatus neon_status = neon_plan.tryRun(
        neon_input, neon_output, neon_workspace.view(), &run_info);
    const cvh::PipelineStatus yuv_status = yuv_plan.tryRun(
        yuv_input_view,
        yuv_output_view,
        yuv_workspace.view(),
        &run_info);
    g_track_allocations.store(false, std::memory_order_relaxed);

    if (!status)
    {
        std::cerr << "pipeline measured run failed: " << status.message()
                  << '\n';
        return 2;
    }
    if (!view_status)
    {
        std::cerr << "borrowed pipeline measured run failed: "
                  << view_status.message() << '\n';
        return 2;
    }
    if (!neon_status)
    {
        std::cerr << "NEON-candidate pipeline measured run failed: "
                  << neon_status.message() << '\n';
        return 2;
    }
    if (!yuv_status)
    {
        std::cerr << "YUV pipeline measured run failed: "
                  << yuv_status.message() << '\n';
        return 2;
    }
    const std::size_t allocations =
        g_allocation_count.load(std::memory_order_relaxed);
    if (allocations != 0)
    {
        std::cerr << "prepared pipeline run allocated " << allocations
                  << " heap block(s)\n";
        return 3;
    }
    return 0;
}
