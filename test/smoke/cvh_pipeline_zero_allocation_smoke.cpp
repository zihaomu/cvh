#include "cvh/pipeline/pipeline.h"

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

    const cvh::PipelineStatus warmup =
        plan.tryRun(input, output, workspace.view(), &run_info);
    if (!warmup)
    {
        std::cerr << "pipeline warmup failed: " << warmup.message() << '\n';
        return 1;
    }

    g_allocation_count.store(0, std::memory_order_relaxed);
    g_track_allocations.store(true, std::memory_order_relaxed);
    const cvh::PipelineStatus status =
        plan.tryRun(input, output, workspace.view(), &run_info);
    g_track_allocations.store(false, std::memory_order_relaxed);

    if (!status)
    {
        std::cerr << "pipeline measured run failed: " << status.message()
                  << '\n';
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
