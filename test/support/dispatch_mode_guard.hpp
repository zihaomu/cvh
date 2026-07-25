#pragma once

#include "cvh/core/detail/dispatch_control.h"

namespace cvh::test
{

class DispatchModeGuard
{
public:
    DispatchModeGuard()
        : previous_(cpu::dispatch_mode())
    {
    }

    explicit DispatchModeGuard(cpu::DispatchMode mode)
        : previous_(cpu::dispatch_mode())
    {
        cpu::set_dispatch_mode(mode);
    }

    ~DispatchModeGuard()
    {
        cpu::set_dispatch_mode(previous_);
    }

    DispatchModeGuard(const DispatchModeGuard&) = delete;
    DispatchModeGuard& operator=(const DispatchModeGuard&) = delete;

private:
    cpu::DispatchMode previous_;
};

}  // namespace cvh::test
