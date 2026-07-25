#pragma once

#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

using namespace cvh;

namespace
{

template<typename T>
void set_raw_bits(T& value, const void* bits)
{
    std::memcpy(&value, bits, sizeof(T));
}

template<typename UInt, typename T>
UInt raw_bits(const T& value)
{
    static_assert(sizeof(UInt) == sizeof(T));
    UInt bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

}  // namespace
