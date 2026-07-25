#pragma once

#include "cvh.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>
#include <limits>

using namespace cvh;

namespace
{

std::uint16_t read_half_bits(const Mat& matrix, int x)
{
    std::uint16_t bits = 0;
    std::memcpy(
        &bits,
        matrix.data + static_cast<size_t>(x) * sizeof(bits),
        sizeof(bits));
    return bits;
}

}  // namespace
