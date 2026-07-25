
#pragma once

#include "cvh/core/mat.h"

#include <string>

namespace cvh
{

// Read a C-contiguous little-endian float32 or int32 NumPy array.
Mat readMatFromNpy(const std::string& path);

}  // namespace cvh
