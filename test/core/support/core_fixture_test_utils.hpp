#pragma once

#include "cvh.h"
#include "test/core/support/floating_point_test_utils.hpp"

#include <string>

namespace cvh::test
{

inline std::string core_data_path(const std::string& filename)
{
    return std::string(M_ROOT_PATH) + "/test/core/data/npy/" + filename;
}

}  // namespace cvh::test
