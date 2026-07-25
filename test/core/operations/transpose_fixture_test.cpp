#include "test/core/support/core_fixture_test_utils.hpp"
#include "test/utils/mat_load.h"

#include <string>

using namespace cvh;

TEST(TransposeFixtureTest, generated_permutations_match_numpy)
{
    {
        const std::string case_name = "transpose_last2_3d";
        const Mat input =
            readMatFromNpy(
                test::core_data_path(case_name + "_i.npy"));
        const Mat expected =
            readMatFromNpy(
                test::core_data_path(case_name + "_o.npy"));
        const Mat actual = transpose(input);
        test::expect_mat_close(
            actual,
            expected,
            1e-6f,
            1e-6f,
            case_name);
    }

    {
        const std::string case_name = "transpose_perm_4d";
        const Mat input =
            readMatFromNpy(
                test::core_data_path(case_name + "_i.npy"));
        const Mat expected =
            readMatFromNpy(
                test::core_data_path(case_name + "_o.npy"));
        const Mat actual = transposeND(input, {0, 2, 1, 3});
        test::expect_mat_close(
            actual,
            expected,
            1e-6f,
            1e-6f,
            case_name);
    }
}
