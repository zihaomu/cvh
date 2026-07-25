#include "test/core/support/transpose_internal_test_utils.hpp"

TEST(TransposeInternalTest, transpose2d_kernel_blocked_matches_reference_for_elem_sizes_and_shapes_in_auto_mode)
{
    const int shapes[][2] = {
        {11, 29},
        {5, 7},
        {13, 29},
        {64, 65},
    };

    const size_t elem_sizes[] = {1, 2, 4, 8};

    for (const auto& shape : shapes)
    {
        for (const size_t elem_size : elem_sizes)
        {
            SCOPED_TRACE(::testing::Message()
                         << "rows=" << shape[0]
                         << ", cols=" << shape[1]
                         << ", elem_size=" << elem_size);
            expect_transpose_kernel_bytes_equal(shape[0], shape[1], elem_size, cpu::DispatchMode::Auto);
        }
    }
}

TEST(TransposeInternalTest, transpose2d_kernel_blocked_matches_reference_in_scalar_only_mode)
{
    const int shapes[][2] = {
        {11, 29},
        {5, 7},
        {13, 29},
        {64, 65},
    };

    const size_t elem_sizes[] = {1, 2, 4, 8};

    for (const auto& shape : shapes)
    {
        for (const size_t elem_size : elem_sizes)
        {
            SCOPED_TRACE(::testing::Message()
                         << "rows=" << shape[0]
                         << ", cols=" << shape[1]
                         << ", elem_size=" << elem_size);
            expect_transpose_kernel_bytes_equal(shape[0], shape[1], elem_size, cpu::DispatchMode::ScalarOnly);
        }
    }
}
