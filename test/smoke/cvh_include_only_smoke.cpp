#include <cvh/cvh.h>
#include <cvh.h>

int main()
{
    cvh::MatShape shape = {2, 3, 4};
    if (cvh::total(shape) != 24)
    {
        return 1;
    }

    int dims_and_sizes[] = {3, 2, 3, 4};
    cvh::MatSize mat_size(dims_and_sizes);
    if (mat_size.dims() != 3 || mat_size[0] != 2 || mat_size[1] != 3 || mat_size[2] != 4)
    {
        return 2;
    }

    return 0;
}
