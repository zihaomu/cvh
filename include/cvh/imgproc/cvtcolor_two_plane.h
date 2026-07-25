#ifndef CVH_IMGPROC_CVTCOLOR_TWO_PLANE_H
#define CVH_IMGPROC_CVTCOLOR_TWO_PLANE_H

#include "cvtcolor.h"

namespace cvh
{

inline void cvtColorTwoPlane(const Mat& y,
                             const Mat& uv,
                             Mat& dst,
                             int code)
{
    if (y.empty() || y.dims != 2 || y.type() != CV_8UC1 ||
        uv.empty() || uv.dims != 2 || uv.type() != CV_8UC2)
    {
        CV_Error(
            Error::StsBadArg,
            "cvtColorTwoPlane expects CV_8UC1 Y and CV_8UC2 UV");
    }
    if ((y.size.p[0] & 1) != 0 || (y.size.p[1] & 1) != 0 ||
        uv.size.p[0] * 2 != y.size.p[0] ||
        uv.size.p[1] * 2 != y.size.p[1])
    {
        CV_Error(
            Error::StsBadSize,
            "cvtColorTwoPlane requires even Y dimensions and half-size UV");
    }
    const bool nv21 =
        code == COLOR_YUV2BGR_NV21 ||
        code == COLOR_YUV2RGB_NV21;
    const bool rgb =
        code == COLOR_YUV2RGB_NV12 ||
        code == COLOR_YUV2RGB_NV21;
    if (!nv21 &&
        code != COLOR_YUV2BGR_NV12 &&
        code != COLOR_YUV2RGB_NV12)
    {
        CV_Error(
            Error::StsBadFlag,
            "cvtColorTwoPlane unsupported conversion code");
    }

    const Mat y_source = y.data == dst.data ? y.clone() : y;
    const Mat uv_source = uv.data == dst.data ? uv.clone() : uv;
    dst.create(y_source.shape(), CV_8UC3);
    const int rows = y_source.size.p[0];
    const int cols = y_source.size.p[1];
    for (int row = 0; row < rows; row += 2)
    {
        const uchar* y_row0 =
            y_source.data +
            static_cast<size_t>(row) * y_source.step(0);
        const uchar* y_row1 =
            y_source.data +
            static_cast<size_t>(row + 1) * y_source.step(0);
        const uchar* uv_row =
            uv_source.data +
            static_cast<size_t>(row / 2) * uv_source.step(0);
        uchar* output0 =
            dst.data + static_cast<size_t>(row) * dst.step(0);
        uchar* output1 =
            dst.data + static_cast<size_t>(row + 1) * dst.step(0);
        for (int col = 0; col < cols; col += 2)
        {
            const size_t uv_index =
                static_cast<size_t>(col / 2) * 2;
            const int first = uv_row[uv_index];
            const int second = uv_row[uv_index + 1];
            const int uu = nv21 ? second : first;
            const int vv = nv21 ? first : second;
            const int d = uu - 128;
            const int e = vv - 128;
            const int blue_chroma = 516 * d + 128;
            const int green_chroma = -100 * d - 208 * e + 128;
            const int red_chroma = 409 * e + 128;

            const uchar* y_rows[2] = {y_row0, y_row1};
            uchar* output_rows[2] = {output0, output1};
            for (int block_row = 0; block_row < 2; ++block_row)
            {
                for (int block_col = 0; block_col < 2; ++block_col)
                {
                    const int x = col + block_col;
                    const int c =
                        std::max(
                            static_cast<int>(y_rows[block_row][x]) - 16,
                            0);
                    const int luminance = 298 * c;
                    const uchar blue = saturate_cast<uchar>(
                        (luminance + blue_chroma) >> 8);
                    const uchar green = saturate_cast<uchar>(
                        (luminance + green_chroma) >> 8);
                    const uchar red = saturate_cast<uchar>(
                        (luminance + red_chroma) >> 8);
                    uchar* pixel =
                        output_rows[block_row] +
                        static_cast<size_t>(x) * 3u;
                    pixel[rgb ? 0 : 2] = red;
                    pixel[1] = green;
                    pixel[rgb ? 2 : 0] = blue;
                }
            }
        }
    }
}

}  // namespace cvh

#endif  // CVH_IMGPROC_CVTCOLOR_TWO_PLANE_H
