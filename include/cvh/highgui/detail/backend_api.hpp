#ifndef CVH_HIGHGUI_DETAIL_BACKEND_API_HPP
#define CVH_HIGHGUI_DETAIL_BACKEND_API_HPP

#include "../../core/mat.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace cvh {
namespace detail {

struct HighguiImage
{
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uchar> pixels;
};

inline bool highgui_headless_mode()
{
    const char* value = std::getenv("CVH_HIGHGUI_HEADLESS");
    return value &&
           (std::strcmp(value, "1") == 0 ||
            std::strcmp(value, "true") == 0 ||
            std::strcmp(value, "TRUE") == 0 ||
            std::strcmp(value, "on") == 0 ||
            std::strcmp(value, "ON") == 0);
}

inline HighguiImage prepare_highgui_image(const Mat& input)
{
    if (input.empty())
    {
        CV_Error(Error::StsBadArg, "imshow: source image must not be empty");
    }
    if (input.dims != 2)
    {
        CV_Error_(
            Error::StsBadArg,
            ("imshow: only 2D Mat is supported, got dims=%d", input.dims));
    }
    if (input.depth() != CV_8U)
    {
        CV_Error_(
            Error::StsBadArg,
            ("imshow: only CV_8U is supported, got depth=%d", input.depth()));
    }

    const int source_channels = input.channels();
    if (source_channels != 1 && source_channels != 3 &&
        source_channels != 4)
    {
        CV_Error_(
            Error::StsBadArg,
            ("imshow: expected 1, 3, or 4 channels, got %d",
             source_channels));
    }

    HighguiImage output;
    output.width = input.size[1];
    output.height = input.size[0];
    output.channels = source_channels == 1 ? 1 : 3;
    output.pixels.resize(
        static_cast<size_t>(output.width) *
        static_cast<size_t>(output.height) *
        static_cast<size_t>(output.channels));

    for (int y = 0; y < output.height; ++y)
    {
        const uchar* source =
            input.data + static_cast<size_t>(y) * input.step(0);
        uchar* destination =
            output.pixels.data() +
            static_cast<size_t>(y) *
                static_cast<size_t>(output.width) *
                static_cast<size_t>(output.channels);

        if (source_channels == output.channels)
        {
            std::memcpy(
                destination,
                source,
                static_cast<size_t>(output.width) *
                    static_cast<size_t>(output.channels));
            continue;
        }

        for (int x = 0; x < output.width; ++x)
        {
            const uchar* source_pixel =
                source + static_cast<size_t>(x) * 4;
            uchar* destination_pixel =
                destination + static_cast<size_t>(x) * 3;
            destination_pixel[0] = source_pixel[0];
            destination_pixel[1] = source_pixel[1];
            destination_pixel[2] = source_pixel[2];
        }
    }

    return output;
}

}  // namespace detail
}  // namespace cvh

#endif  // CVH_HIGHGUI_DETAIL_BACKEND_API_HPP
