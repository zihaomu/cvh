#include "test/imgproc/support/pyramid_color_test_utils.hpp"

namespace
{

enum class BayerChannel
{
    blue,
    green,
    red,
};

BayerChannel bayer_channel_at(int code, int row, int column)
{
    const bool odd_row = (row & 1) != 0;
    const bool odd_column = (column & 1) != 0;
    if (code == COLOR_BayerBG2BGR)
    {
        if (!odd_row && !odd_column) return BayerChannel::red;
        if (odd_row && odd_column) return BayerChannel::blue;
    }
    else if (code == COLOR_BayerGB2BGR)
    {
        if (!odd_row && odd_column) return BayerChannel::red;
        if (odd_row && !odd_column) return BayerChannel::blue;
    }
    else if (code == COLOR_BayerRG2BGR)
    {
        if (!odd_row && !odd_column) return BayerChannel::blue;
        if (odd_row && odd_column) return BayerChannel::red;
    }
    else
    {
        if (!odd_row && odd_column) return BayerChannel::blue;
        if (odd_row && !odd_column) return BayerChannel::red;
    }
    return BayerChannel::green;
}

}  // namespace

TEST(DemosaicingTest, demosaicing_covers_four_bayer_patterns)
{
    const int codes[] = {
        COLOR_BayerBG2BGR,
        COLOR_BayerGB2BGR,
        COLOR_BayerRG2BGR,
        COLOR_BayerGR2BGR,
    };
    for (const int code : codes)
    {
        Mat bayer({7, 9}, CV_8UC1);
        for (int y = 0; y < bayer.size.p[0]; ++y)
        {
            for (int x = 0; x < bayer.size.p[1]; ++x)
            {
                const BayerChannel channel =
                    bayer_channel_at(code, y, x);
                bayer.at<uchar>(y, x) =
                    channel == BayerChannel::blue
                        ? 20
                        : (channel == BayerChannel::green ? 90 : 180);
            }
        }
        Mat color;
        demosaicing(bayer, color, code);
        EXPECT_EQ(color.at<uchar>(3, 4, 0), 20);
        EXPECT_EQ(color.at<uchar>(3, 4, 1), 90);
        EXPECT_EQ(color.at<uchar>(3, 4, 2), 180);
        EXPECT_EQ(color.at<uchar>(0, 0, 1), 90);
    }
    Mat source({3, 3}, CV_8UC1);
    Mat output;
    EXPECT_THROW(demosaicing(source, output, 999), Exception);
    EXPECT_THROW(demosaicing(source, output, COLOR_BayerBG2BGR, 4), Exception);
}
