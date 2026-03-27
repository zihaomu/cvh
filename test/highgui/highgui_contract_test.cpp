#include "cvh.h"
#include "gtest/gtest.h"

#include <filesystem>
#include <string>

using namespace cvh;

namespace
{

int g_imshow_calls = 0;
int g_waitkey_calls = 0;
int g_last_rows = -1;
int g_last_cols = -1;
std::string g_last_name;

void test_imshow_backend(const std::string& winname, const Mat& mat)
{
    ++g_imshow_calls;
    g_last_name = winname;
    g_last_rows = mat.size[0];
    g_last_cols = mat.size[1];
}

int test_waitkey_backend(int delay)
{
    ++g_waitkey_calls;
    return delay + 7;
}

}  // namespace

TEST(Highgui_TEST, imshow_fallback_writes_png_file)
{
    Mat img({3, 4}, CV_8UC3);
    for (int y = 0; y < img.size[0]; ++y)
    {
        for (int x = 0; x < img.size[1]; ++x)
        {
            img.at<uchar>(y, x, 0) = static_cast<uchar>(x * 10 + y);
            img.at<uchar>(y, x, 1) = static_cast<uchar>(x * 3 + y * 7);
            img.at<uchar>(y, x, 2) = static_cast<uchar>(x * 11 + y * 5);
        }
    }

    const std::string winname = "cvh:test/highgui fallback";
    const std::string filename = detail::sanitize_window_name(winname) + ".png";
    const std::filesystem::path out = std::filesystem::current_path() / filename;

    std::filesystem::remove(out);
    ASSERT_FALSE(std::filesystem::exists(out));

    imshow(winname, img);
    ASSERT_TRUE(std::filesystem::exists(out));

    const Mat loaded = imread(out.string(), IMREAD_COLOR);
    ASSERT_FALSE(loaded.empty());
    EXPECT_EQ(loaded.size[0], img.size[0]);
    EXPECT_EQ(loaded.size[1], img.size[1]);

    std::filesystem::remove(out);
}

TEST(Highgui_TEST, waitkey_fallback_returns_minus_one)
{
    EXPECT_EQ(waitKey(1), -1);
    EXPECT_EQ(waitKey(0), -1);
}

TEST(Highgui_TEST, dispatch_registration_overrides_api_calls)
{
    g_imshow_calls = 0;
    g_waitkey_calls = 0;
    g_last_rows = -1;
    g_last_cols = -1;
    g_last_name.clear();

    const detail::ImshowFn old_imshow = detail::imshow_dispatch();
    const detail::WaitKeyFn old_waitkey = detail::waitkey_dispatch();

    detail::register_imshow_backend(&test_imshow_backend);
    detail::register_waitkey_backend(&test_waitkey_backend);

    Mat img({2, 5}, CV_8UC1);
    img = 9;

    imshow("dispatch_case", img);
    const int key = waitKey(33);

    EXPECT_EQ(g_imshow_calls, 1);
    EXPECT_EQ(g_waitkey_calls, 1);
    EXPECT_EQ(g_last_name, "dispatch_case");
    EXPECT_EQ(g_last_rows, 2);
    EXPECT_EQ(g_last_cols, 5);
    EXPECT_EQ(key, 40);

    detail::register_imshow_backend(old_imshow);
    detail::register_waitkey_backend(old_waitkey);
}

TEST(Highgui_TEST, imshow_throws_on_unsupported_depth)
{
    const detail::ImshowFn old_imshow = detail::imshow_dispatch();
    detail::register_imshow_backend(&detail::imshow_fallback);

    Mat fp32({2, 2}, CV_32FC1);
    EXPECT_THROW(imshow("bad_depth", fp32), Exception);

    detail::register_imshow_backend(old_imshow);
}
