#include "cvh.h"
#include "gtest/gtest.h"

using namespace cvh;

TEST(MatShapeDisplay_TEST, shape_semantics_remain_geometry_only)
{
    Mat m({2, 3}, CV_8UC3);

    EXPECT_EQ(m.shape(), (MatShape{2, 3}));
    EXPECT_EQ(m.shapeString(), "[2x3]");
    EXPECT_EQ(m.shapeString(ShapeDisplayOrder::Geometry), "[2x3]");
    EXPECT_EQ(m.shapeString(ShapeDisplayOrder::ChannelFirst), "[3x2x3]");
    EXPECT_EQ(m.displayShapeString(), "[3x2x3]");
}

TEST(MatShapeDisplay_TEST, display_policy_only_applies_channel_first_to_2d_multichannel)
{
    Mat gray({2, 3}, CV_8U);
    EXPECT_EQ(gray.shapeString(ShapeDisplayOrder::ChannelFirst), "[2x3]");
    EXPECT_EQ(gray.displayShapeString(), "[2x3]");

    Mat tensor({2, 3, 4}, CV_8UC3);
    EXPECT_EQ(tensor.shapeString(ShapeDisplayOrder::Geometry), "[2x3x4]");
    EXPECT_EQ(tensor.shapeString(ShapeDisplayOrder::ChannelFirst), "[2x3x4]");
    EXPECT_EQ(tensor.displayShapeString(), "[2x3x4]");
}

TEST(MatShapeDisplay_TEST, print_and_printshape_use_display_policy)
{
    Mat m({2, 3}, CV_8UC3);
    m = 0;

    testing::internal::CaptureStdout();
    m.printShape();
    const std::string shape_out = testing::internal::GetCapturedStdout();
    EXPECT_EQ(shape_out, "shape = [3x2x3]\n");

    testing::internal::CaptureStdout();
    m.print(0);
    const std::string print_out = testing::internal::GetCapturedStdout();
    EXPECT_NE(print_out.find("shape = [3x2x3]\n"), std::string::npos);
}
