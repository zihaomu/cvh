#pragma once

#include "cvh.h"
#include "gtest/gtest.h"

#include <cmath>
#include <vector>

using namespace cvh;

namespace
{

template<typename T>
Mat make_vec_mat(const std::initializer_list<T>& values, int type)
{
    Mat out({1, static_cast<int>(values.size())}, type);
    int idx = 0;
    for (const T v : values)
    {
        out.at<T>(0, idx++) = v;
    }
    return out;
}

template<typename T>
void expect_vec_eq(const Mat& m, const std::initializer_list<T>& values)
{
    ASSERT_EQ(m.size[0], 1);
    ASSERT_EQ(m.size[1], static_cast<int>(values.size()));
    int idx = 0;
    for (const T v : values)
    {
        EXPECT_EQ(m.at<T>(0, idx++), v);
    }
}

void expect_vec_near_f32(const Mat& m, const std::initializer_list<float>& values, float eps = 1e-5f)
{
    ASSERT_EQ(m.type(), CV_32FC1);
    ASSERT_EQ(m.size[0], 1);
    ASSERT_EQ(m.size[1], static_cast<int>(values.size()));
    int idx = 0;
    for (const float v : values)
    {
        EXPECT_NEAR(m.at<float>(0, idx++), v, eps);
    }
}

Mat make_vec_mat_from_doubles(const std::initializer_list<double>& values, int type)
{
    Mat out({1, static_cast<int>(values.size())}, type);
    int idx = 0;
    const int depth = CV_MAT_DEPTH(type);
    for (const double v : values)
    {
        switch (depth)
        {
            case CV_8U:
                out.at<uchar>(0, idx) = saturate_cast<uchar>(v);
                break;
            case CV_8S:
                out.at<schar>(0, idx) = saturate_cast<schar>(v);
                break;
            case CV_16U:
                out.at<ushort>(0, idx) = saturate_cast<ushort>(v);
                break;
            case CV_16S:
                out.at<short>(0, idx) = saturate_cast<short>(v);
                break;
            case CV_32S:
                out.at<int>(0, idx) = saturate_cast<int>(v);
                break;
            case CV_32U:
                out.at<uint>(0, idx) = saturate_cast<uint>(v);
                break;
            case CV_32F:
                out.at<float>(0, idx) = saturate_cast<float>(v);
                break;
            case CV_16F:
                out.at<hfloat>(0, idx) = saturate_cast<hfloat>(v);
                break;
            case CV_64F:
                out.at<double>(0, idx) = v;
                break;
            default:
                CV_Error_(Error::StsNotImplemented, ("Unsupported depth=%d in test helper", depth));
        }
        ++idx;
    }
    return out;
}

double read_vec_value_as_double(const Mat& m, int idx)
{
    switch (m.depth())
    {
        case CV_8U: return static_cast<double>(m.at<uchar>(0, idx));
        case CV_8S: return static_cast<double>(m.at<schar>(0, idx));
        case CV_16U: return static_cast<double>(m.at<ushort>(0, idx));
        case CV_16S: return static_cast<double>(m.at<short>(0, idx));
        case CV_32S: return static_cast<double>(m.at<int>(0, idx));
        case CV_32U: return static_cast<double>(m.at<uint>(0, idx));
        case CV_32F: return static_cast<double>(m.at<float>(0, idx));
        case CV_16F: return static_cast<double>(static_cast<float>(m.at<hfloat>(0, idx)));
        case CV_64F: return m.at<double>(0, idx);
        default:
            CV_Error_(Error::StsNotImplemented, ("Unsupported depth=%d in test helper", m.depth()));
            return 0.0;
    }
}

void expect_vec_match_by_depth(const Mat& m,
                               const std::initializer_list<double>& values,
                               double float_eps = 1e-6,
                               double half_eps = 2e-2)
{
    ASSERT_EQ(m.size[0], 1);
    ASSERT_EQ(m.size[1], static_cast<int>(values.size()));
    int idx = 0;
    for (const double expected : values)
    {
        const double actual = read_vec_value_as_double(m, idx++);
        if (m.depth() == CV_16F)
        {
            EXPECT_NEAR(actual, expected, half_eps);
        }
        else if (m.depth() == CV_32F || m.depth() == CV_64F)
        {
            EXPECT_NEAR(actual, expected, float_eps);
        }
        else
        {
            EXPECT_EQ(actual, expected);
        }
    }
}

}  // namespace
