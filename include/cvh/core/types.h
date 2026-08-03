#ifndef CVH_CORE_TYPES_H
#define CVH_CORE_TYPES_H

#include <algorithm>
#include <cassert>

namespace cvh {

template<typename T>
struct Point_
{
    T x;
    T y;

    constexpr Point_() : x(0), y(0) {}
    constexpr Point_(T x_, T y_) : x(x_), y(y_) {}

    template<typename U>
    constexpr Point_(const Point_<U>& point)
        : x(static_cast<T>(point.x)), y(static_cast<T>(point.y))
    {
    }
};

template<typename T>
inline bool operator==(const Point_<T>& lhs, const Point_<T>& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

template<typename T>
inline bool operator!=(const Point_<T>& lhs, const Point_<T>& rhs)
{
    return !(lhs == rhs);
}

using Point2i = Point_<int>;
using Point2f = Point_<float>;
using Point2d = Point_<double>;
using Point = Point2i;

template<typename T>
struct Size_
{
    T width;
    T height;

    constexpr Size_() : width(0), height(0) {}
    constexpr Size_(T width_, T height_) : width(width_), height(height_) {}

    template<typename U>
    constexpr Size_(const Size_<U>& size)
        : width(static_cast<T>(size.width)), height(static_cast<T>(size.height))
    {
    }

    constexpr bool empty() const { return width <= 0 || height <= 0; }
};

template<typename T>
inline bool operator==(const Size_<T>& lhs, const Size_<T>& rhs)
{
    return lhs.width == rhs.width && lhs.height == rhs.height;
}

template<typename T>
inline bool operator!=(const Size_<T>& lhs, const Size_<T>& rhs)
{
    return !(lhs == rhs);
}

using Size2i = Size_<int>;
using Size2f = Size_<float>;
using Size2d = Size_<double>;
using Size = Size2i;

template<typename T>
struct Rect_
{
    T x;
    T y;
    T width;
    T height;

    constexpr Rect_() : x(0), y(0), width(0), height(0) {}
    constexpr Rect_(T x_, T y_, T width_, T height_)
        : x(x_), y(y_), width(width_), height(height_)
    {
    }
    constexpr Rect_(const Point_<T>& origin, const Size_<T>& size)
        : x(origin.x), y(origin.y), width(size.width), height(size.height)
    {
    }
    constexpr Rect_(const Point_<T>& point1, const Point_<T>& point2)
        : x(std::min(point1.x, point2.x)),
          y(std::min(point1.y, point2.y)),
          width(std::max(point1.x, point2.x) - std::min(point1.x, point2.x)),
          height(std::max(point1.y, point2.y) - std::min(point1.y, point2.y))
    {
    }

    template<typename U>
    constexpr Rect_(const Rect_<U>& rect)
        : x(static_cast<T>(rect.x)),
          y(static_cast<T>(rect.y)),
          width(static_cast<T>(rect.width)),
          height(static_cast<T>(rect.height))
    {
    }

    constexpr Point_<T> tl() const { return Point_<T>(x, y); }
    constexpr Point_<T> br() const { return Point_<T>(x + width, y + height); }
    constexpr Size_<T> size() const { return Size_<T>(width, height); }
    constexpr T area() const { return width * height; }
    constexpr bool empty() const { return width <= 0 || height <= 0; }
    constexpr bool contains(const Point_<T>& point) const
    {
        return x <= point.x && point.x < x + width &&
               y <= point.y && point.y < y + height;
    }
};

template<typename T>
inline bool operator==(const Rect_<T>& lhs, const Rect_<T>& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y &&
           lhs.width == rhs.width && lhs.height == rhs.height;
}

template<typename T>
inline bool operator!=(const Rect_<T>& lhs, const Rect_<T>& rhs)
{
    return !(lhs == rhs);
}

using Rect2i = Rect_<int>;
using Rect2f = Rect_<float>;
using Rect2d = Rect_<double>;
using Rect = Rect2i;

struct Moments
{
    double m00;
    double m10;
    double m01;
    double m20;
    double m11;
    double m02;
    double m30;
    double m21;
    double m12;
    double m03;
    double mu20;
    double mu11;
    double mu02;
    double mu30;
    double mu21;
    double mu12;
    double mu03;
    double nu20;
    double nu11;
    double nu02;
    double nu30;
    double nu21;
    double nu12;
    double nu03;

    constexpr Moments()
        : m00(0), m10(0), m01(0), m20(0), m11(0), m02(0),
          m30(0), m21(0), m12(0), m03(0),
          mu20(0), mu11(0), mu02(0), mu30(0), mu21(0), mu12(0), mu03(0),
          nu20(0), nu11(0), nu02(0), nu30(0), nu21(0), nu12(0), nu03(0)
    {
    }

    constexpr Moments(double m00_, double m10_, double m01_, double m20_,
                      double m11_, double m02_, double m30_, double m21_,
                      double m12_, double m03_)
        : m00(m00_), m10(m10_), m01(m01_), m20(m20_), m11(m11_), m02(m02_),
          m30(m30_), m21(m21_), m12(m12_), m03(m03_),
          mu20(0), mu11(0), mu02(0), mu30(0), mu21(0), mu12(0), mu03(0),
          nu20(0), nu11(0), nu02(0), nu30(0), nu21(0), nu12(0), nu03(0)
    {
    }
};

enum DecompTypes
{
    DECOMP_LU = 0,
};

struct Scalar
{
    double val[4];

    Scalar() : val{0.0, 0.0, 0.0, 0.0} {}
    explicit Scalar(double v0) : val{v0, 0.0, 0.0, 0.0} {}
    Scalar(double v0, double v1, double v2 = 0.0, double v3 = 0.0) : val{v0, v1, v2, v3} {}

    static Scalar all(double v)
    {
        return Scalar(v, v, v, v);
    }

    double& operator[](int i)
    {
        assert(i >= 0 && i < 4);
        return val[i];
    }

    const double& operator[](int i) const
    {
        assert(i >= 0 && i < 4);
        return val[i];
    }
};

inline bool operator==(const Scalar& lhs, const Scalar& rhs)
{
    return lhs.val[0] == rhs.val[0] &&
           lhs.val[1] == rhs.val[1] &&
           lhs.val[2] == rhs.val[2] &&
           lhs.val[3] == rhs.val[3];
}

inline bool operator!=(const Scalar& lhs, const Scalar& rhs)
{
    return !(lhs == rhs);
}

}  // namespace cvh

#endif  // CVH_CORE_TYPES_H
