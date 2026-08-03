#ifndef CVH_IMGPROC_DETAIL_SHAPE_IMPL_HPP
#define CVH_IMGPROC_DETAIL_SHAPE_IMPL_HPP

#include "../shape.h"

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <limits>
#include <type_traits>
#include <utility>

namespace cvh
{
namespace detail
{

template<typename T>
inline void validate_point_values(const std::vector<Point_<T>>& points)
{
    if constexpr (std::is_floating_point<T>::value)
    {
        for (const Point_<T>& point : points)
        {
            if (!std::isfinite(point.x) || !std::isfinite(point.y))
            {
                CV_Error(Error::StsBadArg, "shape point coordinates must be finite");
            }
        }
    }
}

template<typename T>
inline Rect bounding_rect_points(const std::vector<Point_<T>>& points)
{
    if (points.empty())
    {
        return Rect();
    }
    validate_point_values(points);
    const auto floor_to_int = [](T value) {
        const double floored = std::floor(static_cast<double>(value));
        if (floored < static_cast<double>(std::numeric_limits<int>::min()) ||
            floored > static_cast<double>(std::numeric_limits<int>::max()))
        {
            CV_Error(Error::StsOutOfRange, "boundingRect coordinate is outside the integer range");
        }
        return static_cast<int>(floored);
    };
    int minimum_x = floor_to_int(points[0].x);
    int maximum_x = minimum_x;
    int minimum_y = floor_to_int(points[0].y);
    int maximum_y = minimum_y;
    for (size_t index = 1; index < points.size(); ++index)
    {
        const int x = floor_to_int(points[index].x);
        const int y = floor_to_int(points[index].y);
        minimum_x = std::min(minimum_x, x);
        maximum_x = std::max(maximum_x, x);
        minimum_y = std::min(minimum_y, y);
        maximum_y = std::max(maximum_y, y);
    }
    const int64 width = static_cast<int64>(maximum_x) - minimum_x + 1;
    const int64 height = static_cast<int64>(maximum_y) - minimum_y + 1;
    if (width > std::numeric_limits<int>::max() ||
        height > std::numeric_limits<int>::max())
    {
        CV_Error(Error::StsOutOfRange, "boundingRect extent exceeds the integer range");
    }
    return Rect(minimum_x, minimum_y, static_cast<int>(width), static_cast<int>(height));
}

template<typename T>
inline double contour_area_points(const std::vector<Point_<T>>& contour, bool oriented)
{
    validate_point_values(contour);
    if (contour.empty())
    {
        return 0.0;
    }
    double area = 0.0;
    Point_<T> previous = contour.back();
    for (const Point_<T>& point : contour)
    {
        area += static_cast<double>(previous.x) * static_cast<double>(point.y) -
                static_cast<double>(point.x) * static_cast<double>(previous.y);
        previous = point;
    }
    area *= 0.5;
    return oriented ? area : std::fabs(area);
}

template<typename T>
inline double arc_length_points(const std::vector<Point_<T>>& curve, bool closed)
{
    validate_point_values(curve);
    if (curve.size() <= 1)
    {
        return 0.0;
    }
    double length = 0.0;
    size_t previous = closed ? curve.size() - 1 : 0;
    const size_t begin = closed ? 0 : 1;
    for (size_t index = begin; index < curve.size(); ++index)
    {
        const double dx = static_cast<double>(curve[index].x) - curve[previous].x;
        const double dy = static_cast<double>(curve[index].y) - curve[previous].y;
        length += std::sqrt(dx * dx + dy * dy);
        previous = index;
    }
    return length;
}

template<typename T>
inline void approx_poly_dp_points(const std::vector<Point_<T>>& source,
                                  std::vector<Point_<T>>& destination,
                                  double epsilon, bool closed)
{
    validate_point_values(source);
    if (epsilon < 0.0 || !(epsilon < 1e30))
    {
        CV_Error(Error::StsOutOfRange, "approxPolyDP epsilon is invalid");
    }
    destination.clear();
    const int count = static_cast<int>(source.size());
    if (count == 0)
    {
        return;
    }

    struct Slice { int start; int end; };
    std::vector<Slice> stack;
    std::vector<Point_<T>> output(static_cast<size_t>(count));
    int new_count = 0;
    int position = 0;
    bool algorithm_closed = closed;
    int initialization_iterations = 3;
    Slice slice{0, 0};
    Slice right_slice{0, 0};
    Point_<T> start_point(static_cast<T>(-1000000), static_cast<T>(-1000000));
    Point_<T> end_point(0, 0);
    Point_<T> point(0, 0);
    bool within_epsilon = false;
    const double squared_epsilon = epsilon * epsilon;

    auto read_source = [&](Point_<T>& value, int& index) {
        value = source[static_cast<size_t>(index)];
        if (++index >= count) index = 0;
    };

    if (!algorithm_closed)
    {
        right_slice.start = count;
        end_point = source.front();
        start_point = source.back();
        if (start_point != end_point)
        {
            stack.push_back(Slice{0, count - 1});
        }
        else
        {
            algorithm_closed = true;
            initialization_iterations = 1;
        }
    }

    if (algorithm_closed)
    {
        right_slice.start = 0;
        for (int iteration = 0; iteration < initialization_iterations; ++iteration)
        {
            double maximum_distance = 0.0;
            position = (position + right_slice.start) % count;
            read_source(start_point, position);
            for (int index = 1; index < count; ++index)
            {
                read_source(point, position);
                const double dx = static_cast<double>(point.x) - start_point.x;
                const double dy = static_cast<double>(point.y) - start_point.y;
                const double distance = dx * dx + dy * dy;
                if (distance > maximum_distance)
                {
                    maximum_distance = distance;
                    right_slice.start = index;
                }
            }
            within_epsilon = maximum_distance <= squared_epsilon;
        }
        if (!within_epsilon)
        {
            right_slice.end = slice.start = position % count;
            slice.end = right_slice.start = (right_slice.start + slice.start) % count;
            stack.push_back(right_slice);
            stack.push_back(slice);
        }
        else
        {
            output[static_cast<size_t>(new_count++)] = start_point;
        }
    }

    while (!stack.empty())
    {
        slice = stack.back();
        stack.pop_back();
        end_point = source[static_cast<size_t>(slice.end)];
        position = slice.start;
        read_source(start_point, position);
        if (position != slice.end)
        {
            const double dx = static_cast<double>(end_point.x) - start_point.x;
            const double dy = static_cast<double>(end_point.y) - start_point.y;
            const double segment_length_squared = dx * dx + dy * dy;
            double maximum_scaled_distance = 0.0;
            while (position != slice.end)
            {
                read_source(point, position);
                const double px = static_cast<double>(point.x) - start_point.x;
                const double py = static_cast<double>(point.y) - start_point.y;
                const double projection = px * dx + py * dy;
                double distance;
                if (projection < 0)
                {
                    distance = (px * px + py * py) * segment_length_squared;
                }
                else if (projection > segment_length_squared)
                {
                    const double ex = static_cast<double>(point.x) - end_point.x;
                    const double ey = static_cast<double>(point.y) - end_point.y;
                    distance = (ex * ex + ey * ey) * segment_length_squared;
                }
                else
                {
                    const double cross = py * dx - px * dy;
                    distance = cross * cross;
                }
                if (distance > maximum_scaled_distance)
                {
                    maximum_scaled_distance = distance;
                    right_slice.start = (position + count - 1) % count;
                }
            }
            within_epsilon = maximum_scaled_distance <= squared_epsilon * segment_length_squared;
        }
        else
        {
            within_epsilon = true;
            start_point = source[static_cast<size_t>(slice.start)];
        }

        if (within_epsilon)
        {
            output[static_cast<size_t>(new_count++)] = start_point;
        }
        else
        {
            right_slice.end = slice.end;
            slice.end = right_slice.start;
            stack.push_back(right_slice);
            stack.push_back(slice);
        }
    }
    if (!algorithm_closed)
    {
        output[static_cast<size_t>(new_count++)] = source.back();
    }

    const bool cleanup_closed = closed;
    int cleanup_count = new_count;
    position = cleanup_closed ? cleanup_count - 1 : 0;
    auto read_output = [&](Point_<T>& value, int& index) {
        value = output[static_cast<size_t>(index)];
        if (++index >= cleanup_count) index = 0;
    };
    read_output(start_point, position);
    int write_position = position;
    read_output(point, position);
    for (int index = !cleanup_closed;
         index < cleanup_count - !cleanup_closed && new_count > 2; ++index)
    {
        read_output(end_point, position);
        const double dx = static_cast<double>(end_point.x) - start_point.x;
        const double dy = static_cast<double>(end_point.y) - start_point.y;
        const double distance = std::fabs(
            (static_cast<double>(point.x) - start_point.x) * dy -
            (static_cast<double>(point.y) - start_point.y) * dx);
        const double inner =
            (static_cast<double>(point.x) - start_point.x) *
                (static_cast<double>(end_point.x) - point.x) +
            (static_cast<double>(point.y) - start_point.y) *
                (static_cast<double>(end_point.y) - point.y);
        if (distance * distance <= 0.5 * squared_epsilon * (dx * dx + dy * dy) &&
            dx != 0 && dy != 0 && inner >= 0)
        {
            --new_count;
            output[static_cast<size_t>(write_position)] = start_point = end_point;
            if (++write_position >= cleanup_count) write_position = 0;
            read_output(point, position);
            ++index;
            continue;
        }
        output[static_cast<size_t>(write_position)] = start_point = point;
        if (++write_position >= cleanup_count) write_position = 0;
        point = end_point;
    }
    if (!cleanup_closed)
    {
        output[static_cast<size_t>(write_position)] = point;
    }
    destination.assign(output.begin(), output.begin() + new_count);
}

template<typename T>
inline double hull_cross(const Point_<T>& origin, const Point_<T>& lhs,
                         const Point_<T>& rhs)
{
    return (static_cast<double>(lhs.x) - origin.x) *
               (static_cast<double>(rhs.y) - origin.y) -
           (static_cast<double>(lhs.y) - origin.y) *
               (static_cast<double>(rhs.x) - origin.x);
}

template<typename T>
inline void convex_hull_points(const std::vector<Point_<T>>& input,
                               std::vector<Point_<T>>& hull, bool clockwise)
{
    validate_point_values(input);
    hull.clear();
    if (input.empty())
    {
        return;
    }
    std::vector<Point_<T>> points = input;
    std::sort(points.begin(), points.end(), [](const Point_<T>& lhs, const Point_<T>& rhs) {
        return lhs.x < rhs.x || (lhs.x == rhs.x && lhs.y < rhs.y);
    });
    points.erase(std::unique(points.begin(), points.end()), points.end());
    if (points.size() <= 2)
    {
        hull = points;
        if (clockwise) std::reverse(hull.begin(), hull.end());
        return;
    }

    std::vector<Point_<T>> result(points.size() * 2);
    size_t count = 0;
    for (const Point_<T>& point : points)
    {
        while (count >= 2 && hull_cross(result[count - 2], result[count - 1], point) <= 0.0)
        {
            --count;
        }
        result[count++] = point;
    }
    const size_t lower_count = count;
    for (size_t index = points.size() - 1; index-- > 0;)
    {
        const Point_<T>& point = points[index];
        while (count > lower_count &&
               hull_cross(result[count - 2], result[count - 1], point) <= 0.0)
        {
            --count;
        }
        result[count++] = point;
    }
    if (count > 1) --count;
    hull.assign(result.begin(), result.begin() + static_cast<std::ptrdiff_t>(count));
    if (clockwise)
    {
        std::reverse(hull.begin() + 1, hull.end());
    }
}

template<typename T>
inline bool contour_convex_points(const std::vector<Point_<T>>& contour)
{
    validate_point_values(contour);
    const int count = static_cast<int>(contour.size());
    if (count == 0)
    {
        return false;
    }
    Point_<T> previous = contour[static_cast<size_t>((count - 2 + count) % count)];
    Point_<T> current = contour[static_cast<size_t>(count - 1)];
    double dx0 = static_cast<double>(current.x) - previous.x;
    double dy0 = static_cast<double>(current.y) - previous.y;
    int orientation = 0;
    for (int index = 0; index < count; ++index)
    {
        previous = current;
        current = contour[static_cast<size_t>(index)];
        const double dx = static_cast<double>(current.x) - previous.x;
        const double dy = static_cast<double>(current.y) - previous.y;
        const double left = dx * dy0;
        const double right = dy * dx0;
        orientation |= right > left ? 1 : (right < left ? 2 : 3);
        if (orientation == 3)
        {
            return false;
        }
        dx0 = dx;
        dy0 = dy;
    }
    return true;
}

inline void complete_moments(Moments& moments)
{
    double center_x = 0.0;
    double center_y = 0.0;
    double inverse_m00 = 0.0;
    if (std::fabs(moments.m00) > DBL_EPSILON)
    {
        inverse_m00 = 1.0 / moments.m00;
        center_x = moments.m10 * inverse_m00;
        center_y = moments.m01 * inverse_m00;
    }
    moments.mu20 = moments.m20 - moments.m10 * center_x;
    moments.mu11 = moments.m11 - moments.m10 * center_y;
    moments.mu02 = moments.m02 - moments.m01 * center_y;
    moments.mu30 = moments.m30 - center_x * (3.0 * moments.mu20 + center_x * moments.m10);
    const double twice_mu11 = 2.0 * moments.mu11;
    moments.mu21 = moments.m21 - center_x * (twice_mu11 + center_x * moments.m01) - center_y * moments.mu20;
    moments.mu12 = moments.m12 - center_y * (twice_mu11 + center_y * moments.m10) - center_x * moments.mu02;
    moments.mu03 = moments.m03 - center_y * (3.0 * moments.mu02 + center_y * moments.m01);
    const double inverse_sqrt_m00 = std::sqrt(std::fabs(inverse_m00));
    const double scale2 = inverse_m00 * inverse_m00;
    const double scale3 = scale2 * inverse_sqrt_m00;
    moments.nu20 = moments.mu20 * scale2;
    moments.nu11 = moments.mu11 * scale2;
    moments.nu02 = moments.mu02 * scale2;
    moments.nu30 = moments.mu30 * scale3;
    moments.nu21 = moments.mu21 * scale3;
    moments.nu12 = moments.mu12 * scale3;
    moments.nu03 = moments.mu03 * scale3;
}

template<typename T>
inline Moments contour_moments(const std::vector<Point_<T>>& contour)
{
    validate_point_values(contour);
    Moments moments;
    if (contour.empty())
    {
        return moments;
    }
    double a00 = 0, a10 = 0, a01 = 0, a20 = 0, a11 = 0, a02 = 0;
    double a30 = 0, a21 = 0, a12 = 0, a03 = 0;
    double previous_x = contour.back().x;
    double previous_y = contour.back().y;
    double previous_x2 = previous_x * previous_x;
    double previous_y2 = previous_y * previous_y;
    for (const Point_<T>& point : contour)
    {
        const double x = point.x;
        const double y = point.y;
        const double x2 = x * x;
        const double y2 = y * y;
        const double cross = previous_x * y - x * previous_y;
        const double sum_x = previous_x + x;
        const double sum_y = previous_y + y;
        a00 += cross;
        a10 += cross * sum_x;
        a01 += cross * sum_y;
        a20 += cross * (previous_x * sum_x + x2);
        a11 += cross * (previous_x * (sum_y + previous_y) + x * (sum_y + y));
        a02 += cross * (previous_y * sum_y + y2);
        a30 += cross * sum_x * (previous_x2 + x2);
        a03 += cross * sum_y * (previous_y2 + y2);
        a21 += cross * (previous_x2 * (3.0 * previous_y + y) +
                         2.0 * x * previous_x * sum_y + x2 * (previous_y + 3.0 * y));
        a12 += cross * (previous_y2 * (3.0 * previous_x + x) +
                         2.0 * y * previous_y * sum_x + y2 * (previous_x + 3.0 * x));
        previous_x = x;
        previous_y = y;
        previous_x2 = x2;
        previous_y2 = y2;
    }
    if (std::fabs(a00) > FLT_EPSILON)
    {
        const double sign = a00 > 0.0 ? 1.0 : -1.0;
        moments.m00 = a00 * sign * 0.5;
        moments.m10 = a10 * sign / 6.0;
        moments.m01 = a01 * sign / 6.0;
        moments.m20 = a20 * sign / 12.0;
        moments.m11 = a11 * sign / 24.0;
        moments.m02 = a02 * sign / 12.0;
        moments.m30 = a30 * sign / 20.0;
        moments.m21 = a21 * sign / 60.0;
        moments.m12 = a12 * sign / 60.0;
        moments.m03 = a03 * sign / 20.0;
        complete_moments(moments);
    }
    return moments;
}

}  // namespace detail

inline Rect boundingRect(const std::vector<Point>& points) { return detail::bounding_rect_points(points); }
inline Rect boundingRect(const std::vector<Point2f>& points) { return detail::bounding_rect_points(points); }
inline double contourArea(const std::vector<Point>& contour, bool oriented) { return detail::contour_area_points(contour, oriented); }
inline double contourArea(const std::vector<Point2f>& contour, bool oriented) { return detail::contour_area_points(contour, oriented); }
inline double arcLength(const std::vector<Point>& curve, bool closed) { return detail::arc_length_points(curve, closed); }
inline double arcLength(const std::vector<Point2f>& curve, bool closed) { return detail::arc_length_points(curve, closed); }
inline void approxPolyDP(const std::vector<Point>& curve, std::vector<Point>& approximate, double epsilon, bool closed) { detail::approx_poly_dp_points(curve, approximate, epsilon, closed); }
inline void approxPolyDP(const std::vector<Point2f>& curve, std::vector<Point2f>& approximate, double epsilon, bool closed) { detail::approx_poly_dp_points(curve, approximate, epsilon, closed); }
inline void convexHull(const std::vector<Point>& points, std::vector<Point>& hull, bool clockwise) { detail::convex_hull_points(points, hull, clockwise); }
inline void convexHull(const std::vector<Point2f>& points, std::vector<Point2f>& hull, bool clockwise) { detail::convex_hull_points(points, hull, clockwise); }
inline bool isContourConvex(const std::vector<Point>& contour) { return detail::contour_convex_points(contour); }
inline bool isContourConvex(const std::vector<Point2f>& contour) { return detail::contour_convex_points(contour); }
inline Moments moments(const std::vector<Point>& contour) { return detail::contour_moments(contour); }
inline Moments moments(const std::vector<Point2f>& contour) { return detail::contour_moments(contour); }

inline Moments moments(const Mat& image, bool binaryImage)
{
    if (image.empty() || image.dims != 2 || image.type() != CV_8UC1)
    {
        CV_Error(Error::StsUnsupportedFormat, "moments image input supports non-empty CV_8UC1 only");
    }
    Moments result;
    for (int row = 0; row < image.size[0]; ++row)
    {
        for (int column = 0; column < image.size[1]; ++column)
        {
            const uchar pixel = image.at<uchar>(row, column);
            const double value = binaryImage ? (pixel != 0 ? 1.0 : 0.0) : pixel;
            const double x = column;
            const double y = row;
            const double x2 = x * x;
            const double y2 = y * y;
            result.m00 += value;
            result.m10 += value * x;
            result.m01 += value * y;
            result.m20 += value * x2;
            result.m11 += value * x * y;
            result.m02 += value * y2;
            result.m30 += value * x2 * x;
            result.m21 += value * x2 * y;
            result.m12 += value * x * y2;
            result.m03 += value * y2 * y;
        }
    }
    detail::complete_moments(result);
    return result;
}

}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_SHAPE_IMPL_HPP
