#ifndef CVH_IMGPROC_SHAPE_H
#define CVH_IMGPROC_SHAPE_H

#include "../core/mat.h"

#include <vector>

namespace cvh
{

Rect boundingRect(const std::vector<Point>& points);
Rect boundingRect(const std::vector<Point2f>& points);

double contourArea(const std::vector<Point>& contour, bool oriented = false);
double contourArea(const std::vector<Point2f>& contour, bool oriented = false);

double arcLength(const std::vector<Point>& curve, bool closed);
double arcLength(const std::vector<Point2f>& curve, bool closed);

void approxPolyDP(const std::vector<Point>& curve, std::vector<Point>& approximate,
                  double epsilon, bool closed);
void approxPolyDP(const std::vector<Point2f>& curve, std::vector<Point2f>& approximate,
                  double epsilon, bool closed);

void convexHull(const std::vector<Point>& points, std::vector<Point>& hull,
                bool clockwise = false);
void convexHull(const std::vector<Point2f>& points, std::vector<Point2f>& hull,
                bool clockwise = false);

bool isContourConvex(const std::vector<Point>& contour);
bool isContourConvex(const std::vector<Point2f>& contour);

Moments moments(const std::vector<Point>& contour);
Moments moments(const std::vector<Point2f>& contour);
Moments moments(const Mat& image, bool binaryImage = false);

}  // namespace cvh

#include "detail/shape_impl.hpp"

#endif  // CVH_IMGPROC_SHAPE_H
