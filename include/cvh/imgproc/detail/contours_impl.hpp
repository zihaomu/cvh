#ifndef CVH_IMGPROC_DETAIL_CONTOURS_IMPL_HPP
#define CVH_IMGPROC_DETAIL_CONTOURS_IMPL_HPP

#include "../contours.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <deque>
#include <vector>

namespace cvh
{
namespace detail
{

struct ContourWorkspace
{
    int rows;
    int columns;
    std::vector<int> pixels;

    int& at(int row, int column)
    {
        return pixels[static_cast<size_t>(row) * columns + column];
    }

    int at(int row, int column) const
    {
        return pixels[static_cast<size_t>(row) * columns + column];
    }
};

inline std::vector<uchar> exterior_background(const ContourWorkspace& workspace)
{
    std::vector<uchar> exterior(workspace.pixels.size(), 0);
    std::deque<Point> queue;
    exterior[0] = 1;
    queue.emplace_back(0, 0);
    const int dx[4] = {1, 0, -1, 0};
    const int dy[4] = {0, 1, 0, -1};
    while (!queue.empty())
    {
        const Point point = queue.front();
        queue.pop_front();
        for (int direction = 0; direction < 4; ++direction)
        {
            const int x = point.x + dx[direction];
            const int y = point.y + dy[direction];
            if (x < 0 || y < 0 || x >= workspace.columns || y >= workspace.rows)
            {
                continue;
            }
            const size_t index = static_cast<size_t>(y) * workspace.columns + x;
            if (exterior[index] == 0 && workspace.at(y, x) == 0)
            {
                exterior[index] = 1;
                queue.emplace_back(x, y);
            }
        }
    }
    return exterior;
}

inline std::vector<Point> fetch_contour(ContourWorkspace& workspace,
                                        int start_x, int start_y,
                                        bool hole, int method, int label,
                                        Point coordinate_offset)
{
    static constexpr int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    static constexpr int dy[8] = {0, -1, -1, -1, 0, 1, 1, 1};
    int direction_end = hole ? 0 : 4;
    int direction = direction_end;
    int first_x = start_x;
    int first_y = start_y;
    do
    {
        direction = (direction - 1) & 7;
    }
    while (workspace.at(start_y + dy[direction], start_x + dx[direction]) == 0 &&
           direction != direction_end);

    std::vector<Point> contour;
    Point point(start_x - 1 + coordinate_offset.x,
                start_y - 1 + coordinate_offset.y);
    if (direction == direction_end)
    {
        workspace.at(start_y, start_x) = -label;
        contour.push_back(point);
        return contour;
    }

    const int first_neighbor_x = start_x + dx[direction];
    const int first_neighbor_y = start_y + dy[direction];
    int current_x = start_x;
    int current_y = start_y;
    int previous_direction = direction ^ 4;
    for (;;)
    {
        direction_end = direction;
        int search_direction = direction;
        int next_x = current_x;
        int next_y = current_y;
        do
        {
            ++search_direction;
            direction = search_direction & 7;
            next_x = current_x + dx[direction];
            next_y = current_y + dy[direction];
        }
        while (workspace.at(next_y, next_x) == 0 && search_direction < direction_end + 8);

        if (direction != 0 && direction - 1 < direction_end)
        {
            workspace.at(current_y, current_x) = -label;
        }
        else if (workspace.at(current_y, current_x) == 1)
        {
            workspace.at(current_y, current_x) = label;
        }

        if (method == CHAIN_APPROX_NONE || direction != previous_direction)
        {
            contour.push_back(point);
            previous_direction = direction;
        }
        point.x += dx[direction];
        point.y += dy[direction];

        if (next_x == first_x && next_y == first_y &&
            current_x == first_neighbor_x && current_y == first_neighbor_y)
        {
            break;
        }
        current_x = next_x;
        current_y = next_y;
        direction = (direction + 4) & 7;
    }
    return contour;
}

}  // namespace detail

inline void findContours(const Mat& image, std::vector<std::vector<Point>>& contours,
                         int mode, int method, Point offset)
{
    if (image.empty() || image.dims != 2 || image.type() != CV_8UC1)
    {
        CV_Error(Error::StsUnsupportedFormat, "findContours expects non-empty CV_8UC1 image");
    }
    if (mode != RETR_EXTERNAL && mode != RETR_LIST)
    {
        CV_Error(Error::StsBadFlag, "findContours P2-P0 supports RETR_EXTERNAL and RETR_LIST");
    }
    if (method != CHAIN_APPROX_NONE && method != CHAIN_APPROX_SIMPLE)
    {
        CV_Error(Error::StsBadFlag, "findContours P2-P0 supports CHAIN_APPROX_NONE and CHAIN_APPROX_SIMPLE");
    }

    detail::ContourWorkspace workspace;
    workspace.rows = image.size[0] + 2;
    workspace.columns = image.size[1] + 2;
    workspace.pixels.assign(static_cast<size_t>(workspace.rows) * workspace.columns, 0);
    for (int row = 0; row < image.size[0]; ++row)
    {
        for (int column = 0; column < image.size[1]; ++column)
        {
            workspace.at(row + 1, column + 1) = image.at<uchar>(row, column) != 0 ? 1 : 0;
        }
    }
    const std::vector<uchar> exterior = detail::exterior_background(workspace);
    contours.clear();
    int label = 2;
    for (int row = 1; row <= image.size[0]; ++row)
    {
        int previous = 0;
        for (int column = 1; column <= image.size[1] + 1; ++column)
        {
            int pixel = workspace.at(row, column);
            bool hole = false;
            int origin_x = column;
            bool found = previous == 0 && pixel == 1;
            if (!found && pixel == 0 && previous >= 1)
            {
                found = true;
                hole = true;
                origin_x = column - 1;
            }
            if (found)
            {
                const bool is_external = !hole &&
                    exterior[static_cast<size_t>(row) * workspace.columns + origin_x - 1] != 0;
                std::vector<Point> contour = detail::fetch_contour(
                    workspace, origin_x, row, hole, method, label, offset);
                if (mode == RETR_LIST || is_external)
                {
                    contours.push_back(std::move(contour));
                }
                label = label == 127 ? 3 : label + 1;
                pixel = workspace.at(row, column);
            }
            previous = pixel;
        }
    }
    // OpenCV's contour tree inserts newly discovered siblings at the front;
    // RETR_LIST/RETR_EXTERNAL therefore expose reverse raster-discovery order.
    std::reverse(contours.begin(), contours.end());
}

}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_CONTOURS_IMPL_HPP
