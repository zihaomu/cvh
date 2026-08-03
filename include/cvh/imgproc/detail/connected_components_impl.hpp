#ifndef CVH_IMGPROC_DETAIL_CONNECTED_COMPONENTS_IMPL_HPP
#define CVH_IMGPROC_DETAIL_CONNECTED_COMPONENTS_IMPL_HPP

#include "../connected_components.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace cvh
{
namespace detail
{

class LabelUnionFind
{
public:
    LabelUnionFind() : parent_(1, 0) {}

    int make_set()
    {
        const int label = static_cast<int>(parent_.size());
        parent_.push_back(label);
        return label;
    }

    int find(int label)
    {
        int root = label;
        while (parent_[static_cast<size_t>(root)] != root)
        {
            root = parent_[static_cast<size_t>(root)];
        }
        while (parent_[static_cast<size_t>(label)] != label)
        {
            const int next = parent_[static_cast<size_t>(label)];
            parent_[static_cast<size_t>(label)] = root;
            label = next;
        }
        return root;
    }

    void unite(int lhs, int rhs)
    {
        int left_root = find(lhs);
        int right_root = find(rhs);
        if (left_root == right_root)
        {
            return;
        }
        const int root = std::min(left_root, right_root);
        parent_[static_cast<size_t>(left_root)] = root;
        parent_[static_cast<size_t>(right_root)] = root;
    }

private:
    std::vector<int> parent_;
};

inline void validate_connected_components_input(const Mat& image, int connectivity, int ltype)
{
    if (image.empty() || image.dims != 2 || image.type() != CV_8UC1)
    {
        CV_Error(Error::StsBadArg, "connectedComponents expects non-empty CV_8UC1 image");
    }
    if (connectivity != 4 && connectivity != 8)
    {
        CV_Error(Error::StsBadFlag, "connectedComponents connectivity must be 4 or 8");
    }
    if (ltype != CV_32S)
    {
        CV_Error(Error::StsUnsupportedFormat, "connectedComponents P2-P0 supports CV_32S labels only");
    }
}

inline int connected_components_kernel(const Mat& image, Mat& labels, int connectivity)
{
    const int rows = image.size[0];
    const int columns = image.size[1];
    labels.create({rows, columns}, CV_32SC1);
    labels = 0;

    LabelUnionFind sets;
    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0; column < columns; ++column)
        {
            if (image.at<uchar>(row, column) == 0)
            {
                continue;
            }

            int neighbors[4];
            int neighbor_count = 0;
            if (column > 0)
            {
                const int west = labels.at<int>(row, column - 1);
                if (west != 0) neighbors[neighbor_count++] = west;
            }
            if (row > 0)
            {
                if (connectivity == 8 && column > 0)
                {
                    const int north_west = labels.at<int>(row - 1, column - 1);
                    if (north_west != 0) neighbors[neighbor_count++] = north_west;
                }
                const int north = labels.at<int>(row - 1, column);
                if (north != 0) neighbors[neighbor_count++] = north;
                if (connectivity == 8 && column + 1 < columns)
                {
                    const int north_east = labels.at<int>(row - 1, column + 1);
                    if (north_east != 0) neighbors[neighbor_count++] = north_east;
                }
            }

            if (neighbor_count == 0)
            {
                labels.at<int>(row, column) = sets.make_set();
                continue;
            }

            int label = neighbors[0];
            for (int index = 1; index < neighbor_count; ++index)
            {
                label = std::min(label, neighbors[index]);
            }
            labels.at<int>(row, column) = label;
            for (int index = 0; index < neighbor_count; ++index)
            {
                sets.unite(label, neighbors[index]);
            }
        }
    }

    std::unordered_map<int, int> canonical_labels;
    int next_label = 1;
    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0; column < columns; ++column)
        {
            int& label = labels.at<int>(row, column);
            if (label == 0)
            {
                continue;
            }
            const int root = sets.find(label);
            const auto result = canonical_labels.emplace(root, next_label);
            if (result.second)
            {
                ++next_label;
            }
            label = result.first->second;
        }
    }
    return next_label;
}

inline void connected_component_statistics(const Mat& labels, int count,
                                           Mat& stats, Mat& centroids)
{
    stats.create({count, CC_STAT_MAX}, CV_32SC1);
    centroids.create({count, 2}, CV_64FC1);
    std::vector<std::uint64_t> sum_x(static_cast<size_t>(count), 0);
    std::vector<std::uint64_t> sum_y(static_cast<size_t>(count), 0);

    for (int label = 0; label < count; ++label)
    {
        stats.at<int>(label, CC_STAT_LEFT) = INT_MAX;
        stats.at<int>(label, CC_STAT_TOP) = INT_MAX;
        stats.at<int>(label, CC_STAT_WIDTH) = INT_MIN;
        stats.at<int>(label, CC_STAT_HEIGHT) = INT_MIN;
        stats.at<int>(label, CC_STAT_AREA) = 0;
    }

    for (int row = 0; row < labels.size[0]; ++row)
    {
        for (int column = 0; column < labels.size[1]; ++column)
        {
            const int label = labels.at<int>(row, column);
            stats.at<int>(label, CC_STAT_LEFT) =
                std::min(stats.at<int>(label, CC_STAT_LEFT), column);
            stats.at<int>(label, CC_STAT_TOP) =
                std::min(stats.at<int>(label, CC_STAT_TOP), row);
            stats.at<int>(label, CC_STAT_WIDTH) =
                std::max(stats.at<int>(label, CC_STAT_WIDTH), column);
            stats.at<int>(label, CC_STAT_HEIGHT) =
                std::max(stats.at<int>(label, CC_STAT_HEIGHT), row);
            ++stats.at<int>(label, CC_STAT_AREA);
            sum_x[static_cast<size_t>(label)] += static_cast<std::uint64_t>(column);
            sum_y[static_cast<size_t>(label)] += static_cast<std::uint64_t>(row);
        }
    }

    for (int label = 0; label < count; ++label)
    {
        const int area = stats.at<int>(label, CC_STAT_AREA);
        if (area > 0)
        {
            stats.at<int>(label, CC_STAT_WIDTH) =
                stats.at<int>(label, CC_STAT_WIDTH) - stats.at<int>(label, CC_STAT_LEFT) + 1;
            stats.at<int>(label, CC_STAT_HEIGHT) =
                stats.at<int>(label, CC_STAT_HEIGHT) - stats.at<int>(label, CC_STAT_TOP) + 1;
            centroids.at<double>(label, 0) =
                static_cast<double>(sum_x[static_cast<size_t>(label)]) / area;
            centroids.at<double>(label, 1) =
                static_cast<double>(sum_y[static_cast<size_t>(label)]) / area;
        }
        else
        {
            stats.at<int>(label, CC_STAT_LEFT) = -1;
            stats.at<int>(label, CC_STAT_WIDTH) = 0;
            stats.at<int>(label, CC_STAT_HEIGHT) = 0;
            centroids.at<double>(label, 0) = std::numeric_limits<double>::quiet_NaN();
            centroids.at<double>(label, 1) = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

}  // namespace detail

inline int connectedComponents(const Mat& image, Mat& labels, int connectivity, int ltype)
{
    detail::validate_connected_components_input(image, connectivity, ltype);
    return detail::connected_components_kernel(image, labels, connectivity);
}

inline int connectedComponentsWithStats(const Mat& image, Mat& labels, Mat& stats,
                                        Mat& centroids, int connectivity, int ltype)
{
    detail::validate_connected_components_input(image, connectivity, ltype);
    const int count = detail::connected_components_kernel(image, labels, connectivity);
    detail::connected_component_statistics(labels, count, stats, centroids);
    return count;
}

}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_CONNECTED_COMPONENTS_IMPL_HPP
