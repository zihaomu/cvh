#ifndef CVH_IMGPROC_DETAIL_CONNECTED_COMPONENTS_IMPL_HPP
#define CVH_IMGPROC_DETAIL_CONNECTED_COMPONENTS_IMPL_HPP

#include "../connected_components.h"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace cvh
{
namespace detail
{

class LabelUnionFind
{
public:
    explicit LabelUnionFind(std::size_t maximum_labels) : parent_(1, 0)
    {
        parent_.reserve(maximum_labels + 1);
    }

    int make_set()
    {
        const int label = static_cast<int>(parent_.size());
        parent_.push_back(label);
        return label;
    }

    int find(int label)
    {
        int root = label;
        while (parent_[static_cast<std::size_t>(root)] != root)
        {
            root = parent_[static_cast<std::size_t>(root)];
        }
        while (parent_[static_cast<std::size_t>(label)] != label)
        {
            const int next = parent_[static_cast<std::size_t>(label)];
            parent_[static_cast<std::size_t>(label)] = root;
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
        parent_[static_cast<std::size_t>(left_root)] = root;
        parent_[static_cast<std::size_t>(right_root)] = root;
    }

    std::size_t size() const
    {
        return parent_.size();
    }

private:
    std::vector<int> parent_;
};

struct ComponentStatisticsWorkspace
{
    std::vector<int> left;
    std::vector<int> top;
    std::vector<int> right;
    std::vector<int> bottom;
    std::vector<int> area;
    std::vector<std::uint64_t> sum_x;
    std::vector<std::uint64_t> sum_y;

    void create(std::size_t maximum_labels)
    {
        left.assign(maximum_labels, INT_MAX);
        top.assign(maximum_labels, INT_MAX);
        right.assign(maximum_labels, INT_MIN);
        bottom.assign(maximum_labels, INT_MIN);
        area.assign(maximum_labels, 0);
        sum_x.assign(maximum_labels, 0);
        sum_y.assign(maximum_labels, 0);
    }

    void add(int label, int row, int column)
    {
        const std::size_t index = static_cast<std::size_t>(label);
        left[index] = std::min(left[index], column);
        top[index] = std::min(top[index], row);
        right[index] = std::max(right[index], column);
        bottom[index] = std::max(bottom[index], row);
        ++area[index];
        sum_x[index] += static_cast<std::uint64_t>(column);
        sum_y[index] += static_cast<std::uint64_t>(row);
    }

    void write(int count, Mat& stats, Mat& centroids) const
    {
        stats.create({count, CC_STAT_MAX}, CV_32SC1);
        centroids.create({count, 2}, CV_64FC1);
        for (int label = 0; label < count; ++label)
        {
            int* stats_row = reinterpret_cast<int*>(
                stats.data + static_cast<std::size_t>(label) * stats.step(0));
            double* centroid_row = reinterpret_cast<double*>(
                centroids.data +
                static_cast<std::size_t>(label) * centroids.step(0));
            const std::size_t index = static_cast<std::size_t>(label);
            stats_row[CC_STAT_AREA] = area[index];
            if (area[index] > 0)
            {
                stats_row[CC_STAT_LEFT] = left[index];
                stats_row[CC_STAT_TOP] = top[index];
                stats_row[CC_STAT_WIDTH] = right[index] - left[index] + 1;
                stats_row[CC_STAT_HEIGHT] = bottom[index] - top[index] + 1;
                centroid_row[0] =
                    static_cast<double>(sum_x[index]) / area[index];
                centroid_row[1] =
                    static_cast<double>(sum_y[index]) / area[index];
            }
            else
            {
                stats_row[CC_STAT_LEFT] = -1;
                stats_row[CC_STAT_TOP] = INT_MAX;
                stats_row[CC_STAT_WIDTH] = 0;
                stats_row[CC_STAT_HEIGHT] = 0;
                centroid_row[0] = std::numeric_limits<double>::quiet_NaN();
                centroid_row[1] = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
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

inline int connected_components_kernel(const Mat& image,
                                       Mat& labels,
                                       int connectivity,
                                       ComponentStatisticsWorkspace* statistics)
{
    const int rows = image.size[0];
    const int columns = image.size[1];
    labels.create({rows, columns}, CV_32SC1);

    const std::size_t maximum_labels =
        (static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns) + 1) /
        2 + 1;
    LabelUnionFind sets(maximum_labels);
    for (int row = 0; row < rows; ++row)
    {
        const uchar* image_row = image.data +
            static_cast<std::size_t>(row) * image.step(0);
        int* label_row = reinterpret_cast<int*>(
            labels.data + static_cast<std::size_t>(row) * labels.step(0));
        const int* north_row = row > 0
            ? reinterpret_cast<const int*>(
                  labels.data + static_cast<std::size_t>(row - 1) * labels.step(0))
            : nullptr;
        for (int column = 0; column < columns; ++column)
        {
            if (image_row[column] == 0)
            {
                label_row[column] = 0;
                continue;
            }

            int neighbors[4];
            int neighbor_count = 0;
            if (column > 0 && label_row[column - 1] != 0)
            {
                neighbors[neighbor_count++] = label_row[column - 1];
            }
            if (north_row != nullptr)
            {
                if (connectivity == 8 && column > 0 &&
                    north_row[column - 1] != 0)
                {
                    neighbors[neighbor_count++] = north_row[column - 1];
                }
                if (north_row[column] != 0)
                {
                    neighbors[neighbor_count++] = north_row[column];
                }
                if (connectivity == 8 && column + 1 < columns &&
                    north_row[column + 1] != 0)
                {
                    neighbors[neighbor_count++] = north_row[column + 1];
                }
            }

            if (neighbor_count == 0)
            {
                label_row[column] = sets.make_set();
                continue;
            }

            int label = neighbors[0];
            for (int index = 1; index < neighbor_count; ++index)
            {
                label = std::min(label, neighbors[index]);
            }
            label_row[column] = label;
            for (int index = 0; index < neighbor_count; ++index)
            {
                if (neighbors[index] != label)
                {
                    sets.unite(label, neighbors[index]);
                }
            }
        }
    }

    std::vector<int> canonical_labels(sets.size(), 0);
    if (statistics != nullptr)
    {
        statistics->create(sets.size());
    }
    int next_label = 1;
    for (int row = 0; row < rows; ++row)
    {
        int* label_row = reinterpret_cast<int*>(
            labels.data + static_cast<std::size_t>(row) * labels.step(0));
        for (int column = 0; column < columns; ++column)
        {
            int label = label_row[column];
            if (label != 0)
            {
                const int root = sets.find(label);
                int& canonical =
                    canonical_labels[static_cast<std::size_t>(root)];
                if (canonical == 0)
                {
                    canonical = next_label++;
                }
                label = canonical;
                label_row[column] = label;
            }
            if (statistics != nullptr)
            {
                statistics->add(label, row, column);
            }
        }
    }
    return next_label;
}

}  // namespace detail

inline int connectedComponents(const Mat& image, Mat& labels, int connectivity, int ltype)
{
    detail::validate_connected_components_input(image, connectivity, ltype);
    return detail::connected_components_kernel(
        image, labels, connectivity, nullptr);
}

inline int connectedComponentsWithStats(const Mat& image, Mat& labels, Mat& stats,
                                        Mat& centroids, int connectivity, int ltype)
{
    detail::validate_connected_components_input(image, connectivity, ltype);
    detail::ComponentStatisticsWorkspace statistics;
    const int count = detail::connected_components_kernel(
        image, labels, connectivity, &statistics);
    statistics.write(count, stats, centroids);
    return count;
}

}  // namespace cvh

#endif  // CVH_IMGPROC_DETAIL_CONNECTED_COMPONENTS_IMPL_HPP
