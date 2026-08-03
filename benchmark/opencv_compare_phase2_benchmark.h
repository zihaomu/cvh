#ifndef CVH_BENCHMARK_OPENCV_COMPARE_PHASE2_BENCHMARK_H
#define CVH_BENCHMARK_OPENCV_COMPARE_PHASE2_BENCHMARK_H

#include <cstdint>
#include <string>
#include <vector>

namespace cvh_bench_compare {

enum class Phase2OpId
{
    RanduU8C3 = 0,
    RandnU8C3,
    RanduF32C3,
    RandnF32C3,
    RanduU8C1Roi,
    TransformF32C3ToC4,
    PerspectiveTransformF32C3,
    ConnectedComponents,
    ConnectedComponentsWithStats,
    FindContours,
    BoundingRect,
    ContourArea,
    ArcLength,
    ApproxPolyDP,
    ConvexHull,
    IsContourConvex,
    Moments,
    CalcHist,
    CompareHistCorrel,
    CompareHistChiSqr,
    CompareHistIntersect,
    CompareHistBhattacharyya,
    MatchTemplateSqDiff,
    MatchTemplateSqDiffNormed,
    MatchTemplateCCorr,
    MatchTemplateCCorrNormed,
};

struct Phase2BenchmarkConfig
{
    std::string profile;
    int warmup = 1;
    int iters = 1;
    int repeats = 1;
};

struct Phase2BenchmarkResult
{
    std::string suite;
    std::string op;
    std::string variant;
    std::string dispatch_path;
    std::string depth;
    int channels = 1;
    std::string layout = "continuous";
    std::string shape;
    double cvh_ms = 0.0;
    double opencv_ms = 0.0;
    std::string note;
};

double bench_opencv_phase2(Phase2OpId op,
                           int rows,
                           int cols,
                           int point_count,
                           int warmup,
                           int iters,
                           int repeats,
                           std::uint32_t seed);

std::vector<Phase2BenchmarkResult> run_phase2_benchmarks(
    const Phase2BenchmarkConfig& config);

}  // namespace cvh_bench_compare

#endif  // CVH_BENCHMARK_OPENCV_COMPARE_PHASE2_BENCHMARK_H
