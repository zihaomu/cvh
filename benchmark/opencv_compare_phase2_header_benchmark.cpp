#include "common/benchmark_common.h"
#include "cvh.h"
#include "cvh/core/detail/dispatch_control.h"
#include "opencv_compare_phase2_benchmark.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace cvh_bench_compare {
namespace {

volatile double g_phase2_cvh_sink = 0.0;

constexpr int kMicroWarmup = 2;
constexpr int kMicroIterations = 100;
constexpr int kMicroRepeats = 3;

inline void p2_compiler_barrier()
{
#if defined(_MSC_VER)
    _ReadWriteBarrier();
#else
    __asm__ __volatile__("" : : : "memory");
#endif
}

std::uint32_t p2_lcg_next(std::uint32_t state)
{
    return state * 1664525u + 1013904223u;
}

cvh::Mat p2_make_mat(int rows, int cols, int type)
{
    return cvh::Mat({rows, cols}, type);
}

void p2_fill_u8(cvh::Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int scalars_per_row = mat.size[1] * mat.channels();
    for (int row = 0; row < mat.size[0]; ++row)
    {
        unsigned char* output =
            mat.data + static_cast<std::size_t>(row) * mat.step(0);
        for (int index = 0; index < scalars_per_row; ++index)
        {
            state = p2_lcg_next(state);
            output[index] = static_cast<unsigned char>(
                (state >> 24) ^
                static_cast<std::uint32_t>(index + row * 17));
        }
    }
}

void p2_fill_f32(cvh::Mat& mat, std::uint32_t seed)
{
    std::uint32_t state = seed;
    const int scalars_per_row = mat.size[1] * mat.channels();
    for (int row = 0; row < mat.size[0]; ++row)
    {
        float* output = reinterpret_cast<float*>(
            mat.data + static_cast<std::size_t>(row) * mat.step(0));
        for (int index = 0; index < scalars_per_row; ++index)
        {
            state = p2_lcg_next(state);
            output[index] = static_cast<float>(
                static_cast<int>(state & 0xFFFFu) - 32768) /
                4096.0f;
        }
    }
}

void p2_fill_region_mask(cvh::Mat& mask)
{
    for (int row = 0; row < mask.size[0]; ++row)
    {
        for (int col = 0; col < mask.size[1]; ++col)
        {
            mask.at<unsigned char>(row, col) =
                ((row * 17 + col * 29) % 23) < 3 ? 255 : 0;
        }
    }
}

void p2_fill_contour_mask(cvh::Mat& mask)
{
    mask = 0;
    for (int y = 4; y + 12 < mask.size[0]; y += 20)
    {
        for (int x = 4; x + 12 < mask.size[1]; x += 20)
        {
            for (int yy = y; yy < y + 12; ++yy)
            {
                for (int xx = x; xx < x + 12; ++xx)
                {
                    mask.at<unsigned char>(yy, xx) = 255;
                }
            }
        }
    }
}

std::vector<cvh::Point> p2_make_shape_points(int count)
{
    std::vector<cvh::Point> points;
    points.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index)
    {
        points.emplace_back(
            index,
            (index * 37 + (index % 11) * 19) % 1009);
    }
    return points;
}

double p2_checksum(const cvh::Mat& mat)
{
    return static_cast<double>(
        cvh_bench::common::checksum_mat_bytes(mat));
}

double p2_checksum_points(const std::vector<cvh::Point>& points)
{
    double value = static_cast<double>(points.size());
    for (const cvh::Point& point : points)
    {
        value += static_cast<double>(point.x) * 0.5 +
            static_cast<double>(point.y) * 0.25;
    }
    return value;
}

double p2_checksum_contours(
    const std::vector<std::vector<cvh::Point>>& contours)
{
    double value = static_cast<double>(contours.size());
    for (const auto& contour : contours)
    {
        value += p2_checksum_points(contour);
    }
    return value;
}

void p2_calc_hist(const cvh::Mat& image, cvh::Mat& histogram)
{
    const int channel = 0;
    const int histogram_size = 256;
    const float range[] = {0.0f, 256.0f};
    const float* ranges[] = {range};
    cvh::calcHist(
        &image,
        1,
        &channel,
        cvh::Mat(),
        histogram,
        1,
        &histogram_size,
        ranges,
        true,
        false);
}

template<typename RunFn, typename ProbeFn>
double p2_measure_ms(RunFn&& run,
                     ProbeFn&& probe,
                     int warmup,
                     int iters,
                     int repeats)
{
    const auto measured_run = [&]() {
        p2_compiler_barrier();
        run();
        p2_compiler_barrier();
    };
    const auto timing = cvh_bench::common::measure_repeated_ms(
        measured_run, warmup, iters, repeats);
    g_phase2_cvh_sink += static_cast<double>(probe());
    return timing.min_ms;
}

#define P2_BENCH_FUNCTION bench_cvh_phase2
#define P2_NAMESPACE cvh
#define P2_MAT cvh::Mat
#define P2_POINT_TYPE cvh::Point
#include "opencv_compare_phase2_cases.inl"
#undef P2_POINT_TYPE
#undef P2_MAT
#undef P2_NAMESPACE
#undef P2_BENCH_FUNCTION

struct Phase2CaseSpec
{
    Phase2OpId id;
    const char* suite;
    const char* op;
    const char* variant;
    const char* depth;
    int channels;
    const char* layout;
    bool point_shape;
    bool histogram_shape;
    bool template_shape;
    bool micro;
    bool random_stream;
};

const std::vector<Phase2CaseSpec>& phase2_case_specs()
{
    static const std::vector<Phase2CaseSpec> specs = {
        {Phase2OpId::RanduU8C3, "core_mat", "RANDU", "C3", "CV_8U", 3, "continuous", false, false, false, false, true},
        {Phase2OpId::RandnU8C3, "core_mat", "RANDN", "C3", "CV_8U", 3, "continuous", false, false, false, false, true},
        {Phase2OpId::RanduF32C3, "core_mat", "RANDU", "C3", "CV_32F", 3, "continuous", false, false, false, false, true},
        {Phase2OpId::RandnF32C3, "core_mat", "RANDN", "C3", "CV_32F", 3, "continuous", false, false, false, false, true},
        {Phase2OpId::RanduU8C1Roi, "core_mat", "RANDU", "C1", "CV_8U", 1, "roi", false, false, false, false, true},
        {Phase2OpId::TransformF32C3ToC4, "core_mat", "TRANSFORM", "F32_C3_TO_C4", "CV_32F", 3, "continuous", true, false, false, false, false},
        {Phase2OpId::PerspectiveTransformF32C3, "core_mat", "PERSPECTIVE_TRANSFORM", "F32_C3", "CV_32F", 3, "continuous", true, false, false, false, false},
        {Phase2OpId::ConnectedComponents, "imgproc", "CONNECTED_COMPONENTS", "SPARSE_8", "CV_8U", 1, "continuous", false, false, false, false, false},
        {Phase2OpId::ConnectedComponentsWithStats, "imgproc", "CONNECTED_COMPONENTS_WITH_STATS", "SPARSE_8", "CV_8U", 1, "continuous", false, false, false, false, false},
        {Phase2OpId::FindContours, "imgproc", "FIND_CONTOURS", "RETR_LIST_SIMPLE", "CV_8U", 1, "continuous", false, false, false, false, false},
        {Phase2OpId::BoundingRect, "imgproc", "BOUNDING_RECT", "S32_POINTS", "CV_32S", 2, "vector", true, false, false, true, false},
        {Phase2OpId::ContourArea, "imgproc", "CONTOUR_AREA", "S32_POINTS", "CV_32S", 2, "vector", true, false, false, true, false},
        {Phase2OpId::ArcLength, "imgproc", "ARC_LENGTH", "S32_POINTS", "CV_32S", 2, "vector", true, false, false, true, false},
        {Phase2OpId::ApproxPolyDP, "imgproc", "APPROX_POLY_DP", "EPS_1_CLOSED", "CV_32S", 2, "vector", true, false, false, true, false},
        {Phase2OpId::ConvexHull, "imgproc", "CONVEX_HULL", "CCW_POINTS", "CV_32S", 2, "vector", true, false, false, true, false},
        {Phase2OpId::IsContourConvex, "imgproc", "IS_CONTOUR_CONVEX", "S32_POINTS", "CV_32S", 2, "vector", true, false, false, true, false},
        {Phase2OpId::Moments, "imgproc", "MOMENTS", "S32_POINTS", "CV_32S", 2, "vector", true, false, false, true, false},
        {Phase2OpId::CalcHist, "imgproc", "CALC_HIST", "U8C1_256", "CV_8U", 1, "continuous", false, false, false, false, false},
        {Phase2OpId::CompareHistCorrel, "imgproc", "COMPARE_HIST", "METHOD_0", "CV_32F", 1, "continuous", false, true, false, true, false},
        {Phase2OpId::CompareHistChiSqr, "imgproc", "COMPARE_HIST", "METHOD_1", "CV_32F", 1, "continuous", false, true, false, true, false},
        {Phase2OpId::CompareHistIntersect, "imgproc", "COMPARE_HIST", "METHOD_2", "CV_32F", 1, "continuous", false, true, false, true, false},
        {Phase2OpId::CompareHistBhattacharyya, "imgproc", "COMPARE_HIST", "METHOD_3", "CV_32F", 1, "continuous", false, true, false, true, false},
        {Phase2OpId::MatchTemplateSqDiff, "imgproc", "MATCH_TEMPLATE", "METHOD_0", "CV_8U", 1, "continuous", false, false, true, false, false},
        {Phase2OpId::MatchTemplateSqDiffNormed, "imgproc", "MATCH_TEMPLATE", "METHOD_1", "CV_8U", 1, "continuous", false, false, true, false, false},
        {Phase2OpId::MatchTemplateCCorr, "imgproc", "MATCH_TEMPLATE", "METHOD_2", "CV_8U", 1, "continuous", false, false, true, false, false},
        {Phase2OpId::MatchTemplateCCorrNormed, "imgproc", "MATCH_TEMPLATE", "METHOD_3", "CV_8U", 1, "continuous", false, false, true, false, false},
    };
    return specs;
}

}  // namespace

std::vector<Phase2BenchmarkResult> run_phase2_benchmarks(
    const Phase2BenchmarkConfig& config)
{
    const int rows = config.profile == "quick" ? 120 : 240;
    const int cols = config.profile == "quick" ? 160 : 320;
    const int point_count = config.profile == "full" ? 16384 : 4096;
    constexpr std::uint32_t seed = 0x4201u;
    const auto& specs = phase2_case_specs();
    if (specs.size() != 26u)
    {
        throw std::logic_error(
            "P2-P0 Mode B case list must contain 26 rows");
    }

    std::vector<Phase2BenchmarkResult> results;
    results.reserve(specs.size());
    for (const Phase2CaseSpec& spec : specs)
    {
        int warmup = config.warmup;
        int iters = config.iters;
        int repeats = config.repeats;
        if (spec.micro)
        {
            warmup = kMicroWarmup;
            iters = kMicroIterations;
            repeats = kMicroRepeats;
        }

        cvh::cpu::reset_last_dispatch_tag();
        const double cvh_ms = bench_cvh_phase2(
            spec.id,
            rows,
            cols,
            point_count,
            warmup,
            iters,
            repeats,
            seed);
        const cvh::cpu::DispatchTag dispatch_tag =
            cvh::cpu::last_dispatch_tag();
        const double opencv_ms = bench_opencv_phase2(
            spec.id,
            rows,
            cols,
            point_count,
            warmup,
            iters,
            repeats,
            seed);
        if (cvh_ms <= 0.0 || opencv_ms <= 0.0)
        {
            throw std::runtime_error(
                std::string("P2-P0 benchmark failed for ") + spec.op +
                "/" + spec.variant);
        }

        Phase2BenchmarkResult result;
        result.suite = spec.suite;
        result.op = spec.op;
        result.variant = spec.variant;
        const bool point_transform_fastpath =
            spec.id == Phase2OpId::TransformF32C3ToC4 ||
            spec.id == Phase2OpId::PerspectiveTransformF32C3;
        const bool histogram_fastpath =
            spec.id == Phase2OpId::CalcHist ||
            spec.id == Phase2OpId::CompareHistCorrel ||
            spec.id == Phase2OpId::CompareHistChiSqr ||
            spec.id == Phase2OpId::CompareHistIntersect ||
            spec.id == Phase2OpId::CompareHistBhattacharyya;
        const bool random_fastpath =
            spec.id == Phase2OpId::RanduU8C3 ||
            spec.id == Phase2OpId::RandnU8C3 ||
            spec.id == Phase2OpId::RanduF32C3 ||
            spec.id == Phase2OpId::RandnF32C3 ||
            spec.id == Phase2OpId::RanduU8C1Roi;
        if (dispatch_tag == cvh::cpu::DispatchTag::OpenCVUI)
        {
            result.dispatch_path = "opencv_ui";
        }
        else if (dispatch_tag == cvh::cpu::DispatchTag::Scalar &&
                 (point_transform_fastpath || histogram_fastpath || random_fastpath))
        {
            result.dispatch_path = "header_fastpath";
        }
        else
        {
            result.dispatch_path = "public_header_scalar";
        }
        result.depth = spec.depth;
        result.channels = spec.channels;
        result.layout = spec.layout;
        if (spec.histogram_shape)
        {
            result.shape = "256 bins";
        }
        else if (spec.template_shape)
        {
            result.shape = std::to_string(cols) + "x" +
                std::to_string(rows) + "/16x16";
        }
        else if (spec.point_shape)
        {
            result.shape =
                spec.id == Phase2OpId::TransformF32C3ToC4 ||
                    spec.id == Phase2OpId::PerspectiveTransformF32C3
                ? std::to_string(point_count) + "x1"
                : std::to_string(point_count) + " points";
        }
        else
        {
            result.shape =
                std::to_string(cols) + "x" + std::to_string(rows);
        }
        result.cvh_ms = cvh_ms;
        result.opencv_ms = opencv_ms;
        result.note =
            "phase2_p0_representative_case;correctness=upstream_pass";
        if (spec.random_stream)
        {
            result.note +=
                ";random_streams=independent;shape_and_range=aligned";
        }
        if (spec.micro)
        {
            result.note +=
                ";micro_warmup=" + std::to_string(warmup) +
                ";micro_iterations=" + std::to_string(iters) +
                ";micro_repeats=" + std::to_string(repeats);
        }
        if (result.dispatch_path == "public_header_scalar")
        {
            result.note += ";no_ui_fastpath";
        }
        else if (result.dispatch_path == "header_fastpath")
        {
            if (point_transform_fastpath)
                result.note +=
                    ";coefficients=prepacked;channels=specialized;span=continuous";
            else if (histogram_fastpath)
                result.note += ";method=split;rows=typed;accumulator=local";
            else
                result.note +=
                    ";engine=xorshift64star;distributions=hoisted;channels=unrolled;rows=typed";
        }
        results.push_back(std::move(result));
    }
    return results;
}

}  // namespace cvh_bench_compare
