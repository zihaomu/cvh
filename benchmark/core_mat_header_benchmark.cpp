#include "cvh.h"
#include "common/benchmark_common.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cvh_bench {

using Result = common::BenchmarkResult;
constexpr int kBenchmarkSchemaVersion = 2;

struct Args
{
    std::string profile = "quick";
    std::string dispatch = "auto";
    std::string ops;
    int warmup = 3;
    int iters = 10;
    int repeats = 7;
    int threads = 1;
    std::string output_csv;
};

struct ShapeCase
{
    const char* name;
    int rows;
    int cols;
    int type;
};

struct ResultRow
{
    std::string mode = "internal";
    std::string suite = "core_mat";
    std::string module = "core";
    std::string op;
    std::string variant;
    std::string depth;
    int channels = 0;
    std::string layout;
    std::string shape;
    std::size_t elements = 0;
    std::size_t pixels = 0;
    std::string implementation = "cvh_headers";
    std::string dispatch_path = "header_only";
    std::string allocation_mode;
    int warmup = 0;
    int iters = 0;
    int repeats = 0;
    int threads = 1;
    double min_ms = 0.0;
    double median_ms = 0.0;
    double mpix_per_sec = 0.0;
    double melems_per_sec = 0.0;
    double gb_per_sec = 0.0;
    std::uint64_t checksum = 0;
    std::string status = "OK";
    std::string note;
};

volatile std::uint64_t g_sink = 0;

void usage()
{
    std::cout
        << "Usage: cvh_benchmark_core_mat_header "
        << "[--profile quick|stable|full] [--dispatch auto|scalar] "
        << "[--ops GEMM] [--threads N] [--warmup N] [--iters N] [--repeats N] [--output path]\n";
}

Args parse_args(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        const std::string token = argv[i];
        auto next_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
            {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(2);
            }
            return std::string(argv[++i]);
        };

        if (token == "--profile")
        {
            args.profile = next_value("--profile");
        }
        else if (token == "--dispatch")
        {
            args.dispatch = next_value("--dispatch");
        }
        else if (token == "--ops")
        {
            args.ops = next_value("--ops");
        }
        else if (token == "--warmup")
        {
            args.warmup = std::max(0, std::stoi(next_value("--warmup")));
        }
        else if (token == "--iters")
        {
            args.iters = std::max(1, std::stoi(next_value("--iters")));
        }
        else if (token == "--repeats")
        {
            args.repeats = std::max(1, std::stoi(next_value("--repeats")));
        }
        else if (token == "--threads")
        {
            args.threads = std::max(1, std::stoi(next_value("--threads")));
        }
        else if (token == "--output")
        {
            args.output_csv = next_value("--output");
        }
        else if (token == "--help")
        {
            usage();
            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown arg: " << token << "\n";
            std::exit(2);
        }
    }

    if (!common::profile_is_allowed(args.profile, {"quick", "stable", "full"}))
    {
        std::cerr << "Unsupported profile: " << args.profile << "\n";
        std::exit(2);
    }
    if (args.dispatch != "auto" && args.dispatch != "scalar")
    {
        std::cerr << "Unsupported dispatch: " << args.dispatch
                  << " (expected auto/scalar)\n";
        std::exit(2);
    }
    if (!args.ops.empty() && args.ops != "GEMM")
    {
        std::cerr << "Unsupported --ops value: " << args.ops
                  << " (currently supported: GEMM)\n";
        std::exit(2);
    }
    return args;
}

std::string depth_name(int depth)
{
    switch (depth)
    {
        case CV_8U: return "CV_8U";
        case CV_16S: return "CV_16S";
        case CV_32F: return "CV_32F";
        default: return "UNKNOWN";
    }
}

std::string shape_name(const ShapeCase& shape)
{
    std::ostringstream oss;
    oss << shape.cols << "x" << shape.rows << "C" << CV_MAT_CN(shape.type);
    return oss.str();
}

std::string fmt(double value)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << value;
    return oss.str();
}

std::vector<ShapeCase> build_shapes(const std::string& profile)
{
    std::vector<ShapeCase> shapes = {
        {"tiny_u8c1", 32, 32, CV_8UC1},
        {"vga_u8c1", 480, 640, CV_8UC1},
        {"vga_u8c3", 480, 640, CV_8UC3},
        {"vga_f32c1", 480, 640, CV_32FC1},
    };

    if (profile == "stable" || profile == "full")
    {
        shapes.push_back({"hd_u8c1", 1080, 1920, CV_8UC1});
        shapes.push_back({"hd_u8c3", 1080, 1920, CV_8UC3});
        shapes.push_back({"hd_f32c1", 1080, 1920, CV_32FC1});
    }
    if (profile == "full")
    {
        shapes.push_back({"nonaligned_u8c1", 479, 641, CV_8UC1});
        shapes.push_back({"uhd_u8c1", 2160, 3840, CV_8UC1});
        shapes.push_back({"uhd_u8c3", 2160, 3840, CV_8UC3});
    }

    return shapes;
}

void fill_mat(cvh::Mat& mat, std::uint32_t seed)
{
    if (mat.depth() == CV_8U)
    {
        common::fill_mat_u8_lcg(mat, seed);
        return;
    }

    if (mat.depth() == CV_32F)
    {
        const std::size_t count = mat.total() * static_cast<std::size_t>(mat.channels());
        float* ptr = reinterpret_cast<float*>(mat.data);
        for (std::size_t i = 0; i < count; ++i)
        {
            seed = seed * 1664525u + 1013904223u;
            ptr[i] = static_cast<float>(static_cast<int>((seed >> 8) & 0xffffu) - 32768) / 1024.0f;
        }
    }
}

void set_linear_nonzero(cvh::Mat& mat, std::size_t index)
{
    if (mat.channels() != 1 || !mat.isContinuous() || index >= mat.total())
    {
        throw std::runtime_error("invalid nonzero benchmark fixture");
    }

    switch (mat.depth())
    {
        case CV_8U:
            mat.data[index] = 1;
            return;
        case CV_32F:
            reinterpret_cast<float*>(mat.data)[index] = 1.0f;
            return;
        default:
            throw std::runtime_error("unsupported nonzero benchmark depth");
    }
}

template <typename RunFn, typename ChecksumFn>
Result measure(RunFn&& run_once, ChecksumFn&& checksum_fn, const Args& args)
{
    cvh::cpu::reset_last_dispatch_tag();
    const auto timing = common::measure_repeated_ms(run_once, args.warmup, args.iters, args.repeats);
    const std::uint64_t hash = checksum_fn();
    g_sink ^= hash;
    return Result {timing.min_ms, timing.median_ms, hash};
}

ResultRow make_row(const Args& args,
                   const ShapeCase& shape,
                   const std::string& op,
                   const std::string& variant,
                   const std::string& layout,
                   const std::string& allocation_mode,
                   std::size_t bytes_touched,
                   const Result& result)
{
    const std::size_t pixels = static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols);
    const std::size_t elements = pixels * static_cast<std::size_t>(CV_MAT_CN(shape.type));
    ResultRow row;
    row.op = op;
    row.variant = variant;
    row.depth = depth_name(CV_MAT_DEPTH(shape.type));
    row.channels = CV_MAT_CN(shape.type);
    row.layout = layout;
    row.shape = shape_name(shape);
    row.elements = elements;
    row.pixels = pixels;
    row.allocation_mode = allocation_mode;
    row.warmup = args.warmup;
    row.iters = args.iters;
    row.repeats = args.repeats;
    row.threads = args.threads;
    row.min_ms = result.min_ms;
    row.median_ms = result.median_ms;
    row.mpix_per_sec = common::mpix_per_sec(pixels, result.median_ms);
    row.melems_per_sec = result.median_ms > 0.0 ? static_cast<double>(elements) / result.median_ms / 1000.0 : 0.0;
    row.gb_per_sec = result.median_ms > 0.0 ? static_cast<double>(bytes_touched) / result.median_ms / 1.0e6 : 0.0;
    row.checksum = result.checksum;
    return row;
}

template<typename RunFn>
void append_array_op_row(const Args& args,
                         const ShapeCase& shape,
                         const std::string& op,
                         const std::string& variant,
                         std::size_t bytes_touched,
                         cvh::Mat& dst,
                         RunFn&& run,
                         std::vector<ResultRow>& rows)
{
    cvh::cpu::reset_last_dispatch_tag();
    const auto result = measure(
        std::forward<RunFn>(run),
        [&]() { return common::checksum_mat_bytes(dst); },
        args);
    ResultRow row = make_row(
        args,
        shape,
        op,
        variant,
        "continuous",
        "reuse",
        bytes_touched,
        result);
    const cvh::cpu::DispatchTag dispatch_tag = cvh::cpu::last_dispatch_tag();
    if (dispatch_tag != cvh::cpu::DispatchTag::Unknown)
    {
        row.dispatch_path = cvh::cpu::dispatch_tag_name(dispatch_tag);
    }
    rows.push_back(std::move(row));
}

std::uint64_t checksum_doubles(const double* values, std::size_t count)
{
    std::uint64_t hash = common::fnv1a64_basis();
    for (std::size_t i = 0; i < count; ++i)
    {
        std::uint64_t bits = 0;
        std::memcpy(&bits, values + i, sizeof(bits));
        hash = common::fnv1a64_mix_u64(hash, bits);
    }
    return hash;
}

void append_measured_row(const Args& args,
                         const ShapeCase& shape,
                         const std::string& op,
                         const std::string& variant,
                         const std::string& allocation_mode,
                         std::size_t bytes_touched,
                         int threads,
                         const std::string& note,
                         const Result& result,
                         std::vector<ResultRow>& rows)
{
    ResultRow row = make_row(
        args,
        shape,
        op,
        variant,
        "continuous",
        allocation_mode,
        bytes_touched,
        result);
    row.threads = threads;
    row.note = note;
    const cvh::cpu::DispatchTag dispatch_tag = cvh::cpu::last_dispatch_tag();
    if (dispatch_tag != cvh::cpu::DispatchTag::Unknown)
    {
        row.dispatch_path = cvh::cpu::dispatch_tag_name(dispatch_tag);
    }
    rows.push_back(std::move(row));
}

void append_reduction_rows(const Args& args,
                           const ShapeCase& shape,
                           const cvh::Mat& src,
                           std::size_t bytes,
                           std::vector<ResultRow>& rows)
{
    cvh::Mat zeros(src.shape(), src.type());
    zeros.setTo(cvh::Scalar::all(0.0));
    const int saved_threads = cvh::getNumThreads();
    const int thread_counts[] = {1, saved_threads};
    const char* thread_labels[] = {"threads_1", "project_default"};
    for (int mode = 0; mode < 2; ++mode)
    {
        const int threads = thread_counts[mode];
        cvh::setNumThreads(threads);
        const std::string suffix = thread_labels[mode];
        const std::string note =
            "deterministic_header_loop;configured_threads=" +
            std::to_string(threads);
        const bool f32_c1 =
            src.depth() == CV_32F && src.channels() == 1;

        {
            cvh::Scalar value;
            const auto result = measure(
                [&]() { value = cvh::sum(src); },
                [&]() { return checksum_doubles(value.val, 4); },
                args);
            append_measured_row(
                args,
                shape,
                "SUM",
                "all_channels_" + suffix,
                "none",
                bytes,
                threads,
                note,
                result,
                rows);
        }

        {
            cvh::Scalar value;
            const auto result = measure(
                [&]() { value = cvh::mean(src); },
                [&]() { return checksum_doubles(value.val, 4); },
                args);
            append_measured_row(
                args,
                shape,
                "MEAN",
                "all_channels_" + suffix,
                "none",
                bytes,
                threads,
                note,
                result,
                rows);
        }

        {
            cvh::Scalar mean_value;
            cvh::Scalar stddev_value;
            const auto result = measure(
                [&]() {
                    cvh::meanStdDev(src, mean_value, stddev_value);
                },
                [&]() {
                    std::uint64_t hash = checksum_doubles(mean_value.val, 4);
                    return common::fnv1a64_mix_u64(
                        hash, checksum_doubles(stddev_value.val, 4));
                },
                args);
            append_measured_row(
                args,
                shape,
                "MEAN_STDDEV",
                "all_channels_" + suffix,
                "none",
                bytes,
                threads,
                f32_c1
                    ? note + ";statistics=f32_c1_two_pass_ui"
                    : note,
                result,
                rows);
        }

        const struct
        {
            const char* variant;
            int norm_type;
            bool difference;
        } norm_cases[] = {
            {"inf_single_", cvh::NORM_INF, false},
            {"l1_single_", cvh::NORM_L1, false},
            {"l2_", cvh::NORM_L2, false},
            {"inf_diff_zero_", cvh::NORM_INF, true},
            {"l1_diff_zero_", cvh::NORM_L1, true},
            {"l2_diff_zero_", cvh::NORM_L2, true},
        };
        for (const auto& norm_case : norm_cases)
        {
            double value = 0.0;
            const auto result = measure(
                [&]() {
                    value = norm_case.difference
                        ? cvh::norm(
                              src, zeros, norm_case.norm_type)
                        : cvh::norm(src, norm_case.norm_type);
                },
                [&]() { return checksum_doubles(&value, 1); },
                args);
            append_measured_row(
                args,
                shape,
                "NORM",
                std::string(norm_case.variant) + suffix,
                "none",
                norm_case.difference ? bytes * 2 : bytes,
                threads,
                f32_c1 && norm_case.norm_type == cvh::NORM_INF &&
                        !norm_case.difference
                    ? note + ";kernel=f32_single_input_inf_ui"
                    : note,
                result,
                rows);
        }

        if (src.channels() == 1)
        {
            double min_value = 0.0;
            double max_value = 0.0;
            cvh::Point min_location;
            cvh::Point max_location;
            const auto result = measure(
                [&]() {
                    cvh::minMaxLoc(
                        src,
                        &min_value,
                        &max_value,
                        &min_location,
                        &max_location);
                },
                [&]() {
                    const double values[] = {
                        min_value,
                        max_value,
                        static_cast<double>(min_location.x),
                        static_cast<double>(min_location.y),
                        static_cast<double>(max_location.x),
                        static_cast<double>(max_location.y),
                    };
                    return checksum_doubles(values, 6);
                },
                args);
            append_measured_row(
                args,
                shape,
                "MIN_MAX_LOC",
                "first_tie_" + suffix,
                "none",
                bytes,
                threads,
                note,
                result,
                rows);

            int min_indices[2] = {-1, -1};
            int max_indices[2] = {-1, -1};
            const auto index_result = measure(
                [&]() {
                    cvh::minMaxIdx(
                        src,
                        &min_value,
                        &max_value,
                        min_indices,
                        max_indices);
                },
                [&]() {
                    const double values[] = {
                        min_value,
                        max_value,
                        static_cast<double>(min_indices[0]),
                        static_cast<double>(min_indices[1]),
                        static_cast<double>(max_indices[0]),
                        static_cast<double>(max_indices[1]),
                    };
                    return checksum_doubles(values, 6);
                },
                args);
            append_measured_row(
                args,
                shape,
                "MIN_MAX_IDX",
                "first_tie_" + suffix,
                "none",
                bytes,
                threads,
                note,
                index_result,
                rows);
        }

        const struct
        {
            const char* name;
            int rtype;
        } reduce_cases[] = {
            {"sum", cvh::REDUCE_SUM},
            {"avg", cvh::REDUCE_AVG},
            {"max", cvh::REDUCE_MAX},
            {"min", cvh::REDUCE_MIN},
            {"sum2", cvh::REDUCE_SUM2},
        };
        for (int axis = 0; axis <= 1; ++axis)
        {
            for (const auto& reduce_case : reduce_cases)
            {
                cvh::Mat reduced;
                const auto result = measure(
                    [&]() {
                        cvh::reduce(
                            src,
                            reduced,
                            axis,
                            reduce_case.rtype,
                            CV_64F);
                    },
                    [&]() {
                        return common::checksum_mat_bytes(reduced);
                    },
                    args);
                const std::size_t output_elements =
                    static_cast<std::size_t>(
                        axis == 0 ? shape.cols : shape.rows) *
                    static_cast<std::size_t>(src.channels());
                append_measured_row(
                    args,
                    shape,
                    "REDUCE",
                    "axis_" + std::to_string(axis) + "_" +
                        reduce_case.name + "_f64_" + suffix,
                    "reuse",
                    bytes + output_elements * sizeof(double),
                    threads,
                    f32_c1 && axis == 1 &&
                            (reduce_case.rtype == cvh::REDUCE_MAX ||
                             reduce_case.rtype == cvh::REDUCE_MIN)
                        ? note + ";kernel=f32_c1_vector_extrema"
                        : note,
                    result,
                    rows);
            }
        }

        const struct
        {
            const char* variant;
            int norm_type;
            double alpha;
            double beta;
        } normalize_cases[] = {
            {"inf_", cvh::NORM_INF, 1.0, 0.0},
            {"l1_", cvh::NORM_L1, 1.0, 0.0},
            {"l2_", cvh::NORM_L2, 1.0, 0.0},
            {"minmax_", cvh::NORM_MINMAX, -1.0, 1.0},
        };
        for (const auto& normalize_case : normalize_cases)
        {
            cvh::Mat normalized;
            const auto result = measure(
                [&]() {
                    cvh::normalize(
                        src,
                        normalized,
                        normalize_case.alpha,
                        normalize_case.beta,
                        normalize_case.norm_type);
                },
                [&]() { return common::checksum_mat_bytes(normalized); },
                args);
            append_measured_row(
                args,
                shape,
                "NORMALIZE",
                std::string(normalize_case.variant) + suffix,
                "reuse",
                bytes * 2,
                threads,
                note,
                result,
                rows);
        }

        if (src.depth() == CV_32F)
        {
            cvh::Mat applied(src.shape(), src.type());
            const auto result = measure(
                [&]() {
                    if (cvh::detail::reduce_ui::try_apply_normalize(
                            src,
                            applied,
                            cvh::Mat(),
                            0.125,
                            1.5))
                    {
                        cvh::cpu::set_last_dispatch_tag(
                            cvh::cpu::DispatchTag::OpenCVUI);
                    }
                    else
                    {
                        cvh::reduce_detail::apply_normalize(
                            src,
                            applied,
                            cvh::Mat(),
                            0.125,
                            1.5);
                        cvh::cpu::set_last_dispatch_tag(
                            cvh::cpu::DispatchTag::Scalar);
                    }
                },
                [&]() { return common::checksum_mat_bytes(applied); },
                args);
            append_measured_row(
                args,
                shape,
                "NORMALIZE_APPLY_SCALE",
                "f32_to_f32_" + suffix,
                "reuse",
                bytes * 2,
                threads,
                note + ";reduction_excluded",
                result,
                rows);
        }
    }
    cvh::setNumThreads(saved_threads);

    if (src.channels() != 1)
    {
        return;
    }

    const std::vector<int> dims {shape.rows, shape.cols};
    cvh::Mat all_zero(dims, shape.type);
    all_zero.setTo(cvh::Scalar::all(0.0));
    cvh::Mat first_nonzero = all_zero.clone();
    cvh::Mat tail_nonzero = all_zero.clone();
    set_linear_nonzero(first_nonzero, 0);
    set_linear_nonzero(tail_nonzero, tail_nonzero.total() - 1);
    const std::string note = "single_channel;distribution_sensitive";

    {
        int value = 0;
        const auto result = measure(
            [&]() { value = cvh::countNonZero(src); },
            [&]() {
                const double checksum_value = static_cast<double>(value);
                return checksum_doubles(&checksum_value, 1);
            },
            args);
        append_measured_row(
            args,
            shape,
            "COUNT_NON_ZERO",
            "random_dense",
            "none",
            bytes,
            1,
            note,
            result,
            rows);
    }

    const struct
    {
        const char* variant;
        const cvh::Mat* input;
    } has_nonzero_cases[] = {
        {"all_zero", &all_zero},
        {"first_nonzero", &first_nonzero},
        {"tail_nonzero", &tail_nonzero},
    };
    for (const auto& benchmark_case : has_nonzero_cases)
    {
        bool value = false;
        const auto result = measure(
            [&]() { value = cvh::hasNonZero(*benchmark_case.input); },
            [&]() {
                const double checksum_value = value ? 1.0 : 0.0;
                return checksum_doubles(&checksum_value, 1);
            },
            args);
        append_measured_row(
            args,
            shape,
            "HAS_NON_ZERO",
            benchmark_case.variant,
            "none",
            bytes,
            1,
            note,
            result,
            rows);
    }

    const struct
    {
        const char* variant;
        const cvh::Mat* input;
    } find_nonzero_cases[] = {
        {"all_zero", &all_zero},
        {"sparse_tail", &tail_nonzero},
        {"random_dense", &src},
    };
    for (const auto& benchmark_case : find_nonzero_cases)
    {
        std::vector<cvh::Point> points;
        const auto result = measure(
            [&]() {
                cvh::findNonZero(*benchmark_case.input, points);
            },
            [&]() {
                if (points.empty())
                {
                    return common::fnv1a64_basis();
                }
                const double values[] = {
                    static_cast<double>(points.size()),
                    static_cast<double>(points.front().x),
                    static_cast<double>(points.front().y),
                    static_cast<double>(points.back().x),
                    static_cast<double>(points.back().y),
                };
                return checksum_doubles(values, 5);
            },
            args);
        append_measured_row(
            args,
            shape,
            "FIND_NON_ZERO",
            benchmark_case.variant,
            "reuse",
            bytes,
            1,
            note,
            result,
            rows);
    }

    {
        cvh::Mat indices;
        const auto result = measure(
            [&]() { cvh::reduceArgMin(src, indices, 1, false); },
            [&]() { return common::checksum_mat_bytes(indices); },
            args);
        append_measured_row(
            args,
            shape,
            "REDUCE_ARG_MIN",
            "axis_1_first",
            "reuse",
            bytes + static_cast<size_t>(shape.rows) * sizeof(int),
            1,
            note,
            result,
            rows);
    }

    {
        cvh::Mat indices;
        const auto result = measure(
            [&]() { cvh::reduceArgMax(src, indices, 0, true); },
            [&]() { return common::checksum_mat_bytes(indices); },
            args);
        append_measured_row(
            args,
            shape,
            "REDUCE_ARG_MAX",
            "axis_0_last",
            "reuse",
            bytes + static_cast<size_t>(shape.cols) * sizeof(int),
            1,
            note,
            result,
            rows);
    }
}

void append_layout_rows(const Args& args,
                        const ShapeCase& shape,
                        const cvh::Mat& src,
                        std::size_t bytes,
                        std::vector<ResultRow>& rows)
{
    const std::vector<int> dims {shape.rows, shape.cols};
    {
        cvh::Mat mask(dims, CV_8UC1);
        for (int y = 0; y < shape.rows; ++y)
        {
            uchar* mask_row =
                mask.data + static_cast<std::size_t>(y) * mask.step(0);
            for (int x = 0; x < shape.cols; ++x)
            {
                mask_row[x] = ((x + y) & 1) != 0 ? 255 : 0;
            }
        }
        cvh::Mat dst(dims, shape.type);
        dst.setTo(cvh::Scalar::all(0.0));
        append_array_op_row(
            args,
            shape,
            "COPY_TO",
            "partial_mask",
            bytes * 2 + mask.total(),
            dst,
            [&]() { cvh::copyTo(src, dst, mask); },
            rows);
    }

    if (src.channels() > 1)
    {
        cvh::Mat dst;
        append_array_op_row(
            args,
            shape,
            "EXTRACT_CHANNEL",
            "last_channel",
            bytes + src.total() * src.elemSize1(),
            dst,
            [&]() { cvh::extractChannel(src, dst, src.channels() - 1); },
            rows);

        cvh::Mat channel;
        cvh::extractChannel(src, channel, 0);
        cvh::Mat inserted = src.clone();
        append_array_op_row(
            args,
            shape,
            "INSERT_CHANNEL",
            "channel_0_to_last",
            bytes * 2 + channel.total() * channel.elemSize(),
            inserted,
            [&]() {
                cvh::insertChannel(
                    channel, inserted, inserted.channels() - 1);
            },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        std::vector<int> routes;
        routes.reserve(static_cast<std::size_t>(src.channels()) * 2);
        for (int ch = 0; ch < src.channels(); ++ch)
        {
            routes.push_back(ch);
            routes.push_back(src.channels() - 1 - ch);
        }
        append_array_op_row(
            args,
            shape,
            "MIX_CHANNELS",
            "reverse_channels",
            bytes * 2,
            dst,
            [&]() {
                cvh::mixChannels(
                    &src,
                    1,
                    &dst,
                    1,
                    routes.data(),
                    routes.size() / 2);
            },
            rows);
    }

    {
        cvh::Mat dst;
        append_array_op_row(
            args,
            shape,
            "FLIP",
            "horizontal",
            bytes * 2,
            dst,
            [&]() { cvh::flip(src, dst, 1); },
            rows);
    }

    {
        cvh::Mat dst;
        append_array_op_row(
            args,
            shape,
            "FLIP_ND",
            "axis_1",
            bytes * 2,
            dst,
            [&]() { cvh::flipND(src, dst, 1); },
            rows);
    }

    {
        cvh::Mat dst;
        append_array_op_row(
            args,
            shape,
            "TRANSPOSE",
            "last_two",
            bytes * 2,
            dst,
            [&]() { dst = cvh::transpose(src); },
            rows);
    }

    {
        cvh::Mat dst;
        append_array_op_row(
            args,
            shape,
            "ROTATE",
            "clockwise_90",
            bytes * 3,
            dst,
            [&]() { cvh::rotate(src, dst, cvh::ROTATE_90_CLOCKWISE); },
            rows);
    }

    {
        cvh::Mat dst;
        append_array_op_row(
            args,
            shape,
            "REPEAT",
            "2x2",
            bytes * 5,
            dst,
            [&]() { cvh::repeat(src, 2, 2, dst); },
            rows);
    }

    {
        cvh::Mat dst;
        append_array_op_row(
            args,
            shape,
            "HCONCAT",
            "two_equal_inputs",
            bytes * 3,
            dst,
            [&]() { cvh::hconcat(src, src, dst); },
            rows);
    }

    {
        cvh::Mat dst;
        append_array_op_row(
            args,
            shape,
            "VCONCAT",
            "two_equal_inputs",
            bytes * 3,
            dst,
            [&]() { cvh::vconcat(src, src, dst); },
            rows);
    }

    {
        cvh::Mat dst;
        append_array_op_row(
            args,
            shape,
            "BROADCAST",
            "prepend_extent_2",
            bytes * 3,
            dst,
            [&]() {
                cvh::broadcast(
                    src,
                    std::vector<int>({2, shape.rows, shape.cols}),
                    dst);
            },
            rows);
    }
}

void append_gemm_rows(const Args& args, std::vector<ResultRow>& rows)
{
    struct GemmCase
    {
        const char* name;
        int m;
        int k;
        int n;
    };

    std::vector<GemmCase> cases {
        {"square_128", 128, 128, 128},
        {"skinny_m32_k512_n64", 32, 512, 64},
        {"wide_m256_k32_n256", 256, 32, 256},
    };
    if (args.profile == "stable" || args.profile == "full")
    {
        cases.push_back({"square_256", 256, 256, 256});
    }
    if (args.profile == "full")
    {
        cases.push_back({"square_512", 512, 512, 512});
    }

    for (const GemmCase& gemm_case : cases)
    {
        const ShapeCase shape {
            gemm_case.name,
            gemm_case.m,
            gemm_case.n,
            CV_32FC1};
        cvh::Mat a(
            {gemm_case.m, gemm_case.k}, CV_32FC1);
        cvh::Mat b(
            {gemm_case.k, gemm_case.n}, CV_32FC1);
        fill_mat(a, 0x13579BDFu);
        fill_mat(b, 0x2468ACE0u);
        const std::size_t a_bytes = a.total() * a.elemSize();
        const std::size_t b_bytes = b.total() * b.elemSize();
        const std::size_t c_bytes =
            static_cast<std::size_t>(gemm_case.m) *
            static_cast<std::size_t>(gemm_case.n) *
            sizeof(float);
        const std::uint64_t work =
            static_cast<std::uint64_t>(gemm_case.m) *
            static_cast<std::uint64_t>(gemm_case.k) *
            static_cast<std::uint64_t>(gemm_case.n);
        Args gemm_args = args;
        if (work > (16u << 20))
        {
            gemm_args.iters = std::min(
                args.iters,
                std::max(
                    1,
                    static_cast<int>((16u << 20) / work)));
            gemm_args.warmup = std::min(args.warmup, 1);
        }
        const std::string suffix =
            std::string(gemm_case.name);

        cvh::Mat dst;
        append_array_op_row(
            gemm_args,
            shape,
            "GEMM",
            "fp32_nn_end_to_end_" + suffix,
            a_bytes + b_bytes + c_bytes,
            dst,
            [&]() { dst = cvh::gemm(a, b); },
            rows);
        rows.back().allocation_mode = "recreate";
        rows.back().note =
            "component=public_end_to_end;packing=included;"
            "output_allocation=included";

        cvh::GemmPackedB packed_b;
        const auto pack_result = measure(
            [&]() { packed_b = cvh::gemm_pack_b(b); },
            [&]() {
                if (!packed_b.packed_fp32.empty())
                {
                    return common::checksum_bytes(
                        reinterpret_cast<const uchar*>(
                            packed_b.packed_fp32.data()),
                        packed_b.packed_fp32.size() *
                            sizeof(float));
                }
                return common::fnv1a64_basis();
            },
            gemm_args);
        ResultRow pack_row = make_row(
            gemm_args,
            shape,
            "GEMM",
            "fp32_pack_b_only_" + suffix,
            "continuous",
            "precompute",
            b_bytes * 2,
            pack_result);
        pack_row.dispatch_path = "detail_precompute";
        pack_row.note =
            "component=precompute;output_allocation=excluded;"
            "matrix_kernel=excluded";
        rows.push_back(std::move(pack_row));

        append_array_op_row(
            gemm_args,
            shape,
            "GEMM",
            "fp32_nn_pack_once_" + suffix,
            a_bytes + b_bytes + c_bytes,
            dst,
            [&]() { dst = cvh::gemm(a, packed_b); },
            rows);
        rows.back().allocation_mode = "packed_b";
        rows.back().note =
            "component=public_pack_once;packing=excluded;"
            "output_allocation=included";

        cvh::Mat kernel_dst(
            {gemm_case.m, gemm_case.n}, CV_32FC1);
        auto run_kernel = [&]() {
            const bool use_ui =
                cvh::detail::gemm_ui::can_vectorize_nn(
                    gemm_case.n);
            cvh::cpu::set_last_dispatch_tag(
                use_ui ? cvh::cpu::DispatchTag::OpenCVUI
                       : cvh::cpu::DispatchTag::Scalar);
            const float* a_ptr =
                reinterpret_cast<const float*>(a.data);
            const float* b_ptr =
                packed_b.packed_fp32.data();
            float* c_ptr =
                reinterpret_cast<float*>(kernel_dst.data);
            for (int row = 0; row < gemm_case.m; ++row)
            {
                const float* a_row =
                    a_ptr +
                    static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(gemm_case.k);
                float* c_row =
                    c_ptr +
                    static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(gemm_case.n);
                if (use_ui)
                {
                    cvh::detail::gemm_ui::kernel_nn_row_f32(
                        a_row,
                        b_ptr,
                        c_row,
                        gemm_case.n,
                        gemm_case.k);
                }
                else
                {
                    cvh::gemm_kernel_nn_row_scalar(
                        a_row,
                        b_ptr,
                        c_row,
                        gemm_case.n,
                        gemm_case.k);
                }
            }
        };
        append_array_op_row(
            gemm_args,
            shape,
            "GEMM",
            "fp32_nn_kernel_only_" + suffix,
            a_bytes + b_bytes + c_bytes,
            kernel_dst,
            run_kernel,
            rows);
        rows.back().allocation_mode = "precomputed_workspace";
        rows.back().note =
            "component=kernel;packing=excluded;"
            "output_allocation=excluded;shape_dispatch=excluded";

        cvh::Mat b_nt = cvh::transpose(b);
        append_array_op_row(
            gemm_args,
            shape,
            "GEMM",
            "fp32_nt_" + suffix,
            a_bytes + b_bytes + c_bytes,
            dst,
            [&]() {
                dst = cvh::gemm(
                    a, b_nt, false, true);
            },
            rows);
        rows.back().allocation_mode = "recreate";
        rows.back().note =
            "component=public_end_to_end;"
            "output_allocation=included";
    }
}

void print_csv(const std::vector<ResultRow>& rows, std::ostream& os)
{
    common::write_csv_row(
        os,
        {
            "schema_version", "mode", "suite", "module", "op", "variant", "depth", "channels", "layout",
            "shape", "elements", "pixels", "implementation", "dispatch_path", "allocation_mode", "warmup",
            "iters", "repeats", "threads", "min_ms", "median_ms", "mpix_per_sec", "melems_per_sec",
            "gb_per_sec", "checksum", "status", "note",
        });

    for (const auto& row : rows)
    {
        common::write_csv_row(
            os,
            {
                std::to_string(kBenchmarkSchemaVersion),
                row.mode,
                row.suite,
                row.module,
                row.op,
                row.variant,
                row.depth,
                std::to_string(row.channels),
                row.layout,
                row.shape,
                std::to_string(row.elements),
                std::to_string(row.pixels),
                row.implementation,
                row.dispatch_path,
                row.allocation_mode,
                std::to_string(row.warmup),
                std::to_string(row.iters),
                std::to_string(row.repeats),
                std::to_string(row.threads),
                fmt(row.min_ms),
                fmt(row.median_ms),
                fmt(row.mpix_per_sec),
                fmt(row.melems_per_sec),
                fmt(row.gb_per_sec),
                std::to_string(row.checksum),
                row.status,
                row.note,
            });
    }
}

void append_shape_rows(const Args& args, const ShapeCase& shape, std::vector<ResultRow>& rows)
{
    const std::vector<int> dims {shape.rows, shape.cols};
    const std::vector<int> alt_dims {shape.rows + 1, shape.cols};
    const std::size_t bytes = static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols) *
                              static_cast<std::size_t>(CV_MAT_CN(shape.type)) *
                              static_cast<std::size_t>(CV_ELEM_SIZE1(shape.type));

    cvh::Mat src(dims, shape.type);
    fill_mat(src, static_cast<std::uint32_t>(shape.rows * 131 + shape.cols * 17 + CV_MAT_CN(shape.type)));

    {
        cvh::Mat dst(dims, shape.type);
        const auto result = measure(
            [&]() { dst.create(dims, shape.type); },
            [&]() { return common::fnv1a64_mix_u64(common::fnv1a64_basis(), dst.total()); },
            args);
        rows.push_back(make_row(args, shape, "MAT_CREATE", "reuse_same_shape", "none", "reuse", 0, result));
    }

    {
        cvh::Mat dst;
        bool toggle = false;
        const auto result = measure(
            [&]() {
                dst.create(toggle ? dims : alt_dims, shape.type);
                toggle = !toggle;
            },
            [&]() { return common::fnv1a64_mix_u64(common::fnv1a64_basis(), dst.total()); },
            args);
        rows.push_back(make_row(args, shape, "MAT_CREATE", "alternate_shape", "none", "recreate", 0, result));
    }

    {
        cvh::Mat dst(dims, shape.type);
        const auto result = measure(
            [&]() {
                dst.release();
                dst.create(dims, shape.type);
            },
            [&]() { return common::fnv1a64_mix_u64(common::fnv1a64_basis(), dst.total()); },
            args);
        rows.push_back(make_row(args, shape, "MAT_RELEASE_CREATE", "release_then_create", "none", "recreate", 0, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { dst = src.clone(); },
            [&]() { return common::checksum_mat_bytes(dst); },
            args);
        rows.push_back(make_row(args, shape, "MAT_CLONE", "full_copy", "continuous", "recreate", bytes, result));
    }

    {
        cvh::Mat dst;
        const auto result = measure(
            [&]() { src.copyTo(dst); },
            [&]() { return common::checksum_mat_bytes(dst); },
            args);
        rows.push_back(make_row(args, shape, "MAT_COPYTO", "continuous", "continuous", "reuse", bytes, result));
    }

    {
        cvh::Mat parent({shape.rows + 2, shape.cols + 2}, shape.type);
        fill_mat(parent, static_cast<std::uint32_t>(shape.rows * 19 + shape.cols * 23));
        cvh::Mat roi = parent(cvh::Range(1, shape.rows + 1), cvh::Range(1, shape.cols + 1));
        cvh::Mat dst;
        const auto result = measure(
            [&]() { roi.copyTo(dst); },
            [&]() { return common::checksum_mat_bytes(dst); },
            args);
        rows.push_back(make_row(args, shape, "MAT_COPYTO", "roi_to_continuous", "roi", "reuse", bytes, result));
    }

    {
        cvh::Mat dst(dims, shape.type);
        const auto result = measure(
            [&]() { dst.setTo(cvh::Scalar::all(7.0)); },
            [&]() { return common::checksum_mat_bytes(dst); },
            args);
        rows.push_back(make_row(args, shape, "MAT_SETTO", "scalar_all", "continuous", "reuse", bytes, result));
    }

    {
        const int dst_type = CV_MAKETYPE(shape.type == CV_32FC1 ? CV_8U : CV_32F, CV_MAT_CN(shape.type));
        cvh::Mat dst;
        const auto result = measure(
            [&]() { src.convertTo(dst, dst_type); },
            [&]() { return common::checksum_mat_bytes(dst); },
            args);
        rows.push_back(make_row(args, shape, "MAT_CONVERTTO", depth_name(CV_MAT_DEPTH(dst_type)), "continuous", "reuse", bytes, result));
    }

    {
        const std::vector<int> reshaped_dims {shape.rows * shape.cols, 1};
        cvh::Mat view;
        const auto result = measure(
            [&]() { view = src.reshape(reshaped_dims); },
            [&]() { return common::fnv1a64_mix_u64(common::fnv1a64_basis(), view.total()); },
            args);
        rows.push_back(make_row(args, shape, "MAT_RESHAPE", "to_vector_view", "continuous", "none", 0, result));
    }

    cvh::Mat rhs(dims, shape.type);
    fill_mat(rhs, static_cast<std::uint32_t>(shape.rows * 211 + shape.cols * 29 + shape.type));
    const cvh::Scalar scalar_value(3.0, 5.0, 7.0, 11.0);
    cvh::Mat binary_mask(dims, CV_8UC1);
    fill_mat(
        binary_mask,
        static_cast<std::uint32_t>(
            shape.rows * 307 + shape.cols * 43 + shape.type));

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "SCALE_ADD",
            "alpha_0_75",
            bytes * 3,
            dst,
            [&]() { cvh::scaleAdd(src, 0.75, rhs, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "ADD",
            "mat_scalar",
            bytes * 2,
            dst,
            [&]() { cvh::add(src, scalar_value, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "ADD",
            "mat_mat",
            bytes * 3,
            dst,
            [&]() { cvh::add(src, rhs, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "SUBTRACT",
            "scalar_mat",
            bytes * 2,
            dst,
            [&]() { cvh::subtract(scalar_value, src, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "SUBTRACT",
            "mat_mat",
            bytes * 3,
            dst,
            [&]() { cvh::subtract(src, rhs, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "MULTIPLY",
            "mat_scalar",
            bytes * 2,
            dst,
            [&]() { cvh::multiply(src, scalar_value, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "MULTIPLY",
            "mat_mat",
            bytes * 3,
            dst,
            [&]() { cvh::multiply(src, rhs, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "DIVIDE",
            "scalar_mat",
            bytes * 2,
            dst,
            [&]() { cvh::divide(scalar_value, src, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "DIVIDE",
            "mat_mat",
            bytes * 3,
            dst,
            [&]() { cvh::divide(src, rhs, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "ABSDIFF",
            "mat_scalar",
            bytes * 2,
            dst,
            [&]() { cvh::absdiff(src, scalar_value, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "ABSDIFF",
            "mat_mat",
            bytes * 3,
            dst,
            [&]() { cvh::absdiff(src, rhs, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        dst.setTo(cvh::Scalar::all(0xA5));
        append_array_op_row(
            args,
            shape,
            "BITWISE_AND",
            "mat_mat_masked",
            bytes * 3 + binary_mask.total(),
            dst,
            [&]() { cvh::bitwise_and(src, rhs, dst, binary_mask); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        dst.setTo(cvh::Scalar::all(0xA5));
        append_array_op_row(
            args,
            shape,
            "BITWISE_XOR",
            "mat_scalar_masked",
            bytes * 2 + binary_mask.total(),
            dst,
            [&]() {
                cvh::bitwise_xor(src, scalar_value, dst, binary_mask);
            },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "BITWISE_AND",
            "mat_mat_raw_bits",
            bytes * 3,
            dst,
            [&]() { cvh::bitwise_and(src, rhs, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, CV_8UC1);
        const std::size_t mask_bytes =
            static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols);
        append_array_op_row(
            args,
            shape,
            "IN_RANGE",
            "scalar_bounds",
            bytes + mask_bytes,
            dst,
            [&]() {
                cvh::inRange(src, cvh::Scalar::all(-2.5), cvh::Scalar::all(127.5), dst);
            },
            rows);
    }

    {
        cvh::Mat lower(dims, shape.type);
        cvh::Mat upper(dims, shape.type);
        lower.setTo(cvh::Scalar::all(-2.5));
        upper.setTo(cvh::Scalar::all(127.5));
        cvh::Mat dst(dims, CV_8UC1);
        const std::size_t mask_bytes =
            static_cast<std::size_t>(shape.rows) *
            static_cast<std::size_t>(shape.cols);
        append_array_op_row(
            args,
            shape,
            "IN_RANGE",
            "mat_bounds",
            bytes * 3 + mask_bytes,
            dst,
            [&]() { cvh::inRange(src, lower, upper, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "MIN",
            "mat_mat",
            bytes * 3,
            dst,
            [&]() { cvh::min(src, rhs, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "MIN",
            "mat_scalar",
            bytes * 2,
            dst,
            [&]() { cvh::min(src, scalar_value, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "MAX",
            "mat_mat",
            bytes * 3,
            dst,
            [&]() { cvh::max(src, rhs, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, shape.type);
        append_array_op_row(
            args,
            shape,
            "MAX",
            "mat_scalar",
            bytes * 2,
            dst,
            [&]() { cvh::max(src, scalar_value, dst); },
            rows);
    }

    {
        cvh::Mat dst(dims, CV_MAKETYPE(CV_8U, CV_MAT_CN(shape.type)));
        const std::size_t output_bytes =
            static_cast<std::size_t>(shape.rows) * static_cast<std::size_t>(shape.cols) *
            static_cast<std::size_t>(CV_MAT_CN(shape.type));
        append_array_op_row(
            args,
            shape,
            "CONVERT_SCALE_ABS",
            "alpha_1_25_beta_3",
            bytes + output_bytes,
            dst,
            [&]() { cvh::convertScaleAbs(src, dst, 1.25, 3.0); },
            rows);
    }

    if (CV_MAT_DEPTH(shape.type) == CV_32F)
    {
        {
            cvh::Mat dst(dims, CV_MAKETYPE(CV_16S, CV_MAT_CN(shape.type)));
            const std::size_t output_bytes =
                static_cast<std::size_t>(shape.rows) *
                static_cast<std::size_t>(shape.cols) *
                static_cast<std::size_t>(CV_MAT_CN(shape.type)) *
                sizeof(short);
            append_array_op_row(
                args,
                shape,
                "CONVERT_FP16",
                "f32_to_fp16",
                bytes + output_bytes,
                dst,
                [&]() { cvh::convertFp16(src, dst); },
                rows);
        }

        cvh::Mat positive_src = src.clone();
        float* values = reinterpret_cast<float*>(positive_src.data);
        const size_t scalar_count =
            positive_src.total() * static_cast<size_t>(positive_src.channels());
        for (size_t i = 0; i < scalar_count; ++i)
        {
            values[i] = std::fabs(values[i]) + 0.01f;
        }

        {
            cvh::Mat dst(dims, shape.type);
            append_array_op_row(
                args,
                shape,
                "SQRT",
                "positive_f32",
                bytes * 2,
                dst,
                [&]() { cvh::sqrt(positive_src, dst); },
                rows);
        }

        {
            cvh::Mat dst(dims, shape.type);
            append_array_op_row(
                args,
                shape,
                "EXP",
                "f32",
                bytes * 2,
                dst,
                [&]() { cvh::exp(src, dst); },
                rows);
        }

        {
            cvh::Mat dst(dims, shape.type);
            append_array_op_row(
                args,
                shape,
                "LOG",
                "positive_f32",
                bytes * 2,
                dst,
                [&]() { cvh::log(positive_src, dst); },
                rows);
        }

        {
            cvh::Mat dst(dims, shape.type);
            append_array_op_row(
                args,
                shape,
                "POW",
                "power_1_75_f32",
                bytes * 2,
                dst,
                [&]() { cvh::pow(positive_src, 1.75, dst); },
                rows);
        }

        {
            cvh::Mat dst(dims, shape.type);
            append_array_op_row(
                args,
                shape,
                "POW",
                "power_3_f32",
                bytes * 2,
                dst,
                [&]() { cvh::pow(src, 3.0, dst); },
                rows);
        }

        {
            cvh::Mat patched = src.clone();
            append_array_op_row(
                args,
                shape,
                "PATCH_NANS",
                "one_nan_f32",
                bytes * 2,
                patched,
                [&]() {
                    patched.at<float>(0, 0) =
                        std::numeric_limits<float>::quiet_NaN();
                    cvh::patchNaNs(patched, 0.0);
                },
                rows);
        }
    }

    append_reduction_rows(args, shape, src, bytes, rows);
    append_layout_rows(args, shape, src, bytes, rows);
}

ResultRow make_v01_operator_row(const Args& args,
                                const std::string& op,
                                const std::string& variant,
                                const std::string& depth,
                                int channels,
                                const std::string& layout,
                                const std::string& shape,
                                std::size_t elements,
                                std::size_t pixels,
                                std::size_t bytes_touched,
                                const Result& result)
{
    ResultRow row;
    row.op = op;
    row.variant = variant;
    row.depth = depth;
    row.channels = channels;
    row.layout = layout;
    row.shape = shape;
    row.elements = elements;
    row.pixels = pixels;
    row.dispatch_path = "public_header_scalar";
    row.allocation_mode = "reuse";
    row.warmup = args.warmup;
    row.iters = args.iters;
    row.repeats = args.repeats;
    row.threads = args.threads;
    row.min_ms = result.min_ms;
    row.median_ms = result.median_ms;
    row.mpix_per_sec = common::mpix_per_sec(pixels, result.median_ms);
    row.melems_per_sec = result.median_ms > 0.0
        ? static_cast<double>(elements) / result.median_ms / 1000.0
        : 0.0;
    row.gb_per_sec = result.median_ms > 0.0
        ? static_cast<double>(bytes_touched) / result.median_ms / 1.0e6
        : 0.0;
    row.checksum = result.checksum;
    row.note = "scalar baseline; no fast-path claim";
    return row;
}

template<typename RunFn, typename ChecksumFn>
void append_v01_operator_row(const Args& args,
                             ResultRow row,
                             std::size_t bytes_touched,
                             RunFn&& run_once,
                             ChecksumFn&& checksum,
                             std::vector<ResultRow>& rows)
{
    run_once();
    const Result result = measure(
        std::forward<RunFn>(run_once),
        std::forward<ChecksumFn>(checksum),
        args);
    row.warmup = args.warmup;
    row.iters = args.iters;
    row.repeats = args.repeats;
    row.threads = args.threads;
    row.min_ms = result.min_ms;
    row.median_ms = result.median_ms;
    row.mpix_per_sec = common::mpix_per_sec(row.pixels, result.median_ms);
    row.melems_per_sec = result.median_ms > 0.0
        ? static_cast<double>(row.elements) / result.median_ms / 1000.0
        : 0.0;
    row.gb_per_sec = result.median_ms > 0.0
        ? static_cast<double>(bytes_touched) / result.median_ms / 1.0e6
        : 0.0;
    row.checksum = result.checksum;
    rows.push_back(std::move(row));
}

void append_v01_operator_rows(const Args& args, std::vector<ResultRow>& rows)
{
    const int matrix_rows = args.profile == "quick" ? 120 : 240;
    const int matrix_cols = args.profile == "quick" ? 160 : 320;
    const int point_count = args.profile == "full" ? 16384 : 4096;
    const std::string matrix_shape =
        std::to_string(matrix_cols) + "x" + std::to_string(matrix_rows);

    for (int type : {CV_8UC3, CV_32FC3})
    {
        cvh::Mat matrix({matrix_rows, matrix_cols}, type);
        const std::string depth = matrix.depth() == CV_8U ? "CV_8U" : "CV_32F";
        const std::size_t elements = matrix.total() * 3;
        const std::size_t bytes = matrix.total() * matrix.elemSize();
        ResultRow randu_row = make_v01_operator_row(
            args, "RANDU", "C3", depth, 3, "continuous", matrix_shape,
            elements, matrix.total(), bytes, Result {});
        append_v01_operator_row(
            args,
            std::move(randu_row),
            bytes,
            [&]() {
                cvh::randu(
                    matrix,
                    cvh::Scalar::all(0.0),
                    cvh::Scalar::all(127.0));
            },
            [&]() { return common::checksum_mat_bytes(matrix); },
            rows);
        rows.back().dispatch_path = "header_fastpath";
        rows.back().note =
            "lightweight 64-bit engine; channel-unrolled typed span; persistent distributions";

        ResultRow randn_row = make_v01_operator_row(
            args, "RANDN", "C3", depth, 3, "continuous", matrix_shape,
            elements, matrix.total(), bytes, Result {});
        append_v01_operator_row(
            args,
            std::move(randn_row),
            bytes,
            [&]() {
                cvh::randn(
                    matrix,
                    cvh::Scalar::all(64.0),
                    cvh::Scalar::all(12.0));
            },
            [&]() { return common::checksum_mat_bytes(matrix); },
            rows);
        rows.back().dispatch_path = "header_fastpath";
        rows.back().note =
            "lightweight 64-bit engine; channel-unrolled typed span; persistent distributions";
    }

    cvh::Mat storage({matrix_rows + 2, matrix_cols + 3}, CV_8UC1);
    cvh::Mat roi = storage(
        cvh::Range(1, matrix_rows + 1),
        cvh::Range(2, matrix_cols + 2));
    ResultRow roi_row = make_v01_operator_row(
        args, "RANDU", "C1", "CV_8U", 1, "roi", matrix_shape,
        roi.total(), roi.total(), roi.total() * roi.elemSize(), Result {});
    append_v01_operator_row(
        args,
        std::move(roi_row),
        roi.total() * roi.elemSize(),
        [&]() {
            cvh::randu(roi, cvh::Scalar(0.0), cvh::Scalar(256.0));
        },
        [&]() { return common::checksum_mat_bytes(roi); },
        rows);
    rows.back().dispatch_path = "header_fastpath";
    rows.back().note =
        "lightweight 64-bit engine; typed ROI rows; distribution setup hoisted";

    cvh::Mat source({point_count, 1}, CV_32FC3);
    common::fill_mat_f32_lcg(source, 0x8102u);
    cvh::Mat affine({4, 4}, CV_64FC1);
    affine = 0.0f;
    for (int index = 0; index < 4; ++index)
    {
        affine.at<double>(index, index) = 1.0;
    }
    cvh::Mat destination;
    const std::string point_shape = std::to_string(point_count) + "x1";
    ResultRow transform_row = make_v01_operator_row(
        args, "TRANSFORM", "F32_C3_TO_C4", "CV_32F", 3, "continuous",
        point_shape, static_cast<std::size_t>(point_count) * 4,
        static_cast<std::size_t>(point_count),
        source.total() * source.elemSize(), Result {});
    append_v01_operator_row(
        args,
        std::move(transform_row),
        source.total() * source.elemSize(),
        [&]() { cvh::transform(source, destination, affine); },
        [&]() { return common::checksum_mat_bytes(destination); },
        rows);
    rows.back().dispatch_path = "header_fastpath";
    rows.back().note =
        "prepacked coefficients; channel-specialized continuous span";

    cvh::Mat perspective({4, 4}, CV_64FC1);
    perspective = 0.0f;
    for (int index = 0; index < 4; ++index)
    {
        perspective.at<double>(index, index) = 1.0;
    }
    ResultRow perspective_row = make_v01_operator_row(
        args, "PERSPECTIVE_TRANSFORM", "F32_C3", "CV_32F", 3,
        "continuous", point_shape,
        static_cast<std::size_t>(point_count) * 3,
        static_cast<std::size_t>(point_count),
        source.total() * source.elemSize(), Result {});
    append_v01_operator_row(
        args,
        std::move(perspective_row),
        source.total() * source.elemSize(),
        [&]() {
            cvh::perspectiveTransform(source, destination, perspective);
        },
        [&]() { return common::checksum_mat_bytes(destination); },
        rows);
    rows.back().dispatch_path = "header_fastpath";
    rows.back().note =
        "prepacked coefficients; C3 continuous point span";
}

}  // namespace cvh_bench

int main(int argc, char** argv)
{
    const auto args = cvh_bench::parse_args(argc, argv);
    cvh::cpu::set_dispatch_mode(
        args.dispatch == "scalar"
            ? cvh::cpu::DispatchMode::ScalarOnly
            : cvh::cpu::DispatchMode::Auto);
    cvh::setNumThreads(args.threads);
    const auto shapes = cvh_bench::build_shapes(args.profile);
    std::vector<cvh_bench::ResultRow> rows;
    rows.reserve(shapes.size() * 36);

    if (args.ops.empty())
    {
        for (const auto& shape : shapes)
        {
            cvh_bench::append_shape_rows(args, shape, rows);
        }
        cvh_bench::append_v01_operator_rows(args, rows);
    }
    cvh_bench::append_gemm_rows(args, rows);

    cvh_bench::print_csv(rows, std::cout);
    if (!args.output_csv.empty())
    {
        std::ofstream ofs(args.output_csv);
        if (!ofs)
        {
            std::cerr << "Failed to open output: " << args.output_csv << "\n";
            return 4;
        }
        cvh_bench::print_csv(rows, ofs);
    }

    return static_cast<int>(cvh_bench::g_sink & 0u);
}
