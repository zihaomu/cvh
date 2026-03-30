#include "cvh.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace cvh_bench {

using cvh::BinaryOp;

enum class BenchOp
{
    Add = 0,
    Sub,
    Mul,
    Div,
    Mean,
    Max,
    Min,
    And,
    Or,
    Xor,
    Not,
    Mod,
    Bitshift,
    Fmod,
    Atan2,
    Hypot
};

struct Args
{
    std::string profile = "quick";
    int warmup = 2;
    int iters = 20;
    int repeats = 5;
    std::string output_csv;
};

struct ShapeCase
{
    std::string name;
    std::vector<int> dims;
};

struct ResultRow
{
    std::string profile;
    std::string op;
    std::string depth;
    int channels = 0;
    std::string shape;
    std::size_t elements = 0;
    double ms_per_iter = 0.0;
    double melems_per_sec = 0.0;
    double gb_per_sec = 0.0;
};

volatile double g_sink = 0.0;

std::string depth_to_name(int depth)
{
    switch (depth)
    {
        case CV_8U: return "CV_8U";
        case CV_8S: return "CV_8S";
        case CV_16U: return "CV_16U";
        case CV_16S: return "CV_16S";
        case CV_32S: return "CV_32S";
        case CV_32U: return "CV_32U";
        case CV_32F: return "CV_32F";
        case CV_16F: return "CV_16F";
        default: return "UNKNOWN";
    }
}

std::string shape_to_string(const std::vector<int>& dims)
{
    std::ostringstream oss;
    for (std::size_t i = 0; i < dims.size(); ++i)
    {
        if (i > 0)
        {
            oss << "x";
        }
        oss << dims[i];
    }
    return oss.str();
}

const char* op_name(BenchOp op)
{
    switch (op)
    {
        case BenchOp::Add: return "ADD";
        case BenchOp::Sub: return "SUB";
        case BenchOp::Mul: return "MUL";
        case BenchOp::Div: return "DIV";
        case BenchOp::Mean: return "MEAN";
        case BenchOp::Max: return "MAX";
        case BenchOp::Min: return "MIN";
        case BenchOp::And: return "AND";
        case BenchOp::Or: return "OR";
        case BenchOp::Xor: return "XOR";
        case BenchOp::Not: return "NOT";
        case BenchOp::Mod: return "MOD";
        case BenchOp::Bitshift: return "BITSHIFT";
        case BenchOp::Fmod: return "FMOD";
        case BenchOp::Atan2: return "ATAN2";
        case BenchOp::Hypot: return "HYPOT";
    }

    return "UNKNOWN";
}

BinaryOp to_binary_op(BenchOp op)
{
    switch (op)
    {
        case BenchOp::Add: return BinaryOp::ADD;
        case BenchOp::Sub: return BinaryOp::SUB;
        case BenchOp::Mul: return BinaryOp::MUL;
        case BenchOp::Div: return BinaryOp::DIV;
        case BenchOp::Mean: return BinaryOp::MEAN;
        case BenchOp::Max: return BinaryOp::MAX;
        case BenchOp::Min: return BinaryOp::MIN;
        case BenchOp::And: return BinaryOp::AND;
        case BenchOp::Or: return BinaryOp::OR;
        case BenchOp::Xor: return BinaryOp::XOR;
        case BenchOp::Not: return BinaryOp::NOT;
        case BenchOp::Mod: return BinaryOp::MOD;
        case BenchOp::Bitshift: return BinaryOp::BITSHIFT;
        case BenchOp::Fmod: return BinaryOp::FMOD;
        case BenchOp::Atan2: return BinaryOp::ATAN2;
        case BenchOp::Hypot: return BinaryOp::HYPOT;
    }

    return BinaryOp::ADD;
}

bool is_integral_depth(int depth)
{
    return depth == CV_8U || depth == CV_8S || depth == CV_16U || depth == CV_16S ||
           depth == CV_32S || depth == CV_32U;
}

bool is_float_depth(int depth)
{
    return depth == CV_32F || depth == CV_16F;
}

bool op_supported_for_depth(BenchOp op, int depth)
{
    switch (op)
    {
        case BenchOp::And:
        case BenchOp::Or:
        case BenchOp::Xor:
        case BenchOp::Not:
        case BenchOp::Mod:
        case BenchOp::Bitshift:
            return is_integral_depth(depth);
        case BenchOp::Fmod:
        case BenchOp::Atan2:
        case BenchOp::Hypot:
            return is_float_depth(depth);
        default:
            return true;
    }
}

double seed_value(std::size_t idx, int depth, bool rhs, BenchOp op)
{
    if (is_float_depth(depth))
    {
        double base = rhs ? static_cast<double>((idx * 17u) % 401u) * 0.05 - 10.0
                          : static_cast<double>((idx * 131u) % 401u) * 0.05 - 10.0;
        if (op == BenchOp::Bitshift)
        {
            base = static_cast<double>((idx % 4u) + 1u);
        }
        if (rhs && (op == BenchOp::Div || op == BenchOp::Mod || op == BenchOp::Fmod) &&
            std::abs(base) < 1e-6)
        {
            base = 1.25;
        }
        return base;
    }

    if (depth == CV_8U || depth == CV_16U || depth == CV_32U)
    {
        double base = rhs ? static_cast<double>((idx * 17u) % 251u + 1u)
                          : static_cast<double>((idx * 131u) % 251u + 1u);
        if (op == BenchOp::Bitshift && rhs)
        {
            base = static_cast<double>((idx % 4u) + 1u);
        }
        return base;
    }

    double base = rhs ? static_cast<double>((idx * 17u) % 255u) - 127.0
                      : static_cast<double>((idx * 131u) % 255u) - 127.0;
    if (op == BenchOp::Bitshift && rhs)
    {
        base = static_cast<double>((idx % 4u) + 1u);
    }
    if (rhs && (op == BenchOp::Div || op == BenchOp::Mod || op == BenchOp::Fmod) &&
        std::abs(base) < 1e-9)
    {
        base = 3.0;
    }
    return base;
}

void fill_mat(cvh::Mat& mat, bool rhs, BenchOp op)
{
    const std::size_t count = mat.total() * static_cast<std::size_t>(mat.channels());
    switch (mat.depth())
    {
        case CV_8U:
        {
            auto* ptr = reinterpret_cast<uchar*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<uchar>(seed_value(i, CV_8U, rhs, op));
            break;
        }
        case CV_8S:
        {
            auto* ptr = reinterpret_cast<schar*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<schar>(seed_value(i, CV_8S, rhs, op));
            break;
        }
        case CV_16U:
        {
            auto* ptr = reinterpret_cast<ushort*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<ushort>(seed_value(i, CV_16U, rhs, op));
            break;
        }
        case CV_16S:
        {
            auto* ptr = reinterpret_cast<short*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<short>(seed_value(i, CV_16S, rhs, op));
            break;
        }
        case CV_32S:
        {
            auto* ptr = reinterpret_cast<int*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<int>(seed_value(i, CV_32S, rhs, op));
            break;
        }
        case CV_32U:
        {
            auto* ptr = reinterpret_cast<uint*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<uint>(seed_value(i, CV_32U, rhs, op));
            break;
        }
        case CV_32F:
        {
            auto* ptr = reinterpret_cast<float*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<float>(seed_value(i, CV_32F, rhs, op));
            break;
        }
        case CV_16F:
        {
            auto* ptr = reinterpret_cast<hfloat*>(mat.data);
            for (std::size_t i = 0; i < count; ++i) ptr[i] = cvh::saturate_cast<hfloat>(seed_value(i, CV_16F, rhs, op));
            break;
        }
        default:
            break;
    }
}

double probe_checksum(const cvh::Mat& mat)
{
    const std::size_t count = mat.total() * static_cast<std::size_t>(mat.channels());
    if (count == 0)
    {
        return 0.0;
    }

    const std::size_t idx0 = 0;
    const std::size_t idx1 = count / 2;
    const std::size_t idx2 = count - 1;

    auto sample = [&](std::size_t idx) -> double {
        switch (mat.depth())
        {
            case CV_8U: return static_cast<double>(reinterpret_cast<const uchar*>(mat.data)[idx]);
            case CV_8S: return static_cast<double>(reinterpret_cast<const schar*>(mat.data)[idx]);
            case CV_16U: return static_cast<double>(reinterpret_cast<const ushort*>(mat.data)[idx]);
            case CV_16S: return static_cast<double>(reinterpret_cast<const short*>(mat.data)[idx]);
            case CV_32S: return static_cast<double>(reinterpret_cast<const int*>(mat.data)[idx]);
            case CV_32U: return static_cast<double>(reinterpret_cast<const uint*>(mat.data)[idx]);
            case CV_32F: return static_cast<double>(reinterpret_cast<const float*>(mat.data)[idx]);
            case CV_16F: return static_cast<double>(static_cast<float>(reinterpret_cast<const hfloat*>(mat.data)[idx]));
            default: return 0.0;
        }
    };

    return sample(idx0) + sample(idx1) + sample(idx2);
}

void run_one_op(BenchOp op, const cvh::Mat& a, const cvh::Mat& b, cvh::Mat& out)
{
    cvh::binaryFunc(to_binary_op(op), a, b, out);
}

std::vector<ShapeCase> build_shapes(const std::string& profile)
{
    if (profile == "full")
    {
        return {
            {"1d_1m", {1024 * 1024}},
            {"2d_vga", {480, 640}},
            {"2d_fhd", {1080, 1920}},
            {"3d_8x256x256", {8, 256, 256}},
        };
    }

    return {
        {"1d_1m", {1024 * 1024}},
        {"2d_hd", {720, 1280}},
        {"3d_8x256x256", {8, 256, 256}},
    };
}

std::vector<int> build_channels(const std::string& profile)
{
    if (profile == "full")
    {
        return {1, 3, 4};
    }
    return {1, 3};
}

std::vector<int> build_depths(const std::string& profile)
{
    if (profile == "full")
    {
        return {CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32U, CV_32F, CV_16F};
    }
    return {CV_8U, CV_32S, CV_32F, CV_16F};
}

std::vector<BenchOp> build_ops(const std::string& profile)
{
    if (profile == "full")
    {
        return {
            BenchOp::Add, BenchOp::Sub, BenchOp::Mul, BenchOp::Div,
            BenchOp::Mean, BenchOp::Max, BenchOp::Min,
            BenchOp::And, BenchOp::Or, BenchOp::Xor, BenchOp::Not,
            BenchOp::Mod, BenchOp::Bitshift, BenchOp::Fmod,
            BenchOp::Atan2, BenchOp::Hypot};
    }
    return {
        BenchOp::Add, BenchOp::Sub, BenchOp::Mul, BenchOp::Div,
        BenchOp::Mean, BenchOp::Max, BenchOp::Min,
        BenchOp::And, BenchOp::Xor, BenchOp::Mod, BenchOp::Fmod};
}

Args parse_args(int argc, char** argv)
{
    Args args;
    for (int i = 1; i < argc; ++i)
    {
        const std::string token(argv[i]);
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
        else if (token == "--output")
        {
            args.output_csv = next_value("--output");
        }
        else if (token == "--help")
        {
            std::cout
                << "Usage: cvh_benchmark_core_ops [--profile quick|full] [--warmup N] [--iters N] [--repeats N] [--output path]\n";
            std::exit(0);
        }
        else
        {
            std::cerr << "Unknown arg: " << token << "\n";
            std::exit(2);
        }
    }

    if (args.profile != "quick" && args.profile != "full")
    {
        std::cerr << "Unsupported profile: " << args.profile << " (expected quick/full)\n";
        std::exit(2);
    }

    return args;
}

double measure_case(BenchOp op, const cvh::Mat& a, const cvh::Mat& b, cvh::Mat& out, int warmup, int iters, int repeats)
{
    for (int i = 0; i < warmup; ++i)
    {
        run_one_op(op, a, b, out);
    }

    std::vector<double> samples_ms_per_iter;
    samples_ms_per_iter.reserve(static_cast<std::size_t>(repeats));

    using Clock = std::chrono::steady_clock;
    for (int r = 0; r < repeats; ++r)
    {
        const auto t0 = Clock::now();
        for (int i = 0; i < iters; ++i)
        {
            run_one_op(op, a, b, out);
        }
        const auto t1 = Clock::now();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        samples_ms_per_iter.push_back(elapsed_ms / static_cast<double>(iters));
    }

    std::sort(samples_ms_per_iter.begin(), samples_ms_per_iter.end());
    const double median_ms = samples_ms_per_iter[samples_ms_per_iter.size() / 2];
    g_sink += probe_checksum(out);
    return median_ms;
}

void print_csv(const std::vector<ResultRow>& rows, std::ostream& os)
{
    os << "profile,op,depth,channels,shape,elements,ms_per_iter,melems_per_sec,gb_per_sec\n";
    os << std::fixed << std::setprecision(6);
    for (const auto& row : rows)
    {
        os << row.profile << ","
           << row.op << ","
           << row.depth << ","
           << row.channels << ","
           << row.shape << ","
           << row.elements << ","
           << row.ms_per_iter << ","
           << row.melems_per_sec << ","
           << row.gb_per_sec << "\n";
    }
}

}  // namespace cvh_bench

int main(int argc, char** argv)
{
    const auto args = cvh_bench::parse_args(argc, argv);
    const auto shapes = cvh_bench::build_shapes(args.profile);
    const auto channels = cvh_bench::build_channels(args.profile);
    const auto depths = cvh_bench::build_depths(args.profile);
    const auto ops = cvh_bench::build_ops(args.profile);

    std::vector<cvh_bench::ResultRow> rows;
    rows.reserve(512);

    for (const auto& shape_case : shapes)
    {
        for (const int cn : channels)
        {
            for (const int depth : depths)
            {
                const int type = CV_MAKETYPE(depth, cn);
                cvh::Mat a(shape_case.dims, type);
                cvh::Mat b(shape_case.dims, type);
                cvh::Mat out(shape_case.dims, type);

                for (const auto op : ops)
                {
                    if (!cvh_bench::op_supported_for_depth(op, depth))
                    {
                        continue;
                    }

                    cvh_bench::fill_mat(a, false, op);
                    cvh_bench::fill_mat(b, true, op);

                    const double ms_per_iter =
                        cvh_bench::measure_case(op, a, b, out, args.warmup, args.iters, args.repeats);
                    const std::size_t elements = out.total() * static_cast<std::size_t>(out.channels());
                    const std::size_t bytes_per_iter = elements * static_cast<std::size_t>(CV_ELEM_SIZE1(type)) * 3u;
                    const double sec = ms_per_iter / 1000.0;
                    const double melems_per_sec = elements / sec / 1e6;
                    const double gb_per_sec = bytes_per_iter / sec / 1e9;

                    rows.push_back({
                        args.profile,
                        cvh_bench::op_name(op),
                        cvh_bench::depth_to_name(depth),
                        cn,
                        cvh_bench::shape_to_string(shape_case.dims),
                        elements,
                        ms_per_iter,
                        melems_per_sec,
                        gb_per_sec,
                    });
                }
            }
        }
    }

    cvh_bench::print_csv(rows, std::cout);

    if (!args.output_csv.empty())
    {
        std::ofstream ofs(args.output_csv);
        if (!ofs.is_open())
        {
            std::cerr << "Failed to open output file: " << args.output_csv << "\n";
            return 3;
        }
        cvh_bench::print_csv(rows, ofs);
    }

    if (cvh_bench::g_sink == -1.0)
    {
        std::cerr << "unreachable\n";
    }

    return 0;
}
