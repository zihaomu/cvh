#include "pipeline_compare_backend.h"
#include "common/benchmark_common.h"
#include "cvh/core/detail/dispatch_control.h"
#include "cvh/pipeline/pipeline.h"
#include "cvh/recipes/model_input.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef CVH_PIPELINE_PROOF_MANIFEST
#error "CVH_PIPELINE_PROOF_MANIFEST must identify the case manifest"
#endif

namespace cvh_pipeline_proof {
namespace {

struct Args
{
    std::string profile = "quick";
    std::string cache_mode = "both";
    std::string manifest = CVH_PIPELINE_PROOF_MANIFEST;
    std::string output_csv;
    int warmup = 3;
    int iters = 3;
    int repeats = 3;
    int threads = 1;
    int session = 1;
    int ring_mib = 64;
};

struct ErrorMetrics
{
    bool valid = false;
    double max_absolute = 0.0;
    double max_relative = 0.0;
    std::size_t different_values = 0;
    std::size_t first_difference = std::numeric_limits<std::size_t>::max();
};

struct Summary
{
    double minimum_ms = 0.0;
    double p50_ms = 0.0;
    double p90_ms = 0.0;
    double p95_ms = 0.0;
    double maximum_ms = 0.0;
    double coefficient_of_variation = 0.0;
};

struct ImplementationResult
{
    std::string name;
    std::string algorithm_path;
    std::string candidate_route;
    std::string actual_route;
    std::string observed_isa;
    std::string fallback_reason;
    int semantic_stages = -1;
    int execution_groups = -1;
    int full_frame_intermediates = -1;
    int allocations_per_run = -1;
    std::size_t workspace_bytes = 0;
    std::size_t explicit_temporary_bytes = 0;
    std::uint64_t checksum = 0;
    ErrorMetrics validation{};
    std::vector<double> samples_ms;
    Summary summary{};
};

volatile std::uint64_t g_sink = 0;

void usage()
{
    std::cout
        << "Usage: cvh_benchmark_pipeline_compare "
        << "[--profile quick|stable|full] "
        << "[--cache-mode hot|streaming|both] "
        << "[--warmup N] [--iters N] [--repeats N] "
        << "[--threads 1] [--session N] [--ring-mib N] "
        << "[--manifest path] [--output path]\n";
}

Args parseArgs(int argc, char** argv)
{
    Args args;
    for (int index = 1; index < argc; ++index)
    {
        const std::string token = argv[index];
        auto next = [&](const char* option) {
            if (index + 1 >= argc)
            {
                throw std::invalid_argument(
                    std::string("missing value for ") + option);
            }
            return std::string(argv[++index]);
        };
        if (token == "--profile")
        {
            args.profile = next("--profile");
        }
        else if (token == "--cache-mode")
        {
            args.cache_mode = next("--cache-mode");
        }
        else if (token == "--warmup")
        {
            args.warmup = std::max(0, std::stoi(next("--warmup")));
        }
        else if (token == "--iters")
        {
            args.iters = std::max(1, std::stoi(next("--iters")));
        }
        else if (token == "--repeats")
        {
            args.repeats = std::max(1, std::stoi(next("--repeats")));
        }
        else if (token == "--threads")
        {
            args.threads = std::max(1, std::stoi(next("--threads")));
        }
        else if (token == "--session")
        {
            args.session = std::max(1, std::stoi(next("--session")));
        }
        else if (token == "--ring-mib")
        {
            args.ring_mib = std::max(1, std::stoi(next("--ring-mib")));
        }
        else if (token == "--manifest")
        {
            args.manifest = next("--manifest");
        }
        else if (token == "--output")
        {
            args.output_csv = next("--output");
        }
        else if (token == "--help")
        {
            usage();
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("unknown argument: " + token);
        }
    }
    if (args.profile != "quick" && args.profile != "stable" &&
        args.profile != "full")
    {
        throw std::invalid_argument("unsupported profile: " + args.profile);
    }
    if (args.cache_mode != "hot" && args.cache_mode != "streaming" &&
        args.cache_mode != "both")
    {
        throw std::invalid_argument(
            "cache mode must be hot, streaming, or both");
    }
    if (args.threads != 1)
    {
        throw std::invalid_argument(
            "Pipeline proof primary profile currently requires threads=1");
    }
    return args;
}

std::vector<std::string> splitCsv(const std::string& line)
{
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (begin <= line.size())
    {
        const std::size_t end = line.find(',', begin);
        fields.push_back(line.substr(
            begin,
            end == std::string::npos ? std::string::npos : end - begin));
        if (end == std::string::npos)
        {
            break;
        }
        begin = end + 1;
    }
    return fields;
}

int profileRank(const std::string& profile)
{
    if (profile == "quick")
    {
        return 0;
    }
    if (profile == "stable")
    {
        return 1;
    }
    if (profile == "full")
    {
        return 2;
    }
    throw std::invalid_argument("invalid profile in manifest: " + profile);
}

CaseSpec parseCase(const std::vector<std::string>& fields, int line_number)
{
    if (fields.size() != 20)
    {
        throw std::runtime_error(
            "Pipeline manifest line " + std::to_string(line_number) +
            " has " + std::to_string(fields.size()) +
            " columns; expected 20");
    }
    CaseSpec spec;
    spec.id = fields[0];
    spec.min_profile = fields[1];
    spec.input_width = std::stoi(fields[2]);
    spec.input_height = std::stoi(fields[3]);
    if (fields[4] == "BGR8")
    {
        spec.input_format = InputFormat::BGR8;
    }
    else if (fields[4] == "RGB8")
    {
        spec.input_format = InputFormat::RGB8;
    }
    else
    {
        throw std::runtime_error("unsupported input format: " + fields[4]);
    }
    spec.output_width = std::stoi(fields[5]);
    spec.output_height = std::stoi(fields[6]);
    if (fields[7] != "F32")
    {
        throw std::runtime_error(
            "E1 manifest supports only F32 cases, got " + fields[7]);
    }
    if (fields[8] == "NCHW")
    {
        spec.output_layout = OutputLayout::NCHW;
    }
    else if (fields[8] == "NHWC")
    {
        spec.output_layout = OutputLayout::NHWC;
    }
    else
    {
        throw std::runtime_error("unsupported layout: " + fields[8]);
    }
    if (fields[9] == "resize")
    {
        spec.geometry = Geometry::Resize;
    }
    else if (fields[9] == "letterbox")
    {
        spec.geometry = Geometry::Letterbox;
    }
    else
    {
        throw std::runtime_error("unsupported geometry: " + fields[9]);
    }
    if (fields[10] == "nearest")
    {
        spec.interpolation = Interpolation::Nearest;
    }
    else if (fields[10] == "linear")
    {
        spec.interpolation = Interpolation::Linear;
    }
    else
    {
        throw std::runtime_error(
            "unsupported interpolation: " + fields[10]);
    }
    if (fields[11] != "RGB")
    {
        throw std::runtime_error(
            "E1 manifest supports only RGB target color");
    }
    spec.pad_value = std::stof(fields[12]);
    for (int channel = 0; channel < 3; ++channel)
    {
        spec.mean[static_cast<std::size_t>(channel)] =
            std::stof(fields[static_cast<std::size_t>(13 + channel)]);
        spec.stddev[static_cast<std::size_t>(channel)] =
            std::stof(fields[static_cast<std::size_t>(16 + channel)]);
    }
    spec.primary = fields[19] == "true";
    if (spec.id.empty() || spec.input_width <= 0 ||
        spec.input_height <= 0 || spec.output_width <= 0 ||
        spec.output_height <= 0)
    {
        throw std::runtime_error("invalid dimensions/id in manifest");
    }
    for (float stddev : spec.stddev)
    {
        if (!std::isfinite(stddev) || stddev == 0.0f)
        {
            throw std::runtime_error("invalid stddev in manifest");
        }
    }
    return spec;
}

std::vector<CaseSpec> loadCases(const Args& args)
{
    std::ifstream input(args.manifest);
    if (!input)
    {
        throw std::runtime_error(
            "failed to open Pipeline proof manifest: " + args.manifest);
    }
    std::string line;
    if (!std::getline(input, line))
    {
        throw std::runtime_error("Pipeline proof manifest is empty");
    }
    const std::vector<std::string> header = splitCsv(line);
    if (header.size() != 20 || header[0] != "case_id" ||
        header[19] != "primary")
    {
        throw std::runtime_error(
            "Pipeline proof manifest header/schema is invalid");
    }

    std::vector<CaseSpec> cases;
    int line_number = 1;
    while (std::getline(input, line))
    {
        ++line_number;
        if (line.empty() || line[0] == '#')
        {
            continue;
        }
        CaseSpec spec = parseCase(splitCsv(line), line_number);
        if (profileRank(spec.min_profile) <= profileRank(args.profile))
        {
            cases.push_back(std::move(spec));
        }
    }
    if (cases.empty())
    {
        throw std::runtime_error("Pipeline proof profile selected no cases");
    }
    return cases;
}

const char* inputFormatName(InputFormat value)
{
    return value == InputFormat::BGR8 ? "BGR8" : "RGB8";
}

const char* layoutName(OutputLayout value)
{
    return value == OutputLayout::NCHW ? "NCHW" : "NHWC";
}

const char* geometryName(Geometry value)
{
    return value == Geometry::Resize ? "resize" : "letterbox";
}

const char* interpolationName(Interpolation value)
{
    return value == Interpolation::Nearest ? "nearest" : "linear";
}

const char* routeName(cvh::PipelineRoute route)
{
    switch (route)
    {
    case cvh::PipelineRoute::Scalar:
        return "scalar";
    case cvh::PipelineRoute::UniversalIntrinsics:
        return "universal_intrinsics";
    case cvh::PipelineRoute::Neon:
        return "neon";
    case cvh::PipelineRoute::Avx2:
        return "avx2";
    default:
        return "unknown";
    }
}

cvh::Interpolation cvhInterpolation(const CaseSpec& spec)
{
    return spec.interpolation == Interpolation::Nearest
        ? cvh::Interpolation::Nearest
        : cvh::Interpolation::Linear;
}

cvh::Layout cvhLayout(const CaseSpec& spec)
{
    return spec.output_layout == OutputLayout::NCHW
        ? cvh::Layout::NCHW
        : cvh::Layout::NHWC;
}

cvh::PixelFormat cvhPixelFormat(const CaseSpec& spec)
{
    return spec.input_format == InputFormat::BGR8
        ? cvh::PixelFormat::BGR8
        : cvh::PixelFormat::RGB8;
}

cvh::TensorDescriptor outputDescriptor(const CaseSpec& spec)
{
    return spec.output_layout == OutputLayout::NCHW
        ? cvh::tensorDesc<float>(
              {1, 3, spec.output_height, spec.output_width},
              cvh::Layout::NCHW)
        : cvh::tensorDesc<float>(
              {1, spec.output_height, spec.output_width, 3},
              cvh::Layout::NHWC);
}

cvh::PipelinePlan makeFusedPlan(const CaseSpec& spec)
{
    cvh::ModelInputRecipe recipe;
    recipe.input = cvh::imageDesc(
        spec.input_width, spec.input_height, cvhPixelFormat(spec));
    recipe.output = outputDescriptor(spec);
    recipe.color = cvh::Color::RGB;
    recipe.interpolation = cvhInterpolation(spec);
    recipe.geometry = spec.geometry == Geometry::Resize
        ? cvh::ModelInputGeometry::Resize
        : cvh::ModelInputGeometry::Letterbox;
    recipe.letterbox_pad_value[0] = spec.pad_value;
    recipe.letterbox_pad_count = 1;
    recipe.mean = {{spec.mean[0], spec.mean[1], spec.mean[2], 0.0f}};
    recipe.stddev =
        {{spec.stddev[0], spec.stddev[1], spec.stddev[2], 1.0f}};
    recipe.normalize_count = 3;
    return cvh::recipes::modelInput(recipe).prepare();
}

cvh::PipelinePlan makeStagedPlan(const CaseSpec& spec)
{
    cvh::PipelineBuilder builder = cvh::pipe(
        cvh::imageDesc(
            spec.input_width, spec.input_height, cvhPixelFormat(spec)),
        outputDescriptor(spec));
    if (spec.geometry == Geometry::Resize)
    {
        builder.resize(
            spec.output_width,
            spec.output_height,
            cvhInterpolation(spec));
    }
    else
    {
        builder.letterbox(
            spec.output_width,
            spec.output_height,
            spec.pad_value,
            cvhInterpolation(spec));
    }
    builder
        .color(cvh::Color::RGB)
        .normalize(spec.mean, spec.stddev)
        .layout(cvhLayout(spec));
    return builder.prepare();
}

class CvhRunner
{
public:
    CvhRunner(const CaseSpec& spec,
              const std::vector<const std::uint8_t*>& inputs,
              std::size_t input_row_stride,
              const std::vector<float*>& outputs,
              bool fused)
        : spec_(spec),
          inputs_(inputs),
          input_row_stride_(input_row_stride),
          outputs_(outputs),
          plan_(fused ? makeFusedPlan(spec) : makeStagedPlan(spec)),
          workspace_(new cvh::PipelineWorkspace(plan_))
    {
    }

    void setMode(cvh::cpu::DispatchMode mode)
    {
        cvh::cpu::set_dispatch_mode(mode);
    }

    void run(std::size_t frame_index)
    {
        const std::size_t index = frame_index % inputs_.size();
        const std::size_t input_bytes =
            input_row_stride_ * static_cast<std::size_t>(spec_.input_height);
        const cvh::ConstImageView input =
            spec_.input_format == InputFormat::BGR8
            ? cvh::bgr(
                  inputs_[index],
                  input_bytes,
                  spec_.input_width,
                  spec_.input_height,
                  input_row_stride_)
            : cvh::rgb(
                  inputs_[index],
                  input_bytes,
                  spec_.input_width,
                  spec_.input_height,
                  input_row_stride_);
        const std::size_t output_bytes =
            static_cast<std::size_t>(spec_.output_width) *
            static_cast<std::size_t>(spec_.output_height) * 3 *
            sizeof(float);
        cvh::TensorView output =
            spec_.output_layout == OutputLayout::NCHW
            ? cvh::nchw(
                  outputs_[index],
                  output_bytes,
                  1,
                  3,
                  spec_.output_height,
                  spec_.output_width)
            : cvh::nhwc(
                  outputs_[index],
                  output_bytes,
                  1,
                  spec_.output_height,
                  spec_.output_width,
                  3);
        const cvh::PipelineStatus status = plan_.tryRun(
            input, output, workspace_->view(), &last_run_info_);
        if (!status.ok())
        {
            throw std::runtime_error(
                std::string("cvh Pipeline run failed: ") + status.message());
        }
    }

    const cvh::PipelineInfo& info() const { return plan_.info(); }
    const cvh::PipelineRunInfo& runInfo() const { return last_run_info_; }

private:
    CaseSpec spec_;
    std::vector<const std::uint8_t*> inputs_;
    std::size_t input_row_stride_ = 0;
    std::vector<float*> outputs_;
    cvh::PipelinePlan plan_;
    std::unique_ptr<cvh::PipelineWorkspace> workspace_;
    cvh::PipelineRunInfo last_run_info_{};
};

std::uint8_t roundedU8(float value)
{
    const int rounded = static_cast<int>(std::round(value));
    return static_cast<std::uint8_t>(std::clamp(rounded, 0, 255));
}

void computeContentGeometry(const CaseSpec& spec,
                            int& content_width,
                            int& content_height,
                            int& pad_left,
                            int& pad_top)
{
    content_width = spec.output_width;
    content_height = spec.output_height;
    pad_left = 0;
    pad_top = 0;
    if (spec.geometry != Geometry::Letterbox)
    {
        return;
    }
    const float scale = std::min(
        static_cast<float>(spec.output_width) /
            static_cast<float>(spec.input_width),
        static_cast<float>(spec.output_height) /
            static_cast<float>(spec.input_height));
    content_width = std::clamp(
        static_cast<int>(std::floor(
            static_cast<float>(spec.input_width) * scale + 0.5f)),
        1,
        spec.output_width);
    content_height = std::clamp(
        static_cast<int>(std::floor(
            static_cast<float>(spec.input_height) * scale + 0.5f)),
        1,
        spec.output_height);
    pad_left = (spec.output_width - content_width) / 2;
    pad_top = (spec.output_height - content_height) / 2;
}

std::vector<float> independentReference(
    const CaseSpec& spec,
    const std::uint8_t* input,
    std::size_t input_row_stride)
{
    const std::size_t output_values =
        static_cast<std::size_t>(spec.output_width) *
        static_cast<std::size_t>(spec.output_height) * 3;
    std::vector<float> output(output_values);
    int content_width = 0;
    int content_height = 0;
    int pad_left = 0;
    int pad_top = 0;
    computeContentGeometry(
        spec, content_width, content_height, pad_left, pad_top);
    const int pad_right = pad_left + content_width;
    const int pad_bottom = pad_top + content_height;

    for (int y = 0; y < spec.output_height; ++y)
    {
        for (int x = 0; x < spec.output_width; ++x)
        {
            const bool padded = spec.geometry == Geometry::Letterbox &&
                (x < pad_left || x >= pad_right ||
                 y < pad_top || y >= pad_bottom);
            const int geometry_x = x - pad_left;
            const int geometry_y = y - pad_top;
            for (int channel = 0; channel < 3; ++channel)
            {
                std::uint8_t resized = roundedU8(spec.pad_value);
                if (!padded)
                {
                    const int source_channel =
                        spec.input_format == InputFormat::BGR8
                        ? 2 - channel
                        : channel;
                    if (spec.interpolation == Interpolation::Nearest)
                    {
                        const int source_x = std::min(
                            spec.input_width - 1,
                            static_cast<int>(
                                (static_cast<std::int64_t>(geometry_x) *
                                 spec.input_width) /
                                content_width));
                        const int source_y = std::min(
                            spec.input_height - 1,
                            static_cast<int>(
                                (static_cast<std::int64_t>(geometry_y) *
                                 spec.input_height) /
                                content_height));
                        resized = input[
                            static_cast<std::size_t>(source_y) *
                                input_row_stride +
                            static_cast<std::size_t>(source_x * 3 +
                                                     source_channel)];
                    }
                    else
                    {
                        const float scale_x =
                            static_cast<float>(spec.input_width) /
                            static_cast<float>(content_width);
                        const float scale_y =
                            static_cast<float>(spec.input_height) /
                            static_cast<float>(content_height);
                        const float source_x =
                            (static_cast<float>(geometry_x) + 0.5f) *
                                scale_x -
                            0.5f;
                        const float source_y =
                            (static_cast<float>(geometry_y) + 0.5f) *
                                scale_y -
                            0.5f;
                        const int x0 = std::clamp(
                            static_cast<int>(std::floor(source_x)),
                            0,
                            spec.input_width - 1);
                        const int y0 = std::clamp(
                            static_cast<int>(std::floor(source_y)),
                            0,
                            spec.input_height - 1);
                        const int x1 = std::min(x0 + 1, spec.input_width - 1);
                        const int y1 = std::min(y0 + 1, spec.input_height - 1);
                        const float wx = source_x - static_cast<float>(x0);
                        const float wy = source_y - static_cast<float>(y0);
                        const auto value = [&](int sx, int sy) {
                            return static_cast<float>(input[
                                static_cast<std::size_t>(sy) *
                                    input_row_stride +
                                static_cast<std::size_t>(sx * 3 +
                                                         source_channel)]);
                        };
                        const float top =
                            value(x0, y0) + (value(x1, y0) - value(x0, y0)) * wx;
                        const float bottom =
                            value(x0, y1) + (value(x1, y1) - value(x0, y1)) * wx;
                        resized = roundedU8(top + (bottom - top) * wy);
                    }
                }
                const float normalized =
                    (static_cast<float>(resized) -
                     spec.mean[static_cast<std::size_t>(channel)]) /
                    spec.stddev[static_cast<std::size_t>(channel)];
                const std::size_t output_index =
                    spec.output_layout == OutputLayout::NCHW
                    ? (static_cast<std::size_t>(channel) *
                           spec.output_height +
                       y) *
                              spec.output_width +
                          x
                    : (static_cast<std::size_t>(y) * spec.output_width + x) *
                              3 +
                          channel;
                output[output_index] = normalized;
            }
        }
    }
    return output;
}

ErrorMetrics compareOutput(const CaseSpec& spec,
                           const std::vector<float>& reference,
                           const float* actual,
                           bool allow_linear_lsb)
{
    ErrorMetrics metrics;
    metrics.valid = true;
    for (std::size_t index = 0; index < reference.size(); ++index)
    {
        const int channel = spec.output_layout == OutputLayout::NCHW
            ? static_cast<int>(
                  index /
                  (static_cast<std::size_t>(spec.output_width) *
                   static_cast<std::size_t>(spec.output_height)))
            : static_cast<int>(index % 3);
        const double expected = reference[index];
        const double value = actual[index];
        const double absolute = std::fabs(value - expected);
        const double relative = absolute / std::max(1.0, std::fabs(expected));
        metrics.max_absolute = std::max(metrics.max_absolute, absolute);
        metrics.max_relative = std::max(metrics.max_relative, relative);
        const double tolerance = allow_linear_lsb &&
                spec.interpolation == Interpolation::Linear
            ? 1.0 /
                      std::fabs(static_cast<double>(
                          spec.stddev[static_cast<std::size_t>(channel)])) +
                  1e-5
            : 1e-5;
        if (!std::isfinite(value) || absolute > tolerance)
        {
            if (metrics.different_values == 0)
            {
                metrics.first_difference = index;
            }
            ++metrics.different_values;
        }
    }
    metrics.valid = metrics.different_values == 0;
    return metrics;
}

std::uint64_t checksumFloats(const float* values, std::size_t count)
{
    return cvh_bench::common::checksum_bytes(
        reinterpret_cast<const uchar*>(values), count * sizeof(float));
}

double percentile(const std::vector<double>& sorted, double fraction)
{
    if (sorted.empty())
    {
        return 0.0;
    }
    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * weight;
}

Summary summarize(const std::vector<double>& samples)
{
    Summary summary;
    if (samples.empty())
    {
        return summary;
    }
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    summary.minimum_ms = sorted.front();
    summary.p50_ms = percentile(sorted, 0.50);
    summary.p90_ms = percentile(sorted, 0.90);
    summary.p95_ms = percentile(sorted, 0.95);
    summary.maximum_ms = sorted.back();
    const double mean = std::accumulate(
        samples.begin(), samples.end(), 0.0) /
        static_cast<double>(samples.size());
    double variance = 0.0;
    for (double sample : samples)
    {
        const double delta = sample - mean;
        variance += delta * delta;
    }
    variance /= static_cast<double>(samples.size());
    summary.coefficient_of_variation =
        mean > 0.0 ? std::sqrt(variance) / mean : 0.0;
    return summary;
}

std::string samplesString(const std::vector<double>& samples)
{
    std::ostringstream output;
    output << std::fixed << std::setprecision(6);
    for (std::size_t index = 0; index < samples.size(); ++index)
    {
        if (index != 0)
        {
            output << ';';
        }
        output << samples[index];
    }
    return output.str();
}

std::size_t ringCount(const Args& args,
                      const CaseSpec& spec,
                      const std::string& cache_mode)
{
    if (cache_mode == "hot")
    {
        return 1;
    }
    const std::size_t input_bytes =
        static_cast<std::size_t>(spec.input_width) *
        static_cast<std::size_t>(spec.input_height) * 3;
    const std::size_t target_bytes =
        static_cast<std::size_t>(args.ring_mib) * 1024 * 1024;
    return std::max<std::size_t>(
        2, (target_bytes + input_bytes - 1) / input_bytes);
}

std::vector<std::string> selectedCacheModes(const Args& args)
{
    if (args.cache_mode == "both")
    {
        return {"hot", "streaming"};
    }
    return {args.cache_mode};
}

void fillInput(std::vector<std::uint8_t>& input,
               std::uint32_t seed)
{
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        seed = seed * 1664525u + 1013904223u;
        input[index] = static_cast<std::uint8_t>((seed >> 16) & 0xffu);
    }
}

ImplementationResult makeCvhResult(
    const std::string& name,
    const std::string& algorithm_path,
    const CvhRunner& runner,
    std::uint64_t checksum,
    ErrorMetrics validation)
{
    ImplementationResult result;
    result.name = name;
    result.algorithm_path = algorithm_path;
    const cvh::PipelineInfo& info = runner.info();
    const cvh::PipelineRunInfo& run_info = runner.runInfo();
    result.candidate_route = routeName(info.candidate_route);
    result.actual_route = routeName(run_info.actual_route);
    result.observed_isa = routeName(run_info.observed_isa);
    result.fallback_reason = run_info.fallback_reason != nullptr
        ? run_info.fallback_reason
        : "";
    result.semantic_stages = info.semantic_stage_count;
    result.execution_groups = info.execution_group_count;
    result.full_frame_intermediates = info.full_frame_intermediates;
    result.allocations_per_run = info.allocations_per_run;
    result.workspace_bytes = info.workspace_bytes;
    result.explicit_temporary_bytes = info.workspace_bytes;
    result.checksum = checksum;
    result.validation = validation;
    return result;
}

void writeCsvHeader(std::ostream& output)
{
    output
        << "schema_version,mode,suite,case_id,profile,session,cache_mode,"
        << "ring_frames,implementation,input_format,input_shape,output_shape,"
        << "output_type,layout,geometry,interpolation,algorithm_path,"
        << "candidate_route,actual_route,observed_isa,fallback_reason,"
        << "semantic_stages,execution_groups,full_frame_intermediates,"
        << "allocations_per_run,workspace_bytes,explicit_temporary_bytes,"
        << "allocation_mode,"
        << "warmup,iters,repeats,threads,min_ms,p50_ms,p90_ms,p95_ms,max_ms,"
        << "cv,speedup_vs_staged,speedup_vs_opencv,checksum,validation,"
        << "max_abs_error,max_rel_error,different_values,first_difference,"
        << "samples_ms,status,note\n";
}

void writeCsvRow(std::ostream& output,
                 const Args& args,
                 const CaseSpec& spec,
                 const std::string& cache_mode,
                 std::size_t ring_frames,
                 const ImplementationResult& result,
                 double staged_p50,
                 double opencv_p50)
{
    const double speedup_vs_staged = result.summary.p50_ms > 0.0
        ? staged_p50 / result.summary.p50_ms
        : 0.0;
    const double speedup_vs_opencv = result.summary.p50_ms > 0.0
        ? opencv_p50 / result.summary.p50_ms
        : 0.0;
    output << std::fixed << std::setprecision(6)
           << "2,opencv_compare,pipeline," << spec.id << ','
           << args.profile << ',' << args.session << ',' << cache_mode << ','
           << ring_frames << ',' << result.name << ','
           << inputFormatName(spec.input_format) << ','
           << spec.input_width << 'x' << spec.input_height << ','
           << spec.output_width << 'x' << spec.output_height
           << ",F32," << layoutName(spec.output_layout) << ','
           << geometryName(spec.geometry) << ','
           << interpolationName(spec.interpolation) << ','
           << result.algorithm_path << ',' << result.candidate_route << ','
           << result.actual_route << ',' << result.observed_isa << ','
           << (result.fallback_reason.empty() ? "none" : result.fallback_reason)
           << ',';
    const auto write_optional_count = [&](int value) {
        if (value < 0)
        {
            output << "not_reported";
        }
        else
        {
            output << value;
        }
    };
    write_optional_count(result.semantic_stages);
    output << ',';
    write_optional_count(result.execution_groups);
    output << ',';
    write_optional_count(result.full_frame_intermediates);
    output << ',';
    write_optional_count(result.allocations_per_run);
    output << ',' << result.workspace_bytes << ','
           << result.explicit_temporary_bytes
           << ",reuse," << args.warmup << ',' << args.iters << ','
           << args.repeats << ',' << args.threads << ','
           << result.summary.minimum_ms << ',' << result.summary.p50_ms << ','
           << result.summary.p90_ms << ',' << result.summary.p95_ms << ','
           << result.summary.maximum_ms << ','
           << result.summary.coefficient_of_variation << ','
           << speedup_vs_staged << ',' << speedup_vs_opencv << ','
           << result.checksum << ','
           << (result.validation.valid ? "pass" : "fail") << ','
           << result.validation.max_absolute << ','
           << result.validation.max_relative << ','
           << result.validation.different_values << ',';
    if (result.validation.first_difference ==
        std::numeric_limits<std::size_t>::max())
    {
        output << "none";
    }
    else
    {
        output << result.validation.first_difference;
    }
    output << ',' << samplesString(result.samples_ms) << ','
           << (result.summary.coefficient_of_variation <= 0.03
                   ? "ok"
                   : "unstable")
           << ','
           << (spec.primary ? "primary_manifest_case" : "secondary_case")
           << '\n';
}

const char* environmentOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? value : fallback;
}

std::string metadataPath(const std::string& output_csv)
{
    std::string path = output_csv;
    if (path.size() >= 4 && path.compare(path.size() - 4, 4, ".csv") == 0)
    {
        path.replace(path.size() - 4, 4, ".meta.json");
    }
    else
    {
        path += ".meta.json";
    }
    return path;
}

void writeMetadata(const Args& args)
{
    if (args.output_csv.empty())
    {
        return;
    }
    const std::string path = metadataPath(args.output_csv);
    std::ofstream output(path);
    if (!output)
    {
        throw std::runtime_error("failed to open metadata output: " + path);
    }
    cvh_bench::common::BenchmarkMetadata metadata;
    metadata.mode = "opencv_compare";
    metadata.suite = "pipeline";
    metadata.profile = args.profile + ":" + args.cache_mode;
    metadata.implementation =
        "cvh_staged,cvh_fused_scalar,cvh_fused_auto,opencv_explicit";
    metadata.output_csv = args.output_csv;
    metadata.cvh_commit = environmentOr(
        "CVH_BENCHMARK_CVH_COMMIT", "working-tree");
    metadata.opencv_commit = environmentOr(
        "CVH_BENCHMARK_OPENCV_COMMIT", "unknown");
    metadata.opencv_source = environmentOr(
        "CVH_BENCHMARK_OPENCV_SOURCE", "unknown");
    metadata.opencv_build_dir = environmentOr(
        "CVH_BENCHMARK_OPENCV_BUILD_DIR", "unknown");
    metadata.cmake_build_type = cvh_bench::common::build_config();
    metadata.cpu_model = environmentOr("CVH_BENCHMARK_CPU_MODEL", "unknown");
    metadata.warmup = args.warmup;
    metadata.iters = args.iters;
    metadata.repeats = args.repeats;
    metadata.threads = args.threads;
    cvh_bench::common::write_metadata_json(output, metadata);
}

void executeCase(const Args& args,
                 const CaseSpec& spec,
                 const std::string& cache_mode,
                 std::ostream& csv)
{
    const std::size_t input_row_stride =
        static_cast<std::size_t>(spec.input_width) * 3;
    const std::size_t input_bytes =
        input_row_stride * static_cast<std::size_t>(spec.input_height);
    const std::size_t output_values =
        static_cast<std::size_t>(spec.output_width) *
        static_cast<std::size_t>(spec.output_height) * 3;
    const std::size_t ring_frames = ringCount(args, spec, cache_mode);

    std::vector<std::vector<std::uint8_t>> input_storage(ring_frames);
    std::vector<std::vector<float>> output_storage(ring_frames);
    std::vector<const std::uint8_t*> input_pointers;
    std::vector<float*> output_pointers;
    input_pointers.reserve(ring_frames);
    output_pointers.reserve(ring_frames);
    for (std::size_t frame = 0; frame < ring_frames; ++frame)
    {
        input_storage[frame].resize(input_bytes);
        fillInput(
            input_storage[frame],
            0x13579bdu ^ static_cast<std::uint32_t>(frame * 0x9e3779b9u));
        output_storage[frame].resize(output_values);
        // Force physical backing before timing. On demand-zero systems a
        // resize-to-zero alone can leave the first streaming sample paying
        // copy-on-write page faults for caller-owned output buffers.
        std::fill(
            output_storage[frame].begin(),
            output_storage[frame].end(),
            static_cast<float>((frame % 17) + 1));
        input_pointers.push_back(input_storage[frame].data());
        output_pointers.push_back(output_storage[frame].data());
    }

    CvhRunner staged(
        spec, input_pointers, input_row_stride, output_pointers, false);
    CvhRunner fused_scalar(
        spec, input_pointers, input_row_stride, output_pointers, true);
    CvhRunner fused_auto(
        spec, input_pointers, input_row_stride, output_pointers, true);
    std::unique_ptr<OpenCvPipelineRunner> opencv =
        makeOpenCvPipelineRunner(
            spec, input_pointers, input_row_stride, output_pointers);

    const std::vector<float> reference = independentReference(
        spec, input_pointers[0], input_row_stride);

    staged.setMode(cvh::cpu::DispatchMode::ScalarOnly);
    staged.run(0);
    std::vector<float> staged_values = output_storage[0];
    const ErrorMetrics staged_error = compareOutput(
        spec, reference, staged_values.data(), false);
    if (!staged_error.valid)
    {
        throw std::runtime_error(
            spec.id + " staged output failed independent reference");
    }

    fused_scalar.setMode(cvh::cpu::DispatchMode::ScalarOnly);
    fused_scalar.run(0);
    const std::uint64_t fused_scalar_checksum =
        checksumFloats(output_storage[0].data(), output_values);
    if (std::memcmp(
            staged_values.data(),
            output_storage[0].data(),
            output_values * sizeof(float)) != 0)
    {
        throw std::runtime_error(
            spec.id + " fused scalar does not exactly match staged output");
    }
    const ErrorMetrics fused_scalar_error = compareOutput(
        spec, reference, output_storage[0].data(), false);

    fused_auto.setMode(cvh::cpu::DispatchMode::Auto);
    fused_auto.run(0);
    const std::uint64_t fused_auto_checksum =
        checksumFloats(output_storage[0].data(), output_values);
    if (std::memcmp(
            staged_values.data(),
            output_storage[0].data(),
            output_values * sizeof(float)) != 0)
    {
        throw std::runtime_error(
            spec.id + " fused Auto does not exactly match staged output");
    }
    const ErrorMetrics fused_auto_error = compareOutput(
        spec, reference, output_storage[0].data(), false);

    opencv->run(0);
    const std::uint64_t opencv_checksum =
        checksumFloats(output_storage[0].data(), output_values);
    const ErrorMetrics opencv_error = compareOutput(
        spec, reference, output_storage[0].data(), true);
    if (!opencv_error.valid)
    {
        std::ostringstream message;
        message << spec.id << " OpenCV output failed frozen tolerance: "
                << "different_values=" << opencv_error.different_values
                << " first_difference=" << opencv_error.first_difference
                << " max_abs=" << opencv_error.max_absolute;
        throw std::runtime_error(message.str());
    }

    const std::uint64_t staged_checksum =
        checksumFloats(staged_values.data(), output_values);
    std::vector<ImplementationResult> results;
    results.push_back(makeCvhResult(
        "cvh_staged",
        "ordered_staged_color_after_geometry",
        staged,
        staged_checksum,
        staged_error));
    results.push_back(makeCvhResult(
        "cvh_fused_scalar",
        "model_input_direct_store",
        fused_scalar,
        fused_scalar_checksum,
        fused_scalar_error));
    results.push_back(makeCvhResult(
        "cvh_fused_auto",
        "model_input_direct_store",
        fused_auto,
        fused_auto_checksum,
        fused_auto_error));
    ImplementationResult opencv_result;
    opencv_result.name = "opencv_explicit";
    opencv_result.algorithm_path = opencv->algorithmPath();
    opencv_result.candidate_route = "upstream";
    opencv_result.actual_route = "upstream";
    opencv_result.observed_isa = "not_reported";
    opencv_result.semantic_stages = 4;
    opencv_result.full_frame_intermediates =
        opencv->explicitFullFrameIntermediates();
    opencv_result.explicit_temporary_bytes =
        opencv->explicitTemporaryBytes();
    opencv_result.checksum = opencv_checksum;
    opencv_result.validation = opencv_error;
    results.push_back(std::move(opencv_result));

    const std::array<std::array<int, 4>, 4> latin{{
        {{0, 1, 2, 3}},
        {{1, 2, 3, 0}},
        {{2, 3, 0, 1}},
        {{3, 0, 1, 2}},
    }};
    std::array<std::size_t, 4> cursors{{0, 0, 0, 0}};
    auto select_dispatch = [&](int implementation) {
        cvh::cpu::set_dispatch_mode(
            implementation == 2
                ? cvh::cpu::DispatchMode::Auto
                : cvh::cpu::DispatchMode::ScalarOnly);
    };
    auto run_implementation = [&](int implementation) {
        const std::size_t frame = cursors[static_cast<std::size_t>(implementation)]++;
        switch (implementation)
        {
        case 0:
            staged.run(frame);
            break;
        case 1:
            fused_scalar.run(frame);
            break;
        case 2:
            fused_auto.run(frame);
            break;
        case 3:
            opencv->run(frame);
            break;
        default:
            throw std::logic_error("invalid implementation index");
        }
        return frame;
    };

    for (int implementation = 0; implementation < 4; ++implementation)
    {
        select_dispatch(implementation);
        for (int warmup = 0; warmup < args.warmup; ++warmup)
        {
            run_implementation(implementation);
        }
    }
    for (int repeat = 0; repeat < args.repeats; ++repeat)
    {
        const std::array<int, 4>& order =
            latin[static_cast<std::size_t>(repeat % 4)];
        for (int implementation : order)
        {
            select_dispatch(implementation);
            std::size_t last_frame = 0;
            const auto begin = std::chrono::steady_clock::now();
            for (int iteration = 0; iteration < args.iters; ++iteration)
            {
                last_frame = run_implementation(implementation);
            }
            const auto end = std::chrono::steady_clock::now();
            // Consume the output outside the timed region so the optimizer
            // cannot treat the measured stores as dead.
            g_sink ^= checksumFloats(
                output_pointers[last_frame % ring_frames], output_values);
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(end - begin)
                    .count() /
                static_cast<double>(args.iters);
            results[static_cast<std::size_t>(implementation)]
                .samples_ms.push_back(elapsed_ms);
        }
    }

    cvh::cpu::set_dispatch_mode(cvh::cpu::DispatchMode::Auto);
    for (ImplementationResult& result : results)
    {
        result.summary = summarize(result.samples_ms);
    }
    const double staged_p50 = results[0].summary.p50_ms;
    const double opencv_p50 = results[3].summary.p50_ms;

    std::cout << spec.id << ' ' << cache_mode
              << " ring=" << ring_frames
              << " OpenCV=" << openCvPipelineVersion() << '\n';
    for (const ImplementationResult& result : results)
    {
        std::cout << "  " << std::setw(18) << std::left << result.name
                  << " p50=" << std::fixed << std::setprecision(3)
                  << result.summary.p50_ms << " ms"
                  << " vs_staged="
                  << staged_p50 / result.summary.p50_ms << 'x'
                  << " vs_opencv="
                  << opencv_p50 / result.summary.p50_ms << 'x'
                  << " cv=" << result.summary.coefficient_of_variation
                  << " route=" << result.actual_route
                  << '/' << result.observed_isa << '\n';
        writeCsvRow(
            csv,
            args,
            spec,
            cache_mode,
            ring_frames,
            result,
            staged_p50,
            opencv_p50);
        g_sink ^= result.checksum;
    }
}

}  // namespace
}  // namespace cvh_pipeline_proof

int main(int argc, char** argv)
{
    try
    {
        const cvh_pipeline_proof::Args args =
            cvh_pipeline_proof::parseArgs(argc, argv);
        cvh::setNumThreads(args.threads);
        cvh_pipeline_proof::configureOpenCvPipelineThreads(args.threads);
        const std::vector<cvh_pipeline_proof::CaseSpec> cases =
            cvh_pipeline_proof::loadCases(args);

        std::ofstream output_file;
        std::ostream* output = &std::cout;
        if (!args.output_csv.empty())
        {
            if (std::ifstream(args.output_csv).good())
            {
                throw std::runtime_error(
                    "refusing to overwrite CSV output: " + args.output_csv);
            }
            const std::string metadata_path =
                cvh_pipeline_proof::metadataPath(args.output_csv);
            if (std::ifstream(metadata_path).good())
            {
                throw std::runtime_error(
                    "refusing to overwrite metadata output: " +
                    metadata_path);
            }
            output_file.open(args.output_csv);
            if (!output_file)
            {
                throw std::runtime_error(
                    "failed to open CSV output: " + args.output_csv);
            }
            output = &output_file;
        }
        cvh_pipeline_proof::writeCsvHeader(*output);
        for (const cvh_pipeline_proof::CaseSpec& spec : cases)
        {
            for (const std::string& cache_mode :
                 cvh_pipeline_proof::selectedCacheModes(args))
            {
                cvh_pipeline_proof::executeCase(
                    args, spec, cache_mode, *output);
            }
        }
        cvh_pipeline_proof::writeMetadata(args);
        return cvh_pipeline_proof::g_sink == 0xdeadbeefULL ? 3 : 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "pipeline proof benchmark failed: "
                  << exception.what() << '\n';
        return 2;
    }
}
