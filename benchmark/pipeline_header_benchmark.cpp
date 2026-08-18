#include "cvh/pipeline/pipeline.h"
#include "common/benchmark_common.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace cvh_bench {

struct PipelineCase
{
    const char* name;
    int input_width;
    int input_height;
    int output_width;
    int output_height;
    cvh::PixelFormat input_format;
    cvh::Layout output_layout;
    cvh::Interpolation interpolation;
    bool color_stage;
    bool letterbox = false;
    bool yuv = false;
    cvh::PipelineDataType output_type = cvh::PipelineDataType::F32;
};

struct ResultRow
{
    const PipelineCase* shape = nullptr;
    std::string implementation;
    cvh::PipelineInfo info{};
    cvh::PipelineRunInfo run_info{};
    common::TimingResult timing{};
    double speedup_vs_staged = 0.0;
    std::uint64_t checksum = 0;
};

volatile std::uint64_t g_pipeline_sink = 0;

void usage()
{
    std::cout
        << "Usage: cvh_benchmark_pipeline_header "
        << "[--profile quick|stable|full] [--warmup N] "
        << "[--iters N] [--repeats N] [--output path]\n";
}

std::vector<PipelineCase> build_cases(const std::string& profile)
{
    std::vector<PipelineCase> cases{{
        "640x480_bgr_to_224x224_rgb_f32_nchw_linear",
        640,
        480,
        224,
        224,
        cvh::PixelFormat::BGR8,
        cvh::Layout::NCHW,
        cvh::Interpolation::Linear,
        true}};
    if (profile == "stable" || profile == "full")
    {
        cases = {
            {"1280x720_bgr_to_640x640_rgb_f32_nchw_linear", 1280, 720, 640, 640, cvh::PixelFormat::BGR8, cvh::Layout::NCHW, cvh::Interpolation::Linear, true},
            {"1920x1080_bgr_to_640x640_rgb_f32_nchw_linear", 1920, 1080, 640, 640, cvh::PixelFormat::BGR8, cvh::Layout::NCHW, cvh::Interpolation::Linear, true},
            {"640x480_rgb_to_224x224_rgb_f32_nchw_linear", 640, 480, 224, 224, cvh::PixelFormat::RGB8, cvh::Layout::NCHW, cvh::Interpolation::Linear, false},
            {"1280x720_bgr_to_640x640_rgb_f32_nhwc_linear", 1280, 720, 640, 640, cvh::PixelFormat::BGR8, cvh::Layout::NHWC, cvh::Interpolation::Linear, true},
            {"1280x720_bgr_to_640x640_rgb_f32_nchw_nearest", 1280, 720, 640, 640, cvh::PixelFormat::BGR8, cvh::Layout::NCHW, cvh::Interpolation::Nearest, true},
            {"1280x720_bgr_letterbox_640x640_rgb_f32_nchw_nearest", 1280, 720, 640, 640, cvh::PixelFormat::BGR8, cvh::Layout::NCHW, cvh::Interpolation::Nearest, true, true, false},
            {"1280x720_nv12_to_640x640_rgb_f32_nchw_linear", 1280, 720, 640, 640, cvh::PixelFormat::NV12, cvh::Layout::NCHW, cvh::Interpolation::Linear, true, false, true},
            {"1280x720_nv21_to_640x640_rgb_f32_nchw_linear", 1280, 720, 640, 640, cvh::PixelFormat::NV21, cvh::Layout::NCHW, cvh::Interpolation::Linear, true, false, true},
            {"1280x720_bgr_to_640x640_rgb_u8_nchw_nearest", 1280, 720, 640, 640, cvh::PixelFormat::BGR8, cvh::Layout::NCHW, cvh::Interpolation::Nearest, true, false, false, cvh::PipelineDataType::U8},
            {"1280x720_bgr_to_640x640_rgb_s8_nchw_nearest", 1280, 720, 640, 640, cvh::PixelFormat::BGR8, cvh::Layout::NCHW, cvh::Interpolation::Nearest, true, false, false, cvh::PipelineDataType::S8},
            {"640x480_bgr_to_224x224_rgb_s8_nhwc_linear", 640, 480, 224, 224, cvh::PixelFormat::BGR8, cvh::Layout::NHWC, cvh::Interpolation::Linear, true, false, false, cvh::PipelineDataType::S8},
            {"1280x720_nv12_to_640x640_rgb_s8_nchw_linear", 1280, 720, 640, 640, cvh::PixelFormat::NV12, cvh::Layout::NCHW, cvh::Interpolation::Linear, true, false, true, cvh::PipelineDataType::S8},
            {"1280x720_nv21_to_640x640_rgb_s8_nchw_linear", 1280, 720, 640, 640, cvh::PixelFormat::NV21, cvh::Layout::NCHW, cvh::Interpolation::Linear, true, false, true, cvh::PipelineDataType::S8},
        };
    }
    if (profile == "full")
    {
        cases.push_back({
            "641x479_bgr_to_321x239_rgb_f32_nhwc_linear",
            641,
            479,
            321,
            239,
            cvh::PixelFormat::BGR8,
            cvh::Layout::NHWC,
            cvh::Interpolation::Linear,
            true});
    }
    return cases;
}

cvh::TensorDescriptor output_descriptor(const PipelineCase& shape)
{
    cvh::TensorDescriptor descriptor;
    descriptor.data_type = shape.output_type;
    descriptor.layout = shape.output_layout;
    descriptor.dims = 4;
    if (shape.output_layout == cvh::Layout::NCHW)
    {
        descriptor.shape = {
            1, 3, shape.output_height, shape.output_width, 0, 0, 0, 0};
    }
    else
    {
        descriptor.shape = {
            1, shape.output_height, shape.output_width, 3, 0, 0, 0, 0};
    }
    return descriptor;
}

cvh::PipelinePlan make_plan(const PipelineCase& shape, bool fused)
{
    cvh::PipelineBuilder builder = cvh::pipe(
        cvh::imageDesc(
            shape.input_width,
            shape.input_height,
            shape.input_format,
            shape.yuv
                ? cvh::ColorSpec{
                      cvh::ColorMatrix::BT709,
                      cvh::ColorRange::Limited,
                      cvh::ChromaLocation::Left}
                : cvh::ColorSpec{}),
        output_descriptor(shape));
    if (fused && (shape.color_stage || shape.yuv))
    {
        builder.color(cvh::Color::RGB);
    }
    if (shape.letterbox)
    {
        builder.letterbox(
            shape.output_width,
            shape.output_height,
            114.0f,
            shape.interpolation);
    }
    else
    {
        builder.resize(
            shape.output_width, shape.output_height, shape.interpolation);
    }
    if (!fused)
    {
        builder.color(cvh::Color::RGB);
    }
    builder
        .normalize(
            {123.675f, 116.28f, 103.53f},
            {58.395f, 57.12f, 57.375f});
    if (shape.output_type == cvh::PipelineDataType::U8 ||
        shape.output_type == cvh::PipelineDataType::S8)
    {
        builder.quantize(
            shape.output_type,
            0.025f,
            shape.output_type == cvh::PipelineDataType::U8 ? 128 : 0);
    }
    builder.layout(shape.output_layout);
    if (fused)
    {
        builder
            .requireNoFullFrameIntermediate()
            .requireSingleExecutionGroup();
    }
    return builder.prepare();
}

std::string layout_name(cvh::Layout layout)
{
    return layout == cvh::Layout::NCHW ? "NCHW" : "NHWC";
}

std::string interpolation_name(cvh::Interpolation interpolation)
{
    return interpolation == cvh::Interpolation::Nearest
        ? "nearest"
        : "linear";
}

std::string route_name(cvh::PipelineRoute route)
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

std::string depth_name(cvh::PipelineDataType type)
{
    if (type == cvh::PipelineDataType::U8)
    {
        return "CV_8U";
    }
    if (type == cvh::PipelineDataType::S8)
    {
        return "CV_8S";
    }
    return "CV_32F";
}

int mat_type(cvh::PipelineDataType type)
{
    if (type == cvh::PipelineDataType::U8)
    {
        return CV_8UC1;
    }
    if (type == cvh::PipelineDataType::S8)
    {
        return CV_8SC1;
    }
    return CV_32FC1;
}

ResultRow measure_plan(const common::BasicArgs& args,
                       const PipelineCase& shape,
                       const char* implementation,
                       const cvh::PipelinePlan& plan,
                       const cvh::Mat& input,
                       cvh::Mat& output,
                       cvh::PipelineWorkspace& workspace,
                       cvh::cpu::DispatchMode dispatch_mode)
{
    const cvh::cpu::DispatchMode previous_mode =
        cvh::cpu::dispatch_mode();
    cvh::cpu::set_dispatch_mode(dispatch_mode);
    cvh::PipelineRunInfo run_info;
    const common::TimingResult timing = common::measure_repeated_ms(
        [&]() {
            plan.run(input, output, workspace.view(), &run_info);
        },
        args.warmup,
        args.iters,
        args.repeats);
    cvh::cpu::set_dispatch_mode(previous_mode);
    const std::uint64_t checksum = common::checksum_mat_bytes(output);
    g_pipeline_sink ^= checksum;
    return ResultRow{
        &shape,
        implementation,
        plan.info(),
        run_info,
        timing,
        0.0,
        checksum};
}

ResultRow measure_view_plan(const common::BasicArgs& args,
                            const PipelineCase& shape,
                            const char* implementation,
                            const cvh::PipelinePlan& plan,
                            cvh::ConstImageView input,
                            cvh::TensorView output,
                            const cvh::Mat& output_mat,
                            cvh::PipelineWorkspace& workspace,
                            cvh::cpu::DispatchMode dispatch_mode)
{
    const cvh::cpu::DispatchMode previous_mode =
        cvh::cpu::dispatch_mode();
    cvh::cpu::set_dispatch_mode(dispatch_mode);
    cvh::PipelineRunInfo run_info;
    const common::TimingResult timing = common::measure_repeated_ms(
        [&]() {
            plan.run(input, output, workspace.view(), &run_info);
        },
        args.warmup,
        args.iters,
        args.repeats);
    cvh::cpu::set_dispatch_mode(previous_mode);
    const std::uint64_t checksum =
        common::checksum_mat_bytes(output_mat);
    g_pipeline_sink ^= checksum;
    return ResultRow{
        &shape,
        implementation,
        plan.info(),
        run_info,
        timing,
        0.0,
        checksum};
}

void write_csv(const common::BasicArgs& args,
               const std::vector<ResultRow>& rows,
               std::ostream& output)
{
    output
        << "schema_version,mode,suite,module,op,variant,depth,channels,layout,shape,pixels,implementation,candidate_route,dispatch_path,observed_isa,allocation_mode,warmup,iters,repeats,threads,execution_groups,full_frame_intermediates,workspace_bytes,min_ms,median_ms,fps,mpix_per_sec,speedup_vs_staged,checksum,status,note\n";
    output << std::fixed << std::setprecision(6);
    for (const ResultRow& row : rows)
    {
        const PipelineCase& shape = *row.shape;
        const std::size_t pixels =
            static_cast<std::size_t>(shape.output_width) *
            static_cast<std::size_t>(shape.output_height);
        output
            << "1,internal,pipeline,pipeline,"
            << (shape.output_type == cvh::PipelineDataType::U8
                    ? shape.yuv
                        ? "MODEL_INPUT_YUV420_U8"
                        : "MODEL_INPUT_PACKED_U8"
                    : shape.output_type == cvh::PipelineDataType::S8
                        ? shape.yuv
                            ? "MODEL_INPUT_YUV420_S8"
                            : "MODEL_INPUT_PACKED_S8"
                    : shape.yuv
                        ? "MODEL_INPUT_YUV420_F32"
                        : shape.letterbox
                        ? "MODEL_INPUT_PACKED_F32_LETTERBOX"
                        : "MODEL_INPUT_PACKED_F32")
            << ","
            << interpolation_name(shape.interpolation) << ","
            << depth_name(shape.output_type) << ",3,"
            << layout_name(shape.output_layout) << "," << shape.name << ","
            << pixels << "," << row.implementation << ","
            << route_name(row.info.candidate_route) << ","
            << route_name(row.run_info.actual_route) << ","
            << route_name(row.run_info.observed_isa)
            << ",reuse," << args.warmup << "," << args.iters << ","
            << args.repeats << ",1," << row.info.execution_group_count
            << "," << row.info.full_frame_intermediates << ","
            << row.info.workspace_bytes << "," << row.timing.min_ms << ","
            << row.timing.median_ms << ","
            << 1000.0 / row.timing.median_ms << ","
            << common::mpix_per_sec(pixels, row.timing.median_ms) << ","
            << row.speedup_vs_staged << "," << row.checksum
            << ",ok,"
            << (shape.yuv
                    ? shape.input_format == cvh::PixelFormat::NV21
                        ? "nv21_bt709_limited_left_to_rgb"
                        : "nv12_bt709_limited_left_to_rgb"
                    : shape.color_stage ? "bgr_to_rgb" : "rgb_identity")
            << "\n";
    }
}

const char* environment_or(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' ? value : fallback;
}

void write_metadata(const common::BasicArgs& args)
{
    if (args.output_csv.empty())
    {
        return;
    }
    std::string metadata_path = args.output_csv;
    if (metadata_path.size() >= 4 &&
        metadata_path.compare(
            metadata_path.size() - 4, 4, ".csv") == 0)
    {
        metadata_path.replace(
            metadata_path.size() - 4, 4, ".meta.json");
    }
    else
    {
        metadata_path += ".meta.json";
    }
    std::ofstream output(metadata_path);
    common::BenchmarkMetadata metadata;
    metadata.mode = "internal";
    metadata.suite = "pipeline";
    metadata.profile = args.profile;
    metadata.implementation = "staged_p0,fused_scalar,fused_auto";
    metadata.output_csv = args.output_csv;
    metadata.cvh_commit = environment_or(
        "CVH_BENCHMARK_CVH_COMMIT", "working-tree");
    metadata.cmake_build_type = common::build_config();
    metadata.cpu_model = environment_or(
        "CVH_BENCHMARK_CPU_MODEL", "unknown");
    metadata.warmup = args.warmup;
    metadata.iters = args.iters;
    metadata.repeats = args.repeats;
    common::write_metadata_json(output, metadata);
}

}  // namespace cvh_bench

int main(int argc, char** argv)
{
    const cvh_bench::common::BasicArgs args =
        cvh_bench::common::parse_basic_args(
            argc,
            argv,
            cvh_bench::common::BasicArgs{"quick", 2, 3, 5, ""},
            {"quick", "stable", "full"},
            cvh_bench::usage);
    const std::vector<cvh_bench::PipelineCase> cases =
        cvh_bench::build_cases(args.profile);
    std::vector<cvh_bench::ResultRow> rows;
    rows.reserve(cases.size() * 3);

    for (const cvh_bench::PipelineCase& shape : cases)
    {
        if (shape.yuv)
        {
            const std::size_t y_stride =
                static_cast<std::size_t>(shape.input_width) + 16;
            const std::size_t uv_stride =
                static_cast<std::size_t>(shape.input_width) + 16;
            std::vector<uchar> y_plane(
                y_stride * static_cast<std::size_t>(shape.input_height));
            std::vector<uchar> uv_plane(
                uv_stride *
                static_cast<std::size_t>(shape.input_height / 2));
            for (std::size_t index = 0; index < y_plane.size(); ++index)
            {
                y_plane[index] = static_cast<uchar>(index * 37u + 17u);
            }
            for (std::size_t index = 0; index < uv_plane.size(); ++index)
            {
                uv_plane[index] = static_cast<uchar>(index * 29u + 83u);
            }
            const cvh::ColorSpec color_spec{
                cvh::ColorMatrix::BT709,
                cvh::ColorRange::Limited,
                cvh::ChromaLocation::Left};
            const cvh::ConstImageView input =
                shape.input_format == cvh::PixelFormat::NV21
                ? cvh::nv21(
                      y_plane.data(),
                      y_stride,
                      y_plane.size(),
                      uv_plane.data(),
                      uv_stride,
                      uv_plane.size(),
                      shape.input_width,
                      shape.input_height,
                      color_spec)
                : cvh::nv12(
                      y_plane.data(),
                      y_stride,
                      y_plane.size(),
                      uv_plane.data(),
                      uv_stride,
                      uv_plane.size(),
                      shape.input_width,
                      shape.input_height,
                      color_spec);
            const cvh::PipelinePlan fused =
                cvh_bench::make_plan(shape, true);
            cvh::PipelineWorkspace workspace(fused);
            const cvh::TensorDescriptor descriptor =
                cvh_bench::output_descriptor(shape);
            cvh::Mat output(
                descriptor.dims,
                descriptor.shape.data(),
                cvh_bench::mat_type(shape.output_type));
            cvh::TensorView output_view;
            output_view.data = output.data;
            output_view.size_bytes = output.total() * output.elemSize();
            output_view.descriptor = descriptor;
            cvh_bench::ResultRow row = cvh_bench::measure_view_plan(
                args,
                shape,
                "fused_scalar",
                fused,
                input,
                output_view,
                output,
                workspace,
                cvh::cpu::DispatchMode::ScalarOnly);
            rows.push_back(row);
            continue;
        }

        cvh::Mat input(
            {shape.input_height, shape.input_width}, CV_8UC3);
        cvh_bench::common::fill_mat_u8_lcg(
            input,
            static_cast<std::uint32_t>(
                shape.input_width * 31 + shape.input_height * 17));
        const cvh::PipelinePlan staged =
            cvh_bench::make_plan(shape, false);
        const cvh::PipelinePlan fused =
            cvh_bench::make_plan(shape, true);
        cvh::PipelineWorkspace staged_workspace(staged);
        cvh::PipelineWorkspace fused_workspace(fused);
        const cvh::TensorDescriptor descriptor =
            cvh_bench::output_descriptor(shape);
        cvh::Mat staged_output(
            descriptor.dims,
            descriptor.shape.data(),
            cvh_bench::mat_type(shape.output_type));
        cvh::Mat fused_output(
            descriptor.dims,
            descriptor.shape.data(),
            cvh_bench::mat_type(shape.output_type));
        cvh::Mat fused_scalar_output(
            descriptor.dims,
            descriptor.shape.data(),
            cvh_bench::mat_type(shape.output_type));
        staged.run(input, staged_output, staged_workspace.view());
        fused.run(input, fused_output, fused_workspace.view());
        if (!cvh_bench::common::same_mat_bytes(
                staged_output, fused_output))
        {
            std::cerr << "pipeline correctness mismatch: "
                      << shape.name << "\n";
            return 3;
        }

        cvh_bench::ResultRow staged_row = cvh_bench::measure_plan(
            args,
            shape,
            "staged_p0",
            staged,
            input,
            staged_output,
            staged_workspace,
            cvh::cpu::DispatchMode::ScalarOnly);
        cvh_bench::ResultRow fused_scalar_row = cvh_bench::measure_plan(
            args,
            shape,
            "fused_scalar",
            fused,
            input,
            fused_scalar_output,
            fused_workspace,
            cvh::cpu::DispatchMode::ScalarOnly);
        cvh_bench::ResultRow fused_auto_row = cvh_bench::measure_plan(
            args,
            shape,
            "fused_auto",
            fused,
            input,
            fused_output,
            fused_workspace,
            cvh::cpu::DispatchMode::Auto);
        staged_row.speedup_vs_staged = 1.0;
        fused_scalar_row.speedup_vs_staged =
            staged_row.timing.median_ms /
            fused_scalar_row.timing.median_ms;
        fused_auto_row.speedup_vs_staged =
            staged_row.timing.median_ms /
            fused_auto_row.timing.median_ms;
        if (staged_row.checksum != fused_scalar_row.checksum ||
            staged_row.checksum != fused_auto_row.checksum)
        {
            std::cerr << "pipeline measured checksum mismatch: "
                      << shape.name << "\n";
            return 5;
        }
        rows.push_back(staged_row);
        rows.push_back(fused_scalar_row);
        rows.push_back(fused_auto_row);
    }

    cvh_bench::write_csv(args, rows, std::cout);
    if (!args.output_csv.empty())
    {
        std::ofstream output(args.output_csv);
        if (!output)
        {
            std::cerr << "failed to open output: " << args.output_csv << "\n";
            return 4;
        }
        cvh_bench::write_csv(args, rows, output);
        cvh_bench::write_metadata(args);
    }
    return static_cast<int>(cvh_bench::g_pipeline_sink & 0u);
}
