#ifndef CVH_BENCHMARK_PIPELINE_COMPARE_BACKEND_H
#define CVH_BENCHMARK_PIPELINE_COMPARE_BACKEND_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cvh_pipeline_proof {

enum class InputFormat
{
    BGR8,
    RGB8,
};

enum class OutputLayout
{
    NCHW,
    NHWC,
};

enum class Geometry
{
    Resize,
    Letterbox,
};

enum class Interpolation
{
    Nearest,
    Linear,
};

struct CaseSpec
{
    std::string id;
    std::string min_profile;
    int input_width = 0;
    int input_height = 0;
    InputFormat input_format = InputFormat::BGR8;
    int output_width = 0;
    int output_height = 0;
    OutputLayout output_layout = OutputLayout::NCHW;
    Geometry geometry = Geometry::Resize;
    Interpolation interpolation = Interpolation::Linear;
    float pad_value = 114.0f;
    std::array<float, 3> mean{};
    std::array<float, 3> stddev{{1.0f, 1.0f, 1.0f}};
    bool primary = true;
};

class OpenCvPipelineRunner
{
public:
    virtual ~OpenCvPipelineRunner() = default;
    virtual void run(std::size_t frame_index) = 0;
    virtual std::size_t explicitTemporaryBytes() const = 0;
    virtual int explicitFullFrameIntermediates() const = 0;
    virtual const char* algorithmPath() const = 0;
};

void configureOpenCvPipelineThreads(int threads);

std::unique_ptr<OpenCvPipelineRunner> makeOpenCvPipelineRunner(
    const CaseSpec& spec,
    const std::vector<const std::uint8_t*>& inputs,
    std::size_t input_row_stride,
    const std::vector<float*>& outputs);

const char* openCvPipelineVersion();

}  // namespace cvh_pipeline_proof

#endif  // CVH_BENCHMARK_PIPELINE_COMPARE_BACKEND_H
