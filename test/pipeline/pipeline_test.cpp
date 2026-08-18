#include "cvh/pipeline/pipeline.h"
#include "cvh/recipes/model_input.h"
#include "cvh/imgproc/cvtcolor.h"
#include "cvh/imgproc/resize.h"
#include "../support/dispatch_mode_guard.hpp"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

cvh::Mat makeInput()
{
    cvh::Mat input({3, 4}, CV_8UC3);
    for (int y = 0; y < input.size[0]; ++y)
    {
        for (int x = 0; x < input.size[1]; ++x)
        {
            input.at<uchar>(y, x, 0) =
                static_cast<uchar>(10 + y * 17 + x * 3);
            input.at<uchar>(y, x, 1) =
                static_cast<uchar>(20 + y * 11 + x * 5);
            input.at<uchar>(y, x, 2) =
                static_cast<uchar>(30 + y * 7 + x * 9);
        }
    }
    return input;
}

void expectSameFloatMat(const cvh::Mat& lhs, const cvh::Mat& rhs)
{
    ASSERT_EQ(lhs.dims, rhs.dims);
    ASSERT_EQ(lhs.type(), rhs.type());
    for (int dim = 0; dim < lhs.dims; ++dim)
    {
        ASSERT_EQ(lhs.size[dim], rhs.size[dim]);
    }
    ASSERT_TRUE(lhs.isContinuous());
    ASSERT_TRUE(rhs.isContinuous());
    const float* lhs_data = reinterpret_cast<const float*>(lhs.data);
    const float* rhs_data = reinterpret_cast<const float*>(rhs.data);
    const std::size_t value_count =
        lhs.total() * static_cast<std::size_t>(lhs.channels());
    for (std::size_t index = 0; index < value_count; ++index)
    {
        EXPECT_NEAR(lhs_data[index], rhs_data[index], 1e-5f)
            << "index=" << index;
    }
}

void expectSameMat(const cvh::Mat& lhs, const cvh::Mat& rhs)
{
    ASSERT_EQ(lhs.dims, rhs.dims);
    ASSERT_EQ(lhs.type(), rhs.type());
    for (int dim = 0; dim < lhs.dims; ++dim)
    {
        ASSERT_EQ(lhs.size[dim], rhs.size[dim]);
    }
    ASSERT_TRUE(lhs.isContinuous());
    ASSERT_TRUE(rhs.isContinuous());
    if (lhs.depth() == CV_32F)
    {
        expectSameFloatMat(lhs, rhs);
        return;
    }
    ASSERT_TRUE(lhs.depth() == CV_8U || lhs.depth() == CV_8S);
    const std::size_t byte_count = lhs.total() * lhs.elemSize();
    EXPECT_EQ(std::memcmp(lhs.data, rhs.data, byte_count), 0);
}

template <typename T>
T referenceQuantize(float real_value, float scale, int zero_point)
{
    const int minimum = std::is_same<T, uchar>::value ? 0 : -128;
    const int maximum = std::is_same<T, uchar>::value ? 255 : 127;
    if (std::isnan(real_value))
    {
        return static_cast<T>(zero_point);
    }
    if (std::isinf(real_value))
    {
        return static_cast<T>(real_value > 0.0f ? maximum : minimum);
    }
    const double shifted =
        std::round(static_cast<double>(real_value) /
                   static_cast<double>(scale)) +
        static_cast<double>(zero_point);
    if (shifted <= static_cast<double>(minimum))
    {
        return static_cast<T>(minimum);
    }
    if (shifted >= static_cast<double>(maximum))
    {
        return static_cast<T>(maximum);
    }
    return static_cast<T>(static_cast<int>(shifted));
}

uchar referenceRoundU8(float value)
{
    const int rounded = static_cast<int>(std::round(value));
    return static_cast<uchar>(std::clamp(rounded, 0, 255));
}

float referenceYuvChroma(const cvh::ConstImageView& source,
                         int source_x,
                         int source_y,
                         int component)
{
    const int chroma_width = source.descriptor.width / 2;
    const int chroma_height = source.descriptor.height / 2;
    const float coordinate_x =
        source.descriptor.color_spec.chroma_location ==
                cvh::ChromaLocation::Left
        ? static_cast<float>(source_x) / 2.0f
        : (static_cast<float>(source_x) - 0.5f) / 2.0f;
    const float coordinate_y =
        (static_cast<float>(source_y) - 0.5f) / 2.0f;
    const float x = std::clamp(
        coordinate_x, 0.0f, static_cast<float>(chroma_width - 1));
    const float y = std::clamp(
        coordinate_y, 0.0f, static_cast<float>(chroma_height - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, chroma_width - 1);
    const int y1 = std::min(y0 + 1, chroma_height - 1);
    const float weight_x = x - static_cast<float>(x0);
    const float weight_y = y - static_cast<float>(y0);
    const int stored_component =
        source.descriptor.pixel_format == cvh::PixelFormat::NV12
        ? component
        : 1 - component;
    const auto sample = [&](int sample_x, int sample_y) {
        const uchar* row = source.planes[1].data +
            static_cast<std::size_t>(sample_y) *
                source.planes[1].row_stride;
        return static_cast<float>(
            row[sample_x * 2 + stored_component]);
    };
    const float top = sample(x0, y0) +
        (sample(x1, y0) - sample(x0, y0)) * weight_x;
    const float bottom = sample(x0, y1) +
        (sample(x1, y1) - sample(x0, y1)) * weight_x;
    return top + (bottom - top) * weight_y;
}

std::array<uchar, 3> referenceYuvToRgb(
    const cvh::ConstImageView& source,
    int source_x,
    int source_y)
{
    const uchar* y_row = source.planes[0].data +
        static_cast<std::size_t>(source_y) *
            source.planes[0].row_stride;
    float luminance = static_cast<float>(y_row[source_x]);
    float cb = referenceYuvChroma(source, source_x, source_y, 0) -
        128.0f;
    float cr = referenceYuvChroma(source, source_x, source_y, 1) -
        128.0f;
    if (source.descriptor.color_spec.range == cvh::ColorRange::Limited)
    {
        luminance = (luminance - 16.0f) * (255.0f / 219.0f);
        cb *= 255.0f / 224.0f;
        cr *= 255.0f / 224.0f;
    }

    std::array<float, 4> coefficients{
        1.402f, -0.344136f, -0.714136f, 1.772f};
    if (source.descriptor.color_spec.matrix == cvh::ColorMatrix::BT709)
    {
        coefficients =
            {1.5748f, -0.187324f, -0.468124f, 1.8556f};
    }
    else if (source.descriptor.color_spec.matrix ==
             cvh::ColorMatrix::BT2020)
    {
        coefficients =
            {1.4746f, -0.164553f, -0.571353f, 1.8814f};
    }
    return {{
        referenceRoundU8(luminance + coefficients[0] * cr),
        referenceRoundU8(
            luminance + coefficients[1] * cb + coefficients[2] * cr),
        referenceRoundU8(luminance + coefficients[3] * cb)}};
}

std::vector<float> referenceYuvModelInput(
    const cvh::ConstImageView& source,
    int target_width,
    int target_height,
    cvh::Interpolation interpolation,
    cvh::Color output_color,
    cvh::Layout layout,
    const std::array<float, 3>& mean,
    const std::array<float, 3>& stddev)
{
    std::vector<float> output(
        static_cast<std::size_t>(target_width) * target_height * 3);
    const bool nearest = interpolation == cvh::Interpolation::Nearest;
    const float scale_x = static_cast<float>(source.descriptor.width) /
        static_cast<float>(target_width);
    const float scale_y = static_cast<float>(source.descriptor.height) /
        static_cast<float>(target_height);
    for (int y = 0; y < target_height; ++y)
    {
        const float mapped_y =
            (static_cast<float>(y) + 0.5f) * scale_y -
            0.5f;
        const int y0 = nearest
            ? std::min(
                  source.descriptor.height - 1,
                  y * source.descriptor.height / target_height)
            : std::clamp(
                  static_cast<int>(std::floor(mapped_y)),
                  0,
                  source.descriptor.height - 1);
        const int y1 = nearest
            ? y0
            : std::min(y0 + 1, source.descriptor.height - 1);
        const float weight_y = nearest
            ? 0.0f
            : mapped_y - static_cast<float>(y0);
        for (int x = 0; x < target_width; ++x)
        {
            const float mapped_x =
                (static_cast<float>(x) + 0.5f) * scale_x -
                0.5f;
            const int x0 = nearest
                ? std::min(
                      source.descriptor.width - 1,
                      x * source.descriptor.width / target_width)
                : std::clamp(
                      static_cast<int>(std::floor(mapped_x)),
                      0,
                      source.descriptor.width - 1);
            const int x1 = nearest
                ? x0
                : std::min(x0 + 1, source.descriptor.width - 1);
            const float weight_x = nearest
                ? 0.0f
                : mapped_x - static_cast<float>(x0);
            const std::array<uchar, 3> pixel00 =
                referenceYuvToRgb(source, x0, y0);
            const std::array<uchar, 3> pixel01 = nearest
                ? pixel00
                : referenceYuvToRgb(source, x1, y0);
            const std::array<uchar, 3> pixel10 = nearest
                ? pixel00
                : referenceYuvToRgb(source, x0, y1);
            const std::array<uchar, 3> pixel11 = nearest
                ? pixel00
                : referenceYuvToRgb(source, x1, y1);
            for (int channel = 0; channel < 3; ++channel)
            {
                const int source_channel = output_color == cvh::Color::RGB
                    ? channel
                    : 2 - channel;
                float value = static_cast<float>(
                    pixel00[static_cast<std::size_t>(source_channel)]);
                if (!nearest)
                {
                    const float top = value +
                        (static_cast<float>(pixel01[static_cast<std::size_t>(
                             source_channel)]) -
                         value) *
                            weight_x;
                    const float bottom =
                        static_cast<float>(pixel10[static_cast<std::size_t>(
                            source_channel)]) +
                        (static_cast<float>(pixel11[static_cast<std::size_t>(
                             source_channel)]) -
                         static_cast<float>(pixel10[static_cast<std::size_t>(
                             source_channel)])) *
                            weight_x;
                    value = static_cast<float>(
                        referenceRoundU8(
                            top + (bottom - top) * weight_y));
                }
                const std::size_t index = layout == cvh::Layout::NCHW
                    ? (static_cast<std::size_t>(channel) * target_height +
                       y) *
                              target_width +
                          x
                    : (static_cast<std::size_t>(y) * target_width + x) *
                              3 +
                          channel;
                output[index] =
                    (value - mean[static_cast<std::size_t>(channel)]) /
                    stddev[static_cast<std::size_t>(channel)];
            }
        }
    }
    return output;
}

cvh::Mat referenceColor(const cvh::Mat& source,
                        cvh::Color source_color,
                        cvh::Color target_color)
{
    if (source_color == target_color)
    {
        return source.clone();
    }

    int code = -1;
    if ((source_color == cvh::Color::BGR &&
         target_color == cvh::Color::RGB) ||
        (source_color == cvh::Color::RGB &&
         target_color == cvh::Color::BGR))
    {
        code = cvh::COLOR_BGR2RGB;
    }
    else if (source_color == cvh::Color::Gray &&
             (target_color == cvh::Color::BGR ||
              target_color == cvh::Color::RGB))
    {
        code = cvh::COLOR_GRAY2BGR;
    }
    else if (source_color == cvh::Color::BGR &&
             target_color == cvh::Color::Gray)
    {
        code = cvh::COLOR_BGR2GRAY;
    }
    else if (source_color == cvh::Color::RGB &&
             target_color == cvh::Color::Gray)
    {
        code = cvh::COLOR_RGB2GRAY;
    }
    EXPECT_NE(code, -1);
    cvh::Mat target;
    cvh::cvtColor(source, target, code);
    return target;
}

cvh::Mat referenceNormalize(const cvh::Mat& source,
                            const std::array<float, 4>& mean,
                            const std::array<float, 4>& stddev,
                            int count)
{
    cvh::Mat target(
        {source.size[0], source.size[1]},
        CV_MAKETYPE(CV_32F, source.channels()));
    for (int y = 0; y < source.size[0]; ++y)
    {
        const uchar* source_u8 = source.data +
            static_cast<std::size_t>(y) * source.step(0);
        const float* source_f32 = reinterpret_cast<const float*>(source_u8);
        float* target_row = reinterpret_cast<float*>(
            target.data + static_cast<std::size_t>(y) * target.step(0));
        for (int x = 0; x < source.size[1]; ++x)
        {
            for (int channel = 0; channel < source.channels(); ++channel)
            {
                const int offset = x * source.channels() + channel;
                const int parameter = count == 1 ? 0 : channel;
                const float value = source.depth() == CV_8U
                    ? static_cast<float>(source_u8[offset])
                    : source_f32[offset];
                target_row[offset] =
                    (value - mean[static_cast<std::size_t>(parameter)]) /
                    stddev[static_cast<std::size_t>(parameter)];
            }
        }
    }
    return target;
}

cvh::Mat referenceLayout(const cvh::Mat& source, cvh::Layout layout)
{
    const int channels = source.channels();
    const std::array<int, 4> sizes = layout == cvh::Layout::NCHW
        ? std::array<int, 4>{1, channels, source.size[0], source.size[1]}
        : std::array<int, 4>{1, source.size[0], source.size[1], channels};
    cvh::Mat target(4, sizes.data(), CV_MAKETYPE(source.depth(), 1));
    if (layout == cvh::Layout::NHWC)
    {
        for (int y = 0; y < source.size[0]; ++y)
        {
            std::memcpy(
                target.data + static_cast<std::size_t>(y) *
                                  source.size[1] * source.elemSize(),
                source.data + static_cast<std::size_t>(y) * source.step(0),
                static_cast<std::size_t>(source.size[1]) * source.elemSize());
        }
        return target;
    }

    const std::size_t element_size = source.elemSize1();
    for (int y = 0; y < source.size[0]; ++y)
    {
        const uchar* source_row = source.data +
            static_cast<std::size_t>(y) * source.step(0);
        for (int x = 0; x < source.size[1]; ++x)
        {
            for (int channel = 0; channel < channels; ++channel)
            {
                const std::size_t source_offset =
                    static_cast<std::size_t>(x * channels + channel) *
                    element_size;
                const std::size_t target_index =
                    (static_cast<std::size_t>(channel) * source.size[0] + y) *
                        source.size[1] +
                    x;
                std::memcpy(
                    target.data + target_index * element_size,
                    source_row + source_offset,
                    element_size);
            }
        }
    }
    return target;
}

cvh::PipelineDataDescriptor imageDescriptorFor(
    const cvh::Mat& image,
    cvh::Color color)
{
    cvh::ImageDescriptor descriptor;
    descriptor.width = image.size[1];
    descriptor.height = image.size[0];
    descriptor.data_type = image.depth() == CV_8U
        ? cvh::PipelineDataType::U8
        : cvh::PipelineDataType::F32;
    descriptor.color = color;
    descriptor.pixel_format = cvh::PixelFormat::Unknown;
    descriptor.plane_count = 1;
    return descriptor;
}

cvh::PipelineDataDescriptor tensorDescriptorFor(
    const cvh::Mat& tensor,
    cvh::Layout layout)
{
    cvh::TensorDescriptor descriptor;
    descriptor.data_type = tensor.depth() == CV_8U
        ? cvh::PipelineDataType::U8
        : cvh::PipelineDataType::F32;
    descriptor.layout = layout;
    descriptor.dims = tensor.dims;
    for (int dim = 0; dim < tensor.dims; ++dim)
    {
        descriptor.shape[static_cast<std::size_t>(dim)] = tensor.size[dim];
    }
    return descriptor;
}

enum class RandomStageKind
{
    Color,
    Resize,
    Normalize,
};

struct RandomStage
{
    RandomStageKind kind = RandomStageKind::Color;
    cvh::Color color = cvh::Color::BGR;
    int width = 0;
    int height = 0;
    cvh::Interpolation interpolation = cvh::Interpolation::Linear;
    std::array<float, 4> mean{};
    std::array<float, 4> stddev{{1.0f, 1.0f, 1.0f, 1.0f}};
    int count = 0;
};

}  // namespace

TEST(PipelineDescriptorTest, borrowed_helpers_keep_explicit_memory_contracts)
{
    const cvh::ColorSpec color_spec{
        cvh::ColorMatrix::BT709,
        cvh::ColorRange::Limited,
        cvh::ChromaLocation::Left};
    const cvh::ImageDescriptor image =
        cvh::imageDesc(1920, 1080, cvh::PixelFormat::NV12, color_spec);

    EXPECT_TRUE(image.valid());
    EXPECT_EQ(image.plane_count, 2);
    EXPECT_EQ(image.color, cvh::Color::YUV);

    std::array<float, 12> storage{};
    const cvh::TensorView tensor =
        cvh::nchw(storage.data(), sizeof(storage), 1, 3, 2, 2);
    EXPECT_EQ(tensor.data, storage.data());
    EXPECT_EQ(tensor.size_bytes, sizeof(storage));
    EXPECT_EQ(tensor.descriptor.layout, cvh::Layout::NCHW);
    EXPECT_EQ(tensor.descriptor.shape[1], 3);
}

TEST(PipelineOneShotTest, color_normalize_and_layout_follow_written_order)
{
    cvh::Mat input({2, 2}, CV_8UC3);
    for (int y = 0; y < 2; ++y)
    {
        for (int x = 0; x < 2; ++x)
        {
            input.at<uchar>(y, x, 0) =
                static_cast<uchar>(10 + y * 4 + x);
            input.at<uchar>(y, x, 1) =
                static_cast<uchar>(20 + y * 4 + x);
            input.at<uchar>(y, x, 2) =
                static_cast<uchar>(30 + y * 4 + x);
        }
    }

    cvh::Mat output;
    cvh::pipe(input, output)
        .color(cvh::Color::RGB)
        .normalize({1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 4.0f})
        .layout(cvh::Layout::NCHW)
        .run();

    ASSERT_EQ(output.dims, 4);
    EXPECT_EQ(output.type(), CV_32FC1);
    EXPECT_EQ(output.size[0], 1);
    EXPECT_EQ(output.size[1], 3);
    EXPECT_EQ(output.size[2], 2);
    EXPECT_EQ(output.size[3], 2);

    const float* values = reinterpret_cast<const float*>(output.data);
    EXPECT_FLOAT_EQ(values[0], 29.0f);
    EXPECT_FLOAT_EQ(values[4], 9.0f);
    EXPECT_FLOAT_EQ(values[8], 1.75f);
}

TEST(PipelineOneShotTest, changing_operation_order_changes_semantics)
{
    cvh::Mat input({1, 1}, CV_8UC3);
    input.at<uchar>(0, 0, 0) = 10;
    input.at<uchar>(0, 0, 1) = 20;
    input.at<uchar>(0, 0, 2) = 30;

    cvh::Mat color_then_normalize;
    cvh::pipe(input, color_then_normalize)
        .color(cvh::Color::RGB)
        .normalize({1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 4.0f})
        .run();

    cvh::Mat normalize_then_color;
    cvh::pipe(input, normalize_then_color)
        .normalize({1.0f, 2.0f, 3.0f}, {1.0f, 2.0f, 4.0f})
        .color(cvh::Color::RGB)
        .run();

    EXPECT_FLOAT_EQ(color_then_normalize.at<float>(0, 0, 0), 29.0f);
    EXPECT_FLOAT_EQ(normalize_then_color.at<float>(0, 0, 0), 6.75f);
    EXPECT_NE(
        color_then_normalize.at<float>(0, 0, 0),
        normalize_then_color.at<float>(0, 0, 0));
}

TEST(PipelinePreparedTest, prepared_and_one_shot_results_match)
{
    const cvh::Mat input = makeInput();
    const cvh::ImageDescriptor input_desc =
        cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8);
    const cvh::TensorDescriptor output_desc =
        cvh::tensorDesc<float>({1, 3, 2, 2}, cvh::Layout::NCHW);

    const cvh::PipelinePlan plan =
        cvh::pipe(input_desc, output_desc)
            .color(cvh::Color::RGB)
            .resize(2, 2, cvh::Interpolation::Nearest)
            .normalize({1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 8.0f})
            .layout(cvh::Layout::NCHW)
            .requireNoFullFrameIntermediate()
            .requireSingleExecutionGroup()
            .prepare();

    EXPECT_EQ(plan.info().semantic_stage_count, 4);
    EXPECT_EQ(plan.info().execution_group_count, 1);
    EXPECT_EQ(plan.info().full_frame_intermediates, 0);
    EXPECT_EQ(plan.info().allocations_per_run, 0);
    EXPECT_EQ(plan.info().workspace_bytes, 0u);
    EXPECT_EQ(
        plan.info().execution_class,
        cvh::PipelineExecutionClass::FusedTiled);
    EXPECT_EQ(plan.info().candidate_route, cvh::PipelineRoute::Scalar);

    const std::string explanation = plan.explain();
    const std::size_t color_position = explanation.find("[0] color");
    const std::size_t resize_position = explanation.find("[1] resize");
    const std::size_t normalize_position = explanation.find("[2] normalize");
    const std::size_t layout_position = explanation.find("[3] layout");
    ASSERT_NE(color_position, std::string::npos);
    ASSERT_NE(resize_position, std::string::npos);
    ASSERT_NE(normalize_position, std::string::npos);
    ASSERT_NE(layout_position, std::string::npos);
    EXPECT_LT(color_position, resize_position);
    EXPECT_LT(resize_position, normalize_position);
    EXPECT_LT(normalize_position, layout_position);
    EXPECT_NE(
        explanation.find(
            "[0] scalar fused stages 0..3: model-input packed-f32"),
        std::string::npos);

    cvh::Mat prepared_output({1, 3, 2, 2}, CV_32FC1);
    cvh::PipelineWorkspace workspace(plan);
    cvh::PipelineRunInfo run_info;
    plan.run(input, prepared_output, workspace.view(), &run_info);

    EXPECT_EQ(run_info.actual_route, cvh::PipelineRoute::Scalar);
    EXPECT_EQ(run_info.observed_isa, cvh::PipelineRoute::Scalar);
    EXPECT_FALSE(run_info.used_fallback);

    cvh::Mat one_shot_output;
    cvh::pipe(input, one_shot_output)
        .color(cvh::Color::RGB)
        .resize(2, 2, cvh::Interpolation::Nearest)
        .normalize({1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 8.0f})
        .layout(cvh::Layout::NCHW)
        .run();

    expectSameFloatMat(prepared_output, one_shot_output);
}

TEST(PipelinePreparedTest, explain_reports_explicit_execution_groups)
{
    const cvh::PipelinePlan copy_plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8))
            .prepare();

    EXPECT_EQ(copy_plan.info().semantic_stage_count, 0);
    EXPECT_EQ(copy_plan.info().execution_group_count, 1);
    EXPECT_EQ(copy_plan.info().full_frame_intermediates, 0);
    EXPECT_NE(
        copy_plan.explain().find("[0] scalar direct copy"),
        std::string::npos);

    const cvh::PipelinePlan staged_plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::tensorDesc<float>({1, 2, 2, 3}, cvh::Layout::NHWC))
            .resize(2, 2, cvh::Interpolation::Linear)
            .color(cvh::Color::RGB)
            .normalize({1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 8.0f})
            .layout(cvh::Layout::NHWC)
            .prepare();

    EXPECT_EQ(staged_plan.info().semantic_stage_count, 4);
    EXPECT_EQ(staged_plan.info().execution_group_count, 4);
    EXPECT_EQ(staged_plan.info().full_frame_intermediates, 3);
    const std::string explanation = staged_plan.explain();
    EXPECT_NE(
        explanation.find("[0] scalar stages 0..0: resize"),
        std::string::npos);
    EXPECT_NE(
        explanation.find("[1] scalar stages 1..1: color"),
        std::string::npos);
    EXPECT_NE(
        explanation.find("[2] scalar stages 2..2: normalize"),
        std::string::npos);
    EXPECT_NE(
        explanation.find("[3] scalar stages 3..3: layout"),
        std::string::npos);
}

TEST(PipelineFusionTest, packed_f32_matrix_matches_staged_oracle)
{
    const cvh::Mat input = makeInput();
    const std::array<cvh::Interpolation, 2> interpolations{{
        cvh::Interpolation::Nearest,
        cvh::Interpolation::Linear}};
    const std::array<cvh::Layout, 2> layouts{{
        cvh::Layout::NCHW,
        cvh::Layout::NHWC}};

    for (cvh::Interpolation interpolation : interpolations)
    {
        for (cvh::Layout layout : layouts)
        {
            for (bool has_color : {false, true})
            {
                const cvh::TensorDescriptor output_desc =
                    layout == cvh::Layout::NCHW
                    ? cvh::tensorDesc<float>(
                          {1, 3, 2, 3}, cvh::Layout::NCHW)
                    : cvh::tensorDesc<float>(
                          {1, 2, 3, 3}, cvh::Layout::NHWC);
                cvh::PipelineBuilder fused_builder = cvh::pipe(
                    cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
                    output_desc);
                if (has_color)
                {
                    fused_builder.color(cvh::Color::RGB);
                }
                fused_builder.resize(3, 2, interpolation);
                if (layout == cvh::Layout::NCHW)
                {
                    fused_builder.normalize({1.5f}, {2.0f});
                }
                else
                {
                    fused_builder.normalize(
                        {1.0f, 2.0f, 3.0f},
                        {2.0f, 4.0f, 8.0f});
                }
                const cvh::PipelinePlan fused =
                    fused_builder.layout(layout).prepare();

                cvh::PipelineBuilder staged_builder = cvh::pipe(
                    cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
                    output_desc);
                staged_builder
                    .resize(3, 2, interpolation)
                    .color(
                        has_color ? cvh::Color::RGB : cvh::Color::BGR);
                if (layout == cvh::Layout::NCHW)
                {
                    staged_builder.normalize({1.5f}, {2.0f});
                }
                else
                {
                    staged_builder.normalize(
                        {1.0f, 2.0f, 3.0f},
                        {2.0f, 4.0f, 8.0f});
                }
                const cvh::PipelinePlan staged =
                    staged_builder.layout(layout).prepare();

                EXPECT_EQ(fused.info().execution_group_count, 1);
                EXPECT_EQ(fused.info().full_frame_intermediates, 0);
                EXPECT_EQ(
                    fused.info().execution_class,
                    cvh::PipelineExecutionClass::FusedTiled);
                EXPECT_EQ(staged.info().execution_group_count, 4);
                EXPECT_EQ(staged.info().full_frame_intermediates, 3);

                const std::vector<int> output_shape =
                    layout == cvh::Layout::NCHW
                    ? std::vector<int>{1, 3, 2, 3}
                    : std::vector<int>{1, 2, 3, 3};
                cvh::Mat fused_output(output_shape, CV_32FC1);
                cvh::Mat staged_output(output_shape, CV_32FC1);
                cvh::PipelineWorkspace fused_workspace(fused);
                cvh::PipelineWorkspace staged_workspace(staged);
                fused.run(input, fused_output, fused_workspace.view());
                staged.run(input, staged_output, staged_workspace.view());
                expectSameFloatMat(fused_output, staged_output);
            }
        }
    }
}

TEST(PipelineRecipeTest, model_input_is_strict_equivalent_and_fingerprinted)
{
    cvh::ModelInputRecipe recipe;
    recipe.input = cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8);
    recipe.output =
        cvh::tensorDesc<float>({1, 3, 2, 2}, cvh::Layout::NCHW);
    recipe.color = cvh::Color::RGB;
    recipe.interpolation = cvh::Interpolation::Linear;
    recipe.mean = {{1.0f, 2.0f, 3.0f, 0.0f}};
    recipe.stddev = {{2.0f, 4.0f, 8.0f, 1.0f}};
    recipe.normalize_count = 3;

    const cvh::PipelinePlan recipe_plan =
        cvh::recipes::modelInput(recipe)
            .requireNoFullFrameIntermediate()
            .requireSingleExecutionGroup()
            .prepare();
    const cvh::PipelinePlan same_recipe_plan =
        cvh::recipes::modelInput(recipe).prepare();
    EXPECT_STREQ(
        recipe_plan.info().recipe_id,
        "cvh.model_input.packed_f32");
    EXPECT_EQ(recipe_plan.info().recipe_contract_version, 1u);
    EXPECT_NE(recipe_plan.info().recipe_fingerprint, 0u);
    EXPECT_EQ(
        recipe_plan.info().recipe_fingerprint,
        same_recipe_plan.info().recipe_fingerprint);
    EXPECT_EQ(recipe_plan.info().execution_group_count, 1);
    EXPECT_EQ(recipe_plan.info().full_frame_intermediates, 0);
    EXPECT_NE(
        recipe_plan.explain().find(
            "recipe: cvh.model_input.packed_f32 v1"),
        std::string::npos);

    cvh::ModelInputRecipe changed_recipe = recipe;
    changed_recipe.mean[0] += 1.0f;
    const cvh::PipelinePlan changed_plan =
        cvh::recipes::modelInput(changed_recipe).prepare();
    EXPECT_NE(
        recipe_plan.info().recipe_fingerprint,
        changed_plan.info().recipe_fingerprint);

    const cvh::PipelinePlan ordinary_plan =
        cvh::pipe(recipe.input, recipe.output)
            .color(cvh::Color::RGB)
            .resize(2, 2, cvh::Interpolation::Linear)
            .normalize({1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 8.0f})
            .layout(cvh::Layout::NCHW)
            .prepare();
    EXPECT_EQ(ordinary_plan.info().recipe_id, nullptr);
    EXPECT_EQ(ordinary_plan.info().recipe_contract_version, 0u);
    EXPECT_EQ(ordinary_plan.info().recipe_fingerprint, 0u);

    const cvh::Mat input = makeInput();
    cvh::Mat recipe_output({1, 3, 2, 2}, CV_32FC1);
    cvh::Mat ordinary_output({1, 3, 2, 2}, CV_32FC1);
    cvh::PipelineWorkspace recipe_workspace(recipe_plan);
    cvh::PipelineWorkspace ordinary_workspace(ordinary_plan);
    recipe_plan.run(input, recipe_output, recipe_workspace.view());
    ordinary_plan.run(input, ordinary_output, ordinary_workspace.view());
    expectSameFloatMat(recipe_output, ordinary_output);
}

TEST(PipelineRecipeTest, yuv_model_input_is_strict_and_colorspec_fingerprinted)
{
    const cvh::ColorSpec color_spec{
        cvh::ColorMatrix::BT709,
        cvh::ColorRange::Limited,
        cvh::ChromaLocation::Left};
    cvh::ModelInputRecipe recipe;
    recipe.input =
        cvh::imageDesc(4, 4, cvh::PixelFormat::NV12, color_spec);
    recipe.output =
        cvh::tensorDesc<float>({1, 2, 3, 3}, cvh::Layout::NHWC);
    recipe.color = cvh::Color::RGB;
    recipe.interpolation = cvh::Interpolation::Linear;
    recipe.mean = {{1.0f, 2.0f, 3.0f, 0.0f}};
    recipe.stddev = {{2.0f, 4.0f, 8.0f, 1.0f}};
    recipe.normalize_count = 3;

    const cvh::PipelinePlan recipe_plan =
        cvh::recipes::modelInput(recipe).prepare();
    ASSERT_STREQ(
        recipe_plan.info().recipe_id,
        "cvh.model_input.yuv420_f32");
    EXPECT_EQ(recipe_plan.info().recipe_contract_version, 1u);
    EXPECT_EQ(recipe_plan.info().execution_group_count, 1);
    EXPECT_EQ(recipe_plan.info().full_frame_intermediates, 0);
    EXPECT_EQ(recipe_plan.info().workspace_bytes, 0u);
    EXPECT_NE(
        recipe_plan.explain().find(
            "recipe: cvh.model_input.yuv420_f32 v1"),
        std::string::npos);

    cvh::ModelInputRecipe changed_recipe = recipe;
    changed_recipe.input.color_spec.range = cvh::ColorRange::Full;
    const cvh::PipelinePlan changed_plan =
        cvh::recipes::modelInput(changed_recipe).prepare();
    EXPECT_NE(
        recipe_plan.info().recipe_fingerprint,
        changed_plan.info().recipe_fingerprint);

    const cvh::PipelinePlan ordinary_plan =
        cvh::pipe(recipe.input, recipe.output)
            .color(cvh::Color::RGB)
            .resize(3, 2, cvh::Interpolation::Linear)
            .normalize({1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 8.0f})
            .layout(cvh::Layout::NHWC)
            .prepare();
    std::array<uchar, 24> y_storage{};
    std::array<uchar, 12> uv_storage{};
    for (std::size_t index = 0; index < y_storage.size(); ++index)
    {
        y_storage[index] = static_cast<uchar>(17 + index * 7);
    }
    for (std::size_t index = 0; index < uv_storage.size(); ++index)
    {
        uv_storage[index] = static_cast<uchar>(43 + index * 13);
    }
    const cvh::ConstImageView input = cvh::nv12(
        y_storage.data(),
        6,
        y_storage.size(),
        uv_storage.data(),
        6,
        uv_storage.size(),
        4,
        4,
        color_spec);
    std::array<float, 18> recipe_output{};
    std::array<float, 18> ordinary_output{};
    cvh::PipelineWorkspace recipe_workspace(recipe_plan);
    cvh::PipelineWorkspace ordinary_workspace(ordinary_plan);
    recipe_plan.run(
        input,
        cvh::nhwc(
            recipe_output.data(),
            recipe_output.size() * sizeof(float),
            1,
            2,
            3,
            3),
        recipe_workspace.view());
    ordinary_plan.run(
        input,
        cvh::nhwc(
            ordinary_output.data(),
            ordinary_output.size() * sizeof(float),
            1,
            2,
            3,
            3),
        ordinary_workspace.view());
    for (std::size_t index = 0; index < recipe_output.size(); ++index)
    {
        EXPECT_FLOAT_EQ(recipe_output[index], ordinary_output[index]);
    }
}

TEST(PipelineRecipeTest, model_input_rejects_unsupported_contracts)
{
    cvh::ModelInputRecipe recipe;
    recipe.input = cvh::imageDesc(4, 3, cvh::PixelFormat::Gray8);
    recipe.output =
        cvh::tensorDesc<float>({1, 3, 2, 2}, cvh::Layout::NCHW);
    cvh::PipelinePlan plan;
    cvh::PipelineStatus status =
        cvh::recipes::modelInput(recipe).tryPrepare(plan);
    EXPECT_EQ(status.code(), cvh::PipelineStatusCode::Unsupported);
    EXPECT_FALSE(plan.valid());

    recipe.input = cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8);
    recipe.output =
        cvh::tensorDesc<double>({1, 3, 2, 2}, cvh::Layout::NCHW);
    status = cvh::recipes::modelInput(recipe).tryPrepare(plan);
    EXPECT_EQ(status.code(), cvh::PipelineStatusCode::Unsupported);
    EXPECT_FALSE(plan.valid());

    recipe.output =
        cvh::tensorDesc<float>({1, 3, 2, 2}, cvh::Layout::NCHW);
    recipe.normalize_count = 2;
    status = cvh::recipes::modelInput(recipe).tryPrepare(plan);
    EXPECT_EQ(status.code(), cvh::PipelineStatusCode::InvalidOperation);
    EXPECT_FALSE(plan.valid());
}

TEST(PipelineDispatchTest, nearest_half_width_neon_matches_forced_scalar)
{
    cvh::Mat storage({24, 24}, CV_8UC3);
    for (int y = 0; y < storage.size[0]; ++y)
    {
        for (int x = 0; x < storage.size[1]; ++x)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                storage.at<uchar>(y, x, channel) =
                    static_cast<uchar>(
                        (y * 31 + x * 17 + channel * 73) & 0xff);
            }
        }
    }
    const cvh::Mat input =
        storage(cvh::Range::all(), cvh::Range(1, 23));
    ASSERT_FALSE(input.isContinuous());
    const cvh::PipelinePlan plan =
        cvh::pipe(
            cvh::imageDesc(22, 24, cvh::PixelFormat::BGR8),
            cvh::tensorDesc<float>({1, 3, 24, 11}, cvh::Layout::NCHW))
            .color(cvh::Color::RGB)
            .resize(11, 24, cvh::Interpolation::Nearest)
            .normalize({1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 8.0f})
            .layout(cvh::Layout::NCHW)
            .prepare();

    cvh::Mat scalar_output({1, 3, 24, 11}, CV_32FC1);
    cvh::PipelineWorkspace scalar_workspace(plan);
    cvh::PipelineRunInfo scalar_info;
    {
        const cvh::test::DispatchModeGuard guard(
            cvh::cpu::DispatchMode::ScalarOnly);
        cvh::cpu::reset_last_dispatch_tag();
        plan.run(
            input,
            scalar_output,
            scalar_workspace.view(),
            &scalar_info);
        EXPECT_EQ(
            cvh::cpu::last_dispatch_tag(),
            cvh::cpu::DispatchTag::Scalar);
    }
    EXPECT_EQ(scalar_info.actual_route, cvh::PipelineRoute::Scalar);
    EXPECT_EQ(scalar_info.observed_isa, cvh::PipelineRoute::Scalar);
    EXPECT_FALSE(scalar_info.used_fallback);

    cvh::Mat selected_output({1, 3, 24, 11}, CV_32FC1);
    cvh::PipelineWorkspace selected_workspace(plan);
    cvh::PipelineRunInfo selected_info;
    {
        const cvh::test::DispatchModeGuard guard(
            cvh::cpu::DispatchMode::NeonOnly);
        cvh::cpu::reset_last_dispatch_tag();
        plan.run(
            input,
            selected_output,
            selected_workspace.view(),
            &selected_info);
        if (cvh::cpu::neon_runtime_available())
        {
            EXPECT_EQ(
                cvh::cpu::last_dispatch_tag(),
                cvh::cpu::DispatchTag::NEON);
        }
        else
        {
            EXPECT_EQ(
                cvh::cpu::last_dispatch_tag(),
                cvh::cpu::DispatchTag::Scalar);
        }
    }

    if (cvh::cpu::neon_runtime_available())
    {
        EXPECT_EQ(plan.info().candidate_route, cvh::PipelineRoute::Neon);
        EXPECT_EQ(selected_info.actual_route, cvh::PipelineRoute::Neon);
        EXPECT_EQ(selected_info.observed_isa, cvh::PipelineRoute::Neon);
        EXPECT_FALSE(selected_info.used_fallback);
        EXPECT_NE(
            plan.explain().find("candidate route: neon"),
            std::string::npos);
    }
    else
    {
        EXPECT_EQ(plan.info().candidate_route, cvh::PipelineRoute::Scalar);
        EXPECT_EQ(selected_info.actual_route, cvh::PipelineRoute::Scalar);
        EXPECT_EQ(selected_info.observed_isa, cvh::PipelineRoute::Scalar);
        EXPECT_TRUE(selected_info.used_fallback);
        ASSERT_NE(selected_info.fallback_reason, nullptr);
    }
    expectSameFloatMat(scalar_output, selected_output);

    cvh::Mat auto_output({1, 3, 24, 11}, CV_32FC1);
    cvh::PipelineWorkspace auto_workspace(plan);
    cvh::PipelineRunInfo auto_info;
    {
        const cvh::test::DispatchModeGuard guard(
            cvh::cpu::DispatchMode::Auto);
        plan.run(input, auto_output, auto_workspace.view(), &auto_info);
    }
    EXPECT_EQ(
        auto_info.actual_route,
        cvh::cpu::neon_runtime_available()
            ? cvh::PipelineRoute::Neon
            : cvh::PipelineRoute::Scalar);
    expectSameFloatMat(scalar_output, auto_output);

    const cvh::PipelinePlan small_plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::tensorDesc<float>({1, 3, 2, 2}, cvh::Layout::NCHW))
            .resize(2, 2, cvh::Interpolation::Nearest)
            .normalize({0.0f}, {1.0f})
            .layout(cvh::Layout::NCHW)
            .prepare();
    cvh::Mat small_output({1, 3, 2, 2}, CV_32FC1);
    cvh::PipelineWorkspace small_workspace(small_plan);
    cvh::PipelineRunInfo small_info;
    {
        const cvh::test::DispatchModeGuard guard(
            cvh::cpu::DispatchMode::NeonOnly);
        small_plan.run(
            makeInput(),
            small_output,
            small_workspace.view(),
            &small_info);
    }
    EXPECT_EQ(small_info.actual_route, cvh::PipelineRoute::Scalar);
    EXPECT_EQ(small_info.observed_isa, cvh::PipelineRoute::Scalar);
    EXPECT_TRUE(small_info.used_fallback);
    ASSERT_NE(small_info.fallback_reason, nullptr);
}

TEST(PipelineDispatchTest, nearest_letterbox_neon_matches_forced_scalar)
{
    cvh::Mat input({48, 22}, CV_8UC3);
    for (int y = 0; y < input.size[0]; ++y)
    {
        for (int x = 0; x < input.size[1]; ++x)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                input.at<uchar>(y, x, channel) = static_cast<uchar>(
                    y * 17 + x * 9 + channel * 43);
            }
        }
    }
    const cvh::PipelinePlan plan =
        cvh::pipe(
            cvh::imageDesc(22, 48, cvh::PixelFormat::BGR8),
            cvh::tensorDesc<float>({1, 3, 24, 15}, cvh::Layout::NCHW))
            .color(cvh::Color::RGB)
            .letterbox(
                15,
                24,
                {11.0f, 22.0f, 33.0f},
                cvh::Interpolation::Nearest)
            .normalize({1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 8.0f})
            .layout(cvh::Layout::NCHW)
            .prepare();
    ASSERT_TRUE(plan.hasTransform());
    EXPECT_EQ(plan.transform().resized_width, 11);
    EXPECT_EQ(plan.transform().resized_height, 24);
    EXPECT_EQ(plan.transform().pad_left, 2);
    EXPECT_EQ(plan.transform().pad_right, 2);

    cvh::Mat scalar_output({1, 3, 24, 15}, CV_32FC1);
    cvh::Mat neon_output({1, 3, 24, 15}, CV_32FC1);
    cvh::PipelineWorkspace scalar_workspace(plan);
    cvh::PipelineWorkspace neon_workspace(plan);
    cvh::PipelineRunInfo scalar_info;
    cvh::PipelineRunInfo neon_info;
    {
        const cvh::test::DispatchModeGuard guard(
            cvh::cpu::DispatchMode::ScalarOnly);
        plan.run(
            input,
            scalar_output,
            scalar_workspace.view(),
            &scalar_info);
    }
    {
        const cvh::test::DispatchModeGuard guard(
            cvh::cpu::DispatchMode::NeonOnly);
        plan.run(
            input,
            neon_output,
            neon_workspace.view(),
            &neon_info);
    }
    EXPECT_EQ(scalar_info.actual_route, cvh::PipelineRoute::Scalar);
    if (cvh::cpu::neon_runtime_available())
    {
        EXPECT_EQ(plan.info().candidate_route, cvh::PipelineRoute::Neon);
        EXPECT_EQ(neon_info.actual_route, cvh::PipelineRoute::Neon);
        EXPECT_EQ(neon_info.observed_isa, cvh::PipelineRoute::Neon);
        EXPECT_FALSE(neon_info.used_fallback);
    }
    else
    {
        EXPECT_EQ(neon_info.actual_route, cvh::PipelineRoute::Scalar);
        EXPECT_TRUE(neon_info.used_fallback);
    }
    expectSameFloatMat(scalar_output, neon_output);
}

TEST(PipelinePreparedTest, non_contiguous_image_input_is_supported)
{
    cvh::Mat storage({5, 6}, CV_8UC3);
    for (int y = 0; y < storage.size[0]; ++y)
    {
        for (int x = 0; x < storage.size[1]; ++x)
        {
            for (int channel = 0; channel < 3; ++channel)
            {
                storage.at<uchar>(y, x, channel) =
                    static_cast<uchar>(y * 31 + x * 7 + channel);
            }
        }
    }
    const cvh::Mat roi =
        storage(cvh::Range(1, 4), cvh::Range(1, 5));
    ASSERT_FALSE(roi.isContinuous());

    const auto plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::imageDesc(2, 2, cvh::PixelFormat::BGR8))
            .resize(2, 2, cvh::Interpolation::Nearest)
            .prepare();

    cvh::Mat output({2, 2}, CV_8UC3);
    cvh::PipelineWorkspace workspace(plan);
    plan.run(roi, output, workspace.view());

    for (int y = 0; y < 2; ++y)
    {
        const int source_y = (y * 3) / 2;
        for (int x = 0; x < 2; ++x)
        {
            const int source_x = (x * 4) / 2;
            for (int channel = 0; channel < 3; ++channel)
            {
                EXPECT_EQ(
                    output.at<uchar>(y, x, channel),
                    roi.at<uchar>(source_y, source_x, channel));
            }
        }
    }
}

TEST(PipelineBorrowedViewTest, padded_bgr_input_matches_mat_execution)
{
    const cvh::Mat input = makeInput();
    const cvh::PipelinePlan plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::tensorDesc<float>({1, 3, 2, 2}, cvh::Layout::NCHW))
            .color(cvh::Color::RGB)
            .resize(2, 2, cvh::Interpolation::Linear)
            .normalize({1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 8.0f})
            .layout(cvh::Layout::NCHW)
            .prepare();

    cvh::Mat expected({1, 3, 2, 2}, CV_32FC1);
    cvh::PipelineWorkspace mat_workspace(plan);
    plan.run(input, expected, mat_workspace.view());

    constexpr std::size_t row_stride = 17;
    std::array<uchar, row_stride * 3 + 1> padded_input{};
    uchar* const unaligned_data = padded_input.data() + 1;
    for (int y = 0; y < input.size[0]; ++y)
    {
        std::memcpy(
            unaligned_data + static_cast<std::size_t>(y) * row_stride,
            input.data + static_cast<std::size_t>(y) * input.step(0),
            12);
    }
    std::array<float, 12> output{};
    const cvh::ConstImageView input_view = cvh::bgr(
        unaligned_data,
        padded_input.size() - 1,
        4,
        3,
        row_stride);
    const cvh::TensorView output_view =
        cvh::nchw(output.data(), output.size() * sizeof(float), 1, 3, 2, 2);
    cvh::PipelineWorkspace view_workspace(plan);
    cvh::PipelineRunInfo run_info;
    const cvh::PipelineStatus status = plan.tryRun(
        input_view, output_view, view_workspace.view(), &run_info);

    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(run_info.actual_route, cvh::PipelineRoute::Scalar);
    EXPECT_EQ(run_info.observed_isa, cvh::PipelineRoute::Scalar);
    const float* expected_values =
        reinterpret_cast<const float*>(expected.data);
    for (std::size_t index = 0; index < output.size(); ++index)
    {
        EXPECT_NEAR(output[index], expected_values[index], 1e-5f)
            << "index=" << index;
    }
}

TEST(PipelineBorrowedViewTest, rgb_input_and_nhwc_output_match_mat_execution)
{
    const cvh::Mat bgr_input = makeInput();
    const cvh::Mat rgb_input = referenceColor(
        bgr_input, cvh::Color::BGR, cvh::Color::RGB);
    const cvh::PipelinePlan plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::RGB8),
            cvh::tensorDesc<float>({1, 2, 3, 3}, cvh::Layout::NHWC))
            .resize(3, 2, cvh::Interpolation::Nearest)
            .normalize({1.0f, 2.0f, 3.0f}, {2.0f, 4.0f, 8.0f})
            .layout(cvh::Layout::NHWC)
            .prepare();

    cvh::Mat expected({1, 2, 3, 3}, CV_32FC1);
    cvh::PipelineWorkspace mat_workspace(plan);
    plan.run(rgb_input, expected, mat_workspace.view());

    std::array<float, 18> output{};
    const cvh::ConstImageView input_view = cvh::rgb(
        rgb_input.data,
        static_cast<std::size_t>(rgb_input.size[0]) * rgb_input.step(0),
        rgb_input.size[1],
        rgb_input.size[0],
        rgb_input.step(0));
    const cvh::TensorView output_view = cvh::nhwc(
        output.data(), output.size() * sizeof(float), 1, 2, 3, 3);
    cvh::PipelineWorkspace view_workspace(plan);
    const cvh::PipelineStatus status =
        plan.tryRun(input_view, output_view, view_workspace.view());

    ASSERT_TRUE(status.ok()) << status.message();
    const float* expected_values =
        reinterpret_cast<const float*>(expected.data);
    for (std::size_t index = 0; index < output.size(); ++index)
    {
        EXPECT_NEAR(output[index], expected_values[index], 1e-5f)
            << "index=" << index;
    }
}

TEST(PipelineBorrowedViewTest, validation_is_typed_and_never_partially_writes)
{
    const cvh::PipelinePlan plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::tensorDesc<float>({1, 3, 2, 2}, cvh::Layout::NCHW))
            .resize(2, 2, cvh::Interpolation::Nearest)
            .color(cvh::Color::BGR)
            .normalize({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f})
            .layout(cvh::Layout::NCHW)
            .prepare();
    std::array<uchar, 36> input_storage{};
    std::array<float, 12> output_storage{};
    const cvh::ConstImageView valid_input = cvh::bgr(
        input_storage.data(), input_storage.size(), 4, 3, 12);
    const cvh::TensorView valid_output = cvh::nchw(
        output_storage.data(),
        output_storage.size() * sizeof(float),
        1,
        3,
        2,
        2);
    cvh::PipelineWorkspace workspace(plan);

    cvh::ConstImageView wrong_type = valid_input;
    wrong_type.descriptor =
        cvh::imageDesc(4, 3, cvh::PixelFormat::RGB8);
    EXPECT_EQ(
        plan.tryRun(wrong_type, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::TypeMismatch);

    cvh::ConstImageView wrong_shape = valid_input;
    wrong_shape.descriptor.width = 3;
    EXPECT_EQ(
        plan.tryRun(wrong_shape, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::ShapeMismatch);

    cvh::ConstImageView short_stride = valid_input;
    short_stride.planes[0].row_stride = 11;
    EXPECT_EQ(
        plan.tryRun(short_stride, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::BufferTooSmall);

    cvh::ConstImageView short_input = valid_input;
    short_input.planes[0].size_bytes = input_storage.size() - 1;
    EXPECT_EQ(
        plan.tryRun(short_input, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::BufferTooSmall);

    cvh::ConstImageView null_input = valid_input;
    null_input.planes[0].data = nullptr;
    EXPECT_EQ(
        plan.tryRun(null_input, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::BufferTooSmall);

    cvh::ConstImageView inconsistent_planes = valid_input;
    inconsistent_planes.plane_count = 2;
    EXPECT_EQ(
        plan.tryRun(
                inconsistent_planes, valid_output, workspace.view())
            .code(),
        cvh::PipelineStatusCode::InvalidDescriptor);

    cvh::TensorView wrong_output_type = cvh::nchw(
        input_storage.data(), input_storage.size(), 1, 3, 2, 2);
    EXPECT_EQ(
        plan.tryRun(valid_input, wrong_output_type, workspace.view()).code(),
        cvh::PipelineStatusCode::TypeMismatch);

    cvh::TensorView wrong_output_shape = valid_output;
    wrong_output_shape.descriptor.shape[3] = 1;
    EXPECT_EQ(
        plan.tryRun(valid_input, wrong_output_shape, workspace.view()).code(),
        cvh::PipelineStatusCode::ShapeMismatch);

    cvh::TensorView short_output = valid_output;
    short_output.size_bytes -= sizeof(float);
    for (float& value : output_storage)
    {
        value = -77.0f;
    }
    EXPECT_EQ(
        plan.tryRun(valid_input, short_output, workspace.view()).code(),
        cvh::PipelineStatusCode::BufferTooSmall);
    for (float value : output_storage)
    {
        EXPECT_FLOAT_EQ(value, -77.0f);
    }

    cvh::TensorView null_output = valid_output;
    null_output.data = nullptr;
    EXPECT_EQ(
        plan.tryRun(valid_input, null_output, workspace.view()).code(),
        cvh::PipelineStatusCode::BufferTooSmall);

    const cvh::PipelinePlan other_plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::tensorDesc<float>({1, 3, 2, 2}, cvh::Layout::NCHW))
            .resize(2, 2, cvh::Interpolation::Nearest)
            .color(cvh::Color::BGR)
            .normalize({0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f})
            .layout(cvh::Layout::NCHW)
            .prepare();
    cvh::PipelineWorkspace other_workspace(other_plan);
    EXPECT_EQ(
        plan.tryRun(
                valid_input, valid_output, other_workspace.view())
            .code(),
        cvh::PipelineStatusCode::WorkspaceMismatch);

    alignas(float) std::array<uchar, 64> overlapping_storage{};
    const cvh::ConstImageView overlapping_input = cvh::bgr(
        overlapping_storage.data(), 36, 4, 3, 12);
    const cvh::TensorView overlapping_output = cvh::nchw(
        reinterpret_cast<float*>(overlapping_storage.data()),
        48,
        1,
        3,
        2,
        2);
    EXPECT_EQ(
        plan.tryRun(
                overlapping_input, overlapping_output, workspace.view())
            .code(),
        cvh::PipelineStatusCode::AliasingNotSupported);

    const cvh::ConstImageView workspace_input = cvh::bgr(
        static_cast<const uchar*>(workspace.view().data()),
        workspace.size(),
        4,
        3,
        12);
    EXPECT_EQ(
        plan.tryRun(workspace_input, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::AliasingNotSupported);

    const cvh::PipelinePlan image_output_plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::imageDesc(2, 2, cvh::PixelFormat::BGR8))
            .resize(2, 2)
            .prepare();
    cvh::PipelineWorkspace image_output_workspace(image_output_plan);
    EXPECT_EQ(
        image_output_plan
            .tryRun(
                valid_input,
                valid_output,
                image_output_workspace.view())
            .code(),
        cvh::PipelineStatusCode::InvalidDescriptor);
}

TEST(PipelineYuvTest, nv12_nv21_color_contract_matches_independent_reference)
{
    constexpr int source_width = 4;
    constexpr int source_height = 4;
    constexpr int target_width = 3;
    constexpr int target_height = 2;
    constexpr std::size_t y_stride = 7;
    constexpr std::size_t uv_stride = 6;
    std::array<uchar, y_stride * source_height> y_storage{};
    for (int y = 0; y < source_height; ++y)
    {
        for (int x = 0; x < source_width; ++x)
        {
            y_storage[static_cast<std::size_t>(y) * y_stride + x] =
                static_cast<uchar>(24 + y * 41 + x * 17);
        }
    }
    const std::array<std::array<uchar, 2>, 4> chroma{{
        {{84, 196}},
        {{146, 43}},
        {{211, 118}},
        {{57, 232}}}};
    const std::array<cvh::ColorMatrix, 3> matrices{{
        cvh::ColorMatrix::BT601,
        cvh::ColorMatrix::BT709,
        cvh::ColorMatrix::BT2020}};
    const std::array<cvh::ColorRange, 2> ranges{{
        cvh::ColorRange::Limited,
        cvh::ColorRange::Full}};
    const std::array<cvh::ChromaLocation, 2> locations{{
        cvh::ChromaLocation::Center,
        cvh::ChromaLocation::Left}};
    const std::array<cvh::Interpolation, 2> interpolations{{
        cvh::Interpolation::Nearest,
        cvh::Interpolation::Linear}};
    const std::array<cvh::Color, 2> output_colors{{
        cvh::Color::RGB,
        cvh::Color::BGR}};
    const std::array<cvh::Layout, 2> layouts{{
        cvh::Layout::NCHW,
        cvh::Layout::NHWC}};
    const std::array<cvh::PixelFormat, 2> formats{{
        cvh::PixelFormat::NV12,
        cvh::PixelFormat::NV21}};
    const std::array<float, 3> mean{{3.0f, 7.0f, 11.0f}};
    const std::array<float, 3> stddev{{2.0f, 3.0f, 5.0f}};

    for (cvh::PixelFormat format : formats)
    {
        std::array<uchar, uv_stride * (source_height / 2)> uv_storage{};
        for (int y = 0; y < source_height / 2; ++y)
        {
            for (int x = 0; x < source_width / 2; ++x)
            {
                const auto& value = chroma[static_cast<std::size_t>(
                    y * (source_width / 2) + x)];
                const std::size_t offset =
                    static_cast<std::size_t>(y) * uv_stride + x * 2;
                uv_storage[offset] = format == cvh::PixelFormat::NV12
                    ? value[0]
                    : value[1];
                uv_storage[offset + 1] = format == cvh::PixelFormat::NV12
                    ? value[1]
                    : value[0];
            }
        }
        for (cvh::ColorMatrix matrix : matrices)
        {
            for (cvh::ColorRange range : ranges)
            {
                for (cvh::ChromaLocation location : locations)
                {
                    for (cvh::Interpolation interpolation : interpolations)
                    {
                        for (cvh::Color output_color : output_colors)
                        {
                            for (cvh::Layout layout : layouts)
                            {
                                SCOPED_TRACE(::testing::Message()
                                    << "format=" << static_cast<int>(format)
                                    << " matrix=" << static_cast<int>(matrix)
                                    << " range=" << static_cast<int>(range)
                                    << " location=" << static_cast<int>(location)
                                    << " interpolation="
                                    << static_cast<int>(interpolation)
                                    << " color="
                                    << static_cast<int>(output_color)
                                    << " layout=" << static_cast<int>(layout));
                                const cvh::ColorSpec color_spec{
                                    matrix, range, location};
                                const cvh::ConstImageView input =
                                    format == cvh::PixelFormat::NV12
                                    ? cvh::nv12(
                                          y_storage.data(),
                                          y_stride,
                                          y_storage.size(),
                                          uv_storage.data(),
                                          uv_stride,
                                          uv_storage.size(),
                                          source_width,
                                          source_height,
                                          color_spec)
                                    : cvh::nv21(
                                          y_storage.data(),
                                          y_stride,
                                          y_storage.size(),
                                          uv_storage.data(),
                                          uv_stride,
                                          uv_storage.size(),
                                          source_width,
                                          source_height,
                                          color_spec);
                                const cvh::TensorDescriptor output_descriptor =
                                    layout == cvh::Layout::NCHW
                                    ? cvh::tensorDesc<float>(
                                          {1, 3, target_height, target_width},
                                          layout)
                                    : cvh::tensorDesc<float>(
                                          {1, target_height, target_width, 3},
                                          layout);
                                const cvh::PipelinePlan plan =
                                    cvh::pipe(
                                        input.descriptor,
                                        output_descriptor)
                                        .color(output_color)
                                        .resize(
                                            target_width,
                                            target_height,
                                            interpolation)
                                        .normalize(
                                            {mean[0], mean[1], mean[2]},
                                            {stddev[0], stddev[1], stddev[2]})
                                        .layout(layout)
                                        .prepare();
                                EXPECT_EQ(plan.info().semantic_stage_count, 4);
                                EXPECT_EQ(plan.info().execution_group_count, 1);
                                EXPECT_EQ(plan.info().full_frame_intermediates, 0);
                                EXPECT_EQ(plan.info().workspace_bytes, 0u);
                                EXPECT_NE(
                                    plan.explain().find(
                                        "yuv420-model-input"),
                                    std::string::npos);

                                std::array<float,
                                           target_width * target_height * 3>
                                    output{};
                                const cvh::TensorView output_view =
                                    layout == cvh::Layout::NCHW
                                    ? cvh::nchw(
                                          output.data(),
                                          output.size() * sizeof(float),
                                          1,
                                          3,
                                          target_height,
                                          target_width)
                                    : cvh::nhwc(
                                          output.data(),
                                          output.size() * sizeof(float),
                                          1,
                                          target_height,
                                          target_width,
                                          3);
                                cvh::PipelineWorkspace workspace(plan);
                                cvh::PipelineRunInfo run_info;
                                const cvh::PipelineStatus status = plan.tryRun(
                                    input,
                                    output_view,
                                    workspace.view(),
                                    &run_info);
                                ASSERT_TRUE(status.ok()) << status.message();
                                EXPECT_EQ(
                                    run_info.actual_route,
                                    cvh::PipelineRoute::Scalar);
                                EXPECT_EQ(
                                    run_info.observed_isa,
                                    cvh::PipelineRoute::Scalar);
                                const std::vector<float> expected =
                                    referenceYuvModelInput(
                                        input,
                                        target_width,
                                        target_height,
                                        interpolation,
                                        output_color,
                                        layout,
                                        mean,
                                        stddev);
                                ASSERT_EQ(output.size(), expected.size());
                                for (std::size_t index = 0;
                                     index < output.size();
                                     ++index)
                                {
                                    EXPECT_NEAR(
                                        output[index], expected[index], 1e-5f)
                                        << "index=" << index;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(PipelineYuvTest, neutral_black_gray_white_anchors_are_stable)
{
    const std::array<cvh::ColorRange, 2> ranges{{
        cvh::ColorRange::Limited,
        cvh::ColorRange::Full}};
    for (cvh::ColorRange range : ranges)
    {
        const std::array<uchar, 4> y_storage =
            range == cvh::ColorRange::Limited
            ? std::array<uchar, 4>{{16, 126, 235, 64}}
            : std::array<uchar, 4>{{0, 128, 255, 64}};
        const std::array<uchar, 2> uv_storage{{128, 128}};
        const cvh::ColorSpec color_spec{
            cvh::ColorMatrix::BT709,
            range,
            cvh::ChromaLocation::Center};
        const cvh::ConstImageView input = cvh::nv12(
            y_storage.data(),
            2,
            y_storage.size(),
            uv_storage.data(),
            2,
            uv_storage.size(),
            2,
            2,
            color_spec);
        const cvh::PipelinePlan plan =
            cvh::pipe(
                input.descriptor,
                cvh::tensorDesc<float>({1, 3, 2, 2}, cvh::Layout::NCHW))
                .color(cvh::Color::RGB)
                .resize(2, 2, cvh::Interpolation::Nearest)
                .normalize({0.0f}, {1.0f})
                .layout(cvh::Layout::NCHW)
                .prepare();
        std::array<float, 12> output{};
        cvh::PipelineWorkspace workspace(plan);
        plan.run(
            input,
            cvh::nchw(
                output.data(),
                output.size() * sizeof(float),
                1,
                3,
                2,
                2),
            workspace.view());
        const std::array<float, 4> expected =
            range == cvh::ColorRange::Limited
            ? std::array<float, 4>{{0.0f, 128.0f, 255.0f, 56.0f}}
            : std::array<float, 4>{{0.0f, 128.0f, 255.0f, 64.0f}};
        for (int channel = 0; channel < 3; ++channel)
        {
            for (std::size_t pixel = 0; pixel < expected.size(); ++pixel)
            {
                EXPECT_FLOAT_EQ(
                    output[static_cast<std::size_t>(channel) * 4 + pixel],
                    expected[pixel]);
            }
        }
    }
}

TEST(PipelineYuvTest, two_plane_validation_is_typed_and_non_mutating)
{
    const cvh::ColorSpec color_spec{
        cvh::ColorMatrix::BT709,
        cvh::ColorRange::Limited,
        cvh::ChromaLocation::Left};
    const cvh::ImageDescriptor input_descriptor =
        cvh::imageDesc(4, 4, cvh::PixelFormat::NV12, color_spec);
    const cvh::TensorDescriptor output_descriptor =
        cvh::tensorDesc<float>({1, 3, 2, 2}, cvh::Layout::NCHW);
    const cvh::PipelinePlan plan =
        cvh::pipe(input_descriptor, output_descriptor)
            .color(cvh::Color::RGB)
            .resize(2, 2, cvh::Interpolation::Nearest)
            .normalize({0.0f}, {1.0f})
            .layout(cvh::Layout::NCHW)
            .prepare();
    alignas(float) std::array<uchar, 32> y_storage{};
    std::array<uchar, 16> uv_storage{};
    std::array<float, 12> output_storage{};
    const cvh::ConstImageView valid_input = cvh::nv12(
        y_storage.data(),
        6,
        22,
        uv_storage.data(),
        6,
        10,
        4,
        4,
        color_spec);
    const cvh::TensorView valid_output = cvh::nchw(
        output_storage.data(),
        output_storage.size() * sizeof(float),
        1,
        3,
        2,
        2);
    cvh::PipelineWorkspace workspace(plan);

    cvh::ConstImageView wrong_spec = valid_input;
    wrong_spec.descriptor.color_spec.range = cvh::ColorRange::Full;
    EXPECT_EQ(
        plan.tryRun(wrong_spec, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::TypeMismatch);

    cvh::ConstImageView wrong_format = valid_input;
    wrong_format.descriptor.pixel_format = cvh::PixelFormat::NV21;
    EXPECT_EQ(
        plan.tryRun(wrong_format, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::TypeMismatch);

    cvh::ConstImageView inconsistent_planes = valid_input;
    inconsistent_planes.plane_count = 1;
    EXPECT_EQ(
        plan.tryRun(
                inconsistent_planes, valid_output, workspace.view())
            .code(),
        cvh::PipelineStatusCode::InvalidDescriptor);

    cvh::ConstImageView null_y = valid_input;
    null_y.planes[0].data = nullptr;
    EXPECT_EQ(
        plan.tryRun(null_y, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::BufferTooSmall);

    cvh::ConstImageView null_uv = valid_input;
    null_uv.planes[1].data = nullptr;
    EXPECT_EQ(
        plan.tryRun(null_uv, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::BufferTooSmall);

    cvh::ConstImageView short_y_stride = valid_input;
    short_y_stride.planes[0].row_stride = 3;
    EXPECT_EQ(
        plan.tryRun(short_y_stride, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::BufferTooSmall);

    cvh::ConstImageView short_uv_stride = valid_input;
    short_uv_stride.planes[1].row_stride = 3;
    EXPECT_EQ(
        plan.tryRun(short_uv_stride, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::BufferTooSmall);

    cvh::ConstImageView short_y_capacity = valid_input;
    short_y_capacity.planes[0].size_bytes = 21;
    EXPECT_EQ(
        plan.tryRun(short_y_capacity, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::BufferTooSmall);

    cvh::ConstImageView short_uv_capacity = valid_input;
    short_uv_capacity.planes[1].size_bytes = 9;
    EXPECT_EQ(
        plan.tryRun(short_uv_capacity, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::BufferTooSmall);

    cvh::ConstImageView overlapping_planes = valid_input;
    overlapping_planes.planes[1] =
        cvh::ConstPlaneView{y_storage.data() + 8, 6, 10};
    EXPECT_EQ(
        plan.tryRun(
                overlapping_planes, valid_output, workspace.view())
            .code(),
        cvh::PipelineStatusCode::AliasingNotSupported);

    const cvh::TensorView overlapping_output = cvh::nchw(
        reinterpret_cast<float*>(y_storage.data()),
        output_storage.size() * sizeof(float),
        1,
        3,
        2,
        2);
    EXPECT_EQ(
        plan.tryRun(valid_input, overlapping_output, workspace.view()).code(),
        cvh::PipelineStatusCode::AliasingNotSupported);

    for (float& value : output_storage)
    {
        value = -77.0f;
    }
    cvh::ConstImageView final_invalid = valid_input;
    final_invalid.planes[1].size_bytes = 9;
    EXPECT_EQ(
        plan.tryRun(final_invalid, valid_output, workspace.view()).code(),
        cvh::PipelineStatusCode::BufferTooSmall);
    for (float value : output_storage)
    {
        EXPECT_FLOAT_EQ(value, -77.0f);
    }

    cvh::Mat mat_input({4, 4}, CV_8UC3);
    cvh::Mat mat_output({1, 3, 2, 2}, CV_32FC1);
    EXPECT_EQ(
        plan.tryRun(mat_input, mat_output, workspace.view()).code(),
        cvh::PipelineStatusCode::Unsupported);

    EXPECT_THROW(
        cvh::pipe(
            cvh::imageDesc(3, 4, cvh::PixelFormat::NV12, color_spec),
            output_descriptor)
            .color(cvh::Color::RGB)
            .resize(2, 2)
            .normalize({0.0f}, {1.0f})
            .layout(cvh::Layout::NCHW)
            .prepare(),
        cvh::Exception);
    EXPECT_THROW(
        cvh::pipe(input_descriptor, output_descriptor)
            .resize(2, 2)
            .normalize({0.0f}, {1.0f})
            .layout(cvh::Layout::NCHW)
            .prepare(),
        cvh::Exception);
}

TEST(PipelineLetterboxTest, rounding_padding_and_transform_contract_are_stable)
{
    const cvh::Mat input = makeInput();
    const cvh::PipelinePlan plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::imageDesc(8, 9, cvh::PixelFormat::BGR8))
            .letterbox(
                8,
                9,
                {10.0f, 20.0f, 30.0f},
                cvh::Interpolation::Nearest)
            .prepare();
    ASSERT_TRUE(plan.hasTransform());
    const cvh::PipelineTransform& transform = plan.transform();
    EXPECT_FLOAT_EQ(transform.scale, 2.0f);
    EXPECT_EQ(transform.resized_width, 8);
    EXPECT_EQ(transform.resized_height, 6);
    EXPECT_EQ(transform.pad_left, 0);
    EXPECT_EQ(transform.pad_right, 0);
    EXPECT_EQ(transform.pad_top, 1);
    EXPECT_EQ(transform.pad_bottom, 2);
    EXPECT_FLOAT_EQ(transform.scale_x, 2.0f);
    EXPECT_FLOAT_EQ(transform.scale_y, 2.0f);

    const cvh::PipelinePoint target_origin =
        transform.sourceToTarget({0.0f, 0.0f});
    const cvh::PipelinePoint target_end =
        transform.sourceToTarget({4.0f, 3.0f});
    EXPECT_FLOAT_EQ(target_origin.x, 0.0f);
    EXPECT_FLOAT_EQ(target_origin.y, 1.0f);
    EXPECT_FLOAT_EQ(target_end.x, 8.0f);
    EXPECT_FLOAT_EQ(target_end.y, 7.0f);
    const cvh::PipelinePoint source_end =
        transform.targetToSource(target_end);
    EXPECT_FLOAT_EQ(source_end.x, 4.0f);
    EXPECT_FLOAT_EQ(source_end.y, 3.0f);
    EXPECT_TRUE(transform.isPadding({0.0f, 0.0f}));
    EXPECT_FALSE(transform.isPadding({0.0f, 1.0f}));
    EXPECT_FALSE(transform.isPadding({7.99f, 6.99f}));
    EXPECT_TRUE(transform.isPadding({0.0f, 7.0f}));

    cvh::Mat output({9, 8}, CV_8UC3);
    cvh::PipelineWorkspace workspace(plan);
    plan.run(input, output, workspace.view());
    for (int y = 0; y < output.size[0]; ++y)
    {
        for (int x = 0; x < output.size[1]; ++x)
        {
            const bool padding = y < 1 || y >= 7;
            for (int channel = 0; channel < 3; ++channel)
            {
                const uchar expected = padding
                    ? static_cast<uchar>(10 + channel * 10)
                    : input.at<uchar>(
                          (y - 1) * 3 / 6,
                          x * 4 / 8,
                          channel);
                EXPECT_EQ(output.at<uchar>(y, x, channel), expected)
                    << "y=" << y << " x=" << x
                    << " channel=" << channel;
            }
        }
    }

    const cvh::PipelinePlan half_up =
        cvh::pipe(
            cvh::imageDesc(2, 4, cvh::PixelFormat::BGR8),
            cvh::imageDesc(5, 5, cvh::PixelFormat::BGR8))
            .letterbox(5, 5)
            .prepare();
    EXPECT_FLOAT_EQ(half_up.transform().scale, 1.25f);
    EXPECT_EQ(half_up.transform().resized_width, 3);
    EXPECT_EQ(half_up.transform().resized_height, 5);
    EXPECT_EQ(half_up.transform().pad_left, 1);
    EXPECT_EQ(half_up.transform().pad_right, 1);
}

TEST(PipelineLetterboxTest, packed_fused_matches_staged_reference)
{
    const cvh::Mat input = makeInput();
    const std::array<cvh::Interpolation, 2> interpolations{{
        cvh::Interpolation::Nearest,
        cvh::Interpolation::Linear}};
    const std::array<cvh::Layout, 2> layouts{{
        cvh::Layout::NCHW,
        cvh::Layout::NHWC}};
    for (cvh::Interpolation interpolation : interpolations)
    {
        for (cvh::Layout layout : layouts)
        {
            const cvh::TensorDescriptor output_descriptor =
                layout == cvh::Layout::NCHW
                ? cvh::tensorDesc<float>({1, 3, 9, 8}, layout)
                : cvh::tensorDesc<float>({1, 9, 8, 3}, layout);
            const cvh::PipelinePlan fused =
                cvh::pipe(
                    cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
                    output_descriptor)
                    .color(cvh::Color::RGB)
                    .letterbox(
                        8,
                        9,
                        {11.0f, 22.0f, 33.0f},
                        interpolation)
                    .normalize(
                        {1.0f, 2.0f, 3.0f},
                        {2.0f, 4.0f, 8.0f})
                    .layout(layout)
                    .prepare();
            EXPECT_EQ(fused.info().execution_group_count, 1);
            EXPECT_EQ(fused.info().full_frame_intermediates, 0);
            EXPECT_EQ(fused.info().workspace_bytes, 0u);
            EXPECT_TRUE(fused.hasTransform());

            const cvh::Mat rgb = referenceColor(
                input, cvh::Color::BGR, cvh::Color::RGB);
            const cvh::PipelinePlan staged_letterbox =
                cvh::pipe(
                    cvh::imageDesc(4, 3, cvh::PixelFormat::RGB8),
                    cvh::imageDesc(8, 9, cvh::PixelFormat::RGB8))
                    .letterbox(
                        8,
                        9,
                        {11.0f, 22.0f, 33.0f},
                        interpolation)
                    .prepare();
            cvh::Mat padded({9, 8}, CV_8UC3);
            cvh::PipelineWorkspace staged_workspace(staged_letterbox);
            staged_letterbox.run(
                rgb, padded, staged_workspace.view());
            const cvh::Mat normalized = referenceNormalize(
                padded,
                {{1.0f, 2.0f, 3.0f, 0.0f}},
                {{2.0f, 4.0f, 8.0f, 1.0f}},
                3);
            const cvh::Mat expected = referenceLayout(normalized, layout);
            const std::vector<int> shape = layout == cvh::Layout::NCHW
                ? std::vector<int>{1, 3, 9, 8}
                : std::vector<int>{1, 9, 8, 3};
            cvh::Mat output(shape, CV_32FC1);
            cvh::PipelineWorkspace fused_workspace(fused);
            fused.run(input, output, fused_workspace.view());
            expectSameFloatMat(output, expected);
        }
    }
}

TEST(PipelineLetterboxTest, yuv_recipe_fuses_and_matches_reference)
{
    const cvh::ColorSpec color_spec{
        cvh::ColorMatrix::BT601,
        cvh::ColorRange::Limited,
        cvh::ChromaLocation::Center};
    cvh::ModelInputRecipe recipe;
    recipe.input =
        cvh::imageDesc(4, 4, cvh::PixelFormat::NV21, color_spec);
    recipe.output =
        cvh::tensorDesc<float>({1, 3, 6, 8}, cvh::Layout::NCHW);
    recipe.color = cvh::Color::RGB;
    recipe.interpolation = cvh::Interpolation::Linear;
    recipe.geometry = cvh::ModelInputGeometry::Letterbox;
    recipe.letterbox_pad_value = {{9.0f, 19.0f, 29.0f, 0.0f}};
    recipe.letterbox_pad_count = 3;
    recipe.mean = {{1.0f, 2.0f, 3.0f, 0.0f}};
    recipe.stddev = {{2.0f, 4.0f, 8.0f, 1.0f}};
    recipe.normalize_count = 3;
    const cvh::PipelinePlan plan =
        cvh::recipes::modelInput(recipe).prepare();
    ASSERT_STREQ(
        plan.info().recipe_id,
        "cvh.model_input.yuv420_f32_letterbox");
    EXPECT_EQ(plan.info().execution_group_count, 1);
    EXPECT_EQ(plan.info().full_frame_intermediates, 0);
    ASSERT_TRUE(plan.hasTransform());
    EXPECT_EQ(plan.transform().resized_width, 6);
    EXPECT_EQ(plan.transform().resized_height, 6);
    EXPECT_EQ(plan.transform().pad_left, 1);
    EXPECT_EQ(plan.transform().pad_right, 1);

    std::array<uchar, 24> y_storage{};
    std::array<uchar, 12> vu_storage{};
    for (std::size_t index = 0; index < y_storage.size(); ++index)
    {
        y_storage[index] = static_cast<uchar>(22 + index * 9);
    }
    for (std::size_t index = 0; index < vu_storage.size(); ++index)
    {
        vu_storage[index] = static_cast<uchar>(38 + index * 11);
    }
    const cvh::ConstImageView input = cvh::nv21(
        y_storage.data(),
        6,
        y_storage.size(),
        vu_storage.data(),
        6,
        vu_storage.size(),
        4,
        4,
        color_spec);
    const std::array<float, 3> mean{{1.0f, 2.0f, 3.0f}};
    const std::array<float, 3> stddev{{2.0f, 4.0f, 8.0f}};
    const std::vector<float> content = referenceYuvModelInput(
        input,
        6,
        6,
        cvh::Interpolation::Linear,
        cvh::Color::RGB,
        cvh::Layout::NCHW,
        mean,
        stddev);
    std::array<float, 3 * 6 * 8> expected{};
    for (int channel = 0; channel < 3; ++channel)
    {
        const float pad =
            (recipe.letterbox_pad_value[
                 static_cast<std::size_t>(channel)] -
             mean[static_cast<std::size_t>(channel)]) /
            stddev[static_cast<std::size_t>(channel)];
        for (int y = 0; y < 6; ++y)
        {
            for (int x = 0; x < 8; ++x)
            {
                const std::size_t target_index =
                    (static_cast<std::size_t>(channel) * 6 + y) * 8 + x;
                expected[target_index] = x == 0 || x == 7
                    ? pad
                    : content[(static_cast<std::size_t>(channel) * 6 + y) *
                                  6 +
                              (x - 1)];
            }
        }
    }
    std::array<float, 3 * 6 * 8> output{};
    cvh::PipelineWorkspace workspace(plan);
    plan.run(
        input,
        cvh::nchw(
            output.data(),
            output.size() * sizeof(float),
            1,
            3,
            6,
            8),
        workspace.view());
    for (std::size_t index = 0; index < output.size(); ++index)
    {
        EXPECT_NEAR(output[index], expected[index], 1e-5f)
            << "index=" << index;
    }

    cvh::ModelInputRecipe changed = recipe;
    changed.letterbox_pad_value[0] += 1.0f;
    EXPECT_NE(
        plan.info().recipe_fingerprint,
        cvh::recipes::modelInput(changed)
            .prepare()
            .info()
            .recipe_fingerprint);
}

TEST(PipelineLetterboxTest, invalid_contracts_fail_during_prepare)
{
    const cvh::ImageDescriptor input =
        cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8);
    const cvh::ImageDescriptor output =
        cvh::imageDesc(8, 9, cvh::PixelFormat::BGR8);
    EXPECT_THROW(
        cvh::pipe(input, output)
            .letterbox(8, 9, {1.0f, 2.0f})
            .prepare(),
        cvh::Exception);
    EXPECT_THROW(
        cvh::pipe(input, output)
            .letterbox(
                8,
                9,
                std::numeric_limits<float>::quiet_NaN())
            .prepare(),
        cvh::Exception);
    EXPECT_THROW(
        cvh::pipe(input, output)
            .letterbox(8, 9)
            .letterbox(8, 9)
            .prepare(),
        cvh::Exception);

    const cvh::PipelinePlan no_transform =
        cvh::pipe(
            input,
            cvh::imageDesc(8, 9, cvh::PixelFormat::BGR8))
            .resize(8, 9)
            .prepare();
    EXPECT_FALSE(no_transform.hasTransform());
    EXPECT_THROW(no_transform.transform(), cvh::Exception);
}

TEST(PipelineQuantizeTest, ties_non_finite_and_saturation_are_stable)
{
    const std::array<float, 9> values{{
        -std::numeric_limits<float>::infinity(),
        -255.0f,
        -3.0f,
        -1.0f,
        std::numeric_limits<float>::quiet_NaN(),
        1.0f,
        3.0f,
        255.0f,
        std::numeric_limits<float>::infinity()}};
    cvh::Mat input({1, static_cast<int>(values.size())}, CV_32FC1);
    std::memcpy(input.data, values.data(), sizeof(values));
    cvh::ImageDescriptor input_descriptor;
    input_descriptor.width = static_cast<int>(values.size());
    input_descriptor.height = 1;
    input_descriptor.data_type = cvh::PipelineDataType::F32;
    input_descriptor.color = cvh::Color::Gray;
    input_descriptor.plane_count = 1;

    for (cvh::PipelineDataType target_type :
         {cvh::PipelineDataType::U8, cvh::PipelineDataType::S8})
    {
        cvh::ImageDescriptor output_descriptor = input_descriptor;
        output_descriptor.data_type = target_type;
        const int zero_point =
            target_type == cvh::PipelineDataType::U8 ? 128 : 0;
        const cvh::PipelinePlan plan =
            cvh::pipe(input_descriptor, output_descriptor)
                .quantize(target_type, 2.0f, zero_point)
                .prepare();
        cvh::Mat output(
            {1, static_cast<int>(values.size())},
            target_type == cvh::PipelineDataType::U8
                ? CV_8UC1
                : CV_8SC1);
        cvh::PipelineWorkspace workspace(plan);
        plan.run(input, output, workspace.view());
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            if (target_type == cvh::PipelineDataType::U8)
            {
                EXPECT_EQ(
                    output.at<uchar>(0, static_cast<int>(index)),
                    referenceQuantize<uchar>(
                        values[index], 2.0f, zero_point));
            }
            else
            {
                EXPECT_EQ(
                    output.at<schar>(0, static_cast<int>(index)),
                    referenceQuantize<schar>(
                        values[index], 2.0f, zero_point));
            }
        }
    }
}

TEST(PipelineQuantizeTest, packed_direct_store_matches_staged_reference)
{
    const cvh::Mat input = makeInput();
    for (cvh::PipelineDataType target_type :
         {cvh::PipelineDataType::U8, cvh::PipelineDataType::S8})
    {
        for (bool letterbox : {false, true})
        {
            for (cvh::Interpolation interpolation :
                 {cvh::Interpolation::Nearest,
                  cvh::Interpolation::Linear})
            {
                for (cvh::Layout layout :
                     {cvh::Layout::NCHW, cvh::Layout::NHWC})
                {
                    const int width = letterbox ? 6 : 3;
                    const int height = letterbox ? 4 : 2;
                    cvh::TensorDescriptor output_descriptor;
                    output_descriptor.data_type = target_type;
                    output_descriptor.layout = layout;
                    output_descriptor.dims = 4;
                    output_descriptor.shape = layout == cvh::Layout::NCHW
                        ? std::array<int, cvh::PIPELINE_MAX_DIM>{
                              1, 3, height, width, 0, 0, 0, 0}
                        : std::array<int, cvh::PIPELINE_MAX_DIM>{
                              1, height, width, 3, 0, 0, 0, 0};
                    const int zero_point =
                        target_type == cvh::PipelineDataType::U8 ? 128 : 0;

                    cvh::PipelineBuilder fused_builder = cvh::pipe(
                        cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
                        output_descriptor);
                    fused_builder.color(cvh::Color::RGB);
                    if (letterbox)
                    {
                        fused_builder.letterbox(
                            width,
                            height,
                            114.0f,
                            interpolation);
                    }
                    else
                    {
                        fused_builder.resize(width, height, interpolation);
                    }
                    const cvh::PipelinePlan fused = fused_builder
                        .normalize({128.0f}, {64.0f})
                        .quantize(target_type, 0.02f, zero_point)
                        .layout(layout)
                        .prepare();

                    cvh::PipelineBuilder staged_builder = cvh::pipe(
                        cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
                        output_descriptor);
                    if (letterbox)
                    {
                        staged_builder.letterbox(
                            width,
                            height,
                            114.0f,
                            interpolation);
                    }
                    else
                    {
                        staged_builder.resize(width, height, interpolation);
                    }
                    const cvh::PipelinePlan staged = staged_builder
                        .color(cvh::Color::RGB)
                        .normalize({128.0f}, {64.0f})
                        .quantize(target_type, 0.02f, zero_point)
                        .layout(layout)
                        .prepare();
                    EXPECT_EQ(fused.info().execution_group_count, 1);
                    EXPECT_EQ(fused.info().full_frame_intermediates, 0);
                    EXPECT_EQ(fused.info().workspace_bytes, 0u);
                    EXPECT_EQ(staged.info().execution_group_count, 5);
                    EXPECT_NE(
                        fused.explain().find("model-input quantized"),
                        std::string::npos);

                    const std::vector<int> shape =
                        layout == cvh::Layout::NCHW
                        ? std::vector<int>{1, 3, height, width}
                        : std::vector<int>{1, height, width, 3};
                    const int type =
                        target_type == cvh::PipelineDataType::U8
                        ? CV_8UC1
                        : CV_8SC1;
                    cvh::Mat fused_output(shape, type);
                    cvh::Mat staged_output(shape, type);
                    cvh::PipelineWorkspace fused_workspace(fused);
                    cvh::PipelineWorkspace staged_workspace(staged);
                    fused.run(
                        input, fused_output, fused_workspace.view());
                    staged.run(
                        input, staged_output, staged_workspace.view());
                    expectSameMat(fused_output, staged_output);
                }
            }
        }
    }
}

TEST(PipelineQuantizeTest, yuv_s8_recipe_matches_independent_reference)
{
    const cvh::ColorSpec color_spec{
        cvh::ColorMatrix::BT2020,
        cvh::ColorRange::Full,
        cvh::ChromaLocation::Left};
    cvh::ModelInputRecipe recipe;
    recipe.input =
        cvh::imageDesc(4, 4, cvh::PixelFormat::NV12, color_spec);
    recipe.output =
        cvh::tensorDesc<signed char>({1, 2, 3, 3}, cvh::Layout::NHWC);
    recipe.color = cvh::Color::BGR;
    recipe.interpolation = cvh::Interpolation::Linear;
    recipe.mean = {{128.0f, 120.0f, 112.0f, 0.0f}};
    recipe.stddev = {{64.0f, 60.0f, 56.0f, 1.0f}};
    recipe.normalize_count = 3;
    recipe.quantize_scale = 0.025f;
    recipe.quantize_zero_point = -3;
    const cvh::PipelinePlan plan =
        cvh::recipes::modelInput(recipe).prepare();
    ASSERT_STREQ(plan.info().recipe_id, "cvh.model_input.yuv420_s8");
    EXPECT_EQ(plan.info().execution_group_count, 1);
    EXPECT_EQ(plan.info().full_frame_intermediates, 0);
    EXPECT_NE(
        plan.explain().find("yuv420-model-input quantized"),
        std::string::npos);

    cvh::ModelInputRecipe changed = recipe;
    changed.quantize_scale = 0.05f;
    EXPECT_NE(
        plan.info().recipe_fingerprint,
        cvh::recipes::modelInput(changed)
            .prepare()
            .info()
            .recipe_fingerprint);

    std::array<uchar, 24> y_storage{};
    std::array<uchar, 12> uv_storage{};
    for (std::size_t index = 0; index < y_storage.size(); ++index)
    {
        y_storage[index] = static_cast<uchar>(19 + index * 11);
    }
    for (std::size_t index = 0; index < uv_storage.size(); ++index)
    {
        uv_storage[index] = static_cast<uchar>(37 + index * 17);
    }
    const cvh::ConstImageView input = cvh::nv12(
        y_storage.data(),
        6,
        y_storage.size(),
        uv_storage.data(),
        6,
        uv_storage.size(),
        4,
        4,
        color_spec);
    const std::array<float, 3> mean{{128.0f, 120.0f, 112.0f}};
    const std::array<float, 3> stddev{{64.0f, 60.0f, 56.0f}};
    const std::vector<float> real_values = referenceYuvModelInput(
        input,
        3,
        2,
        cvh::Interpolation::Linear,
        cvh::Color::BGR,
        cvh::Layout::NHWC,
        mean,
        stddev);
    std::array<schar, 18> output{};
    cvh::PipelineWorkspace workspace(plan);
    const cvh::PipelineStatus status = plan.tryRun(
        input,
        cvh::nhwc(
            output.data(),
            output.size() * sizeof(schar),
            1,
            2,
            3,
            3),
        workspace.view());
    ASSERT_TRUE(status.ok()) << status.message();
    for (std::size_t index = 0; index < output.size(); ++index)
    {
        EXPECT_EQ(
            output[index],
            referenceQuantize<schar>(
                real_values[index],
                recipe.quantize_scale,
                recipe.quantize_zero_point))
            << "index=" << index;
    }
}

TEST(PipelineQuantizeTest, invalid_parameters_fail_during_prepare)
{
    const cvh::ImageDescriptor input =
        cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8);
    const cvh::TensorDescriptor output =
        cvh::tensorDesc<uchar>({1, 3, 2, 2}, cvh::Layout::NCHW);
    const auto make_builder = [&]() {
        return cvh::pipe(input, output)
            .resize(2, 2)
            .normalize({0.0f}, {1.0f});
    };
    EXPECT_THROW(
        make_builder()
            .quantize(cvh::PipelineDataType::U8, 0.0f, 0)
            .layout(cvh::Layout::NCHW)
            .prepare(),
        cvh::Exception);
    EXPECT_THROW(
        make_builder()
            .quantize(cvh::PipelineDataType::U8, -1.0f, 0)
            .layout(cvh::Layout::NCHW)
            .prepare(),
        cvh::Exception);
    EXPECT_THROW(
        make_builder()
            .quantize(
                cvh::PipelineDataType::U8,
                std::numeric_limits<float>::quiet_NaN(),
                0)
            .layout(cvh::Layout::NCHW)
            .prepare(),
        cvh::Exception);
    EXPECT_THROW(
        make_builder()
            .quantize(cvh::PipelineDataType::U8, 1.0f, 256)
            .layout(cvh::Layout::NCHW)
            .prepare(),
        cvh::Exception);
    EXPECT_THROW(
        make_builder()
            .quantize(cvh::PipelineDataType::S8, 1.0f, -129)
            .layout(cvh::Layout::NCHW)
            .prepare(),
        cvh::Exception);
    EXPECT_THROW(
        make_builder()
            .quantize(cvh::PipelineDataType::F32, 1.0f, 0)
            .layout(cvh::Layout::NCHW)
            .prepare(),
        cvh::Exception);
    EXPECT_THROW(
        cvh::pipe(input, output)
            .quantize(cvh::PipelineDataType::U8, 1.0f, 0)
            .layout(cvh::Layout::NCHW)
            .prepare(),
        cvh::Exception);
}

TEST(PipelineQuantizeTest, borrowed_u8_s8_capacity_and_alias_are_checked)
{
    std::array<uchar, 36> input_storage{};
    const cvh::ConstImageView input_view = cvh::bgr(
        input_storage.data(), input_storage.size(), 4, 3, 12);
    for (cvh::PipelineDataType target_type :
         {cvh::PipelineDataType::U8, cvh::PipelineDataType::S8})
    {
        const cvh::Layout layout =
            target_type == cvh::PipelineDataType::U8
            ? cvh::Layout::NCHW
            : cvh::Layout::NHWC;
        cvh::TensorDescriptor descriptor;
        descriptor.data_type = target_type;
        descriptor.layout = layout;
        descriptor.dims = 4;
        descriptor.shape = layout == cvh::Layout::NCHW
            ? std::array<int, cvh::PIPELINE_MAX_DIM>{
                  1, 3, 2, 2, 0, 0, 0, 0}
            : std::array<int, cvh::PIPELINE_MAX_DIM>{
                  1, 2, 2, 3, 0, 0, 0, 0};
        const cvh::PipelinePlan plan =
            cvh::pipe(input_view.descriptor, descriptor)
                .resize(2, 2, cvh::Interpolation::Nearest)
                .normalize({128.0f}, {64.0f})
                .quantize(
                    target_type,
                    0.025f,
                    target_type == cvh::PipelineDataType::U8 ? 128 : 0)
                .layout(layout)
                .prepare();
        std::array<uchar, 12> output_storage{};
        cvh::TensorView output_view;
        output_view.data = output_storage.data();
        output_view.size_bytes = output_storage.size();
        output_view.descriptor = descriptor;
        cvh::PipelineWorkspace workspace(plan);
        ASSERT_TRUE(
            plan.tryRun(input_view, output_view, workspace.view()).ok());

        output_storage.fill(0x5a);
        cvh::TensorView short_output = output_view;
        short_output.size_bytes -= 1;
        EXPECT_EQ(
            plan.tryRun(input_view, short_output, workspace.view()).code(),
            cvh::PipelineStatusCode::BufferTooSmall);
        for (uchar value : output_storage)
        {
            EXPECT_EQ(value, 0x5a);
        }

        cvh::TensorView overlapping_output = output_view;
        overlapping_output.data = input_storage.data();
        EXPECT_EQ(
            plan.tryRun(
                    input_view, overlapping_output, workspace.view())
                .code(),
            cvh::PipelineStatusCode::AliasingNotSupported);
    }
}

TEST(PipelineValidationTest, invalid_order_fails_during_prepare)
{
    cvh::PipelineBuilder builder =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::tensorDesc<uchar>({1, 3, 2, 2}, cvh::Layout::NCHW))
            .layout(cvh::Layout::NCHW)
            .resize(2, 2);

    cvh::PipelinePlan plan;
    const cvh::PipelineStatus status = builder.tryPrepare(plan);
    EXPECT_FALSE(status.ok());
    EXPECT_FALSE(plan.valid());
    EXPECT_EQ(status.code(), cvh::PipelineStatusCode::InvalidOperation);
    EXPECT_EQ(status.stage(), 1);
    EXPECT_NE(
        std::string(status.message()).find(
            "pipeline stage 1 \"resize\": expected Image"),
        std::string::npos);
}

TEST(PipelineValidationTest, hard_requirements_never_silently_fallback)
{
    EXPECT_THROW(
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::imageDesc(2, 2, cvh::PixelFormat::RGB8))
            .color(cvh::Color::RGB)
            .resize(2, 2)
            .requireNoFullFrameIntermediate()
            .prepare(),
        cvh::Exception);

    cvh::PipelineBuilder required_builder =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::imageDesc(2, 2, cvh::PixelFormat::RGB8))
            .color(cvh::Color::RGB)
            .resize(2, 2)
            .requireSingleExecutionGroup();
    cvh::PipelinePlan rejected_plan;
    const cvh::PipelineStatus requirement_status =
        required_builder.tryPrepare(rejected_plan);
    EXPECT_EQ(
        requirement_status.code(),
        cvh::PipelineStatusCode::RequirementNotSatisfied);
    EXPECT_EQ(requirement_status.stage(), 0);
    EXPECT_FALSE(rejected_plan.valid());

    const cvh::PipelinePlan direct =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::imageDesc(2, 2, cvh::PixelFormat::BGR8))
            .resize(2, 2)
            .requireSingleExecutionGroup()
            .prepare();
    EXPECT_EQ(direct.info().execution_group_count, 1);
    EXPECT_EQ(direct.info().full_frame_intermediates, 0);
}

TEST(PipelineValidationTest, prepared_run_rejects_wrong_output_and_alias)
{
    const cvh::Mat input = makeInput();
    const cvh::PipelinePlan plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::imageDesc(2, 2, cvh::PixelFormat::BGR8))
            .resize(2, 2)
            .prepare();
    cvh::PipelineWorkspace workspace(plan);

    cvh::Mat wrong_output({3, 2}, CV_8UC3);
    const cvh::PipelineStatus wrong_status =
        plan.tryRun(input, wrong_output, workspace.view());
    EXPECT_EQ(
        wrong_status.code(), cvh::PipelineStatusCode::ShapeMismatch);

    cvh::Mat wrong_type({2, 2}, CV_32FC3);
    const cvh::PipelineStatus wrong_type_status =
        plan.tryRun(input, wrong_type, workspace.view());
    EXPECT_EQ(
        wrong_type_status.code(), cvh::PipelineStatusCode::TypeMismatch);

    const cvh::PipelinePlan copy_plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8))
            .prepare();
    cvh::PipelineWorkspace copy_workspace(copy_plan);
    cvh::Mat alias = input;
    const cvh::PipelineStatus alias_status =
        copy_plan.tryRun(input, alias, copy_workspace.view());
    EXPECT_EQ(
        alias_status.code(),
        cvh::PipelineStatusCode::AliasingNotSupported);
}

TEST(PipelineValidationTest, one_shot_rejects_same_mat_without_mutating_input)
{
    cvh::Mat input = makeInput();
    uchar* const original_data = input.data;
    const int original_rows = input.size[0];
    const int original_cols = input.size[1];

    EXPECT_THROW(
        cvh::pipe(input, input)
            .resize(2, 2)
            .run(),
        cvh::Exception);
    EXPECT_EQ(input.data, original_data);
    EXPECT_EQ(input.size[0], original_rows);
    EXPECT_EQ(input.size[1], original_cols);
}

TEST(PipelinePropertyTest, deterministic_legal_chains_match_sequential_reference)
{
    std::mt19937 random(0xC0FFEEu);
    constexpr int kChainCount = 64;
    int fused_plan_count = 0;
    int staged_plan_count = 0;

    for (int chain = 0; chain < kChainCount; ++chain)
    {
        SCOPED_TRACE(::testing::Message() << "chain=" << chain);
        const int input_height = 1 + static_cast<int>(random() % 9u);
        const int input_width = 1 + static_cast<int>(random() % 11u);
        cvh::Mat input({input_height, input_width}, CV_8UC3);
        for (int y = 0; y < input_height; ++y)
        {
            for (int x = 0; x < input_width; ++x)
            {
                for (int channel = 0; channel < 3; ++channel)
                {
                    input.at<uchar>(y, x, channel) =
                        static_cast<uchar>(
                            (chain * 29 + y * 17 + x * 11 + channel * 53) &
                            0xff);
                }
            }
        }

        cvh::Mat reference = input.clone();
        cvh::Color current_color = cvh::Color::BGR;
        std::vector<RandomStage> stages;
        const int stage_count = chain < 2
            ? 2
            : 1 + static_cast<int>(random() % 6u);
        stages.reserve(static_cast<std::size_t>(stage_count));

        for (int stage_index = 0;
             stage_index < stage_count;
             ++stage_index)
        {
            RandomStage stage;
            const int choice = chain == 0
                ? stage_index + 1
                : chain == 1
                    ? 2 - stage_index
                    : static_cast<int>(random() % 3u);
            if (choice == 0)
            {
                stage.kind = RandomStageKind::Color;
                const std::array<cvh::Color, 3> colors{
                    cvh::Color::Gray,
                    cvh::Color::BGR,
                    cvh::Color::RGB};
                stage.color = colors[random() % colors.size()];
                reference =
                    referenceColor(reference, current_color, stage.color);
                current_color = stage.color;
            }
            else if (choice == 1)
            {
                stage.kind = RandomStageKind::Resize;
                stage.width = 1 + static_cast<int>(random() % 12u);
                stage.height = 1 + static_cast<int>(random() % 10u);
                stage.interpolation = (random() & 1u) == 0u
                    ? cvh::Interpolation::Nearest
                    : cvh::Interpolation::Linear;
                cvh::Mat resized;
                cvh::resize(
                    reference,
                    resized,
                    cvh::Size(stage.width, stage.height),
                    0.0,
                    0.0,
                    static_cast<int>(stage.interpolation));
                reference = resized;
            }
            else
            {
                stage.kind = RandomStageKind::Normalize;
                const int channels = reference.channels();
                stage.count = ((random() & 1u) == 0u) ? 1 : channels;
                for (int parameter = 0;
                     parameter < stage.count;
                     ++parameter)
                {
                    stage.mean[static_cast<std::size_t>(parameter)] =
                        static_cast<float>(random() % 17u) * 0.25f;
                    stage.stddev[static_cast<std::size_t>(parameter)] =
                        0.5f + static_cast<float>(random() % 7u) * 0.25f;
                }
                reference = referenceNormalize(
                    reference, stage.mean, stage.stddev, stage.count);
            }
            stages.push_back(stage);
        }

        const bool has_layout = chain < 2 || (random() % 3u) != 0u;
        const cvh::Layout layout = (random() & 1u) == 0u
            ? cvh::Layout::NCHW
            : cvh::Layout::NHWC;
        cvh::PipelineDataDescriptor output_descriptor;
        if (has_layout)
        {
            reference = referenceLayout(reference, layout);
            output_descriptor = tensorDescriptorFor(reference, layout);
        }
        else
        {
            output_descriptor =
                imageDescriptorFor(reference, current_color);
        }

        cvh::PipelineBuilder builder = cvh::pipe(
            cvh::imageDesc(
                input_width, input_height, cvh::PixelFormat::BGR8),
            output_descriptor);
        for (const RandomStage& stage : stages)
        {
            switch (stage.kind)
            {
            case RandomStageKind::Color:
                builder.color(stage.color);
                break;
            case RandomStageKind::Resize:
                builder.resize(
                    stage.width, stage.height, stage.interpolation);
                break;
            case RandomStageKind::Normalize:
                if (stage.count == 1)
                {
                    builder.normalize(
                        {stage.mean[0]}, {stage.stddev[0]});
                }
                else
                {
                    builder.normalize(
                        {stage.mean[0], stage.mean[1], stage.mean[2]},
                        {stage.stddev[0],
                         stage.stddev[1],
                         stage.stddev[2]});
                }
                break;
            }
        }
        if (has_layout)
        {
            builder.layout(layout);
        }

        const cvh::PipelinePlan plan = builder.prepare();
        if (plan.info().execution_class ==
            cvh::PipelineExecutionClass::FusedTiled)
        {
            ++fused_plan_count;
        }
        else
        {
            ++staged_plan_count;
        }
        cvh::PipelineWorkspace workspace(plan);
        std::vector<int> output_shape;
        output_shape.reserve(static_cast<std::size_t>(reference.dims));
        for (int dim = 0; dim < reference.dims; ++dim)
        {
            output_shape.push_back(reference.size[dim]);
        }
        cvh::Mat output(output_shape, reference.type());
        const cvh::PipelineStatus status =
            plan.tryRun(input, output, workspace.view());
        ASSERT_TRUE(status.ok()) << status.message();
        expectSameMat(output, reference);
    }
    EXPECT_GT(fused_plan_count, 0);
    EXPECT_GT(staged_plan_count, 0);
}

TEST(PipelinePreparedTest, immutable_plan_runs_concurrently_with_separate_workspaces)
{
    const cvh::Mat input = makeInput();
    const cvh::PipelinePlan plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::tensorDesc<float>({1, 3, 2, 3}, cvh::Layout::NCHW))
            .color(cvh::Color::RGB)
            .resize(3, 2)
            .normalize({1.0f, 2.0f, 3.0f}, {2.0f, 3.0f, 4.0f})
            .layout(cvh::Layout::NCHW)
            .prepare();

    cvh::PipelineWorkspace first_workspace(plan);
    cvh::PipelineWorkspace second_workspace(plan);
    cvh::Mat first_output({1, 3, 2, 3}, CV_32FC1);
    cvh::Mat second_output({1, 3, 2, 3}, CV_32FC1);
    cvh::PipelineStatus first_status;
    cvh::PipelineStatus second_status;

    std::thread first([&]() {
        first_status =
            plan.tryRun(input, first_output, first_workspace.view());
    });
    std::thread second([&]() {
        second_status =
            plan.tryRun(input, second_output, second_workspace.view());
    });
    first.join();
    second.join();

    ASSERT_TRUE(first_status.ok()) << first_status.message();
    ASSERT_TRUE(second_status.ok()) << second_status.message();
    expectSameFloatMat(first_output, second_output);
}

TEST(PipelineBorrowedViewTest, immutable_plan_runs_concurrently_with_separate_workspaces)
{
    const cvh::Mat input = makeInput();
    const cvh::PipelinePlan plan =
        cvh::pipe(
            cvh::imageDesc(4, 3, cvh::PixelFormat::BGR8),
            cvh::tensorDesc<float>({1, 3, 2, 3}, cvh::Layout::NCHW))
            .color(cvh::Color::RGB)
            .resize(3, 2)
            .normalize({1.0f, 2.0f, 3.0f}, {2.0f, 3.0f, 4.0f})
            .layout(cvh::Layout::NCHW)
            .prepare();
    const cvh::ConstImageView input_view = cvh::bgr(
        input.data,
        static_cast<std::size_t>(input.size[0]) * input.step(0),
        input.size[1],
        input.size[0],
        input.step(0));
    std::array<float, 18> first_output{};
    std::array<float, 18> second_output{};
    const cvh::TensorView first_view = cvh::nchw(
        first_output.data(),
        first_output.size() * sizeof(float),
        1,
        3,
        2,
        3);
    const cvh::TensorView second_view = cvh::nchw(
        second_output.data(),
        second_output.size() * sizeof(float),
        1,
        3,
        2,
        3);
    cvh::PipelineWorkspace first_workspace(plan);
    cvh::PipelineWorkspace second_workspace(plan);
    cvh::PipelineStatus first_status;
    cvh::PipelineStatus second_status;

    std::thread first([&]() {
        first_status =
            plan.tryRun(input_view, first_view, first_workspace.view());
    });
    std::thread second([&]() {
        second_status =
            plan.tryRun(input_view, second_view, second_workspace.view());
    });
    first.join();
    second.join();

    ASSERT_TRUE(first_status.ok()) << first_status.message();
    ASSERT_TRUE(second_status.ok()) << second_status.message();
    for (std::size_t index = 0; index < first_output.size(); ++index)
    {
        EXPECT_FLOAT_EQ(first_output[index], second_output[index]);
    }
}
