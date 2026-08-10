#include "cvh/pipeline/pipeline.h"
#include "cvh/imgproc/cvtcolor.h"
#include "cvh/imgproc/resize.h"
#include "gtest/gtest.h"

#include <array>
#include <cstddef>
#include <cstring>
#include <random>
#include <string>
#include <thread>
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
    ASSERT_EQ(lhs.depth(), CV_8U);
    const std::size_t byte_count = lhs.total() * lhs.elemSize();
    EXPECT_EQ(std::memcmp(lhs.data, rhs.data, byte_count), 0);
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
            .prepare();

    EXPECT_EQ(plan.info().semantic_stage_count, 4);
    EXPECT_EQ(plan.info().execution_group_count, 4);
    EXPECT_EQ(plan.info().full_frame_intermediates, 3);
    EXPECT_EQ(plan.info().allocations_per_run, 0);
    EXPECT_GT(plan.info().workspace_bytes, 0u);
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
        const int stage_count = 1 + static_cast<int>(random() % 6u);
        stages.reserve(static_cast<std::size_t>(stage_count));

        for (int stage_index = 0;
             stage_index < stage_count;
             ++stage_index)
        {
            RandomStage stage;
            const int choice = static_cast<int>(random() % 3u);
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

        const bool has_layout = (random() % 3u) != 0u;
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
