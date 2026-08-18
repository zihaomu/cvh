#ifndef CVH_RECIPES_MODEL_INPUT_H
#define CVH_RECIPES_MODEL_INPUT_H

#include "../pipeline/pipeline.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <utility>

namespace cvh {

enum class ModelInputGeometry
{
    Resize,
    Letterbox,
};

struct ModelInputRecipe
{
    ImageDescriptor input{};
    TensorDescriptor output{};
    Color color = Color::RGB;
    Interpolation interpolation = Interpolation::Linear;
    ModelInputGeometry geometry = ModelInputGeometry::Resize;
    std::array<float, 4> letterbox_pad_value{{114.0f, 0.0f, 0.0f, 0.0f}};
    int letterbox_pad_count = 1;
    std::array<float, 4> mean{};
    std::array<float, 4> stddev{{1.0f, 1.0f, 1.0f, 1.0f}};
    int normalize_count = 3;
    float quantize_scale = 1.0f;
    int quantize_zero_point = 0;
};

namespace recipe_detail {

inline std::uint64_t fingerprintMix(std::uint64_t hash,
                                    std::uint64_t value)
{
    hash ^= value;
    hash *= 1099511628211ull;
    return hash;
}

inline std::uint64_t fingerprintFloat(std::uint64_t hash, float value)
{
    std::uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float fingerprint width");
    std::memcpy(&bits, &value, sizeof(bits));
    return fingerprintMix(hash, bits);
}

inline std::uint64_t modelInputFingerprint(
    const ModelInputRecipe& recipe)
{
    std::uint64_t hash = 1469598103934665603ull;
    hash = fingerprintMix(hash, 1);
    hash = fingerprintMix(hash, recipe.input.width);
    hash = fingerprintMix(hash, recipe.input.height);
    hash = fingerprintMix(
        hash, static_cast<std::uint64_t>(recipe.input.data_type));
    hash = fingerprintMix(
        hash, static_cast<std::uint64_t>(recipe.input.color));
    hash = fingerprintMix(
        hash, static_cast<std::uint64_t>(recipe.input.pixel_format));
    hash = fingerprintMix(hash, recipe.input.plane_count);
    hash = fingerprintMix(
        hash, static_cast<std::uint64_t>(recipe.input.color_spec.matrix));
    hash = fingerprintMix(
        hash, static_cast<std::uint64_t>(recipe.input.color_spec.range));
    hash = fingerprintMix(
        hash,
        static_cast<std::uint64_t>(
            recipe.input.color_spec.chroma_location));
    hash = fingerprintMix(
        hash, static_cast<std::uint64_t>(recipe.output.data_type));
    hash = fingerprintMix(
        hash, static_cast<std::uint64_t>(recipe.output.layout));
    hash = fingerprintMix(hash, recipe.output.dims);
    for (int dim = 0; dim < recipe.output.dims; ++dim)
    {
        hash = fingerprintMix(
            hash,
            recipe.output.shape[static_cast<std::size_t>(dim)]);
    }
    hash = fingerprintMix(hash, static_cast<std::uint64_t>(recipe.color));
    hash = fingerprintMix(
        hash, static_cast<std::uint64_t>(recipe.interpolation));
    if (recipe.geometry == ModelInputGeometry::Letterbox)
    {
        hash = fingerprintMix(hash, 0x4c4554544552424full);
        hash = fingerprintMix(hash, recipe.letterbox_pad_count);
        for (int index = 0; index < recipe.letterbox_pad_count; ++index)
        {
            hash = fingerprintFloat(
                hash,
                recipe.letterbox_pad_value[
                    static_cast<std::size_t>(index)]);
        }
    }
    hash = fingerprintMix(hash, recipe.normalize_count);
    for (int index = 0; index < recipe.normalize_count; ++index)
    {
        hash = fingerprintFloat(
            hash, recipe.mean[static_cast<std::size_t>(index)]);
        hash = fingerprintFloat(
            hash, recipe.stddev[static_cast<std::size_t>(index)]);
    }
    if (recipe.output.data_type == PipelineDataType::U8 ||
        recipe.output.data_type == PipelineDataType::S8)
    {
        hash = fingerprintFloat(hash, recipe.quantize_scale);
        hash = fingerprintMix(hash, static_cast<std::uint64_t>(
            static_cast<std::int64_t>(recipe.quantize_zero_point)));
    }
    return hash;
}

inline void validateModelInputRecipe(const ModelInputRecipe& recipe,
                                     int& output_width,
                                     int& output_height)
{
    if (!recipe.input.valid() || !recipe.output.valid())
    {
        CV_Error(
            Error::StsBadArg,
            "modelInput recipe descriptor is invalid");
    }
    const bool bgr =
        recipe.input.pixel_format == PixelFormat::BGR8 &&
        recipe.input.color == Color::BGR;
    const bool rgb =
        recipe.input.pixel_format == PixelFormat::RGB8 &&
        recipe.input.color == Color::RGB;
    const bool yuv420 =
        (recipe.input.pixel_format == PixelFormat::NV12 ||
         recipe.input.pixel_format == PixelFormat::NV21) &&
        recipe.input.color == Color::YUV &&
        recipe.input.plane_count == 2 &&
        recipe.input.width % 2 == 0 &&
        recipe.input.height % 2 == 0;
    if (recipe.input.data_type != PipelineDataType::U8 ||
        ((!bgr && !rgb) && !yuv420) ||
        ((bgr || rgb) && recipe.input.plane_count != 1))
    {
        CV_Error(
            Error::StsNotImplemented,
            "modelInput recipe supports packed BGR8/RGB8 or even-sized two-plane NV12/NV21 input");
    }
    if (recipe.color != Color::BGR && recipe.color != Color::RGB)
    {
        CV_Error(
            Error::StsNotImplemented,
            "modelInput recipe target color must be BGR or RGB");
    }
    if (recipe.interpolation != Interpolation::Nearest &&
        recipe.interpolation != Interpolation::Linear)
    {
        CV_Error(
            Error::StsNotImplemented,
            "modelInput recipe interpolation is unsupported");
    }
    if (recipe.geometry != ModelInputGeometry::Resize &&
        recipe.geometry != ModelInputGeometry::Letterbox)
    {
        CV_Error(
            Error::StsNotImplemented,
            "modelInput recipe geometry is unsupported");
    }
    if (recipe.geometry == ModelInputGeometry::Letterbox)
    {
        if (recipe.letterbox_pad_count != 1 &&
            recipe.letterbox_pad_count != 3)
        {
            CV_Error(
                Error::StsBadArg,
                "modelInput letterbox pad count must be 1 or 3");
        }
        for (int index = 0;
             index < recipe.letterbox_pad_count;
             ++index)
        {
            if (!std::isfinite(recipe.letterbox_pad_value[
                    static_cast<std::size_t>(index)]))
            {
                CV_Error(
                    Error::StsBadArg,
                    "modelInput letterbox pad values must be finite");
            }
        }
    }
    if ((recipe.output.data_type != PipelineDataType::F32 &&
         recipe.output.data_type != PipelineDataType::U8 &&
         recipe.output.data_type != PipelineDataType::S8) ||
        recipe.output.dims != 4 ||
        (recipe.output.layout != Layout::NCHW &&
         recipe.output.layout != Layout::NHWC))
    {
        CV_Error(
            Error::StsNotImplemented,
            "modelInput recipe supports rank-4 F32/U8/S8 NCHW/NHWC output");
    }
    if (recipe.output.shape[0] != 1)
    {
        CV_Error(
            Error::StsNotImplemented,
            "modelInput recipe supports batch=1");
    }
    if (recipe.output.layout == Layout::NCHW)
    {
        if (recipe.output.shape[1] != 3)
        {
            CV_Error(
                Error::StsNotImplemented,
                "modelInput NCHW output must have three channels");
        }
        output_height = recipe.output.shape[2];
        output_width = recipe.output.shape[3];
    }
    else
    {
        if (recipe.output.shape[3] != 3)
        {
            CV_Error(
                Error::StsNotImplemented,
                "modelInput NHWC output must have three channels");
        }
        output_height = recipe.output.shape[1];
        output_width = recipe.output.shape[2];
    }
    if (recipe.normalize_count != 1 && recipe.normalize_count != 3)
    {
        CV_Error(
            Error::StsBadArg,
            "modelInput normalize_count must be 1 or 3");
    }
    for (int index = 0; index < recipe.normalize_count; ++index)
    {
        const float mean = recipe.mean[static_cast<std::size_t>(index)];
        const float stddev =
            recipe.stddev[static_cast<std::size_t>(index)];
        if (!std::isfinite(mean) || !std::isfinite(stddev) ||
            stddev == 0.0f)
        {
            CV_Error(
                Error::StsBadArg,
            "modelInput mean/stddev must be finite and stddev non-zero");
        }
    }
    if (recipe.output.data_type == PipelineDataType::U8 ||
        recipe.output.data_type == PipelineDataType::S8)
    {
        if (!std::isfinite(recipe.quantize_scale) ||
            recipe.quantize_scale <= 0.0f)
        {
            CV_Error(
                Error::StsBadArg,
                "modelInput quantize scale must be finite and positive");
        }
        const int minimum =
            recipe.output.data_type == PipelineDataType::U8 ? 0 : -128;
        const int maximum =
            recipe.output.data_type == PipelineDataType::U8 ? 255 : 127;
        if (recipe.quantize_zero_point < minimum ||
            recipe.quantize_zero_point > maximum)
        {
            CV_Error(
                Error::StsOutOfRange,
                "modelInput quantize zero point is outside target range");
        }
    }
}

}  // namespace recipe_detail

class ModelInputRecipeBuilder
{
public:
    explicit ModelInputRecipeBuilder(ModelInputRecipe recipe)
        : recipe_(std::move(recipe))
    {
    }

    ModelInputRecipeBuilder& preferFusion() { return *this; }
    ModelInputRecipeBuilder& requireNoFullFrameIntermediate()
    {
        return *this;
    }
    ModelInputRecipeBuilder& requireSingleExecutionGroup()
    {
        return *this;
    }

    PipelinePlan prepare() const
    {
        int output_width = 0;
        int output_height = 0;
        recipe_detail::validateModelInputRecipe(
            recipe_, output_width, output_height);
        PipelineBuilder builder = pipe(recipe_.input, recipe_.output);
        if (recipe_.input.color != recipe_.color)
        {
            builder.color(recipe_.color);
        }
        if (recipe_.geometry == ModelInputGeometry::Letterbox)
        {
            if (recipe_.letterbox_pad_count == 1)
            {
                builder.letterbox(
                    output_width,
                    output_height,
                    recipe_.letterbox_pad_value[0],
                    recipe_.interpolation);
            }
            else
            {
                builder.letterbox(
                    output_width,
                    output_height,
                    {recipe_.letterbox_pad_value[0],
                     recipe_.letterbox_pad_value[1],
                     recipe_.letterbox_pad_value[2]},
                    recipe_.interpolation);
            }
        }
        else
        {
            builder.resize(
                output_width, output_height, recipe_.interpolation);
        }
        if (recipe_.normalize_count == 1)
        {
            builder.normalize(
                {recipe_.mean[0]}, {recipe_.stddev[0]});
        }
        else
        {
            builder.normalize(
                {recipe_.mean[0], recipe_.mean[1], recipe_.mean[2]},
                {recipe_.stddev[0],
                 recipe_.stddev[1],
                 recipe_.stddev[2]});
        }
        if (recipe_.output.data_type == PipelineDataType::U8 ||
            recipe_.output.data_type == PipelineDataType::S8)
        {
            builder.quantize(
                recipe_.output.data_type,
                recipe_.quantize_scale,
                recipe_.quantize_zero_point);
        }
        builder
            .layout(recipe_.output.layout)
            .requireNoFullFrameIntermediate()
            .requireSingleExecutionGroup();
        const bool yuv =
            recipe_.input.pixel_format == PixelFormat::NV12 ||
            recipe_.input.pixel_format == PixelFormat::NV21;
        const bool letterbox =
            recipe_.geometry == ModelInputGeometry::Letterbox;
        const PipelineDataType output_type = recipe_.output.data_type;
        if (yuv)
        {
            builder.recipe_id_ = output_type == PipelineDataType::U8
                ? letterbox
                    ? "cvh.model_input.yuv420_u8_letterbox"
                    : "cvh.model_input.yuv420_u8"
                : output_type == PipelineDataType::S8
                    ? letterbox
                        ? "cvh.model_input.yuv420_s8_letterbox"
                        : "cvh.model_input.yuv420_s8"
                    : letterbox
                        ? "cvh.model_input.yuv420_f32_letterbox"
                        : "cvh.model_input.yuv420_f32";
        }
        else
        {
            builder.recipe_id_ = output_type == PipelineDataType::U8
                ? letterbox
                    ? "cvh.model_input.packed_u8_letterbox"
                    : "cvh.model_input.packed_u8"
                : output_type == PipelineDataType::S8
                    ? letterbox
                        ? "cvh.model_input.packed_s8_letterbox"
                        : "cvh.model_input.packed_s8"
                    : letterbox
                        ? "cvh.model_input.packed_f32_letterbox"
                        : "cvh.model_input.packed_f32";
        }
        builder.recipe_contract_version_ = 1;
        builder.recipe_fingerprint_ =
            recipe_detail::modelInputFingerprint(recipe_);
        return builder.prepare();
    }

    PipelineStatus tryPrepare(PipelinePlan& output) const
    {
        output = PipelinePlan();
        try
        {
            output = prepare();
            return PipelineStatus();
        }
        catch (const Exception& exception)
        {
            return detail::prepareExceptionStatus(exception);
        }
        catch (const std::exception& exception)
        {
            return PipelineStatus::failure(
                PipelineStatusCode::InternalError,
                -1,
                exception.what());
        }
    }

private:
    ModelInputRecipe recipe_{};
};

namespace recipes {

inline ModelInputRecipeBuilder modelInput(ModelInputRecipe recipe)
{
    return ModelInputRecipeBuilder(std::move(recipe));
}

}  // namespace recipes
}  // namespace cvh

#endif  // CVH_RECIPES_MODEL_INPUT_H
