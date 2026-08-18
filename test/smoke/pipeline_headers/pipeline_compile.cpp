#include "cvh/pipeline/pipeline.h"
#include "cvh/recipes/model_input.h"

int cvh_pipeline_pipeline_header_compile()
{
    const auto image =
        cvh::imageDesc(2, 2, cvh::PixelFormat::BGR8);
    cvh::ModelInputRecipe recipe;
    recipe.input = image;
    recipe.output =
        cvh::tensorDesc<signed char>({1, 3, 1, 1}, cvh::Layout::NCHW);
    recipe.quantize_scale = 0.025f;
    recipe.quantize_zero_point = -3;
    const cvh::PipelinePlan plan =
        cvh::recipes::modelInput(recipe).prepare();
    return image.valid() && plan.info().recipe_contract_version == 1
        ? 0
        : 1;
}
