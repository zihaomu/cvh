#include "pipeline_compare_backend.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace cvh_pipeline_proof {
namespace {

std::size_t matBytes(const cv::Mat& mat)
{
    return mat.empty() ? 0 : mat.total() * mat.elemSize();
}

class ExplicitOpenCvPipelineRunner final : public OpenCvPipelineRunner
{
public:
    ExplicitOpenCvPipelineRunner(
        CaseSpec spec,
        const std::vector<const std::uint8_t*>& inputs,
        std::size_t input_row_stride,
        const std::vector<float*>& outputs)
        : spec_(std::move(spec))
    {
        if (inputs.empty() || inputs.size() != outputs.size())
        {
            throw std::invalid_argument(
                "OpenCV Pipeline runner requires matching non-empty rings");
        }
        const std::size_t minimum_stride =
            static_cast<std::size_t>(spec_.input_width) * 3;
        if (input_row_stride < minimum_stride)
        {
            throw std::invalid_argument(
                "OpenCV Pipeline input stride is too small");
        }

        input_frames_.reserve(inputs.size());
        for (const std::uint8_t* input : inputs)
        {
            if (input == nullptr)
            {
                throw std::invalid_argument(
                    "OpenCV Pipeline input pointer is null");
            }
            input_frames_.emplace_back(
                spec_.input_height,
                spec_.input_width,
                CV_8UC3,
                const_cast<std::uint8_t*>(input),
                input_row_stride);
        }

        computeGeometry();
        allocateIntermediates();
        prepareOutputs(outputs);
        prepareNormalizeTransform();
    }

    void run(std::size_t frame_index) override
    {
        const std::size_t index = frame_index % input_frames_.size();
        const cv::Mat& input = input_frames_[index];
        cv::resize(
            input,
            resized_content_,
            cv::Size(content_width_, content_height_),
            0.0,
            0.0,
            spec_.interpolation == Interpolation::Nearest
                ? cv::INTER_NEAREST
                : cv::INTER_LINEAR);

        const cv::Mat* geometry_output = &resized_content_;
        if (spec_.geometry == Geometry::Letterbox)
        {
            cv::copyMakeBorder(
                resized_content_,
                letterboxed_,
                pad_top_,
                pad_bottom_,
                pad_left_,
                pad_right_,
                cv::BORDER_CONSTANT,
                cv::Scalar(
                    spec_.pad_value,
                    spec_.pad_value,
                    spec_.pad_value));
            geometry_output = &letterboxed_;
        }

        if (spec_.output_layout == OutputLayout::NCHW)
        {
            cv::split(*geometry_output, split_u8_);
            for (int target_channel = 0; target_channel < 3;
                 ++target_channel)
            {
                const int source_channel =
                    spec_.input_format == InputFormat::BGR8
                    ? 2 - target_channel
                    : target_channel;
                const float inverse = 1.0f /
                    spec_.stddev[static_cast<std::size_t>(target_channel)];
                split_u8_[static_cast<std::size_t>(source_channel)].convertTo(
                    nchw_outputs_[index]
                        [static_cast<std::size_t>(target_channel)],
                    CV_32F,
                    inverse,
                    -spec_.mean[static_cast<std::size_t>(target_channel)] *
                        inverse);
            }
        }
        else
        {
            geometry_output->convertTo(converted_, CV_32FC3);
            cv::transform(
                converted_, nhwc_outputs_[index], normalize_transform_);
        }
    }

    std::size_t explicitTemporaryBytes() const override
    {
        return temporary_bytes_;
    }

    int explicitFullFrameIntermediates() const override
    {
        return full_frame_intermediates_;
    }

    const char* algorithmPath() const override
    {
        return algorithm_path_.c_str();
    }

private:
    void computeGeometry()
    {
        content_width_ = spec_.output_width;
        content_height_ = spec_.output_height;
        if (spec_.geometry != Geometry::Letterbox)
        {
            return;
        }
        const float scale = std::min(
            static_cast<float>(spec_.output_width) /
                static_cast<float>(spec_.input_width),
            static_cast<float>(spec_.output_height) /
                static_cast<float>(spec_.input_height));
        content_width_ = std::clamp(
            static_cast<int>(std::floor(
                static_cast<float>(spec_.input_width) * scale + 0.5f)),
            1,
            spec_.output_width);
        content_height_ = std::clamp(
            static_cast<int>(std::floor(
                static_cast<float>(spec_.input_height) * scale + 0.5f)),
            1,
            spec_.output_height);
        const int horizontal = spec_.output_width - content_width_;
        const int vertical = spec_.output_height - content_height_;
        pad_left_ = horizontal / 2;
        pad_right_ = horizontal - pad_left_;
        pad_top_ = vertical / 2;
        pad_bottom_ = vertical - pad_top_;
    }

    void allocateIntermediates()
    {
        resized_content_.create(content_height_, content_width_, CV_8UC3);
        if (spec_.geometry == Geometry::Letterbox)
        {
            letterboxed_.create(
                spec_.output_height, spec_.output_width, CV_8UC3);
        }
        full_frame_intermediates_ =
            1 + (spec_.geometry == Geometry::Letterbox ? 1 : 0);
        if (spec_.output_layout == OutputLayout::NCHW)
        {
            split_u8_.resize(3);
            for (cv::Mat& plane : split_u8_)
            {
                plane.create(
                    spec_.output_height, spec_.output_width, CV_8UC1);
            }
            ++full_frame_intermediates_;
            algorithm_path_ = spec_.geometry == Geometry::Letterbox
                ? "resize_copyMakeBorder_split_u8_convert_normalize_planes"
                : "resize_split_u8_convert_normalize_planes";
        }
        else
        {
            converted_.create(
                spec_.output_height, spec_.output_width, CV_32FC3);
            ++full_frame_intermediates_;
            algorithm_path_ = spec_.geometry == Geometry::Letterbox
                ? "resize_copyMakeBorder_convert_transform_rgb_normalize"
                : "resize_convert_transform_rgb_normalize";
        }
        temporary_bytes_ =
            matBytes(resized_content_) + matBytes(letterboxed_) +
            matBytes(converted_);
        for (const cv::Mat& plane : split_u8_)
        {
            temporary_bytes_ += matBytes(plane);
        }
    }

    void prepareOutputs(const std::vector<float*>& outputs)
    {
        if (spec_.output_layout == OutputLayout::NHWC)
        {
            nhwc_outputs_.reserve(outputs.size());
            for (float* output : outputs)
            {
                if (output == nullptr)
                {
                    throw std::invalid_argument(
                        "OpenCV Pipeline output pointer is null");
                }
                nhwc_outputs_.emplace_back(
                    spec_.output_height,
                    spec_.output_width,
                    CV_32FC3,
                    output);
            }
            return;
        }

        const std::size_t plane_values =
            static_cast<std::size_t>(spec_.output_width) *
            static_cast<std::size_t>(spec_.output_height);
        nchw_outputs_.reserve(outputs.size());
        for (float* output : outputs)
        {
            if (output == nullptr)
            {
                throw std::invalid_argument(
                    "OpenCV Pipeline output pointer is null");
            }
            std::vector<cv::Mat> planes;
            planes.reserve(3);
            for (int channel = 0; channel < 3; ++channel)
            {
                planes.emplace_back(
                    spec_.output_height,
                    spec_.output_width,
                    CV_32FC1,
                    output + static_cast<std::size_t>(channel) *
                        plane_values);
            }
            nchw_outputs_.push_back(std::move(planes));
        }
    }

    void prepareNormalizeTransform()
    {
        normalize_transform_ = cv::Mat::zeros(3, 4, CV_32F);
        for (int target_channel = 0; target_channel < 3; ++target_channel)
        {
            const int source_channel =
                spec_.input_format == InputFormat::BGR8
                ? 2 - target_channel
                : target_channel;
            const float inverse = 1.0f /
                spec_.stddev[static_cast<std::size_t>(target_channel)];
            normalize_transform_.at<float>(
                target_channel, source_channel) = inverse;
            normalize_transform_.at<float>(target_channel, 3) =
                -spec_.mean[static_cast<std::size_t>(target_channel)] *
                inverse;
        }
    }

    CaseSpec spec_;
    std::vector<cv::Mat> input_frames_;
    cv::Mat resized_content_;
    cv::Mat letterboxed_;
    cv::Mat converted_;
    std::vector<cv::Mat> split_u8_;
    cv::Mat normalize_transform_;
    std::vector<cv::Mat> nhwc_outputs_;
    std::vector<std::vector<cv::Mat>> nchw_outputs_;
    int content_width_ = 0;
    int content_height_ = 0;
    int pad_left_ = 0;
    int pad_right_ = 0;
    int pad_top_ = 0;
    int pad_bottom_ = 0;
    std::size_t temporary_bytes_ = 0;
    int full_frame_intermediates_ = 0;
    std::string algorithm_path_;
};

}  // namespace

void configureOpenCvPipelineThreads(int threads)
{
    cv::setNumThreads(std::max(1, threads));
    cv::setUseOptimized(true);
}

std::unique_ptr<OpenCvPipelineRunner> makeOpenCvPipelineRunner(
    const CaseSpec& spec,
    const std::vector<const std::uint8_t*>& inputs,
    std::size_t input_row_stride,
    const std::vector<float*>& outputs)
{
    return std::unique_ptr<OpenCvPipelineRunner>(
        new ExplicitOpenCvPipelineRunner(
            spec, inputs, input_row_stride, outputs));
}

const char* openCvPipelineVersion()
{
    return CV_VERSION;
}

}  // namespace cvh_pipeline_proof
