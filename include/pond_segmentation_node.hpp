#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace pond_segmentation
{

struct PreprocessInfo
{
  float scale = 1.0f;
  int pad_left = 0;
  int pad_top = 0;
  int resized_width = 0;
  int resized_height = 0;
  int original_width = 0;
  int original_height = 0;
};

class PondSegmentationNode : public rclcpp::Node
{
public:
  explicit PondSegmentationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  enum class InferenceMode
  {
    kUnknown,
    kSemantic,
    kYoloSegmentation,
  };

  std::string model_path_;
  std::vector<std::string> camera_topics_;
  std::string output_topic_;
  float conf_threshold_;
  float nms_threshold_;
  float mask_threshold_;
  int input_width_;
  int input_height_;
  bool rotate_image_180_;
  double segmentation_hz_;
  int foreground_class_id_;
  std::chrono::steady_clock::duration segmentation_period_;
  InferenceMode inference_mode_{InferenceMode::kUnknown};

  Ort::Env                             ort_env_;
  Ort::SessionOptions                  session_options_;
  std::unique_ptr<Ort::Session>        session_;
  Ort::AllocatorWithDefaultOptions     allocator_;
  std::vector<std::string>             input_names_storage_;
  std::vector<std::string>             output_names_storage_;
  std::vector<const char *>            input_names_;
  std::vector<const char *>            output_names_;
  std::vector<int64_t>                 input_shape_;
  std::vector<float>                   input_buffer_;

  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> image_subs_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr                 mask_pub_;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_segmentation_time_;

  void loadModel();
  cv::Mat preprocess(const cv::Mat & image, PreprocessInfo & info) const;
  cv::Mat postprocess(
    const std::vector<Ort::Value> & outputs,
    const PreprocessInfo & info,
    int original_width,
    int original_height) const;
  cv::Mat postprocessSemantic(
    const Ort::Value & output,
    const PreprocessInfo & info,
    int original_width,
    int original_height) const;
  cv::Mat postprocessYoloSegmentation(
    const std::vector<Ort::Value> & outputs,
    const PreprocessInfo & info,
    int original_width,
    int original_height) const;
  void imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr & msg,
    const std::string & camera_topic);
};

}  // namespace pond_segmentation
