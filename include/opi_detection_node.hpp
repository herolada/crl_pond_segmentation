#pragma once

#include <memory>
#include <string>
#include <chrono>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <vision_msgs/msg/bounding_box2_d_array.hpp>
#include <vision_msgs/msg/bounding_box2_d.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
// #include <onnxruntime/core/session/onnxruntime_cxx_api.h>

namespace opi_detection
{

struct Detection
{
  float x, y, w, h;   // centre-x, centre-y, width, height (pixels)
  float confidence;
  int   class_id;
};

class OpiDetectionNode : public rclcpp::Node
{
public:
  explicit OpiDetectionNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  // ── Parameters ────────────────────────────────────────────────────────────
  std::string model_path_;
  std::vector<std::string> camera_topics_;
  std::vector<std::string> class_names_;
  float conf_threshold_;
  float nms_threshold_;
  int   input_width_;
  int   input_height_;
  bool  rotate_image_180_;
  double detection_hz_;
  bool  enable_openvino_ep_;
  std::chrono::steady_clock::duration detection_period_;

  // ── ONNX Runtime ──────────────────────────────────────────────────────────
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

  // ── ROS I/O ───────────────────────────────────────────────────────────────
  // One subscriber per camera topic (stored so they are not destroyed)
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> image_subs_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr         camera_info_sub_;
  rclcpp::Publisher<vision_msgs::msg::BoundingBox2DArray>::SharedPtr    bbox_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr                 bbox_img_pub_;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point> last_detection_time_;

  // ── Helpers ───────────────────────────────────────────────────────────────
  void loadModel();
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg,
                     const std::string & camera_topic);
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg);

  cv::Mat preprocess(const cv::Mat & image, float & scale_x, float & scale_y) const;
  std::vector<Detection> postprocess(const std::vector<float> & output,
                                     float scale_x, float scale_y,
                                     int orig_w, int orig_h) const;
  std::string classLabel(int class_id) const;
};

}  // namespace opi_detection
