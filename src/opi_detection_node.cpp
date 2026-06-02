#include "opi_detection_node.hpp"

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <stdexcept>

namespace opi_detection
{

// ─────────────────────────────────────────────────────────────────────────────
OpiDetectionNode::OpiDetectionNode(const rclcpp::NodeOptions & options)
: Node("opi_detection_node", options),
  ort_env_(ORT_LOGGING_LEVEL_WARNING, "opi_detection")
{
  // ── Parameters ──────────────────────────────────────────────────────────
  model_path_     = declare_parameter<std::string>("model_path", "models/yolov11s.onnx");
  camera_topics_  = declare_parameter<std::vector<std::string>>(
                      "camera_topics", std::vector<std::string>{"luxonis/oak/rgb/image_raw"});
  class_names_    = declare_parameter<std::vector<std::string>>(
                      "class_names", std::vector<std::string>{"adr_hazard_panel"});
  conf_threshold_ = static_cast<float>(declare_parameter<double>("conf_threshold", 0.40));
  nms_threshold_  = static_cast<float>(declare_parameter<double>("nms_threshold",  0.45));
  input_width_    = declare_parameter<int>("input_width",  640);
  input_height_   = declare_parameter<int>("input_height", 640);
  rotate_image_180_ = declare_parameter<bool>("rotate_image_180", false);
  detection_hz_   = declare_parameter<double>("detection_hz", 0.0);
  enable_openvino_ep_ = declare_parameter<bool>("enable_openvino_ep", true);

  if (detection_hz_ > 0.0) {
    detection_period_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(1.0 / detection_hz_));
  } else {
    detection_period_ = std::chrono::steady_clock::duration::zero();
  }

  std::string camera_info_topic =
    declare_parameter<std::string>("camera_info_topic", "luxonis/oak/rgb/camera_info");
  std::string output_topic =
    declare_parameter<std::string>("output_topic", "opi/detections");

  // ── Load model ──────────────────────────────────────────────────────────
  loadModel();

  // ── Subscribers (one per camera + one CameraInfo) ───────────────────────
  for (const auto & topic : camera_topics_) {
    auto sub = create_subscription<sensor_msgs::msg::Image>(
      topic, rclcpp::SensorDataQoS(),
      [this, topic](const sensor_msgs::msg::Image::ConstSharedPtr & msg) {
        imageCallback(msg, topic);
      });
    image_subs_.push_back(sub);
    RCLCPP_INFO(get_logger(), "Subscribed to camera topic: %s", topic.c_str());
  }

  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_info_topic, rclcpp::SensorDataQoS(),
    std::bind(&OpiDetectionNode::cameraInfoCallback, this, std::placeholders::_1));

  // ── Publisher ────────────────────────────────────────────────────────────
  bbox_pub_ = create_publisher<vision_msgs::msg::BoundingBox2DArray>(
    output_topic, rclcpp::SystemDefaultsQoS());

  bbox_img_pub_ = create_publisher<sensor_msgs::msg::Image>(
    output_topic + "/image_raw", rclcpp::SystemDefaultsQoS());

  RCLCPP_INFO(get_logger(), "OPI detection node ready. Model: %s", model_path_.c_str());
  if (detection_period_ == std::chrono::steady_clock::duration::zero()) {
    RCLCPP_INFO(get_logger(), "Detection runs on every received image.");
  } else {
    RCLCPP_INFO(get_logger(), "Detection rate limited to %.2f Hz per camera topic.", detection_hz_);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
std::string OpiDetectionNode::classLabel(int class_id) const
{
  if (class_id >= 0 && class_id < static_cast<int>(class_names_.size())) {
    return class_names_[static_cast<size_t>(class_id)];
  }
  return std::to_string(class_id);
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiDetectionNode::loadModel()
{
  // OpenVINO EP options
  std::unordered_map<std::string, std::string> ov_options;
  ov_options["device_type"]            = "CPU";
  ov_options["precision"]              = "FP32";
  ov_options["num_of_threads"]         = "8";
  ov_options["cache_dir"]              = "/tmp/ov_cache";
  ov_options["enable_opencl_throttling"] = "false";

  if (enable_openvino_ep_) {
    try {
      session_options_.AppendExecutionProvider("OpenVINO", ov_options);
      RCLCPP_INFO(get_logger(), "OpenVINO EP enabled");
    } catch (const Ort::Exception & e) {
      RCLCPP_WARN(
        get_logger(),
        "OpenVINO EP is not available in the linked ONNX Runtime build, so the node will run on CPU: %s",
        e.what());
      RCLCPP_WARN(
        get_logger(),
        "Set `enable_openvino_ep:=false` to skip this warning, or install an ONNX Runtime build with OpenVINO support.");
    }
  } else {
    RCLCPP_INFO(get_logger(), "OpenVINO EP disabled by parameter; running with CPU execution provider");
  }

  // Keep these as fallback
  session_options_.SetIntraOpNumThreads(std::thread::hardware_concurrency());
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  session_options_.EnableCpuMemArena();
  session_options_.EnableMemPattern();
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  session_ = std::make_unique<Ort::Session>(
    ort_env_, model_path_.c_str(), session_options_);

  // Collect input/output names
  for (size_t i = 0; i < session_->GetInputCount(); ++i) {
    auto name_ptr = session_->GetInputNameAllocated(i, allocator_);
    input_names_storage_.emplace_back(name_ptr.get());
  }
  for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
    auto name_ptr = session_->GetOutputNameAllocated(i, allocator_);
    output_names_storage_.emplace_back(name_ptr.get());
  }
  for (const auto & s : input_names_storage_)  input_names_.push_back(s.c_str());
  for (const auto & s : output_names_storage_) output_names_.push_back(s.c_str());

  // Input shape: [1, 3, H, W]
  input_shape_ = {1, 3, input_height_, input_width_};
  input_buffer_.resize(1 * 3 * input_height_ * input_width_);

  RCLCPP_INFO(get_logger(), "ONNX model loaded. Inputs: %zu  Outputs: %zu",
              input_names_.size(), output_names_.size());
}

// ─────────────────────────────────────────────────────────────────────────────
cv::Mat OpiDetectionNode::preprocess(const cv::Mat & image,
                                     float & scale_x, float & scale_y) const
{
  // Letterbox resize to [input_height_ x input_width_]
  int orig_h = image.rows, orig_w = image.cols;
  scale_x = static_cast<float>(orig_w) / static_cast<float>(input_width_);
  scale_y = static_cast<float>(orig_h) / static_cast<float>(input_height_);

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(input_width_, input_height_));

  // BGR -> RGB, uint8 -> float32 normalised to [0,1]
  cv::Mat rgb;
  cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

  // HWC -> CHW blob
  cv::Mat blob = cv::dnn::blobFromImage(rgb);  // returns 1×C×H×W
  return blob;
}

// ─────────────────────────────────────────────────────────────────────────────
// YOLOv8 output shape: [1, num_classes+4, num_anchors]
// Each column: [cx, cy, w, h, cls0_score, cls1_score, ...]
std::vector<Detection> OpiDetectionNode::postprocess(
  const std::vector<float> & raw,
  float scale_x, float scale_y,
  int /*orig_w*/, int /*orig_h*/) const
{
  // Determine number of predictions from output size.
  // YOLOv8s default: 8400 anchors for 640×640 input.
  // Output flat layout: (4 + num_classes) × num_anchors stored row-major.
  // We don't know num_classes at compile time, so derive from raw.size().
  const int num_anchors  = 8400;  // standard YOLOv8 at 640
  const int row_stride   = static_cast<int>(raw.size()) / num_anchors;
  const int num_classes  = row_stride - 4;

  std::vector<cv::Rect>  boxes;
  std::vector<float>     scores;
  std::vector<int>       class_ids;
  std::vector<Detection> detections;

  for (int a = 0; a < num_anchors; ++a) {
    // Find the best class score for this anchor
    float best_score = 0.0f;
    int   best_cls   = 0;
    for (int c = 0; c < num_classes; ++c) {
      float s = raw[static_cast<size_t>((4 + c) * num_anchors + a)];
      if (s > best_score) { best_score = s; best_cls = c; }
    }
    if (best_score < conf_threshold_) continue;

    float cx = raw[static_cast<size_t>(0 * num_anchors + a)] * scale_x;
    float cy = raw[static_cast<size_t>(1 * num_anchors + a)] * scale_y;
    float bw = raw[static_cast<size_t>(2 * num_anchors + a)] * scale_x;
    float bh = raw[static_cast<size_t>(3 * num_anchors + a)] * scale_y;

    int x = static_cast<int>(cx - bw / 2.0f);
    int y = static_cast<int>(cy - bh / 2.0f);
    int w = static_cast<int>(bw);
    int h = static_cast<int>(bh);

    boxes.emplace_back(x, y, w, h);
    scores.push_back(best_score);
    class_ids.push_back(best_cls);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, indices);

  for (int idx : indices) {
    Detection d;
    d.x          = static_cast<float>(boxes[idx].x + boxes[idx].width  / 2);
    d.y          = static_cast<float>(boxes[idx].y + boxes[idx].height / 2);
    d.w          = static_cast<float>(boxes[idx].width);
    d.h          = static_cast<float>(boxes[idx].height);
    d.confidence = scores[idx];
    d.class_id   = class_ids[idx];
    detections.push_back(d);
  }
  return detections;
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiDetectionNode::imageCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg,
  const std::string & camera_topic)
{
  const auto callback_start = std::chrono::steady_clock::now();

  if (detection_period_ != std::chrono::steady_clock::duration::zero()) {
    const auto now = std::chrono::steady_clock::now();
    auto & last_detection_time = last_detection_time_[camera_topic];
    if (last_detection_time.time_since_epoch().count() != 0 &&
        now - last_detection_time < detection_period_) {
      return;
    }
    last_detection_time = now;
  }

  cv::Mat image;
  try {
    image = cv_bridge::toCvCopy(msg, "bgr8")->image;
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  if (rotate_image_180_) {
    cv::rotate(image, image, cv::ROTATE_180);
  }

  const auto preprocess_start = std::chrono::steady_clock::now();
  float scale_x, scale_y;
  cv::Mat blob = preprocess(image, scale_x, scale_y);
  const auto preprocess_end = std::chrono::steady_clock::now();

  // Run inference
  const auto inference_start = std::chrono::steady_clock::now();
  Ort::MemoryInfo mem_info =
    Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
  //   mem_info,
  //   reinterpret_cast<float *>(blob.data),
  //   static_cast<size_t>(blob.total()),
  //   input_shape_.data(),
  //   input_shape_.size());
  std::memcpy(input_buffer_.data(), blob.data, input_buffer_.size() * sizeof(float));
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    mem_info,
    input_buffer_.data(),
    input_buffer_.size(),
    input_shape_.data(),
    input_shape_.size());

  auto output_tensors = session_->Run(
    Ort::RunOptions{nullptr},
    input_names_.data(),
    &input_tensor,
    1,
    output_names_.data(),
    output_names_.size());
  const auto inference_end = std::chrono::steady_clock::now();

  float * raw_ptr    = output_tensors[0].GetTensorMutableData<float>();
  size_t  raw_size   = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
  std::vector<float> raw_output(raw_ptr, raw_ptr + raw_size);

  const auto postprocess_start = std::chrono::steady_clock::now();
  auto detections = postprocess(raw_output, scale_x, scale_y, image.cols, image.rows);
  const auto postprocess_end = std::chrono::steady_clock::now();

  // Publish
  vision_msgs::msg::BoundingBox2DArray bbox_array;
  bbox_array.header = msg->header;

  for (const auto & det : detections) {
    vision_msgs::msg::BoundingBox2D bbox;
    bbox.center.position.x  = det.x;
    bbox.center.position.y  = det.y;
    bbox.size_x             = det.w;
    bbox.size_y             = det.h;
    bbox_array.boxes.push_back(bbox);

    int x1 = det.x - det.w / 2,  y1 = det.y - det.h / 2;
    int x2 = det.x + det.w / 2,  y2 = det.y + det.h / 2;

    cv::rectangle(image, {x1, y1}, {x2, y2}, {0, 0, 255}, 2);

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << det.confidence;
    const std::string label = ss.str();

    cv::putText(image, label, {x1, y1 - 8}, cv::FONT_HERSHEY_SIMPLEX, 1.6, {0, 0, 255}, 3.3);
  }

  bbox_pub_->publish(bbox_array);

  // Publish detection as image.
  bbox_img_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", image).toImageMsg());

  const auto callback_end = std::chrono::steady_clock::now();
  const auto preprocess_ms =
    std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
  const auto inference_ms =
    std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
  const auto postprocess_ms =
    std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
  const auto total_ms =
    std::chrono::duration<double, std::milli>(callback_end - callback_start).count();

  RCLCPP_INFO(
    get_logger(),
    "[%s] timing: preprocess=%.2f ms, inference=%.2f ms, postprocess=%.2f ms, total=%.2f ms, detections=%zu",
    camera_topic.c_str(),
    preprocess_ms,
    inference_ms,
    postprocess_ms,
    total_ms,
    detections.size());
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiDetectionNode::cameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
{
  // Store the latest CameraInfo for use if needed (e.g. undistortion).
  // Currently consumed by the localization node, but kept here for completeness.
  (void)msg;
}

}  // namespace opi_detection

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<opi_detection::OpiDetectionNode>());
  rclcpp::shutdown();
  return 0;
}
