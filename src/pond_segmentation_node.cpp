#include "pond_segmentation_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace pond_segmentation
{
namespace
{

float sigmoid(float value)
{
  return 1.0f / (1.0f + std::exp(-value));
}

int clampInt(int value, int low, int high)
{
  return std::max(low, std::min(value, high));
}

std::string shapeToString(const std::vector<int64_t> & shape)
{
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      ss << ", ";
    }
    ss << shape[i];
  }
  ss << "]";
  return ss.str();
}

}  // namespace

PondSegmentationNode::PondSegmentationNode(const rclcpp::NodeOptions & options)
: Node("pond_segmentation_node", options),
  ort_env_(ORT_LOGGING_LEVEL_WARNING, "pond_segmentation")
{
  model_path_ = declare_parameter<std::string>("model_path", "models/best_int8.onnx");
  camera_topics_ = declare_parameter<std::vector<std::string>>(
    "camera_topics", std::vector<std::string>{"luxonis/oak/rgb/image_raw"});
  output_topic_ = declare_parameter<std::string>("output_topic", "pond/segmentation");
  conf_threshold_ = static_cast<float>(declare_parameter<double>("conf_threshold", 0.25));
  nms_threshold_ = static_cast<float>(declare_parameter<double>("nms_threshold", 0.45));
  mask_threshold_ = static_cast<float>(declare_parameter<double>("mask_threshold", 0.50));
  input_width_ = declare_parameter<int>("input_width", 320);
  input_height_ = declare_parameter<int>("input_height", 320);
  rotate_image_180_ = declare_parameter<bool>("rotate_image_180", false);
  segmentation_hz_ = declare_parameter<double>("segmentation_hz", 0.0);
  foreground_class_id_ = declare_parameter<int>("foreground_class_id", 1);

  if (segmentation_hz_ > 0.0) {
    segmentation_period_ = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(1.0 / segmentation_hz_));
  } else {
    segmentation_period_ = std::chrono::steady_clock::duration::zero();
  }

  loadModel();

  for (const auto & topic : camera_topics_) {
    auto sub = create_subscription<sensor_msgs::msg::Image>(
      topic, rclcpp::SensorDataQoS(),
      [this, topic](const sensor_msgs::msg::Image::ConstSharedPtr & msg) {
        imageCallback(msg, topic);
      });
    image_subs_.push_back(sub);
    RCLCPP_INFO(get_logger(), "Subscribed to image topic: %s", topic.c_str());
  }

  mask_pub_ = create_publisher<sensor_msgs::msg::Image>(
    output_topic_ + "/image_raw", rclcpp::SystemDefaultsQoS());

  RCLCPP_INFO(get_logger(), "Segmentation node ready. Model: %s", model_path_.c_str());
  RCLCPP_INFO(
    get_logger(),
    "Publishing binary mask on: %s",
    (output_topic_ + "/image_raw").c_str());
}

void PondSegmentationNode::loadModel()
{
  session_options_.SetIntraOpNumThreads(
    std::max(1u, std::thread::hardware_concurrency()));
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  session_options_.EnableCpuMemArena();
  session_options_.EnableMemPattern();
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  session_ = std::make_unique<Ort::Session>(
    ort_env_, model_path_.c_str(), session_options_);

  input_names_storage_.clear();
  output_names_storage_.clear();
  input_names_.clear();
  output_names_.clear();

  for (size_t i = 0; i < session_->GetInputCount(); ++i) {
    auto name_ptr = session_->GetInputNameAllocated(i, allocator_);
    input_names_storage_.emplace_back(name_ptr.get());
  }
  for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
    auto name_ptr = session_->GetOutputNameAllocated(i, allocator_);
    output_names_storage_.emplace_back(name_ptr.get());
  }
  for (const auto & s : input_names_storage_) {
    input_names_.push_back(s.c_str());
  }
  for (const auto & s : output_names_storage_) {
    output_names_.push_back(s.c_str());
  }

  if (input_names_.empty() || output_names_.empty()) {
    throw std::runtime_error("ONNX model has no inputs or outputs");
  }

  auto input_info = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
  auto input_shape = input_info.GetShape();
  const bool input_shape_valid =
    input_shape.size() == 4 &&
    std::all_of(input_shape.begin(), input_shape.end(), [](int64_t dim) { return dim > 0; });

  if (input_shape_valid) {
    input_shape_ = input_shape;
    if (input_shape_[2] > 0) {
      input_height_ = static_cast<int>(input_shape_[2]);
    }
    if (input_shape_[3] > 0) {
      input_width_ = static_cast<int>(input_shape_[3]);
    }
  } else {
    input_shape_ = {1, 3, input_height_, input_width_};
  }

  input_buffer_.resize(static_cast<size_t>(input_shape_[0] * input_shape_[1] * input_shape_[2] * input_shape_[3]));

  // Use output count to determine inference mode.
  // Dynamic-shape ONNX exports return empty shape vectors at load time, so
  // inspecting ranks here is unreliable. YOLO-seg always has exactly 2 outputs
  // (predictions + mask prototypes); semantic models have 1.
  const size_t output_count = session_->GetOutputCount();
  if (output_count == 1) {
    inference_mode_ = InferenceMode::kSemantic;
  } else if (output_count == 2) {
    inference_mode_ = InferenceMode::kYoloSegmentation;
  } else {
    inference_mode_ = InferenceMode::kUnknown;
  }

  RCLCPP_INFO(get_logger(), "ONNX model loaded successfully.");
  RCLCPP_INFO(
    get_logger(),
    "Input tensor shape: %s",
    shapeToString(input_shape_).c_str());
  for (size_t i = 0; i < output_count; ++i) {
    auto shape = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    RCLCPP_INFO(
      get_logger(),
      "Output[%zu] shape: %s%s",
      i,
      shapeToString(shape).c_str(),
      shape.empty() ? " (dynamic/unknown at load time)" : "");
  }
  RCLCPP_INFO(
    get_logger(),
    "Inference mode: %s",
    inference_mode_ == InferenceMode::kSemantic ? "semantic" :
    inference_mode_ == InferenceMode::kYoloSegmentation ? "yolo-segmentation" : "unknown");
}

cv::Mat PondSegmentationNode::preprocess(const cv::Mat & image, PreprocessInfo & info) const
{
  info.original_width = image.cols;
  info.original_height = image.rows;

  if (image.empty()) {
    return {};
  }

  const float scale_w = static_cast<float>(input_width_) / static_cast<float>(image.cols);
  const float scale_h = static_cast<float>(input_height_) / static_cast<float>(image.rows);
  info.scale = std::min(scale_w, scale_h);
  info.resized_width = std::max(1, static_cast<int>(std::round(image.cols * info.scale)));
  info.resized_height = std::max(1, static_cast<int>(std::round(image.rows * info.scale)));
  info.pad_left = (input_width_ - info.resized_width) / 2;
  info.pad_top = (input_height_ - info.resized_height) / 2;

  cv::Mat resized;
  cv::resize(image, resized, cv::Size(info.resized_width, info.resized_height), 0.0, 0.0, cv::INTER_LINEAR);

  cv::Mat letterboxed;
  cv::copyMakeBorder(
    resized,
    letterboxed,
    info.pad_top,
    input_height_ - info.pad_top - info.resized_height,
    info.pad_left,
    input_width_ - info.pad_left - info.resized_width,
    cv::BORDER_CONSTANT,
    cv::Scalar(114, 114, 114));

  return cv::dnn::blobFromImage(
    letterboxed,
    1.0 / 255.0,
    cv::Size(),
    cv::Scalar(),
    true,
    false,
    CV_32F);
}

cv::Mat PondSegmentationNode::postprocess(
  const std::vector<Ort::Value> & outputs,
  const PreprocessInfo & info,
  int original_width,
  int original_height) const
{
  if (outputs.empty()) {
    return cv::Mat::zeros(original_height, original_width, CV_8UC1);
  }

  if (inference_mode_ == InferenceMode::kSemantic || outputs.size() == 1) {
    return postprocessSemantic(outputs[0], info, original_width, original_height);
  }

  if (inference_mode_ == InferenceMode::kYoloSegmentation) {
    return postprocessYoloSegmentation(outputs, info, original_width, original_height);
  }

  RCLCPP_WARN(get_logger(), "Unknown ONNX output layout. Returning empty mask.");
  return cv::Mat::zeros(original_height, original_width, CV_8UC1);
}

cv::Mat PondSegmentationNode::postprocessSemantic(
  const Ort::Value & output,
  const PreprocessInfo & info,
  int original_width,
  int original_height) const
{
  auto tensor_info = output.GetTensorTypeAndShapeInfo();
  const auto shape = tensor_info.GetShape();
  if (shape.size() != 4) {
    RCLCPP_WARN(get_logger(), "Semantic output must be rank-4, got %s", shapeToString(shape).c_str());
    return cv::Mat::zeros(original_height, original_width, CV_8UC1);
  }

  const float * data = output.GetTensorData<float>();
  const int64_t dim1 = shape[1];
  const int64_t dim2 = shape[2];
  const int64_t dim3 = shape[3];

  bool nchw = false;
  bool nhwc = false;
  if (dim1 <= 64 && dim2 > 4 && dim3 > 4) {
    nchw = true;
  } else if (dim3 <= 64 && dim1 > 4 && dim2 > 4) {
    nhwc = true;
  } else {
    nchw = true;
  }

  cv::Mat mask_model;

  if (nchw) {
    const int channels = static_cast<int>(dim1);
    const int height = static_cast<int>(dim2);
    const int width = static_cast<int>(dim3);
    const int class_id = clampInt(foreground_class_id_, 0, std::max(0, channels - 1));

    if (channels == 1) {
      cv::Mat logits(height, width, CV_32F, const_cast<float *>(data));
      cv::Mat probabilities = logits.clone();
      for (int y = 0; y < probabilities.rows; ++y) {
        float * row = probabilities.ptr<float>(y);
        for (int x = 0; x < probabilities.cols; ++x) {
          row[x] = sigmoid(row[x]);
        }
      }
      cv::threshold(probabilities, mask_model, mask_threshold_, 255.0, cv::THRESH_BINARY);
      mask_model.convertTo(mask_model, CV_8U);
    } else {
      const size_t plane_size = static_cast<size_t>(height * width);
      const float * channel_data = data + static_cast<size_t>(class_id) * plane_size;
      cv::Mat logits(height, width, CV_32F, const_cast<float *>(channel_data));
      cv::Mat probabilities = logits.clone();
      for (int y = 0; y < probabilities.rows; ++y) {
        float * row = probabilities.ptr<float>(y);
        for (int x = 0; x < probabilities.cols; ++x) {
          row[x] = sigmoid(row[x]);
        }
      }
      cv::threshold(probabilities, mask_model, mask_threshold_, 255.0, cv::THRESH_BINARY);
      mask_model.convertTo(mask_model, CV_8U);
    }
  } else if (nhwc) {
    const int channels = static_cast<int>(dim3);
    const int height = static_cast<int>(dim1);
    const int width = static_cast<int>(dim2);
    const int class_id = clampInt(foreground_class_id_, 0, std::max(0, channels - 1));

    cv::Mat probabilities(height, width, CV_32F);
    for (int y = 0; y < height; ++y) {
      float * row = probabilities.ptr<float>(y);
      for (int x = 0; x < width; ++x) {
        const size_t index = static_cast<size_t>((y * width + x) * channels + class_id);
        row[x] = sigmoid(data[index]);
      }
    }

    cv::threshold(probabilities, mask_model, mask_threshold_, 255.0, cv::THRESH_BINARY);
    mask_model.convertTo(mask_model, CV_8U);
  }

  if (mask_model.empty()) {
    return cv::Mat::zeros(original_height, original_width, CV_8UC1);
  }

  cv::Mat resized;
  cv::resize(mask_model, resized, cv::Size(input_width_, input_height_), 0.0, 0.0, cv::INTER_NEAREST);

  const cv::Rect roi(
    info.pad_left,
    info.pad_top,
    std::max(1, info.resized_width),
    std::max(1, info.resized_height));

  cv::Mat cropped = resized(roi).clone();
  cv::Mat mask_original;
  cv::resize(cropped, mask_original, cv::Size(original_width, original_height), 0.0, 0.0, cv::INTER_NEAREST);

  if (mask_original.type() != CV_8UC1) {
    mask_original.convertTo(mask_original, CV_8UC1);
  }
  return mask_original;
}

cv::Mat PondSegmentationNode::postprocessYoloSegmentation(
  const std::vector<Ort::Value> & outputs,
  const PreprocessInfo & info,
  int original_width,
  int original_height) const
{
  const Ort::Value * prediction_output = nullptr;
  const Ort::Value * proto_output = nullptr;
  std::vector<int64_t> prediction_shape;
  std::vector<int64_t> proto_shape;

  for (const auto & output : outputs) {
    const auto shape = output.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() == 3 && !prediction_output) {
      prediction_output = &output;
      prediction_shape = shape;
    } else if (shape.size() == 4 && !proto_output) {
      proto_output = &output;
      proto_shape = shape;
    }
  }

  if (!prediction_output || !proto_output) {
    RCLCPP_WARN(get_logger(), "YOLO segmentation outputs were not recognized.");
    return cv::Mat::zeros(original_height, original_width, CV_8UC1);
  }

  const float * prediction_data = prediction_output->GetTensorData<float>();
  const float * proto_data = proto_output->GetTensorData<float>();

  const int64_t pred_dim1 = prediction_shape[1];
  const int64_t pred_dim2 = prediction_shape[2];
  const bool channels_first = pred_dim1 < pred_dim2;
  const int64_t num_predictions = channels_first ? pred_dim2 : pred_dim1;
  const int64_t attribute_count = channels_first ? pred_dim1 : pred_dim2;

  int64_t mask_dim = 0;
  int mask_height = 0;
  int mask_width = 0;
  if (proto_shape[1] > 4 && proto_shape[2] > 4 && proto_shape[3] > 4) {
    mask_dim = proto_shape[1];
    mask_height = static_cast<int>(proto_shape[2]);
    mask_width = static_cast<int>(proto_shape[3]);
  } else {
    RCLCPP_WARN(get_logger(), "YOLO mask prototype has unexpected shape %s", shapeToString(proto_shape).c_str());
    return cv::Mat::zeros(original_height, original_width, CV_8UC1);
  }

  const int64_t num_classes = attribute_count - 4 - mask_dim;
  if (num_classes <= 0) {
    RCLCPP_WARN(
      get_logger(),
      "YOLO prediction tensor does not contain class scores and mask coefficients: %s",
      shapeToString(prediction_shape).c_str());
    return cv::Mat::zeros(original_height, original_width, CV_8UC1);
  }

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<std::vector<float>> coeffs;
  boxes.reserve(static_cast<size_t>(num_predictions));
  scores.reserve(static_cast<size_t>(num_predictions));
  coeffs.reserve(static_cast<size_t>(num_predictions));

  for (int64_t i = 0; i < num_predictions; ++i) {
    const float * row = nullptr;
    auto attr = [&](int64_t index) -> float {
      if (channels_first) {
        return prediction_data[static_cast<size_t>(index) * static_cast<size_t>(num_predictions) +
                               static_cast<size_t>(i)];
      }
      if (!row) {
        row = prediction_data + static_cast<size_t>(i) * static_cast<size_t>(attribute_count);
      }
      return row[index];
    };

    const float cx = attr(0);
    const float cy = attr(1);
    const float width = attr(2);
    const float height = attr(3);

    float best_score = 0.0f;
    for (int64_t c = 0; c < num_classes; ++c) {
      const float class_score = attr(4 + c);
      if (class_score > best_score) {
        best_score = class_score;
      }
    }

    if (best_score < conf_threshold_) {
      continue;
    }

    const float x1 = cx - width * 0.5f;
    const float y1 = cy - height * 0.5f;
    const float x2 = cx + width * 0.5f;
    const float y2 = cy + height * 0.5f;

    const float scale = info.scale > 0.0f ? info.scale : 1.0f;
    int left = static_cast<int>(std::round((x1 - info.pad_left) / scale));
    int top = static_cast<int>(std::round((y1 - info.pad_top) / scale));
    int right = static_cast<int>(std::round((x2 - info.pad_left) / scale));
    int bottom = static_cast<int>(std::round((y2 - info.pad_top) / scale));

    left = clampInt(left, 0, original_width - 1);
    top = clampInt(top, 0, original_height - 1);
    right = clampInt(right, 0, original_width - 1);
    bottom = clampInt(bottom, 0, original_height - 1);

    const int box_width = std::max(1, right - left);
    const int box_height = std::max(1, bottom - top);
    boxes.emplace_back(left, top, box_width, box_height);
    scores.push_back(best_score);

    std::vector<float> detection_coeffs(static_cast<size_t>(mask_dim));
    for (int64_t m = 0; m < mask_dim; ++m) {
      detection_coeffs[static_cast<size_t>(m)] = attr(4 + num_classes + m);
    }
    coeffs.push_back(std::move(detection_coeffs));
  }

  if (boxes.empty()) {
    return cv::Mat::zeros(original_height, original_width, CV_8UC1);
  }

  std::vector<int> kept_indices;
  cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, kept_indices);

  cv::Mat final_mask = cv::Mat::zeros(original_height, original_width, CV_8UC1);

  const size_t proto_plane_size = static_cast<size_t>(mask_height * mask_width);
  cv::Mat proto_mat(
    static_cast<int>(mask_dim),
    static_cast<int>(proto_plane_size),
    CV_32F,
    const_cast<float *>(proto_data));

  for (int index : kept_indices) {
    const auto & det_coeffs = coeffs[static_cast<size_t>(index)];
    cv::Mat coeff_row(1, static_cast<int>(mask_dim), CV_32F, const_cast<float *>(det_coeffs.data()));

    cv::Mat mask_flat = coeff_row * proto_mat;
    cv::Mat mask_small = mask_flat.reshape(1, mask_height).clone();
    for (int y = 0; y < mask_small.rows; ++y) {
      float * row = mask_small.ptr<float>(y);
      for (int x = 0; x < mask_small.cols; ++x) {
        row[x] = sigmoid(row[x]);
      }
    }

    cv::Mat mask_input;
    cv::resize(
      mask_small,
      mask_input,
      cv::Size(input_width_, input_height_),
      0.0,
      0.0,
      cv::INTER_LINEAR);

    const cv::Rect roi(
      info.pad_left,
      info.pad_top,
      std::max(1, info.resized_width),
      std::max(1, info.resized_height));

    cv::Mat mask_unpadded = mask_input(roi).clone();
    cv::Mat mask_original;
    cv::resize(
      mask_unpadded,
      mask_original,
      cv::Size(original_width, original_height),
      0.0,
      0.0,
      cv::INTER_LINEAR);

    cv::Mat binary_mask;
    cv::threshold(mask_original, binary_mask, mask_threshold_, 255.0, cv::THRESH_BINARY);
    binary_mask.convertTo(binary_mask, CV_8U);
    cv::bitwise_or(final_mask, binary_mask, final_mask);
  }

  return final_mask;
}

void PondSegmentationNode::imageCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg,
  const std::string & camera_topic)
{
  const auto callback_start = std::chrono::steady_clock::now();

  if (segmentation_period_ != std::chrono::steady_clock::duration::zero()) {
    const auto now = std::chrono::steady_clock::now();
    auto & last_segmentation_time = last_segmentation_time_[camera_topic];
    if (last_segmentation_time.time_since_epoch().count() != 0 &&
        now - last_segmentation_time < segmentation_period_) {
      return;
    }
    last_segmentation_time = now;
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
  PreprocessInfo preprocess_info;
  cv::Mat blob = preprocess(image, preprocess_info);
  if (blob.empty()) {
    RCLCPP_WARN(get_logger(), "Preprocessing failed for camera topic %s", camera_topic.c_str());
    return;
  }
  const auto preprocess_end = std::chrono::steady_clock::now();

  const auto inference_start = std::chrono::steady_clock::now();
  std::memcpy(
    input_buffer_.data(),
    blob.data,
    input_buffer_.size() * sizeof(float));

  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
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

  const auto postprocess_start = std::chrono::steady_clock::now();
  cv::Mat mask = postprocess(output_tensors, preprocess_info, image.cols, image.rows);
  const auto postprocess_end = std::chrono::steady_clock::now();

  if (mask.empty()) {
    mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
  }

  if (mask.type() != CV_8UC1) {
    mask.convertTo(mask, CV_8UC1);
  }

  sensor_msgs::msg::Image::SharedPtr mask_msg =
    cv_bridge::CvImage(msg->header, "mono8", mask).toImageMsg();
  mask_pub_->publish(*mask_msg);

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
    "[%s] timing: preprocess=%.2f ms, inference=%.2f ms, postprocess=%.2f ms, total=%.2f ms",
    camera_topic.c_str(),
    preprocess_ms,
    inference_ms,
    postprocess_ms,
    total_ms);
}

}  // namespace pond_segmentation

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pond_segmentation::PondSegmentationNode>());
  rclcpp::shutdown();
  return 0;
}
