#include "opi_localization_node.hpp"

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

namespace opi_localization
{

// ─────────────────────────────────────────────────────────────────────────────
OpiLocalizationNode::OpiLocalizationNode(const rclcpp::NodeOptions & options)
: Node("opi_localization_node", options)
{
  // ── Parameters ──────────────────────────────────────────────────────────
  placard_width_m_  = declare_parameter<double>("placard_width_m",  0.40);
  placard_height_m_ = declare_parameter<double>("placard_height_m", 0.30);
  map_frame_        = declare_parameter<std::string>("map_frame",    "map");
  camera_frame_     = declare_parameter<std::string>("camera_frame", "camera_optical_frame");

  std::string camera_info_topic =
    declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");
  std::string bbox_topic   = declare_parameter<std::string>("bbox_topic",   "opi/detections");
  std::string output_topic = declare_parameter<std::string>("output_topic", "opi/positions_raw");

  // ── Model points (placard corners, origin at centre, z=0) ────────────────
  // Order: top-left, top-right, bottom-right, bottom-left (CCW when facing placard)
  float hw = static_cast<float>(placard_width_m_  / 2.0);
  float hh = static_cast<float>(placard_height_m_ / 2.0);
  model_points_ = {
    cv::Point3f(-hw,  hh, 0.0f),  // top-left
    cv::Point3f( hw,  hh, 0.0f),  // top-right
    cv::Point3f( hw, -hh, 0.0f),  // bottom-right
    cv::Point3f(-hw, -hh, 0.0f),  // bottom-left
  };

  // ── TF ──────────────────────────────────────────────────────────────────
  tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // ── ROS I/O ─────────────────────────────────────────────────────────────
  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_info_topic, rclcpp::SensorDataQoS(),
    std::bind(&OpiLocalizationNode::cameraInfoCallback, this, std::placeholders::_1));

  bbox_sub_ = create_subscription<vision_msgs::msg::BoundingBox2DArray>(
    bbox_topic, rclcpp::SystemDefaultsQoS(),
    std::bind(&OpiLocalizationNode::bboxCallback, this, std::placeholders::_1));

  pose_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
    output_topic, rclcpp::SystemDefaultsQoS());

  RCLCPP_INFO(get_logger(),
    "OPI localization node ready. Placard size: %.2f × %.2f m",
    placard_width_m_, placard_height_m_);
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiLocalizationNode::cameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
{
  if (camera_info_received_) return;  // only need it once (or re-update on change)

  camera_frame_ = msg->header.frame_id;

  camera_matrix_ = (cv::Mat_<double>(3, 3) <<
    msg->k[0], msg->k[1], msg->k[2],
    msg->k[3], msg->k[4], msg->k[5],
    msg->k[6], msg->k[7], msg->k[8]);

  dist_coeffs_ = cv::Mat(1, static_cast<int>(msg->d.size()), CV_64F);
  for (size_t i = 0; i < msg->d.size(); ++i)
    dist_coeffs_.at<double>(0, static_cast<int>(i)) = msg->d[i];

  camera_info_received_ = true;
  RCLCPP_INFO(get_logger(), "CameraInfo received and stored.");
}

// ─────────────────────────────────────────────────────────────────────────────
bool OpiLocalizationNode::solvePlacard(
  const vision_msgs::msg::BoundingBox2D & bbox,
  cv::Vec3d & rvec, cv::Vec3d & tvec) const
{
  // Derive the four image corners from the centre + size bounding box.
  // Order must match model_points_: TL, TR, BR, BL.
  float cx = static_cast<float>(bbox.center.position.x);
  float cy = static_cast<float>(bbox.center.position.y);
  float hw = static_cast<float>(bbox.size_x / 2.0);
  float hh = static_cast<float>(bbox.size_y / 2.0);

  std::vector<cv::Point2f> image_points = {
    {cx - hw, cy - hh},  // top-left
    {cx + hw, cy - hh},  // top-right
    {cx + hw, cy + hh},  // bottom-right
    {cx - hw, cy + hh},  // bottom-left
  };

  std::vector<cv::Point3f> obj_pts(model_points_.begin(), model_points_.end());

  // IPPE (Infinitesimal Plane-based Pose Estimation) – best for planar targets.
  std::vector<cv::Vec3d> rvecs, tvecs;
  std::vector<double>    repr_errors;

  bool ok = cv::solvePnPGeneric(
    obj_pts, image_points,
    camera_matrix_, dist_coeffs_,
    rvecs, tvecs,
    false, cv::SOLVEPNP_IPPE,
    cv::noArray(), cv::noArray(),
    repr_errors);

  if (!ok || rvecs.empty()) return false;

  // IPPE returns two solutions; pick the one with lower reprojection error
  // that also has positive z (in front of camera).
  int best = 0;
  if (rvecs.size() > 1) {
    bool z0_positive = (tvecs[0][2] > 0.0);
    bool z1_positive = (tvecs[1][2] > 0.0);
    if (!z0_positive && z1_positive) {
      best = 1;
    } else if (z0_positive && z1_positive) {
      best = (repr_errors[0] <= repr_errors[1]) ? 0 : 1;
    }
  }

  if (tvecs[best][2] <= 0.0) {
    RCLCPP_DEBUG(get_logger(), "solvePnP: no valid solution with positive z.");
    return false;
  }

  rvec = rvecs[best];
  tvec = tvecs[best];
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
bool OpiLocalizationNode::transformToMap(
  const cv::Vec3d & /*rvec*/, const cv::Vec3d & tvec,
  const std_msgs::msg::Header & header,
  geometry_msgs::msg::Pose & pose_in_map) const
{
  // Build a PoseStamped in camera frame from the translation vector.
  geometry_msgs::msg::PoseStamped pose_camera;
  pose_camera.header.frame_id = camera_frame_;
  pose_camera.header.stamp    = header.stamp;
  pose_camera.pose.position.x = tvec[0];
  pose_camera.pose.position.y = tvec[1];
  pose_camera.pose.position.z = tvec[2];

  // Convert rotation vector to quaternion via Rodrigues
  cv::Mat rot_mat;
  cv::Rodrigues(cv::Mat(3, 1, CV_64F, const_cast<double *>(tvec.val)), rot_mat);
  // (Use rvec for orientation but tvec for position – clarify variable name below)
  pose_camera.pose.orientation.w = 1.0;  // placeholder; orientation not consumed downstream

  geometry_msgs::msg::PoseStamped pose_map;
  try {
    tf_buffer_->transform(pose_camera, pose_map, map_frame_,
                          tf2::durationFromSec(0.1));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(get_logger(), "TF transform failed: %s", ex.what());
    return false;
  }

  pose_in_map = pose_map.pose;
  return true;
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiLocalizationNode::bboxCallback(
  const vision_msgs::msg::BoundingBox2DArray::ConstSharedPtr & msg)
{
  if (!camera_info_received_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
      "Waiting for CameraInfo …");
    return;
  }
  if (msg->boxes.empty()) return;

  geometry_msgs::msg::PoseArray pose_array;
  pose_array.header.stamp    = msg->header.stamp;
  pose_array.header.frame_id = map_frame_;

  for (const auto & bbox : msg->boxes) {
    cv::Vec3d rvec, tvec;
    if (!solvePlacard(bbox, rvec, tvec)) continue;

    geometry_msgs::msg::Pose pose_map;
    if (!transformToMap(rvec, tvec, msg->header, pose_map)) continue;

    pose_array.poses.push_back(pose_map);
  }

  if (!pose_array.poses.empty()) {
    pose_pub_->publish(pose_array);
  }
}

}  // namespace opi_localization

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<opi_localization::OpiLocalizationNode>());
  rclcpp::shutdown();
  return 0;
}
