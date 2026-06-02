#pragma once

#include <array>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <vision_msgs/msg/bounding_box2_d_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <opencv2/opencv.hpp>

namespace opi_localization
{

class OpiLocalizationNode : public rclcpp::Node
{
public:
  explicit OpiLocalizationNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  // ── Parameters ────────────────────────────────────────────────────────────
  double placard_width_m_;   // physical width of the OPI placard [m]
  double placard_height_m_;  // physical height of the OPI placard [m]
  std::string map_frame_;
  std::string camera_frame_;

  // ── State ─────────────────────────────────────────────────────────────────
  // 3-D model points of the placard corners in the placard local frame
  // (origin at centre, z = 0 plane, x right, y up).
  std::array<cv::Point3f, 4> model_points_;

  // Latest camera intrinsics
  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  bool    camera_info_received_{false};

  // ── TF ────────────────────────────────────────────────────────────────────
  std::shared_ptr<tf2_ros::Buffer>            tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // ── ROS I/O ───────────────────────────────────────────────────────────────
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr       camera_info_sub_;
  rclcpp::Subscription<vision_msgs::msg::BoundingBox2DArray>::SharedPtr bbox_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr          pose_pub_;

  // ── Callbacks ─────────────────────────────────────────────────────────────
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg);
  void bboxCallback(const vision_msgs::msg::BoundingBox2DArray::ConstSharedPtr & msg);

  // ── Helpers ───────────────────────────────────────────────────────────────
  // Attempt to solve PnP with IPPE for a single bounding box.
  // Returns true and fills pose_in_camera if successful.
  bool solvePlacard(const vision_msgs::msg::BoundingBox2D & bbox,
                    cv::Vec3d & rvec, cv::Vec3d & tvec) const;

  // Transform a pose from camera frame to map frame via TF.
  bool transformToMap(const cv::Vec3d & rvec, const cv::Vec3d & tvec,
                      const std_msgs::msg::Header & header,
                      geometry_msgs::msg::Pose & pose_in_map) const;
};

}  // namespace opi_localization
