#include "opi_tracker_node.hpp"

#include <limits>
#include <cmath>
#include <sstream>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/wait_for_message.hpp>

#include <sys/stat.h>
#include <sys/types.h>
#define MKDIR(path) mkdir(path, 0755)

namespace opi_tracker
{

// ─────────────────────────────────────────────────────────────────────────────
OpiTrackerNode::OpiTrackerNode(const rclcpp::NodeOptions & options)
: Node("opi_tracker_node", options)
{
  // ── Parameters ──────────────────────────────────────────────────────────
  cluster_radius_m_ = declare_parameter<double>("cluster_radius_m", 0.75);
  min_count_        = static_cast<uint32_t>(declare_parameter<int>("min_count", 5));
  prune_timeout_s_  = declare_parameter<double>("prune_timeout_s", 60.0);
  opi_reached_distance_ = declare_parameter<double>("opi_reached_distance", 1.0);
  map_frame_        = declare_parameter<std::string>("map_frame",   "map");
  double publish_hz = declare_parameter<double>("publish_hz", 2.0);

  std::string input_topic = declare_parameter<std::string>("input_topic",  "opi/positions_raw");
  tracked_topic_          = declare_parameter<std::string>("tracked_topic", "opi/tracked");
  goals_topic_            = declare_parameter<std::string>("goals_topic", "opi/goals");
  marker_topic_           = declare_parameter<std::string>("marker_topic", "opi/markers");
  std::string odom_topic  = declare_parameter<std::string>("odom_topic", "/odom");
  image_topic_  = declare_parameter("image_topic", "/luxonis/oak/rgb/image_raw");
  img_save_path_ = declare_parameter("img_save_path", "~/opi_images/");
  MKDIR(img_save_path_.c_str());

  // ── TF ──────────────────────────────────────────────────────────────────
  tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  // ── ROS I/O ─────────────────────────────────────────────────────────────
  positions_sub_ = create_subscription<geometry_msgs::msg::PoseArray>(
    input_topic, rclcpp::SystemDefaultsQoS(),
    std::bind(&OpiTrackerNode::positionsCallback, this, std::placeholders::_1));

  odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
    odom_topic, rclcpp::SystemDefaultsQoS(),
    std::bind(&OpiTrackerNode::odomCallback, this, std::placeholders::_1));

  hypotheses_pub_ = create_publisher<pond_segmentation::msg::TrackedPoseArray>(
    tracked_topic_, rclcpp::SystemDefaultsQoS());

  unvisited_pub_ = create_publisher<geometry_msgs::msg::PoseArray>(
    goals_topic_, rclcpp::SystemDefaultsQoS());

  markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
    marker_topic_, rclcpp::SystemDefaultsQoS());

  auto period = std::chrono::duration<double>(1.0 / publish_hz);
  publish_timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    std::bind(&OpiTrackerNode::publishTimerCallback, this));

  RCLCPP_INFO(get_logger(),
    "OPI tracker node ready. cluster_radius=%.2f m, min_count=%u, prune=%.1f s, reached=%.2f m",
    cluster_radius_m_, min_count_, prune_timeout_s_, opi_reached_distance_);
}

// ─────────────────────────────────────────────────────────────────────────────
OpiHypothesis * OpiTrackerNode::findNearest(const Eigen::Vector3d & meas)
{
  double         best_dist = cluster_radius_m_;
  OpiHypothesis * best_hyp  = nullptr;

  for (auto & [id, hyp] : hypotheses_) {
    double d = (hyp.centroid - meas).norm();
    if (d < best_dist) {
      best_dist = d;
      best_hyp  = &hyp;
    }
  }
  return best_hyp;
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::pruneHypotheses(const rclcpp::Time & now)
{
  std::vector<uint32_t> to_remove;
  for (const auto & [id, hyp] : hypotheses_) {
    double age = (now - hyp.last_seen).seconds();
    if (age > prune_timeout_s_) {
      to_remove.push_back(id);
    }
  }
  for (uint32_t id : to_remove) {
    RCLCPP_INFO(get_logger(), "Pruned stale OPI hypothesis id=%u", id);
    hypotheses_.erase(id);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::positionsCallback(
  const geometry_msgs::msg::PoseArray::ConstSharedPtr & msg)
{
  rclcpp::Time stamp(msg->header.stamp);
  // pruneHypotheses(stamp);

  geometry_msgs::msg::PoseStamped robot_pose;
  const bool can_check_visited =
    have_robot_pose_ && getRobotPoseInFrame(msg->header.frame_id, stamp, robot_pose);

  for (const auto & pose : msg->poses) {
    Eigen::Vector3d meas(pose.position.x, pose.position.y, pose.position.z);

    OpiHypothesis * hyp = findNearest(meas);

    if (hyp != nullptr) {
      // Update running mean (online / incremental mean)
      ++hyp->count;
      hyp->centroid += (meas - hyp->centroid) / static_cast<double>(hyp->count);
      hyp->last_seen = stamp;
    } else {
      // Create new hypothesis
      OpiHypothesis new_hyp;
      new_hyp.id        = next_id_++;
      new_hyp.centroid  = meas;
      new_hyp.count     = 1;
      new_hyp.last_seen = stamp;
      new_hyp.frame     = msg->header.frame_id;
      hypotheses_[new_hyp.id] = new_hyp;
      RCLCPP_INFO(get_logger(), "New OPI hypothesis id=%u at [%.2f, %.2f, %.2f]",
                  new_hyp.id, meas.x(), meas.y(), meas.z());
      takePhoto(new_hyp.id, "new");
    }

    if (hyp == nullptr) {
      hyp = &hypotheses_.at(next_id_ - 1);
    }

    if (can_check_visited && hyp != nullptr && !hyp->visited) {
      const double dx = hyp->centroid.x() - robot_pose.pose.position.x;
      const double dy = hyp->centroid.y() - robot_pose.pose.position.y;
      const double distance_xy = std::hypot(dx, dy);
      if (distance_xy <= opi_reached_distance_) {
        hyp->visited = true;
        RCLCPP_INFO(
          get_logger(),
          "Marked OPI hypothesis id=%u visited at XY distance %.2f m",
          hyp->id, distance_xy);
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::odomCallback(const nav_msgs::msg::Odometry::ConstSharedPtr & msg)
{
  latest_robot_pose_.header = msg->header;
  latest_robot_pose_.pose = msg->pose.pose;
  have_robot_pose_ = true;

  geometry_msgs::msg::PoseStamped robot_pose_transformed;

  for (auto & [id,hyp] : hypotheses_) {

    bool can_check_visited = true;

    if (robot_pose_transformed == geometry_msgs::msg::PoseStamped() || robot_pose_transformed.header.frame_id != hyp.frame) {

      rclcpp::Time stamp(msg->header.stamp);
      can_check_visited =
        have_robot_pose_ && getRobotPoseInFrame(hyp.frame, stamp, robot_pose_transformed);

    }
      
    if (can_check_visited && !hyp.visited) {
      const double dx = hyp.centroid.x() - robot_pose_transformed.pose.position.x;
      const double dy = hyp.centroid.y() - robot_pose_transformed.pose.position.y;
      const double distance_xy = std::hypot(dx, dy);

      if (distance_xy <= opi_reached_distance_) {
        hyp.visited = true;
        RCLCPP_INFO(
          get_logger(),
          "Marked OPI hypothesis id=%u visited at XY distance %.2f m",
          hyp.id, distance_xy);
        takePhoto(hyp.id, "closeup");
      } else {
        RCLCPP_INFO(
          get_logger(),
          "OPI too far id=%u at XY distance %.2f m",
          hyp.id, distance_xy);
      }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::publishTimerCallback()
{
  rclcpp::Time now = get_clock()->now();

  pond_segmentation::msg::TrackedPoseArray out;
  geometry_msgs::msg::PoseArray unvisited_out;
  out.header.stamp    = now;
  out.header.frame_id = map_frame_;
  unvisited_out.header = out.header;

  for (const auto & [id, hyp] : hypotheses_) {
    if (hyp.count < min_count_) continue;

    pond_segmentation::msg::TrackedPose tracked_pose;
    tracked_pose.id = id;
    tracked_pose.pose.position.x    = hyp.centroid.x();
    tracked_pose.pose.position.y    = hyp.centroid.y();
    tracked_pose.pose.position.z    = hyp.centroid.z();
    tracked_pose.pose.orientation.w = 1.0;
    tracked_pose.visited = hyp.visited;
    out.poses.push_back(tracked_pose);

    if (!hyp.visited) {
      unvisited_out.poses.push_back(tracked_pose.pose);
    }
  }

  hypotheses_pub_->publish(out);
  unvisited_pub_->publish(unvisited_out);
  publishMarkers(now);
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::takePhoto(int id, std::string specifier)
{
  RCLCPP_INFO(get_logger(), "Trying to take a photo of OPI %d.", id);
  sensor_msgs::msg::Image img_msg; 
  auto timeout = std::chrono::seconds(1);
  bool received_msg = rclcpp::wait_for_message(img_msg, shared_from_this(), image_topic_, timeout);
  if (received_msg) {
    RCLCPP_INFO(get_logger(), "Sucessfully taken photo of OPI %d!", id);
  } else {
    RCLCPP_WARN(get_logger(), "Failed to take photo of the OPI! wait_for_msg did not receive a msg in %ld s",timeout.count());
    return;
  }

  // save the photo to a file
  cv_bridge::CvImageConstPtr cv_img = cv_bridge::toCvShare(std::make_shared<sensor_msgs::msg::Image>(img_msg), img_msg.encoding);

  if(cv_img->image.empty()) {
    RCLCPP_ERROR(get_logger(), "The image is empty.");
    return;
  }

  std::string separator =
    (!img_save_path_.empty() &&
     img_save_path_.back() != '/' &&
     img_save_path_.back() != '\\')
        ? "/"
        : "";

  std::string save_path =
      img_save_path_ + separator + "OPI_" + std::to_string(id) + "_" + specifier + ".png";
  cv::imwrite(save_path, cv_img->image);
}

// ─────────────────────────────────────────────────────────────────────────────
bool OpiTrackerNode::getRobotPoseInFrame(
  const std::string & target_frame,
  const rclcpp::Time & /*stamp*/,
  geometry_msgs::msg::PoseStamped & robot_pose) const
{
  if (!have_robot_pose_) {
    return false;
  }

  if (target_frame.empty()) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000, "Cannot compare OPI and robot pose: empty target frame.");
    return false;
  }

  if (latest_robot_pose_.header.frame_id == target_frame) {
    robot_pose = latest_robot_pose_;
    return true;
  }

  try {
    tf_buffer_->transform(
      latest_robot_pose_, robot_pose, target_frame, tf2::durationFromSec(0.1));
    return true;
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Failed to transform robot pose from '%s' to '%s': %s",
      latest_robot_pose_.header.frame_id.c_str(), target_frame.c_str(), ex.what());
    return false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
void OpiTrackerNode::publishMarkers(const rclcpp::Time & stamp)
{
  visualization_msgs::msg::MarkerArray marker_array;

  // Delete all old markers first
  visualization_msgs::msg::Marker delete_all;
  delete_all.action = visualization_msgs::msg::Marker::DELETEALL;
  delete_all.header.frame_id = map_frame_;
  delete_all.header.stamp    = stamp;
  marker_array.markers.push_back(delete_all);

  for (const auto & [id, hyp] : hypotheses_) {
    if (hyp.count < min_count_) continue;

    // Sphere at centroid
    visualization_msgs::msg::Marker sphere;
    sphere.header.frame_id = map_frame_;
    sphere.header.stamp    = stamp;
    sphere.ns              = "opi_hypotheses";
    sphere.id              = static_cast<int>(id);
    sphere.type            = visualization_msgs::msg::Marker::SPHERE;
    sphere.action          = visualization_msgs::msg::Marker::ADD;
    sphere.pose.position.x = hyp.centroid.x();
    sphere.pose.position.y = hyp.centroid.y();
    sphere.pose.position.z = hyp.centroid.z();
    sphere.pose.orientation.w = 1.0;
    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.5;
    sphere.color.r = 0.74f; sphere.color.g = 0.25f; sphere.color.b = 0.74f;
    sphere.color.a = 0.9f;
    sphere.lifetime = rclcpp::Duration(0, 0);  // persist until deleted

    // Text label above sphere
    visualization_msgs::msg::Marker text;
    text.header        = sphere.header;
    text.ns            = "opi_labels";
    text.id            = static_cast<int>(id);
    text.type          = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text.action        = visualization_msgs::msg::Marker::ADD;
    text.pose          = sphere.pose;
    text.pose.position.z += 0.6;
    text.scale.z       = 0.40;
    text.color.r = text.color.g = text.color.b = 0.16f;
    text.color.a = 1.0f;
    std::ostringstream ss;
    ss << "OPI #" << id << "\n(n=" << hyp.count << ")";
    text.text = ss.str();
    text.lifetime = rclcpp::Duration(0, 0);

    marker_array.markers.push_back(sphere);
    marker_array.markers.push_back(text);
  }

  markers_pub_->publish(marker_array);
}

}  // namespace opi_tracker

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<opi_tracker::OpiTrackerNode>());
  rclcpp::shutdown();
  return 0;
}
