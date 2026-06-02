cmake_minimum_required(VERSION 3.8)
project(pond_segmentation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackedPose.msg"
  "msg/TrackedPoseArray.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

# ONNX Runtime – adjust ONNXRUNTIME_ROOT if installed elsewhere
set(ONNXRUNTIME_ROOT "/opt/onnxruntime")
#  CACHE PATH "ONNX Runtime install prefix")
# find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib REQUIRED)
# find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
          # HINTS ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session REQUIRED)

# ── DETECTION NODE ────────────────────────────────────────────────────────────────────
add_executable(opi_detection_node src/opi_detection_node.cpp)

target_include_directories(opi_detection_node PRIVATE
  include
  # ${ONNXRUNTIME_INCLUDE}
  ${ONNXRUNTIME_ROOT}/include
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(opi_detection_node
  rclcpp
  sensor_msgs
  vision_msgs
  cv_bridge
  image_transport
  OpenCV
)

target_link_libraries(opi_detection_node
  # ${ONNXRUNTIME_LIB}
  ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
  ${OpenCV_LIBS}
)

# ── LOCALIZATION NODE ────────────────────────────────────────────────────────────────────
add_executable(opi_localization_node src/opi_localization_node.cpp)
 
target_include_directories(opi_localization_node PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
 
ament_target_dependencies(opi_localization_node
  rclcpp
  sensor_msgs
  vision_msgs
  nav_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
  OpenCV
)
 
target_link_libraries(opi_localization_node ${OpenCV_LIBS})

# ── TRACKER NODE ────────────────────────────────────────────────────────────────────
add_executable(opi_tracker_node src/opi_tracker_node.cpp)
 
target_include_directories(opi_tracker_node PRIVATE
  include
  ${EIGEN3_INCLUDE_DIRS}
)
 
ament_target_dependencies(opi_tracker_node
  rclcpp
  cv_bridge
  geometry_msgs
  nav_msgs
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
  std_msgs
)

rosidl_target_interfaces(opi_tracker_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")


install(TARGETS opi_detection_node 
                opi_localization_node
                opi_tracker_node
        DESTINATION lib/${PROJECT_NAME})
        
install(DIRECTORY launch models DESTINATION share/${PROJECT_NAME} OPTIONAL)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
