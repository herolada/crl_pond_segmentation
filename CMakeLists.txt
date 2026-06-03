cmake_minimum_required(VERSION 3.8)
project(pond_segmentation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# ONNX Runtime – install to /opt/onnxruntime as described in README
set(ONNXRUNTIME_ROOT "/opt/onnxruntime/onnxruntime-linux-x64-1.26.0")

add_executable(pond_segmentation_node src/pond_segmentation_node.cpp)

target_include_directories(pond_segmentation_node PRIVATE
  include
  ${ONNXRUNTIME_ROOT}/include
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(pond_segmentation_node
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
)

target_link_libraries(pond_segmentation_node
  ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
  ${OpenCV_LIBS}
)

install(TARGETS pond_segmentation_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch models
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

ament_package()
