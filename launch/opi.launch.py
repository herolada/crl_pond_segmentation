"""Launch the commander node with parameters."""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    declared_args = [
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="simulation/bag or not",
        ),
        DeclareLaunchArgument(
            "rotate_image_180",
            default_value="true",
            description="Rotate the subscribed image by 180 degrees before detection",
        ),
        DeclareLaunchArgument(
            "detection_hz",
            default_value="0.0",
            description="Maximum object detection frequency per subscribed camera topic; 0.0 processes every frame",
        ),
        DeclareLaunchArgument(
            "enable_openvino_ep",
            default_value="true",
            description="Try to enable the OpenVINO execution provider if the linked ONNX Runtime build supports it",
        ),
    ]

    return LaunchDescription(
        declared_args +
        [Node(
            package="pond_segmentation",
            executable="opi_detection_node",
            name="opi_detection_node",
            output="screen",
            # prefix=['valgrind --tool=callgrind --dump-instr=yes -v --instr-atstart=no'],
            parameters=[{
                "model_path": get_package_share_directory("pond_segmentation")+"/models/yolov11s.onnx",
                "camera_topics": ["camera/image_raw"],
                "conf_threshold": 0.40,
                "nms_threshold": 0.45,
                "input_width": 640,
                "input_height": 640,
                "rotate_image_180": LaunchConfiguration("rotate_image_180"),
                "detection_hz": LaunchConfiguration("detection_hz"),
                "enable_openvino_ep": LaunchConfiguration("enable_openvino_ep"),
                "camera_info_topic": "camera/camera_info",
                "output_topic": "opi/detections",
                "use_sim_time": LaunchConfiguration("use_sim_time")}],
            remappings=[
                ("camera/image_raw", "/basler_front/image_raw"),
                ("camera/camera_info", "/basler_front/camera_info"),
            ],
        ),
        Node(
            package="pond_segmentation",
            executable="opi_localization_node",
            name="opi_localization_node",
            output="screen",
            parameters=[{
                "placard_width_m": 0.40,
                "placard_height_m": 0.30,
                "map_frame": "map",
                "camera_frame": "pylon_camera",
                "camera_info_topic": "camera/camera_info",
                "bbox_topic": "opi/detections",
                "output_topic": "opi/positions_raw",
                "use_sim_time": LaunchConfiguration("use_sim_time")}],
            remappings=[
                ("camera/camera_info", "/basler_front/camera_info"),
                ("opi/detections", "/opi/detections"),
                ("opi/positions_raw", "/opi/positions_raw"),
            ],
        ),
        Node(
            package="pond_segmentation",
            executable="opi_tracker_node",
            name="opi_tracker_node",
            output="screen",
            parameters=[{
                "opi_reached_distance": 3.0,
                "cluster_radius_m": 5.0,
                "min_count": 5,
                "prune_timeout_s": 60.0, # not used
                "map_frame": "map",
                "publish_hz": 2.0,
                "input_topic": "opi/positions_raw",
                "tracked_topic": "opi/tracked",
                "goals_topic": "opi/goals",
                "marker_topic": "opi/markers",
                "odom_topic": "/liorf/mapping/baselink_odometry",
                "image_topic": "/basler_front/image_color",
                "img_save_path": "/home/robot/opi_images/",
                "use_sim_time": LaunchConfiguration("use_sim_time")}],
            remappings=[
                ("opi/positions_raw", "/opi/positions_raw"),
                ("opi/goals", "/opi/goals"),
                ("opi/tracked", "/opi/tracked"),
                ("opi/markers", "/opi/markers"),
            ],
        )]
    )
