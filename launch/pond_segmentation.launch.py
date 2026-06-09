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
            description="Use /clock from a bag or simulator instead of wall time",
        ),
        DeclareLaunchArgument(
            "rotate_image_180",
            default_value="false",
            description="Rotate the input image 180° before inference (for upside-down cameras)",
        ),
        DeclareLaunchArgument(
            "segmentation_hz",
            default_value="0.0",
            description="Max segmentation rate in Hz per camera topic; 0.0 means every frame",
        ),
    ]

    return LaunchDescription(
        declared_args +
        [Node(
            package="pond_segmentation",
            executable="pond_segmentation_node",
            name="pond_segmentation_node",
            output="screen",
            parameters=[{
                # Path to the ONNX model file installed with the package
                "model_path": get_package_share_directory("pond_segmentation") + "/models/best_dgx.onnx",

                # List of image topics to run inference on (one mask published per topic)
                "camera_topics": ["camera/image_raw"],

                # Minimum confidence score for a YOLO detection to be kept before NMS
                "conf_threshold": 0.4,

                # IoU overlap threshold for Non-Maximum Suppression
                "nms_threshold": 0.45,

                # Sigmoid output threshold for binarising the mask (0 = background, 255 = water)
                "mask_threshold": 0.4,

                # Model input resolution — must match what the ONNX was exported with
                "input_width": 320,
                "input_height": 320,

                # Rotate the input image 180° before inference (set via launch arg)
                "rotate_image_180": LaunchConfiguration("rotate_image_180"),

                # Max inference rate per camera topic; set via launch arg (0.0 = every frame)
                "segmentation_hz": LaunchConfiguration("segmentation_hz"),

                # Base output topic; mask is published on <output_topic>/image_raw (mono8)
                "output_topic": "pond/segmentation",

                "use_sim_time": LaunchConfiguration("use_sim_time"),
            }],
            remappings=[
                # Remap the generic camera topic to the actual hardware topic
                ("camera/image_raw", "/basler_front/image_color"),
            ],
        )]
    )
