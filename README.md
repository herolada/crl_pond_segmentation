# pond_segmentation

ROS 2 node that runs a YOLOv11n-seg model (single "water" class) on camera images and publishes a binary segmentation mask.

## Dependencies

### ONNX Runtime

```bash
wget https://github.com/microsoft/onnxruntime/releases/download/v1.26.0/onnxruntime-linux-x64-1.26.0.tgz
tar -xzf onnxruntime-linux-x64-1.26.0.tgz
sudo mv onnxruntime-linux-x64-1.26.0 /opt/onnxruntime
```

### ROS 2 packages

```
rclcpp  sensor_msgs  cv_bridge  libopencv-dev
```

## Build

```bash
colcon build --packages-select pond_segmentation
```

## Usage

```bash
ros2 run pond_segmentation pond_segmentation_node \
  --ros-args \
  -p model_path:=<path/to/best_int8.onnx> \
  -p camera_topics:="['luxonis/oak/rgb/image_raw']" \
  -p output_topic:=pond/segmentation
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `models/best_int8.onnx` | Path to the ONNX model |
| `camera_topics` | `['luxonis/oak/rgb/image_raw']` | List of input image topics |
| `output_topic` | `pond/segmentation` | Base topic for the mask output |
| `conf_threshold` | `0.25` | YOLO detection confidence threshold |
| `nms_threshold` | `0.45` | NMS IoU threshold |
| `mask_threshold` | `0.50` | Sigmoid threshold for binarising the mask |
| `input_width` | `320` | Model input width (overridden by model shape if static) |
| `input_height` | `320` | Model input height (overridden by model shape if static) |
| `rotate_image_180` | `false` | Rotate input 180° before inference |
| `segmentation_hz` | `0.0` | Max inference rate in Hz (0 = process every frame) |
| `foreground_class_id` | `1` | Class index used for semantic models (ignored in YOLO mode) |

### Published topics

| Topic | Type | Description |
|---|---|---|
| `<output_topic>/image_raw` | `sensor_msgs/Image` (mono8) | Binary segmentation mask (255 = water, 0 = background) |

## Model inference modes

The node auto-detects the model type from output tensor shapes at startup:

- **YOLO segmentation** (YOLOv8/v11 `-seg`): two outputs, rank-3 predictions + rank-4 prototypes
- **Semantic segmentation**: single rank-4 output (NCHW or NHWC), thresholded per class
