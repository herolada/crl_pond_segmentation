# Pond Segmentation Training

This folder contains a small Ultralytics training pipeline for fine-tuning a pond / water segmentation model and exporting the best checkpoint to ONNX for later C++/ROS 2 inference.

## Possible improvements to try
Anything that will make the model faster, more accurate, or overcome the difference between the distribution of the training data and what the real robot sees.

Some ideas from the top of my head (feel free to try anything else though):
- Data
    - find more public segmentation datasets
    - make a custom dataset from public images
    - make a synthetic dataset from robot bag files
    - better data augmentation
- Model
    - different model (different YOLO, or another model e.g. FAST R-CNN)
    - different size of model (currently using 's' models)
    - better hyper-parameters (lr, bs, loss weights, weight decay, schedule, etc.)


## Install

I used `uv` for environment managment.

1. install uv (https://docs.astral.sh/uv/getting-started/installation/)
2. (inside scripts/) `uv sync`

## Train

From the workspace root:

```bash
uv run python pond_segmentation/scripts/train.py --data pond_segmentation/data --tensorboard
```

By default the script merges every `data/*/data.yaml` dataset it finds under `pond_segmentation/data`, rewrites the labels into a unified training view, and uses a single foreground class called `water`.

Or change model (should get automatically downloaded, if provided by the latest `ultralytics`):

```bash
uv run python pond_segmentation/scripts/train.py --model yolo11m-seg.pt --class-mode preserve --tensorboard
```

## Monitor Training

Launch TensorBoard against the run directory:

```bash
uv run tensorboard --logdir pond_segmentation/runs/segmentation_training
```

The training run also saves validation comparison images under:

```text
pond_segmentation/runs/segmentation_training/<run_name>/val_previews/
```

Each preview shows ground-truth masks side by side with model predictions so you can track qualitative progress during fine-tuning.

## Evaluate on Test Data

Run a trained checkpoint against a test image folder and a matching label folder:

```bash
uv run python pond_segmentation/scripts/evaluate.py \
  --model pond_segmentation/runs/segmentation_training/yolo11n-seg/weights/best.pt \
  --images /path/to/test/images \
  --labels /path/to/test/labels
```

The script:

- runs Ultralytics validation on a temporary test split so you get the built-in segmentation metrics
- computes extra pixel-level and instance-level metrics from the predictions vs. ground truth
- randomly selects preview examples and packs them into a contact-sheet montage in `pond_segmentation/runs/segmentation_evaluation/<model_stem>_previews/`
- also saves the individual preview tiles in `pond_segmentation/runs/segmentation_evaluation/<model_stem>_previews/examples/`
- writes the combined metrics to `pond_segmentation/runs/segmentation_evaluation/<model_stem>_metrics.json`
