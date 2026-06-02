#!/usr/bin/env python3
"""Train and export a YOLO segmentation model for pond-like water regions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = PACKAGE_DIR / "data"
DEFAULT_RUNS_DIR = PACKAGE_DIR / "runs" / "segmentation_training"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path


@dataclass(frozen=True)
class DatasetSource:
    root: Path
    yaml_path: Path
    names: list[str]


@dataclass(frozen=True)
class PreparedDataset:
    yaml_path: Path
    root_dir: Path
    val_samples: list[Sample]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a pretrained Ultralytics YOLO segmentation model on a merged pond / water "
            "dataset and export the best checkpoint to ONNX."
        )
    )
    parser.add_argument(
        "--model",
        default="yolo11n-seg.pt",
        help="Pretrained Ultralytics segmentation checkpoint or model alias, e.g. yolo11s-seg.pt.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Path to a single Ultralytics dataset YAML or to the parent data directory. "
            "If a directory is provided, every child folder with a data.yaml file is merged."
        ),
    )
    parser.add_argument("--imgsz", type=int, default=320, help="Training and export image size.")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="0", help='Training device, e.g. "0", "0,1", or "cpu".')
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Parent directory for training outputs.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional run name. Defaults to the model stem, e.g. yolo11s-seg.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=4,
        help="How many validation images to visualize.",
    )
    parser.add_argument(
        "--preview-interval",
        type=int,
        default=5,
        help="Save validation prediction-vs-ground-truth previews every N epochs.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for preview images.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold for preview images.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version for export.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Export the ONNX model with dynamic axes.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the exported ONNX graph if the environment supports it.",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Also log custom metrics and preview images to TensorBoard.",
    )
    parser.add_argument(
        "--class-mode",
        choices=("single", "preserve"),
        default="single",
        help=(
            "How to merge labels from different source datasets. "
            "'single' maps every foreground mask to one class; 'preserve' keeps source class names."
        ),
    )
    parser.add_argument(
        "--single-class-name",
        default="water",
        help="Class name to use when --class-mode single is selected.",
    )
    return parser.parse_args()


def read_dataset_config(dataset_yaml: Path) -> dict:
    with dataset_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Dataset config at {dataset_yaml} is not a YAML mapping.")
    return data


def normalize_split_name(split: str) -> str:
    return "val" if split == "valid" else split


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "sample"


def unique_sample_stem(source: DatasetSource, split: str, image_path: Path) -> str:
    relative = image_path.relative_to(source.root)
    relative_without_suffix = relative.with_suffix("")
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:10]
    parts = [source.root.name, split, *relative_without_suffix.parts, digest]
    return slugify("__".join(parts))


def resolve_source_root_from_yaml(dataset_yaml: Path, data_config: dict) -> Path:
    dataset_root = Path(data_config.get("path", dataset_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (dataset_yaml.parent / dataset_root).resolve()
    return dataset_root


def resolve_split_dir(source: DatasetSource, data_config: dict, split: str) -> Path | None:
    normalized_split = normalize_split_name(split)

    # Prefer the split folders that actually exist next to the dataset root.
    # This matches the Roboflow-style layout in this repo:
    #   <dataset>/train/images
    #   <dataset>/valid/images
    #   <dataset>/test/images
    candidate_dirs: list[Path] = []
    if normalized_split == "val":
        candidate_dirs.extend([source.root / "val", source.root / "valid"])
    else:
        candidate_dirs.append(source.root / normalized_split)

    # Fall back to the paths declared in the YAML if they point somewhere else.
    split_key = split if split in data_config else ("val" if split == "valid" and "val" in data_config else None)
    if split_key is not None:
        split_value = data_config.get(split_key)
        if split_value is not None:
            split_path = Path(split_value)
            if not split_path.is_absolute():
                dataset_root = resolve_source_root_from_yaml(source.yaml_path, data_config)
                split_path = (dataset_root / split_path).resolve()

            if split_path.name == "images":
                candidate_dirs.append(split_path.parent)
            elif split_path.is_dir():
                candidate_dirs.append(split_path)
            elif split_path.parent.is_dir():
                candidate_dirs.append(split_path.parent)

    for candidate_dir in candidate_dirs:
        if (candidate_dir / "images").exists():
            return candidate_dir

    return None


def discover_dataset_sources(data_path: Path) -> list[DatasetSource]:
    if data_path.is_file():
        if data_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"Unsupported dataset file type: {data_path}")
        data_config = read_dataset_config(data_path)
        names = data_config.get("names") or []
        if not isinstance(names, list):
            raise ValueError(f"Expected 'names' to be a list in {data_path}")
        return [DatasetSource(root=resolve_source_root_from_yaml(data_path, data_config), yaml_path=data_path, names=[str(name) for name in names])]

    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    child_sources: list[DatasetSource] = []
    for candidate in sorted(data_path.iterdir()):
        if not candidate.is_dir() or candidate.name.startswith("."):
            continue
        yaml_path = candidate / "data.yaml"
        if not yaml_path.exists():
            continue
        data_config = read_dataset_config(yaml_path)
        names = data_config.get("names") or []
        if not isinstance(names, list):
            raise ValueError(f"Expected 'names' to be a list in {yaml_path}")
        child_sources.append(
            DatasetSource(
                root=resolve_source_root_from_yaml(yaml_path, data_config),
                yaml_path=yaml_path,
                names=[str(name) for name in names],
            )
        )

    if child_sources:
        return child_sources

    yaml_path = data_path / "data.yaml"
    if yaml_path.exists():
        data_config = read_dataset_config(yaml_path)
        names = data_config.get("names") or []
        if not isinstance(names, list):
            raise ValueError(f"Expected 'names' to be a list in {yaml_path}")
        return [DatasetSource(root=resolve_source_root_from_yaml(yaml_path, data_config), yaml_path=yaml_path, names=[str(name) for name in names])]

    raise FileNotFoundError(
        f"No dataset YAML files found in {data_path}. Expected either a YAML file or child folders with data.yaml."
    )


def iter_split_images(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        return []
    return sorted(path for path in split_dir.rglob("*") if path.is_file() and is_image_file(path))


def source_label_path(split_dir: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(split_dir / "images")
    return split_dir / "labels" / relative.with_suffix(".txt")


def parse_label_line(line: str) -> tuple[int, list[float]] | None:
    parts = line.strip().split()
    if len(parts) < 7 or len(parts) % 2 == 0:
        return None
    try:
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return None
    return class_id, coords


def remap_label_file(
    source: DatasetSource,
    source_label_path: Path,
    destination_label_path: Path,
    class_mode: str,
    global_class_names: list[str],
    global_class_lookup: dict[str, int],
) -> None:
    destination_label_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_label_path.exists():
        destination_label_path.write_text("", encoding="utf-8")
        return

    rewritten_lines: list[str] = []
    with source_label_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parsed = parse_label_line(raw_line)
            if parsed is None:
                continue
            class_id, coords = parsed
            if class_mode == "single":
                mapped_class_id = 0
            else:
                if 0 <= class_id < len(source.names):
                    class_name = source.names[class_id]
                else:
                    class_name = f"class_{class_id}"
                if class_name not in global_class_lookup:
                    global_class_lookup[class_name] = len(global_class_names)
                    global_class_names.append(class_name)
                mapped_class_id = global_class_lookup[class_name]
            coord_text = " ".join(f"{value:g}" for value in coords)
            rewritten_lines.append(f"{mapped_class_id} {coord_text}")

    destination_label_path.write_text("\n".join(rewritten_lines) + ("\n" if rewritten_lines else ""), encoding="utf-8")


def link_or_copy_file(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        destination_path.unlink()
    try:
        destination_path.symlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, destination_path)


def build_merged_dataset(
    data_path: Path,
    project_dir: Path,
    class_mode: str,
    single_class_name: str,
) -> PreparedDataset:
    sources = discover_dataset_sources(data_path)
    cache_root = Path(
        tempfile.mkdtemp(
            prefix="pond_segmentation_dataset_",
            dir=str(project_dir / "_dataset_cache"),
        )
    )

    merged_root = cache_root / "merged"
    merged_root.mkdir(parents=True, exist_ok=True)

    global_class_names: list[str] = [single_class_name] if class_mode == "single" else []
    global_class_lookup: dict[str, int] = {single_class_name: 0} if class_mode == "single" else {}
    split_counts: dict[str, int] = {split: 0 for split in ("train", "val", "test")}
    val_samples: list[Sample] = []

    for split in ("train", "val", "test"):
        (merged_root / split / "images").mkdir(parents=True, exist_ok=True)
        (merged_root / split / "labels").mkdir(parents=True, exist_ok=True)

    for source in sources:
        data_config = read_dataset_config(source.yaml_path)
        for source_split in ("train", "val", "valid", "test"):
            split_dir = resolve_split_dir(source, data_config, source_split)
            if split_dir is None:
                continue
            normalized_split = normalize_split_name(source_split)
            image_dir = split_dir / "images"
            if not image_dir.exists():
                continue

            for image_path in iter_split_images(image_dir):
                unique_stem = unique_sample_stem(source, normalized_split, image_path)
                destination_image = merged_root / normalized_split / "images" / f"{unique_stem}{image_path.suffix.lower()}"
                destination_label = merged_root / normalized_split / "labels" / f"{unique_stem}.txt"
                source_label = source_label_path(split_dir, image_path)

                link_or_copy_file(image_path, destination_image)
                remap_label_file(
                    source=source,
                    source_label_path=source_label,
                    destination_label_path=destination_label,
                    class_mode=class_mode,
                    global_class_names=global_class_names,
                    global_class_lookup=global_class_lookup,
                )

                split_counts[normalized_split] += 1
                if normalized_split == "val" and len(val_samples) < 64:
                    val_samples.append(Sample(image_path=destination_image, label_path=destination_label))

    if split_counts["train"] == 0:
        raise FileNotFoundError(
            f"No training images were found under {data_path}. Expected at least one child dataset with train/images."
        )
    if split_counts["val"] == 0:
        raise FileNotFoundError(
            f"No validation images were found under {data_path}. Expected at least one child dataset with val/valid/images."
        )

    dataset_yaml_path = merged_root / "data.yaml"
    dataset_yaml = {
        "path": str(merged_root),
        "train": "train/images",
        "val": "val/images",
        "nc": len(global_class_names) if class_mode == "preserve" else 1,
        "names": global_class_names if class_mode == "preserve" else [single_class_name],
    }
    if split_counts["test"] > 0:
        dataset_yaml["test"] = "test/images"

    with dataset_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_yaml, handle, sort_keys=False)

    return PreparedDataset(yaml_path=dataset_yaml_path, root_dir=merged_root, val_samples=val_samples)


def load_segmentation_labels(label_path: Path, image_shape: tuple[int, int, int]) -> list[tuple[int, np.ndarray]]:
    height, width = image_shape[:2]
    labels: list[tuple[int, np.ndarray]] = []
    if not label_path.exists():
        return labels

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_label_line(line)
            if parsed is None:
                continue
            class_id, coords = parsed
            points = np.array(coords, dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= width
            points[:, 1] *= height
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            labels.append((class_id, points.astype(np.int32)))
    return labels


def draw_polygons(
    image: np.ndarray,
    polygons: Iterable[tuple[int, np.ndarray]],
    color: tuple[int, int, int],
    prefix: str,
) -> np.ndarray:
    annotated = image.copy()
    overlay = image.copy()

    for class_id, points in polygons:
        if points.size == 0:
            continue
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(annotated, [points], isClosed=True, color=color, thickness=2)
        x, y = points[0].tolist()
        cv2.putText(
            annotated,
            f"{prefix}:{class_id}",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0.0, annotated)
    return annotated


def draw_ground_truth(image: np.ndarray, polygons: Iterable[tuple[int, np.ndarray]]) -> np.ndarray:
    return draw_polygons(image, polygons, (0, 255, 0), "gt")


def draw_predictions(image: np.ndarray, result) -> np.ndarray:
    annotated = image.copy()

    if getattr(result, "masks", None) is not None and result.masks.xy is not None:
        overlay = image.copy()
        boxes = result.boxes
        for index, polygon in enumerate(result.masks.xy):
            if polygon is None or len(polygon) == 0:
                continue
            points = np.asarray(polygon, dtype=np.int32)
            if points.ndim != 2 or points.shape[0] < 3:
                continue
            cv2.fillPoly(overlay, [points], (0, 0, 255))
            cv2.polylines(annotated, [points], isClosed=True, color=(0, 0, 255), thickness=2)
            class_id = int(boxes.cls[index].item()) if boxes is not None and index < len(boxes) else 0
            conf = float(boxes.conf[index].item()) if boxes is not None and index < len(boxes) else 0.0
            x, y = points[0].tolist()
            cv2.putText(
                annotated,
                f"pred:{class_id} {conf:.2f}",
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0.0, annotated)
        return annotated

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].detach().cpu().numpy().astype(int).tolist()
        class_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"pred:{class_id} {conf:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def stack_preview(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    gt_panel = cv2.copyMakeBorder(ground_truth, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    pred_panel = cv2.copyMakeBorder(prediction, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(gt_panel, "Ground Truth", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(pred_panel, "Prediction", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return np.hstack((gt_panel, pred_panel))


def collect_samples(image_dir: Path, label_dir: Path, limit: int) -> list[Sample]:
    image_paths = sorted(path for path in image_dir.rglob("*") if path.is_file() and is_image_file(path))
    samples: list[Sample] = []
    fallback_samples: list[Sample] = []

    for image_path in image_paths:
        relative = image_path.relative_to(image_dir)
        label_path = label_dir / relative.with_suffix(".txt")
        sample = Sample(image_path=image_path, label_path=label_path)
        fallback_samples.append(sample)
        if label_path.exists() and label_path.read_text(encoding="utf-8").strip():
            samples.append(sample)
        if len(samples) >= limit:
            break

    if not samples:
        samples = fallback_samples[:limit]

    if not samples:
        raise FileNotFoundError(f"No validation image/label pairs found in {image_dir} and {label_dir}.")
    return samples


class TrainingMonitor:
    """Ultralytics callback bundle for TensorBoard logging and preview exports."""

    def __init__(self, args: argparse.Namespace, val_samples: list[Sample]) -> None:
        self.args = args
        self.val_samples = val_samples
        self.writer: SummaryWriter | None = None
        self.run_dir: Path | None = None
        self.preview_dir: Path | None = None

    def on_train_start(self, trainer) -> None:
        self.run_dir = Path(trainer.save_dir)
        self.preview_dir = self.run_dir / "val_previews"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        if self.args.tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            self.writer.add_text("train/model", str(self.args.model))
            self.writer.add_text("train/data", str(self.args.data.resolve()))
            self.writer.add_text("train/hparams", json.dumps(training_overrides(self.args), indent=2))

    def on_fit_epoch_end(self, trainer) -> None:
        if not self.writer:
            return
        metrics = getattr(trainer, "metrics", {}) or {}
        epoch = int(trainer.epoch) + 1
        for key, value in metrics.items():
            if isinstance(value, (float, int)):
                self.writer.add_scalar(key, value, epoch)

    def on_model_save(self, trainer) -> None:
        epoch = int(trainer.epoch) + 1
        if epoch % self.args.preview_interval != 0 and epoch != int(self.args.epochs):
            return
        self._render_previews(trainer, epoch)

    def on_train_end(self, trainer) -> None:
        self._render_previews(trainer, int(trainer.epoch) + 1, force_best=True)
        if self.writer:
            self.writer.flush()
            self.writer.close()

    def _render_previews(self, trainer, epoch: int, force_best: bool = False) -> None:
        if self.preview_dir is None:
            return

        checkpoint_path = Path(trainer.best if force_best and Path(trainer.best).exists() else trainer.last)
        predictor = YOLO(str(checkpoint_path))
        epoch_dir = self.preview_dir / f"epoch_{epoch:03d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        for sample in self.val_samples:
            image_bgr = cv2.imread(str(sample.image_path))
            if image_bgr is None:
                continue

            gt_polygons = load_segmentation_labels(sample.label_path, image_bgr.shape)
            prediction = predictor.predict(
                source=str(sample.image_path),
                imgsz=self.args.imgsz,
                conf=self.args.conf,
                iou=self.args.iou,
                verbose=False,
                device=self.args.device,
            )[0]

            preview = stack_preview(
                draw_ground_truth(image_bgr, gt_polygons),
                draw_predictions(image_bgr, prediction),
            )
            preview_path = epoch_dir / f"{sample.image_path.stem}_preview.jpg"
            cv2.imwrite(str(preview_path), preview)

            if self.writer:
                rgb_preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                chw_preview = np.transpose(rgb_preview, (2, 0, 1))
                self.writer.add_image(
                    tag=f"val_previews/{sample.image_path.stem}",
                    img_tensor=chw_preview,
                    global_step=epoch,
                )


def training_overrides(args: argparse.Namespace) -> dict:
    run_name = args.name or Path(args.model).stem
    return {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "project": str(args.project.resolve()),
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "cos_lr": True,
        "weight_decay": 0.001,
        "hsv_h": 0.7,
        "hsv_s": 0.7,
        "hsv_v": 0.9,
        "degrees": 45.0,
        "translate": 0.1,
        "scale": 1.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic": 0.5,
        "mixup": 0.1,
        "copy_paste": 0.0,
        "copy_paste_mode": "flip",
        "auto_augment": "randaugment",
        "erasing": 0.4,
        "crop_fraction": 1.0,
        "plots": True,
        "save": True,
        "save_period": args.preview_interval,
        "val": True,
        "verbose": True,
    }


def main() -> None:
    args = parse_args()
    args.data = args.data.resolve()
    args.project = args.project.resolve()
    args.project.mkdir(parents=True, exist_ok=True)
    (args.project / "_dataset_cache").mkdir(parents=True, exist_ok=True)

    prepared_dataset = build_merged_dataset(
        data_path=args.data,
        project_dir=args.project,
        class_mode=args.class_mode,
        single_class_name=args.single_class_name,
    )
    args.data = prepared_dataset.yaml_path
    val_samples = prepared_dataset.val_samples[: args.preview_count]
    if not val_samples:
        val_image_dir = prepared_dataset.root_dir / "val" / "images"
        val_label_dir = prepared_dataset.root_dir / "val" / "labels"
        val_samples = collect_samples(val_image_dir, val_label_dir, args.preview_count)

    model = YOLO(args.model)
    monitor = TrainingMonitor(args, val_samples)
    model.add_callback("on_train_start", monitor.on_train_start)
    model.add_callback("on_fit_epoch_end", monitor.on_fit_epoch_end)
    model.add_callback("on_model_save", monitor.on_model_save)
    model.add_callback("on_train_end", monitor.on_train_end)

    results = model.train(**training_overrides(args))

    best_checkpoint = Path(results.save_dir) / "weights" / "best.pt"
    export_model = YOLO(str(best_checkpoint))
    export_path = export_model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
    )

    print(f"Training complete. Best checkpoint: {best_checkpoint}")
    print(f"ONNX export complete: {export_path}")
    if args.tensorboard:
        print(f"TensorBoard logs: {Path(results.save_dir) / 'tensorboard'}")
    print(f"Validation previews: {Path(results.save_dir) / 'val_previews'}")


if __name__ == "__main__":
    main()
