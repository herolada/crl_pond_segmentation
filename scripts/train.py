#!/usr/bin/env python3
"""Train and export a YOLO segmentation model for pond-like water regions."""

from __future__ import annotations
import random
import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = PACKAGE_DIR / "data"
DEFAULT_RUNS_DIR = PACKAGE_DIR / "runs" / "segmentation_training"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_batch_size(value: str) -> int | float:
    """Accept either a fixed integer batch size or Ultralytics' AutoBatch memory fraction."""
    try:
        if "." in value:
            batch_fraction = float(value)
            if not 0.0 < batch_fraction <= 1.0:
                raise argparse.ArgumentTypeError("AutoBatch fractions must be in the range (0.0, 1.0].")
            return batch_fraction
        batch_size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Batch must be an integer or a GPU memory fraction like 0.75.") from exc

    if batch_size <= 0:
        raise argparse.ArgumentTypeError("Batch size must be positive.")
    return batch_size


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
    parser.add_argument(
        "--batch",
        type=parse_batch_size,
        default=16,
        help=(
            "Batch size. Use an integer for a fixed batch, or a float in (0, 1] to let "
            "Ultralytics AutoBatch target that fraction of GPU memory. "
            "NOTE: AutoBatch requires cudnn.benchmark=False, which conflicts with "
            "throughput-optimised training. Use an explicit integer batch size when possible."
        ),
    )
    parser.add_argument("--device", default="0", help='Training device, e.g. "0", "0,1", or "cpu".')
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Data loader workers. Lower values use much less RAM in WSL; raise if the GPU is still starved.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Number of batches each data loader worker preloads. Lower values reduce RAM pressure.",
    )
    parser.add_argument(
        "--buffer-images",
        type=int,
        default=128,
        help="Cap Ultralytics' in-RAM augmentation image buffer. Lower values reduce RAM during mosaic training.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin CUDA transfer buffers. Disable with --no-pin-memory if host RAM is still tight.",
    )
    parser.add_argument(
        "--cache",
        choices=("false", "ram", "disk"),
        default="false",
        help="Ultralytics image cache mode. Keep false for lowest RAM use; disk can help if storage is fast.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable deterministic CUDA ops. Disabled by default for better GPU throughput.",
    )
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
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
    parser.add_argument(
        "--fix-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Scan and fix label files that have mismatched box/segment counts before training. "
            "Lines with a bounding-box entry but no polygon (or vice-versa) are removed. "
            "Enabled by default; use --no-fix-labels to skip."
        ),
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=None,
        help=(
            "Initial learning rate. Defaults to 0.001 when not set. "
            "Pass the value suggested by --lr-finder here to use it for a full training run."
        ),
    )
    parser.add_argument(
        "--lr-finder",
        action="store_true",
        help=(
            "Run a learning-rate range test (Leslie Smith) instead of a full training run. "
            "Sweeps lr exponentially from --lr-finder-start-lr to --lr-finder-end-lr, records "
            "the smoothed loss, saves a plot, and prints a suggested lr0. "
            "Re-run without this flag (and with --lr0 <suggested>) to do the actual training."
        ),
    )
    parser.add_argument(
        "--lr-finder-start-lr",
        type=float,
        default=1e-7,
        help="Minimum learning rate for the LR range test.",
    )
    parser.add_argument(
        "--lr-finder-end-lr",
        type=float,
        default=10.0,
        help="Maximum learning rate for the LR range test.",
    )
    parser.add_argument(
        "--lr-finder-num-iter",
        type=int,
        default=200,
        help="Maximum number of mini-batches to run during the LR range test.",
    )
    parser.add_argument(
        "--lr-finder-stop-factor",
        type=float,
        default=4.0,
        help="Abort the range test when smoothed loss exceeds best_loss * stop_factor (divergence guard).",
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
    """Return a short, unique filename stem for a merged-dataset sample.

    The full path is hashed so that the stem is always exactly 24 characters:
    an 8-char dataset prefix + '_' + 16-char SHA-1 hex digest.  This keeps
    merged paths well within Windows' 260-char MAX_PATH limit even when the
    project directory is deeply nested.
    """
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:16]
    # Take up to 8 slugified chars from the dataset root name as a human-
    # readable prefix so files remain identifiable in a file browser.
    prefix = slugify(source.root.name)[:8].rstrip("_")
    return f"{prefix}_{digest}"


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
    """Parse one YOLO segmentation label line.

    A valid segmentation line has the form:
        <class_id> x1 y1 x2 y2 ... xN yN
    where N >= 3, so at minimum 7 values (1 class + 6 coords).
    The coordinate count must be even (pairs of x,y).

    Returns None for any line that doesn't meet these criteria so that
    detect-only boxes (5 values) and malformed lines are silently skipped.
    This prevents the Ultralytics "segment counts != box counts" crash.
    """
    parts = line.strip().split()
    if len(parts) < 7:
        # Fewer than 3 polygon points — could be a bounding-box-only line (5
        # values) or garbage.  Skip it so segment counts stay in sync.
        return None
    # Coordinate count must be even (x,y pairs).
    if (len(parts) - 1) % 2 != 0:
        return None
    try:
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return None
    return class_id, coords


def _count_label_file_issues(label_path: Path) -> int:
    """Return the number of lines in *label_path* that would be dropped by parse_label_line."""
    if not label_path.exists():
        return 0
    count = 0
    with label_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and parse_label_line(line) is None:
                count += 1
    return count


def fix_label_files(labels_dir: Path) -> tuple[int, int]:
    """Remove non-polygon lines from every .txt file under *labels_dir*.

    Returns (files_changed, lines_removed).
    """
    files_changed = 0
    lines_removed = 0
    for label_path in sorted(labels_dir.rglob("*.txt")):
        kept: list[str] = []
        dropped = 0
        with label_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                if not raw_line.strip():
                    continue
                if parse_label_line(raw_line) is None:
                    dropped += 1
                else:
                    kept.append(raw_line.rstrip())
        if dropped:
            label_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
            files_changed += 1
            lines_removed += dropped
    return files_changed, lines_removed


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
                # Skip bounding-box-only or malformed lines to keep segment
                # counts consistent with box counts inside each label file.
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


def _windows_longpath(p: Path) -> str:
    r"""Return a \\?\-prefixed absolute path string to bypass MAX_PATH on Windows."""
    resolved = str(p.resolve())
    if resolved.startswith("\\\\"):
        # UNC path  ->  \\?\UNC\<server>\<share>\...
        return "\\\\?\\UNC\\" + resolved[2:]
    # Regular drive path  ->  \\?\C:\...
    return "\\\\?\\" + resolved


def link_or_copy_file(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() or destination_path.is_symlink():
        destination_path.unlink()
    try:
        destination_path.symlink_to(source_path)
    except (OSError, NotImplementedError):
        # Symlinks require Developer Mode or admin rights on Windows; fall back
        # to a plain copy.  Use the \?\  long-path prefix on Windows so that
        # paths longer than MAX_PATH (260 chars) work without a registry tweak.
        if os.name == "nt":
            import ctypes
            src = _windows_longpath(source_path)
            dst = _windows_longpath(destination_path)
            ok = ctypes.windll.kernel32.CopyFileW(src, dst, False)
            if not ok:
                err = ctypes.get_last_error()
                raise OSError(err, f"CopyFileW failed (error {err}): {src!r} -> {dst!r}")
        else:
            shutil.copy2(source_path, destination_path)


def cache_override(cache_mode: str) -> bool | str:
    return False if cache_mode == "false" else cache_mode


def configure_training_runtime(args: argparse.Namespace) -> None:
    """Tune process-wide CPU/GPU settings before Ultralytics builds loaders and models."""
    cv2.setNumThreads(0)

    using_cuda = str(args.device).lower() != "cpu" and torch.cuda.is_available()
    if using_cuda:
        # AutoBatch requires cudnn.benchmark=False to compute memory usage
        # reliably.  When the user supplies a float batch size we therefore
        # force benchmark off regardless of --deterministic.
        using_autobatch = isinstance(args.batch, float)
        if using_autobatch:
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = not args.deterministic

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def patch_ultralytics_dataloader(args: argparse.Namespace) -> None:
    """Expose Ultralytics dataloader pin-memory and prefetch knobs without modifying site-packages."""
    from torch.utils.data import distributed
    from ultralytics.data import build as data_build
    from ultralytics.models.yolo.detect import train as detect_train

    def build_dataloader(
        dataset,
        batch: int,
        workers: int,
        shuffle: bool = True,
        rank: int = -1,
        drop_last: bool = False,
        pin_memory: bool = True,
    ):
        batch = min(batch, len(dataset))
        if getattr(dataset, "max_buffer_length", 0):
            dataset.max_buffer_length = min(dataset.max_buffer_length, args.buffer_images)
        cuda_device_count = torch.cuda.device_count()
        max_workers = (os.cpu_count() or 1) // max(cuda_device_count, 1)
        worker_count = max(0, min(max_workers, workers))
        sampler = (
            None
            if rank == -1
            else distributed.DistributedSampler(dataset, shuffle=shuffle)
            if shuffle
            else data_build.ContiguousDistributedSampler(dataset)
        )
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + data_build.RANK)
        return data_build.InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch,
            shuffle=shuffle and sampler is None,
            num_workers=worker_count,
            sampler=sampler,
            prefetch_factor=args.prefetch_factor if worker_count > 0 else None,
            pin_memory=cuda_device_count > 0 and pin_memory and args.pin_memory,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=data_build.seed_worker,
            generator=generator,
            drop_last=drop_last and len(dataset) % batch != 0,
        )

    data_build.build_dataloader = build_dataloader
    detect_train.build_dataloader = build_dataloader


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
                src_label = source_label_path(split_dir, image_path)

                link_or_copy_file(image_path, destination_image)
                remap_label_file(
                    source=source,
                    source_label_path=src_label,
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
            if isinstance(sample, tuple):
                image_path = Path(sample[0])
                label_path = Path(sample[1])
            else:
                image_path = sample.image_path
                label_path = sample.label_path

            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue

            gt_polygons = load_segmentation_labels(label_path, image_bgr.shape)
            prediction = predictor.predict(
                source=str(image_path),
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
            preview_path = epoch_dir / f"{image_path.stem}_preview.jpg"
            cv2.imwrite(str(preview_path), preview)

            if self.writer:
                rgb_preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                chw_preview = np.transpose(rgb_preview, (2, 0, 1))
                self.writer.add_image(
                    tag=f"val_previews/{image_path.stem}",
                    img_tensor=chw_preview,
                    global_step=epoch,
                )


class _LrFinderStop(BaseException):
    """Raised from an Ultralytics callback to abort the LR range test mid-epoch."""


class LrFinderMonitor:
    """Records exponentially-increasing LR and the corresponding smoothed loss."""

    def __init__(self, start_lr: float, end_lr: float, num_iter: int, stop_factor: float) -> None:
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iter = num_iter
        self.stop_factor = stop_factor
        self._lr_mult = (end_lr / start_lr) ** (1.0 / num_iter)
        self._iter = 0
        self._beta = 0.98
        self._avg_loss = 0.0
        self._best_loss = float("inf")
        self.lrs: list[float] = []
        self.losses: list[float] = []
        self.run_dir: Path | None = None

    def on_train_start(self, trainer) -> None:
        self.run_dir = Path(trainer.save_dir)

    def on_train_batch_start(self, trainer) -> None:
        # Override the LR that Ultralytics' warmup scheduler may have just set.
        current_lr = self.start_lr * (self._lr_mult ** self._iter)
        for pg in trainer.optimizer.param_groups:
            pg["lr"] = current_lr

    def on_train_batch_end(self, trainer) -> None:
        current_lr = self.start_lr * (self._lr_mult ** self._iter)
        loss = float(trainer.loss.detach().cpu())

        # Exponential moving average with bias correction (same as fast.ai LR finder).
        self._avg_loss = self._beta * self._avg_loss + (1 - self._beta) * loss
        smoothed = self._avg_loss / (1 - self._beta ** (self._iter + 1))

        self.lrs.append(current_lr)
        self.losses.append(smoothed)

        if smoothed < self._best_loss:
            self._best_loss = smoothed

        self._iter += 1
        if self._iter >= self.num_iter or smoothed > self.stop_factor * self._best_loss:
            raise _LrFinderStop

    def suggest_lr(self) -> float | None:
        """Return the LR at the point of steepest loss descent."""
        if len(self.losses) < 5:
            return None
        log_lrs = np.log10(self.lrs)
        gradients = np.gradient(np.array(self.losses), log_lrs)
        # Exclude the final 10 % of points (divergence zone) before searching.
        cutoff = max(1, len(gradients) - max(5, len(gradients) // 10))
        return self.lrs[int(np.argmin(gradients[:cutoff]))]

    def save_plot(self, output_path: Path) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping LR finder plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogx(self.lrs, self.losses)
        ax.set_xlabel("Learning Rate (log scale)")
        ax.set_ylabel("Smoothed Loss")
        ax.set_title("LR Finder — Loss vs Learning Rate")
        ax.grid(True, which="both", alpha=0.4)

        suggested = self.suggest_lr()
        if suggested is not None:
            ax.axvline(x=suggested, color="red", linestyle="--", label=f"Suggested LR: {suggested:.2e}")
            ax.legend()

        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)


def run_lr_finder(args: argparse.Namespace, prepared_dataset: PreparedDataset) -> None:
    """Run an LR range test, save a loss-vs-LR plot, and print the suggested lr0."""
    monitor = LrFinderMonitor(
        start_lr=args.lr_finder_start_lr,
        end_lr=args.lr_finder_end_lr,
        num_iter=args.lr_finder_num_iter,
        stop_factor=args.lr_finder_stop_factor,
    )

    run_name = (args.name or Path(args.model).stem) + "_lr_finder"
    overrides = {
        **training_overrides(args),
        "data": str(prepared_dataset.yaml_path),
        "epochs": 1,
        "patience": 0,
        "save": False,
        "val": False,
        "plots": False,
        "name": run_name,
        "verbose": False,
    }

    model = YOLO(args.model)
    model.add_callback("on_train_start", monitor.on_train_start)
    model.add_callback("on_train_batch_start", monitor.on_train_batch_start)
    model.add_callback("on_train_batch_end", monitor.on_train_batch_end)

    print(
        f"LR finder: sweeping lr {args.lr_finder_start_lr:.1e} → {args.lr_finder_end_lr:.1e} "
        f"over up to {args.lr_finder_num_iter} batches …"
    )
    try:
        model.train(**overrides)
    except _LrFinderStop:
        pass
    except Exception as exc:
        if not monitor.lrs:
            raise
        # Absorb finalisation errors that Ultralytics may raise after the mid-epoch abort.
        print(f"LR finder: training interrupted ({exc})")

    if not monitor.lrs:
        print("LR finder: no data collected — check your dataset and model configuration.")
        return

    if monitor.run_dir is not None:
        plot_path = monitor.run_dir / "lr_finder.png"
        monitor.save_plot(plot_path)
        print(f"LR finder plot: {plot_path}")

    suggested = monitor.suggest_lr()
    if suggested is not None:
        print(f"LR finder: suggested lr0 = {suggested:.2e}  (point of steepest loss descent)")
        print(f"           Re-run with --lr0 {suggested:.2e} to train with this learning rate.")
    else:
        print("LR finder: not enough data to suggest a learning rate — try a wider range.")


def training_overrides(args: argparse.Namespace) -> dict:
    run_name = args.name or Path(args.model).stem
    return {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "cache": cache_override(args.cache),
        "patience": args.patience,
        "project": str(args.project.resolve()),
        "name": run_name,
        "exist_ok": True,
        "pretrained": True,
        "amp": True,
        "deterministic": args.deterministic,
        "compile": False,
        "optimizer": "AdamW",
        "lr0": args.lr0 if args.lr0 is not None else 0.001,
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
        # "crop_fraction" removed — deprecated in recent Ultralytics versions.
        "plots": True,
        "save": True,
        "save_period": args.preview_interval,
        "val": True,
        "verbose": True,
    }


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("--workers must be zero or greater.")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be at least 1.")
    if args.buffer_images < 1:
        raise ValueError("--buffer-images must be at least 1.")

    args.data = args.data.resolve()
    args.project = args.project.resolve()
    args.project.mkdir(parents=True, exist_ok=True)
    (args.project / "_dataset_cache").mkdir(parents=True, exist_ok=True)
    configure_training_runtime(args)
    patch_ultralytics_dataloader(args)

    prepared_dataset = build_merged_dataset(
        data_path=args.data,
        project_dir=args.project,
        class_mode=args.class_mode,
        single_class_name=args.single_class_name,
    )

    # Optionally clean label files in the merged dataset before training to
    # ensure box and segment counts are equal.  This prevents the Ultralytics
    # "WARNING Box and segment counts should be equal" crash.
    if args.fix_labels:
        for split in ("train", "val", "test"):
            labels_dir = prepared_dataset.root_dir / split / "labels"
            if labels_dir.exists():
                files_changed, lines_removed = fix_label_files(labels_dir)
                if files_changed:
                    print(
                        f"fix_labels [{split}]: removed {lines_removed} non-polygon "
                        f"line(s) from {files_changed} file(s)."
                    )
        # Invalidate any stale Ultralytics label caches so they are rebuilt
        # from the cleaned files.
        for cache_file in prepared_dataset.root_dir.rglob("labels.cache"):
            cache_file.unlink(missing_ok=True)

    if args.lr_finder:
        run_lr_finder(args, prepared_dataset)
        return

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