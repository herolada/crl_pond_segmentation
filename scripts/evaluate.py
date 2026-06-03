#!/usr/bin/env python3
"""Run a trained YOLO segmentation model on a test set and report evaluation metrics."""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "runs" / "segmentation_evaluation"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class InstanceMask:
    class_id: int
    mask: np.ndarray
    score: float | None = None


@dataclass(frozen=True)
class EvaluationSample:
    image_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a trained Ultralytics YOLO segmentation model on a test set, "
            "compare predictions against ground truth, and save evaluation previews."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to a trained YOLO segmentation checkpoint, e.g. runs/.../weights/best.pt.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to the test image root. Nested folders are allowed.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to the test label root. Labels should mirror the image directory structure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for rendered previews and metrics JSON output.",
    )
    parser.add_argument("--imgsz", type=int, default=320, help="Inference and validation image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for predictions and validation.")
    parser.add_argument("--device", default="0", help='Inference device, e.g. "0", "0,1", or "cpu".')
    parser.add_argument(
        "--preview-count",
        type=int,
        default=8,
        help="How many random examples to render as side-by-side ground-truth vs prediction previews.",
    )
    parser.add_argument(
        "--preview-cols",
        type=int,
        default=2,
        help="How many columns to use when packing preview pairs into a montage.",
    )
    parser.add_argument(
        "--preview-seed",
        type=int,
        default=None,
        help="Optional random seed for preview sampling. Leave unset for a fresh random subset each run.",
    )
    parser.add_argument(
        "--match-iou",
        type=float,
        default=0.5,
        help="IoU threshold used when matching predicted instances against ground truth masks.",
    )
    parser.add_argument(
        "--no-class-aware",
        action="store_true",
        help="Match instances without requiring the predicted class to equal the ground-truth class.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def normalize_names(names: object) -> dict[int, str]:
    if isinstance(names, dict):
        items = sorted((int(key), str(value)) for key, value in names.items())
        return {key: value for key, value in items}
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    return {0: "object"}


def as_name_list(names: dict[int, str]) -> list[str]:
    if not names:
        return ["object"]
    return [names[index] for index in sorted(names)]

def letterbox(image: np.ndarray, size: int, color=(24, 24, 24)) -> np.ndarray:
    h, w = image.shape[:2]

    scale = min(size / w, size / h)
    nw = int(round(w * scale))
    nh = int(round(h * scale))

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((size, size, 3), color, dtype=np.uint8)

    x = (size - nw) // 2
    y = (size - nh) // 2

    canvas[y:y+nh, x:x+nw] = resized
    return canvas

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


def iter_image_paths(image_root: Path) -> list[Path]:
    if not image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {image_root}")
    return sorted(path for path in image_root.rglob("*") if path.is_file() and is_image_file(path))


def collect_samples(image_root: Path, label_root: Path) -> list[EvaluationSample]:
    samples: list[EvaluationSample] = []
    for image_path in iter_image_paths(image_root):
        relative = image_path.relative_to(image_root)
        label_path = label_root / relative.with_suffix(".txt")
        samples.append(EvaluationSample(image_path=image_path, label_path=label_path))

    if not samples:
        raise FileNotFoundError(f"No images found under {image_root}")
    return samples


def read_labels(label_path: Path, image_shape: tuple[int, int, int]) -> list[InstanceMask]:
    height, width = image_shape[:2]
    instances: list[InstanceMask] = []
    if not label_path.exists():
        return instances

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
            polygon = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(polygon, [points.astype(np.int32)], 1)
            instances.append(InstanceMask(class_id=class_id, mask=polygon.astype(bool)))

    return instances


def prediction_instances(result, image_shape: tuple[int, int, int]) -> list[InstanceMask]:
    masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if masks is None or masks.xy is None:
        return []

    height, width = image_shape[:2]
    instances: list[InstanceMask] = []
    for index, polygon in enumerate(masks.xy):
        if polygon is None:
            continue
        points = np.asarray(polygon, dtype=np.int32)
        if points.ndim != 2 or points.shape[0] < 3:
            continue
        canvas = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(canvas, [points], 1)
        class_id = int(boxes.cls[index].item()) if boxes is not None and index < len(boxes) else 0
        conf = float(boxes.conf[index].item()) if boxes is not None and index < len(boxes) else None
        instances.append(InstanceMask(class_id=class_id, mask=canvas.astype(bool), score=conf))
    return instances


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum(dtype=np.float64)
    union = np.logical_or(mask_a, mask_b).sum(dtype=np.float64)
    if union == 0:
        return 0.0
    return float(intersection / union)


def aggregate_foreground_metrics(
    ground_truth: list[InstanceMask],
    predictions: list[InstanceMask],
    image_shape: tuple[int, int, int],
) -> dict[str, float]:
    height, width = image_shape[:2]
    gt_union = np.zeros((height, width), dtype=bool)
    pred_union = np.zeros((height, width), dtype=bool)

    for instance in ground_truth:
        gt_union |= instance.mask
    for instance in predictions:
        pred_union |= instance.mask

    tp = np.logical_and(gt_union, pred_union).sum(dtype=np.float64)
    fp = np.logical_and(~gt_union, pred_union).sum(dtype=np.float64)
    fn = np.logical_and(gt_union, ~pred_union).sum(dtype=np.float64)
    denom_iou = tp + fp + fn
    denom_precision = tp + fp
    denom_recall = tp + fn
    dice_denom = (2 * tp) + fp + fn

    return {
        "pixel_tp": float(tp),
        "pixel_fp": float(fp),
        "pixel_fn": float(fn),
        "pixel_precision": float(tp / denom_precision) if denom_precision else 0.0,
        "pixel_recall": float(tp / denom_recall) if denom_recall else 0.0,
        "pixel_iou": float(tp / denom_iou) if denom_iou else 0.0,
        "pixel_dice": float((2 * tp) / dice_denom) if dice_denom else 0.0,
    }


def greedy_match_instances(
    ground_truth: list[InstanceMask],
    predictions: list[InstanceMask],
    class_aware: bool,
    iou_threshold: float,
) -> tuple[list[float], dict[int, dict[str, float]]]:
    class_stats: dict[int, dict[str, float]] = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "iou_sum": 0.0})
    matched_ious: list[float] = []

    gt_by_class: dict[int, list[int]] = defaultdict(list)
    pred_by_class: dict[int, list[int]] = defaultdict(list)
    classes = set()
    for index, instance in enumerate(ground_truth):
        gt_by_class[instance.class_id].append(index)
        classes.add(instance.class_id)
    for index, instance in enumerate(predictions):
        pred_by_class[instance.class_id].append(index)
        classes.add(instance.class_id)

    candidate_classes = sorted(classes) if class_aware else [None]
    for class_id in candidate_classes:
        gt_indices = gt_by_class[class_id] if class_aware else list(range(len(ground_truth)))
        pred_indices = pred_by_class[class_id] if class_aware else list(range(len(predictions)))
        if class_aware:
            stats_key = int(class_id)
        else:
            stats_key = -1

        if not gt_indices and not pred_indices:
            continue
        if not gt_indices:
            class_stats[stats_key]["fp"] += float(len(pred_indices))
            continue
        if not pred_indices:
            class_stats[stats_key]["fn"] += float(len(gt_indices))
            continue

        iou_matrix = np.zeros((len(gt_indices), len(pred_indices)), dtype=np.float32)
        for gt_row, gt_index in enumerate(gt_indices):
            for pred_col, pred_index in enumerate(pred_indices):
                iou_matrix[gt_row, pred_col] = mask_iou(ground_truth[gt_index].mask, predictions[pred_index].mask)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        while True:
            best_value = -1.0
            best_pair: tuple[int, int] | None = None
            for gt_row in range(len(gt_indices)):
                if gt_row in matched_gt:
                    continue
                for pred_col in range(len(pred_indices)):
                    if pred_col in matched_pred:
                        continue
                    value = float(iou_matrix[gt_row, pred_col])
                    if value > best_value:
                        best_value = value
                        best_pair = (gt_row, pred_col)

            if best_pair is None or best_value < iou_threshold:
                break

            gt_row, pred_col = best_pair
            matched_gt.add(gt_row)
            matched_pred.add(pred_col)
            matched_ious.append(best_value)
            class_stats[stats_key]["tp"] += 1.0
            class_stats[stats_key]["iou_sum"] += best_value

        class_stats[stats_key]["fp"] += float(len(pred_indices) - len(matched_pred))
        class_stats[stats_key]["fn"] += float(len(gt_indices) - len(matched_gt))

    return matched_ious, class_stats


def class_metrics_from_stats(class_stats: dict[int, dict[str, float]], class_names: dict[int, str]) -> dict[str, dict[str, float]]:
    per_class: dict[str, dict[str, float]] = {}
    for class_id, stats in sorted(class_stats.items(), key=lambda item: item[0]):
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        key = class_names.get(class_id, f"class_{class_id}")
        per_class[key] = {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "mean_matched_iou": float(stats["iou_sum"] / tp) if tp else 0.0,
        }
    return per_class


def draw_instances(
    image_bgr: np.ndarray,
    instances: Iterable[InstanceMask],
    color: tuple[int, int, int],
    label_prefix: str,
    class_names: dict[int, str],
) -> np.ndarray:
    annotated = image_bgr.copy()
    overlay = image_bgr.copy()

    for instance in instances:
        mask = instance.mask.astype(bool)
        if not mask.any():
            continue
        overlay[mask] = color
        ys, xs = np.where(mask)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        class_name = class_names.get(instance.class_id, f"class_{instance.class_id}")
        score_text = f" {instance.score:.2f}" if instance.score is not None else ""
        cv2.putText(
            annotated,
            f"{label_prefix}:{class_name}{score_text}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0.0, annotated)
    return annotated


def stack_preview(ground_truth: np.ndarray, prediction: np.ndarray, title: str) -> np.ndarray:
    header = np.zeros((42, ground_truth.shape[1] + prediction.shape[1], 3), dtype=np.uint8)
    header[:] = (24, 24, 24)
    cv2.putText(header, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
    left = cv2.copyMakeBorder(ground_truth, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    right = cv2.copyMakeBorder(prediction, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    cv2.putText(left, "Ground Truth", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(right, "Prediction", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    return np.vstack((header, np.hstack((left, right))))


def pad_to_size(image: np.ndarray, height: int, width: int) -> np.ndarray:
    pad_bottom = max(0, height - image.shape[0])
    pad_right = max(0, width - image.shape[1])
    return cv2.copyMakeBorder(image, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(24, 24, 24))


def build_montage(images: list[np.ndarray], cols: int, title: str) -> np.ndarray:
    if not images:
        raise ValueError("Cannot build a montage without preview images.")
    cols = max(1, cols)
    rows = math.ceil(len(images) / cols)
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    padding = 12
    header_height = 48
    canvas_height = header_height + rows * max_height + (rows - 1) * padding
    canvas_width = cols * max_width + (cols - 1) * padding
    canvas = np.full((canvas_height, canvas_width, 3), 24, dtype=np.uint8)
    cv2.putText(canvas, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)

    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        x = col * (max_width + padding)
        y = header_height + row * (max_height + padding)
        padded = pad_to_size(image, max_height, max_width)
        canvas[y : y + max_height, x : x + max_width] = padded

    return canvas


def build_temporary_ultralytics_dataset(
    image_root: Path,
    label_root: Path,
    class_names: dict[int, str],
    output_dir: Path,
) -> tuple[tempfile.TemporaryDirectory, Path]:
    temp_root = tempfile.TemporaryDirectory(prefix="pond_segmentation_eval_", dir=str(output_dir))
    temp_root_path = Path(temp_root.name)
    dataset_root = temp_root_path / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)

    test_images = dataset_root / "test" / "images"
    test_labels = dataset_root / "test" / "labels"
    test_images.parent.mkdir(parents=True, exist_ok=True)

    try:
        test_images.symlink_to(image_root)
    except OSError:
        shutil.copytree(image_root, test_images, dirs_exist_ok=True)

    try:
        test_labels.symlink_to(label_root)
    except OSError:
        shutil.copytree(label_root, test_labels, dirs_exist_ok=True)

    data_yaml = {
        "path": str(dataset_root),
        "train": "test/images",
        "val": "test/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": as_name_list(class_names),
    }
    data_yaml_path = dataset_root / "data.yaml"
    with data_yaml_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data_yaml, handle, sort_keys=False)

    return temp_root, data_yaml_path


def render_preview_panel(
    image_bgr: np.ndarray,
    ground_truth: list[InstanceMask],
    predictions: list[InstanceMask],
    class_names: dict[int, str],
    sample_metrics: dict[str, float],
    imgsz: int,
) -> np.ndarray:
    gt_panel = draw_instances(image_bgr, ground_truth, (0, 255, 0), "gt", class_names)
    pred_panel = draw_instances(image_bgr, predictions, (0, 0, 255), "pred", class_names)

    gt_panel = letterbox(gt_panel, imgsz)
    pred_panel = letterbox(pred_panel, imgsz)

    panel = stack_preview(gt_panel, pred_panel, "Segmentation Evaluation")
    summary = (
        f"IoU={sample_metrics['pixel_iou']:.3f}  Dice={sample_metrics['pixel_dice']:.3f}  "
        f"InstF1={sample_metrics['instance_f1']:.3f}"
    )
    cv2.putText(panel, summary, (12, panel.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_samples(args.images.resolve(), args.labels.resolve())
    model = YOLO(str(args.model.resolve()))
    class_names = normalize_names(getattr(model, "names", None))

    temp_dataset, data_yaml_path = build_temporary_ultralytics_dataset(
        image_root=args.images.resolve(),
        label_root=args.labels.resolve(),
        class_names=class_names,
        output_dir=args.output_dir,
    )

    try:
        val_results = model.val(
            data=str(data_yaml_path),
            split="test",
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            plots=False,
            save_json=False,
            save_txt=False,
            project=str(args.output_dir),
            name=f"{Path(args.model).stem}_ultralytics_val",
            exist_ok=True,
        )
        preview_dir = args.output_dir / f"{Path(args.model).stem}_previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_examples_dir = preview_dir / "examples"
        preview_examples_dir.mkdir(parents=True, exist_ok=True)

        total_pixel_tp = 0.0
        total_pixel_fp = 0.0
        total_pixel_fn = 0.0
        total_matches: list[float] = []
        total_class_stats: dict[int, dict[str, float]] = defaultdict(lambda: {"tp": 0.0, "fp": 0.0, "fn": 0.0, "iou_sum": 0.0})
        preview_rng = random.Random(args.preview_seed)
        preview_count = max(0, min(args.preview_count, len(samples)))
        preview_samples = preview_rng.sample(samples, k=preview_count) if preview_count else []
        preview_sample_paths = {sample.image_path for sample in preview_samples}
        preview_panels: list[tuple[Path, np.ndarray]] = []

        for sample in samples:
            image_bgr = cv2.imread(str(sample.image_path))
            if image_bgr is None:
                continue

            ground_truth = read_labels(sample.label_path, image_bgr.shape)
            prediction = model.predict(
                source=str(sample.image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )[0]
            predictions = prediction_instances(prediction, image_bgr.shape)

            image_pixel_metrics = aggregate_foreground_metrics(ground_truth, predictions, image_bgr.shape)
            matches, class_stats = greedy_match_instances(
                ground_truth=ground_truth,
                predictions=predictions,
                class_aware=not args.no_class_aware,
                iou_threshold=args.match_iou,
            )
            sample_instance_tp = sum(stats["tp"] for stats in class_stats.values())
            sample_instance_fp = sum(stats["fp"] for stats in class_stats.values())
            sample_instance_fn = sum(stats["fn"] for stats in class_stats.values())
            sample_instance_precision = (
                sample_instance_tp / (sample_instance_tp + sample_instance_fp)
                if (sample_instance_tp + sample_instance_fp)
                else 0.0
            )
            sample_instance_recall = (
                sample_instance_tp / (sample_instance_tp + sample_instance_fn)
                if (sample_instance_tp + sample_instance_fn)
                else 0.0
            )
            sample_instance_f1 = (
                2 * sample_instance_precision * sample_instance_recall / (sample_instance_precision + sample_instance_recall)
                if (sample_instance_precision + sample_instance_recall)
                else 0.0
            )

            total_pixel_tp += image_pixel_metrics["pixel_tp"]
            total_pixel_fp += image_pixel_metrics["pixel_fp"]
            total_pixel_fn += image_pixel_metrics["pixel_fn"]
            total_matches.extend(matches)

            for class_id, stats in class_stats.items():
                bucket = total_class_stats[class_id]
                bucket["tp"] += stats["tp"]
                bucket["fp"] += stats["fp"]
                bucket["fn"] += stats["fn"]
                bucket["iou_sum"] += stats["iou_sum"]

            if sample.image_path in preview_sample_paths:
                preview = render_preview_panel(
                    image_bgr=image_bgr,
                    ground_truth=ground_truth,
                    predictions=predictions,
                    class_names=class_names,
                    sample_metrics={
                        "pixel_iou": image_pixel_metrics["pixel_iou"],
                        "pixel_dice": image_pixel_metrics["pixel_dice"],
                        "instance_f1": sample_instance_f1,
                    },
                    imgsz=args.imgsz,
                )
                preview_panels.append((sample.image_path, preview))

        for image_path, preview in preview_panels:
            preview_path = preview_examples_dir / f"{image_path.stem}_preview.jpg"
            cv2.imwrite(str(preview_path), preview)

        if preview_panels:
            montage = build_montage(
                images=[panel for _, panel in preview_panels],
                cols=max(1, args.preview_cols),
                title=f"Random preview pairs ({len(preview_panels)} samples)",
            )
            montage_path = preview_dir / f"{Path(args.model).stem}_preview_montage.jpg"
            cv2.imwrite(str(montage_path), montage)

        total_pixel_precision = total_pixel_tp / (total_pixel_tp + total_pixel_fp) if (total_pixel_tp + total_pixel_fp) else 0.0
        total_pixel_recall = total_pixel_tp / (total_pixel_tp + total_pixel_fn) if (total_pixel_tp + total_pixel_fn) else 0.0
        total_pixel_iou = total_pixel_tp / (total_pixel_tp + total_pixel_fp + total_pixel_fn) if (total_pixel_tp + total_pixel_fp + total_pixel_fn) else 0.0
        total_pixel_dice = (2 * total_pixel_tp) / ((2 * total_pixel_tp) + total_pixel_fp + total_pixel_fn) if ((2 * total_pixel_tp) + total_pixel_fp + total_pixel_fn) else 0.0

        instance_tp = sum(stats["tp"] for stats in total_class_stats.values())
        instance_fp = sum(stats["fp"] for stats in total_class_stats.values())
        instance_fn = sum(stats["fn"] for stats in total_class_stats.values())
        instance_precision = instance_tp / (instance_tp + instance_fp) if (instance_tp + instance_fp) else 0.0
        instance_recall = instance_tp / (instance_tp + instance_fn) if (instance_tp + instance_fn) else 0.0
        instance_f1 = (2 * instance_precision * instance_recall / (instance_precision + instance_recall)) if (instance_precision + instance_recall) else 0.0
        mean_matched_iou = float(np.mean(total_matches)) if total_matches else 0.0

        metrics = {
            "ultralytics": getattr(val_results, "results_dict", {}),
            "pixel": {
                "precision": total_pixel_precision,
                "recall": total_pixel_recall,
                "iou": total_pixel_iou,
                "dice": total_pixel_dice,
                "tp": total_pixel_tp,
                "fp": total_pixel_fp,
                "fn": total_pixel_fn,
            },
            "instance": {
                "precision": instance_precision,
                "recall": instance_recall,
                "f1": instance_f1,
                "mean_matched_iou": mean_matched_iou,
            },
            "per_class_instances": class_metrics_from_stats(total_class_stats, class_names),
            "data": {
                "model": str(args.model.resolve()),
                "images": str(args.images.resolve()),
                "labels": str(args.labels.resolve()),
                "num_images": len(samples),
            },
        }

        metrics_path = args.output_dir / f"{Path(args.model).stem}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)

        print("Ultralytics validation metrics:")
        for key, value in metrics["ultralytics"].items():
            if isinstance(value, (float, int)) and not math.isnan(float(value)):
                print(f"  {key}: {float(value):.6f}")
            else:
                print(f"  {key}: {value}")

        print("Custom pixel metrics:")
        print(f"  precision: {total_pixel_precision:.6f}")
        print(f"  recall:    {total_pixel_recall:.6f}")
        print(f"  iou:       {total_pixel_iou:.6f}")
        print(f"  dice:      {total_pixel_dice:.6f}")

        print("Custom instance metrics:")
        print(f"  precision:        {instance_precision:.6f}")
        print(f"  recall:           {instance_recall:.6f}")
        print(f"  f1:               {instance_f1:.6f}")
        print(f"  mean matched IoU: {mean_matched_iou:.6f}")

        print(f"Metrics JSON: {metrics_path}")
        if preview_panels:
            print(f"Preview montage: {montage_path}")
        print(f"Preview images: {preview_examples_dir}")
    finally:
        temp_dataset.cleanup()


if __name__ == "__main__":
    main()
