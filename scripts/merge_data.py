from pathlib import Path
import shutil
import yaml

# Folder containing your 5 datasets
SOURCE_ROOT = Path(r"F:\prace\crl_pond_segmentation\data")

# Output merged dataset
DEST_ROOT = Path(r"F:\prace\crl_pond_segmentation\data\merged")

splits = ["train", "val", "test"]

# Create destination folders
for split in splits:
    (DEST_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
    (DEST_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


def convert_label_file(src_label, dst_label):
    """
    Convert every YOLO class id to class 0.
    Works for bbox, segmentation, pose, OBB etc.
    """
    with open(src_label, "r") as f:
        lines = f.readlines()

    converted = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        # Replace class id with 0
        parts[0] = "0"

        converted.append(" ".join(parts))

    with open(dst_label, "w") as f:
        f.write("\n".join(converted))


for dataset_dir in SOURCE_ROOT.iterdir():
    if not dataset_dir.is_dir():
        continue

    print(f"Processing {dataset_dir.name}")

    for split in splits:

        # Handle both "val" and "valid"
        src_split = split

        if split == "val" and not (dataset_dir / "val").exists():
            if (dataset_dir / "valid").exists():
                src_split = "valid"

        image_dir = dataset_dir / src_split / "images"
        label_dir = dataset_dir / src_split / "labels"

        # Copy images
        if image_dir.exists():
            for img in image_dir.iterdir():
                if img.is_file():
                    new_name = f"{dataset_dir.name}_{img.name}"
                    shutil.copy2(
                        img,
                        DEST_ROOT / split / "images" / new_name
                    )

        # Copy + convert labels
        if label_dir.exists():
            for lbl in label_dir.iterdir():
                if lbl.is_file():

                    new_name = f"{dataset_dir.name}_{lbl.name}"

                    convert_label_file(
                        lbl,
                        DEST_ROOT / split / "labels" / new_name
                    )

# Create final data.yaml
data_yaml = {
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "nc": 1,
    "names": ["water"]
}

with open(DEST_ROOT / "data.yaml", "w") as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False)

print("Merge complete.")