from pathlib import Path
from ultralytics import YOLO

best_checkpoint = Path("../models/best.pt")
export_model = YOLO(str(best_checkpoint))
export_path = export_model.export(
    format="onnx",
    imgsz=320,
    simplify=True,
    int8=False,
    data="../merged_data/data.yaml"
)