"""
training/yolov8/train_yolov8.py

Fine-tune YOLOv8s (Ultralytics, COCO-pretrained) on the traffic/urban-security
VOC subset, using the YOLO-format dataset produced by data_prep/voc_to_yolo.py.

Why this model: represents the "YOLO family" (single-stage, anchor-free in
v8) branch of the 3-way comparison. yolov8s (small) is chosen over yolov8n/m
as the accuracy/speed middle ground that trains quickly on a single T4.

Usage:
    python train_yolov8.py \
        --data /content/yolo_data/data.yaml \
        --output_dir /content/runs/yolov8 \
        --epochs 50 --imgsz 640 --batch 16
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))  # -> training/
from common.metrics_utils import save_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml from data_prep/voc_to_yolo.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights", default="yolov8s.pt", help="pretrained checkpoint to fine-tune from")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15, help="early-stopping patience (epochs)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)  # downloads the pretrained checkpoint on first use

    train_start = time.time()
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(out_dir),
        name="train",
        exist_ok=True,
        patience=args.patience,
    )
    train_time_min = (time.time() - train_start) / 60

    # final evaluation on the held-out TEST split (not val) for a fair
    # comparison against Faster R-CNN / RT-DETR's own test-set numbers
    val_results = model.val(data=args.data, split="test")
    metrics = {k: float(v) for k, v in val_results.results_dict.items()}

    speed = val_results.speed  # {"preprocess": ms, "inference": ms, "postprocess": ms}, per image
    inference_ms = speed.get("inference")
    fps = (1000.0 / inference_ms) if inference_ms else None

    n_params = sum(p.numel() for p in model.model.parameters())

    results = {
        "model_name": "YOLOv8s",
        "paper": "Redmon et al. YOLO lineage (Ultralytics implementation, no single canonical v8 paper)",
        "num_params": n_params,
        "train_time_min": train_time_min,
        "test_metrics": metrics,
        "fps": fps,
        "latency_ms": inference_ms,
        "epochs": args.epochs,
        "batch_size": args.batch,
    }
    save_metrics(out_dir / "metrics.json", results)
    print("Done. Results in", out_dir)


if __name__ == "__main__":
    main()
