"""
training/rtdetr/train_rtdetr.py

Fine-tune RT-DETR (Ultralytics, COCO-pretrained) on the traffic/urban-security
VOC subset, using the SAME YOLO-format dataset as YOLOv8 (data_prep/voc_to_yolo.py)
-- ultralytics exposes RT-DETR through the same Model API as YOLO, so no
separate data conversion is needed.

Why this model: represents the "Transformer-based" family in the 3-way
comparison. Paper: Lv, Xu, Zhao et al., "DETRs Beat YOLOs on Real-Time Object
Detection", 2023. This is specifically chosen over the original DETR because
(a) vanilla DETR needs ~500 epochs from scratch to converge, infeasible on a
single Colab T4 session, and (b) the RT-DETR paper's own benchmark reports
53.1/54.3 AP on COCO at 108/74 FPS measured directly on a T4 GPU -- i.e. the
paper itself validates real-time feasibility on exactly the GPU this project
runs on.

Usage:
    python train_rtdetr.py \
        --data /content/yolo_data/data.yaml \
        --output_dir /content/runs/rtdetr \
        --epochs 40 --imgsz 640 --batch 8
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

from ultralytics import RTDETR

sys.path.append(str(Path(__file__).resolve().parent.parent))  # -> training/
from common.metrics_utils import save_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml from data_prep/voc_to_yolo.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights", default="rtdetr-l.pt", help="pretrained checkpoint to fine-tune from")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="RT-DETR is heavier than YOLOv8s -> smaller default batch for T4")
    ap.add_argument("--patience", type=int, default=15)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = RTDETR(args.weights)  # downloads the pretrained checkpoint on first use

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

    val_results = model.val(data=args.data, split="test")
    metrics = {k: float(v) for k, v in val_results.results_dict.items()}

    speed = val_results.speed
    inference_ms = speed.get("inference")
    fps = (1000.0 / inference_ms) if inference_ms else None

    n_params = sum(p.numel() for p in model.model.parameters())

    results = {
        "model_name": "RT-DETR-L",
        "paper": "Lv et al., DETRs Beat YOLOs on Real-Time Object Detection, 2023",
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
