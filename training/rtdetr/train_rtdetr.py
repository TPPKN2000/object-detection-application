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

--- PATCH NOTES ------------------------------------------------------------
1. PREFLIGHT CHECK on the data.yaml (train/val/test keys present + each
   resolved path actually exists and contains >=1 image) BEFORE model.train()
   is called. This directly targets the failure mode seen with the YOLOv8
   run in this project: training completed successfully (2h on a T4) and
   THEN crashed at the final `model.val(split="test")` because the data.yaml
   being used didn't have a valid "test" mapping. With this check, that same
   class of bug now fails in a couple of seconds, before any GPU time is spent.
2. --workers and --cache exposed explicitly (previously implicit ultralytics
   defaults). `--cache ram` (or `disk`) decodes each image once and reuses it
   for later epochs instead of re-reading from disk every epoch -- worth
   trying given the "Slow image access detected" warning ultralytics printed
   during the YOLOv8 run in this same environment. Off by default since it
   trades some RAM/disk for speed; explicitly opt-in.
None of this changes epochs, image size, augmentation, or model architecture.
-----------------------------------------------------------------------------

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

import yaml
from ultralytics import RTDETR

sys.path.append(str(Path(__file__).resolve().parent.parent))  # -> training/
from common.metrics_utils import save_metrics


def preflight_check_yolo_yaml(data_yaml_path: str, required_splits=("train", "val", "test")) -> None:
    """Fail fast (seconds) instead of failing after hours of training.

    Verifies the data.yaml has every required split key AND that the
    resolved image directory exists and contains at least one image, for
    all of `required_splits`. This is exactly the class of bug that hit the
    YOLOv8 run in this project: training ran for the full 2 hours and only
    then crashed on `model.val(split="test")` because the data.yaml being
    used (a stale/committed copy) didn't map "test" to a real path.
    """
    yaml_path = Path(data_yaml_path)
    if not yaml_path.exists():
        raise SystemExit(f"[preflight] data.yaml not found at '{yaml_path}'.")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    base = Path(data.get("path", yaml_path.parent)).expanduser()
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    problems = []
    for split in required_splits:
        if split not in data or not data[split]:
            problems.append(f"  - '{split}' key missing from data.yaml entirely")
            continue
        split_dir = (base / data[split]).resolve()
        if not split_dir.exists():
            problems.append(f"  - '{split}' -> '{split_dir}' does not exist")
            continue
        has_image = any(p.suffix.lower() in img_exts for p in split_dir.iterdir())
        if not has_image:
            problems.append(f"  - '{split}' -> '{split_dir}' exists but contains no images")

    if problems:
        raise SystemExit(
            "[preflight] data.yaml failed validation BEFORE training started:\n"
            + "\n".join(problems)
            + f"\n\nyaml resolved 'path' = {base}\n"
            "This usually means the data.yaml in use is stale (e.g. committed to the repo from an "
            "earlier session) rather than freshly generated in THIS session, or images are missing/"
            "symlinks are broken. Re-run the Phase A data_prep cells fresh in this session and point "
            "--data at the freshly written data.yaml, rather than a previously-committed copy."
        )
    print(f"[preflight] data.yaml OK -- {', '.join(required_splits)} all resolve to non-empty dirs "
          f"(path={base})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to data.yaml from data_prep/voc_to_yolo.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--weights", default="rtdetr-l.pt", help="pretrained checkpoint to fine-tune from")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8, help="RT-DETR is heavier than YOLOv8s -> smaller default batch for T4")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--workers", type=int, default=8, help="dataloader workers (ultralytics default is 8)")
    ap.add_argument("--cache", default=None, choices=[None, "ram", "disk"],
                     help="cache decoded images after the first epoch to avoid re-reading from disk "
                          "every epoch. Worth trying given the slow-image-access warning seen in this "
                          "project's YOLOv8 run. Off by default (uses more RAM/disk).")
    ap.add_argument("--skip_preflight", action="store_true",
                     help="skip the fast data.yaml sanity check (not recommended)")
    args = ap.parse_args()

    if not args.skip_preflight:
        preflight_check_yolo_yaml(args.data)

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
        workers=args.workers,
        cache=args.cache,
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
