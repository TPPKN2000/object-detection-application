"""
training/yolov8/train_yolov8.py

Fine-tune YOLOv8s (Ultralytics, COCO-pretrained) on the traffic/urban-security
VOC subset, using the YOLO-format dataset produced by data_prep/voc_to_yolo.py.

Why this model: represents the "YOLO family" (single-stage, anchor-free in
v8) branch of the 3-way comparison. yolov8s (small) is chosen over yolov8n/m
as the accuracy/speed middle ground that trains quickly on a single T4.

--- PATCH NOTES -------------------------------------------------------------
Same fix as train_rtdetr.py, applied here for consistency (and for any future
re-run of this script): a preflight check on data.yaml BEFORE model.train()
is called. This is a direct fix for what actually happened in this project --
the last run trained the full 50 epochs successfully (2h on a T4) and only
THEN crashed at `model.val(split="test")` because the data.yaml in use
mapped to a stale/incomplete path (missing "test" split). With this check,
that failure now happens in seconds, before any GPU time is spent.
--cache exposed too, given the "Slow image access detected" warning
ultralytics printed during the actual run in this environment.
------------------------------------------------------------------------------

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

import yaml
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent.parent))  # -> training/
from common.metrics_utils import save_metrics


def preflight_check_yolo_yaml(data_yaml_path: str, required_splits=("train", "val", "test")) -> None:
    """See train_rtdetr.py::preflight_check_yolo_yaml -- identical check,
    duplicated here (rather than imported) so this script stays runnable
    standalone with only `common/` as a shared dependency, matching the
    rest of this repo's style."""
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
    ap.add_argument("--weights", default="yolov8s.pt", help="pretrained checkpoint to fine-tune from")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15, help="early-stopping patience (epochs)")
    ap.add_argument("--workers", type=int, default=8, help="dataloader workers (ultralytics default is 8)")
    ap.add_argument("--cache", default=None, choices=[None, "ram", "disk"],
                     help="cache decoded images after the first epoch to avoid re-reading from disk "
                          "every epoch. Worth trying given the slow-image-access warning seen in this "
                          "project's actual run. Off by default (uses more RAM/disk).")
    ap.add_argument("--skip_preflight", action="store_true",
                     help="skip the fast data.yaml sanity check (not recommended)")
    args = ap.parse_args()

    if not args.skip_preflight:
        preflight_check_yolo_yaml(args.data)

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
        workers=args.workers,
        cache=args.cache,
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
