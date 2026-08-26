"""
training/faster_rcnn/train_faster_rcnn.py

Fine-tune torchvision Faster R-CNN (ResNet50-FPN v2, COCO-pretrained) on the
"person, bicycle, car, motorbike, bus, train" traffic/urban-security subset
of PASCAL VOC (see data_prep/).

Paper (why this model): Ren, He, Girshick, Sun, "Faster R-CNN: Towards
Real-Time Object Detection with Region Proposal Networks", NeurIPS 2015 --
the classic two-stage CNN detector, representing the "CNN-based" family in
this project's 3-way comparison.

We fine-tune from COCO-pretrained weights (not from scratch) specifically so
training converges in a handful of epochs on a single Colab T4.

--- PATCH NOTES (speed/robustness pass, same training scope/quality) -------
1. PREFLIGHT CHECK: right after datasets are built (before any GPU work),
   we open the first few images of train/val/test to confirm paths resolve.
   This turns a "burn 2h of T4 then crash at eval" failure into a "fail in
   under 5 seconds" failure -- directly targeting the symlink/stale-path
   issue that hit the YOLOv8 run in this project.
2. num_workers default raised 2 -> 4 (still overridable), + persistent_workers
   + prefetch_factor, to fix the CPU-bound data-loading bottleneck that was
   capping throughput at ~6.2-6.5 img/s instead of the 9-14 img/s the
   original 8-batch/600-1000px config should reach on a T4.
3. cudnn.benchmark = True on CUDA (free, usually net positive even with the
   per-batch size variation from aspect-ratio-preserving resize).
4. Removed the per-ITERATION .item() sync for the running loss -- only
   syncs at print_freq boundaries and at epoch end now.
5. Optional --channels_last flag (default OFF): experimental, torchvision
   detection models aren't guaranteed to benefit/behave identically, so this
   is opt-in, not a default.
None of these change epoch count, image resolution, batch composition, or
any other factor that affects the trained model's final accuracy.
-----------------------------------------------------------------------------

Usage:
    python train_faster_rcnn.py \
        --train_json /content/coco/train.json \
        --val_json   /content/coco/val.json \
        --test_json  /content/coco/test.json \
        --output_dir /content/runs/faster_rcnn \
        --epochs 15 --batch_size 4 --lr 0.005
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

sys.path.append(str(Path(__file__).resolve().parent.parent))  # -> training/
from common.coco_dataset import CocoStyleDetection, collate_fn
from common.transforms import build_transforms
from common.metrics_utils import benchmark_fps, count_params, save_metrics


def get_model(num_classes_with_bg: int, pretrained: bool = True, min_size: int = 800, max_size: int = 1333):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, min_size=min_size, max_size=max_size)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    return model


def preflight_check(name: str, dataset: "CocoStyleDetection", n_sample: int = 5) -> None:
    """Fail fast (seconds) instead of failing after hours of training.
    Opens the first n_sample images of a split via PIL and confirms the
    abs_path actually resolves to a readable image. This is exactly the
    class of bug that made the YOLOv8 run in this project burn its full
    training time only to crash at the final test-set eval because of a
    stale/broken data.yaml -> image path chain.
    """
    if len(dataset) == 0:
        raise SystemExit(f"[preflight] {name}: dataset is EMPTY (0 images) -- "
                          f"check the json/path passed in before starting training.")
    n = min(n_sample, len(dataset))
    for i in range(n):
        img_info = dataset.images[dataset.image_ids[i]]
        abs_path = img_info["abs_path"]
        try:
            with Image.open(abs_path) as im:
                im.verify()
        except Exception as e:
            raise SystemExit(
                f"[preflight] {name}: FAILED to open image #{i} at '{abs_path}' ({e}).\n"
                f"This almost always means the dataset was prepared in a DIFFERENT session/path "
                f"than the one this script is running in now (e.g. a symlink target that no longer "
                f"exists, or a stale committed json/yaml). Regenerate coco/yolo_data fresh in THIS "
                f"session (re-run the Phase A cells) before training, rather than reusing a "
                f"previously-committed copy."
            )
    print(f"[preflight] {name}: OK ({len(dataset)} images, sampled {n} readable)")


@torch.no_grad()
def evaluate(model, data_loader, device) -> dict:
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy")
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        preds = [{k: v.detach().cpu() for k, v in o.items()} for o in outputs]
        gts = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        metric.update(preds, gts)
    result = metric.compute()
    # keep only scalar entries (drop e.g. per-class tensors) for clean JSON
    return {k: float(v) for k, v in result.items() if getattr(v, "numel", lambda: 1)() == 1}


def train_one_epoch(model, optimizer, data_loader, device, epoch: int, scaler=None, print_freq: int = 20) -> float:
    model.train()
    n_batches = len(data_loader)
    use_amp = scaler is not None
    epoch_start = time.time()

    # Accumulate loss as a GPU tensor and only sync to host (.item()) at
    # print boundaries / epoch end, instead of every single iteration.
    running_loss = torch.zeros((), device=device)

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.detach()

        if i % print_freq == 0:
            loss_value = loss.item()  # only sync here, not every iteration
            elapsed = time.time() - epoch_start
            img_per_sec = ((i + 1) * data_loader.batch_size) / max(elapsed, 1e-6)
            print(f"  epoch {epoch} iter {i}/{n_batches} loss={loss_value:.4f} "
                  f"({img_per_sec:.1f} img/s -- if this is <10 img/s on a T4, something is wrong, see README)")

    return (running_loss / max(1, n_batches)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--val_json", required=True)
    ap.add_argument("--test_json", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--num_workers", type=int, default=4,
                     help="raised from the old default of 2: on the T4 img/s logged during the last "
                          "run, throughput plateaued at ~6.2-6.5 img/s (vs. 9-14 expected) which points "
                          "to a CPU-bound data-loading bottleneck at 2 workers. Check `!nproc` in Colab "
                          "to see how many are actually available and raise further if you have >4 cores.")
    ap.add_argument("--prefetch_factor", type=int, default=4,
                     help="only used when --num_workers > 0")
    ap.add_argument("--channels_last", action="store_true",
                     help="EXPERIMENTAL: convert model+inputs to channels_last memory format for "
                          "potential Tensor Core speedup on the T4. Off by default -- torchvision "
                          "detection models are not guaranteed to fully support/benefit from this, "
                          "test it on a short run before trusting it for the full training.")
    ap.add_argument("--skip_preflight", action="store_true",
                     help="skip the fast dataset-path sanity check (not recommended)")
    ap.add_argument("--no_pretrained", action="store_true",
                     help="train from scratch instead of COCO-pretrained weights (NOT recommended on T4)")
    ap.add_argument("--min_size", type=int, default=800, help="shorter-side resize target (torchvision default 800)")
    ap.add_argument("--max_size", type=int, default=1333, help="longer-side cap (torchvision default 1333); "
                     "lowering both reduces T4 memory/time at some accuracy cost")
    ap.add_argument("--amp", type=lambda x: x.lower() != "false", default=True,
                     help="mixed-precision training (default on) -- roughly 1.5-2x faster on a T4's Tensor Cores")
    ap.add_argument("--allow_cpu", action="store_true",
                     help="explicitly allow training on CPU (VERY slow, ~50-100x slower than a T4). "
                          "Without this flag the script exits immediately if no GPU is detected, "
                          "instead of silently training on CPU for hours.")
    ap.add_argument("--patience", type=int, default=5,
                     help="stop early if val mAP@0.5 doesn't improve for this many epochs "
                          "(set to a number >= --epochs to disable). Mirrors ultralytics' patience "
                          "so all 3 pipelines behave consistently under a tight time budget.")
    ap.add_argument("--resume", default=None,
                     help="path to a checkpoint.pt written by a previous (interrupted) run of THIS "
                          "script. Restores model/optimizer/scheduler/scaler state and continues from "
                          "the next epoch, so a Colab disconnect only costs the current in-progress "
                          "epoch, not the whole run. Pass the SAME --train_json/--val_json/--test_json/"
                          "--min_size/--max_size/etc used originally -- only --epochs and --patience are "
                          "safe to change on resume.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cpu" and not args.allow_cpu:
        raise SystemExit(
            "\n*** No GPU detected (torch.cuda.is_available() == False) ***\n"
            "Training Faster R-CNN on CPU is ~50-100x slower than a T4 and will take many hours.\n"
            "This almost always means the Colab runtime type isn't actually set to a GPU yet --\n"
            "go to Runtime > Change runtime type > T4 GPU, then Runtime > Restart session, then re-run\n"
            "the notebook from the clone/pip-install cells (a runtime change requires a restart to take effect).\n"
            "If you really want to proceed on CPU anyway (e.g. a tiny debug run), pass --allow_cpu."
        )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # free speedup for the (mostly-)fixed input size range

    train_ds = CocoStyleDetection(args.train_json, transforms=build_transforms(train=True))
    val_ds = CocoStyleDetection(args.val_json, transforms=build_transforms(train=False))
    test_ds = CocoStyleDetection(args.test_json, transforms=build_transforms(train=False))
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"num_classes(+bg)={train_ds.num_classes_with_background}")

    if not args.skip_preflight:
        preflight_check("train", train_ds)
        preflight_check("val", val_ds)
        preflight_check("test", test_ds)

    loader_kwargs = dict(num_workers=args.num_workers, collate_fn=collate_fn,
                          pin_memory=(device.type == "cuda"))
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = get_model(train_ds.num_classes_with_background, pretrained=not args.no_pretrained,
                       min_size=args.min_size, max_size=args.max_size)
    model.to(device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)
        print("channels_last: ON (experimental)")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 2), gamma=0.1)
    scaler = GradScaler(device.type, enabled=(args.amp and device.type == "cuda"))
    print(f"AMP enabled: {scaler.is_enabled()}")

    best_map50 = -1.0
    epochs_since_improvement = 0
    best_model_state = None
    history = []
    start_epoch = 0
    prior_train_time_min = 0.0

    if args.resume:
        print(f"resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_map50 = ckpt["best_map50"]
        epochs_since_improvement = ckpt["epochs_since_improvement"]
        best_model_state = ckpt.get("best_model")
        history = ckpt.get("history", [])
        prior_train_time_min = ckpt.get("cumulative_train_time_min", 0.0)
        print(f"resumed at epoch {start_epoch}, best_map50 so far={best_map50:.4f}, "
              f"prior training time={prior_train_time_min:.1f} min")

    train_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler=scaler)
        lr_scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        map50 = val_metrics.get("map_50", -1.0)
        print(f"epoch {epoch}: train_loss={avg_loss:.4f} val_map50={map50:.4f}")
        history.append({"epoch": epoch, "train_loss": avg_loss, **val_metrics})
        if map50 > best_map50:
            best_map50 = map50
            epochs_since_improvement = 0
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_model_state, out_dir / "best.pt")
        else:
            epochs_since_improvement += 1

        # Full state checkpoint written EVERY epoch (not just on improvement) so a Colab
        # disconnect never costs more than the current in-progress epoch. This single file
        # is enough to --resume in a fresh session (it embeds the best weights too, in case
        # best.pt itself wasn't downloaded separately).
        torch.save({
            "model": model.state_dict(),
            "best_model": best_model_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "best_map50": best_map50,
            "epochs_since_improvement": epochs_since_improvement,
            "history": history,
            "cumulative_train_time_min": prior_train_time_min + (time.time() - train_start) / 60,
        }, out_dir / "checkpoint.pt")

        if epochs_since_improvement >= args.patience:
            print(f"early stopping: val_map50 hasn't improved for {args.patience} epochs "
                  f"(best={best_map50:.4f} at earlier epoch)")
            break
    train_time_min = prior_train_time_min + (time.time() - train_start) / 60

    torch.save(model.state_dict(), out_dir / "last.pt")
    # evaluate the BEST checkpoint (by val mAP50) on the held-out test set. Prefer the
    # in-memory copy (works even right after a --resume that finished with 0 new epochs);
    # fall back to reading best.pt from disk otherwise.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print("test metrics (best checkpoint):", test_metrics)

    # single-image (batch=1) FPS/latency benchmark on the same device
    model.eval()
    n_sample = min(20, len(test_ds))
    sample_imgs = [test_ds[i][0].to(device) for i in range(n_sample)]

    def predict_fn(img):
        with torch.no_grad():
            return model([img])

    fps, latency_ms = benchmark_fps(predict_fn, sample_imgs, device)

    results = {
        "model_name": "Faster R-CNN (ResNet50-FPN v2)",
        "paper": "Ren et al., Faster R-CNN, NeurIPS 2015",
        "num_params": count_params(model),
        "train_time_min": train_time_min,
        "best_val_map50": best_map50,
        "test_metrics": test_metrics,
        "fps": fps,
        "latency_ms": latency_ms,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
    }
    save_metrics(out_dir / "metrics.json", results)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Done. Results in", out_dir)


if __name__ == "__main__":
    main()
