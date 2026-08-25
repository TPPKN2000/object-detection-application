"""
training/compare_results.py

Gather metrics.json from the 3 training runs and produce a single comparison
table (Markdown + JSON) plus bar charts, ready to paste into the report
(Requirement 1: so sanh do chinh xac, toc do, do phuc tap, kha nang ung dung).

Usage:
    python compare_results.py \
        --runs faster_rcnn=/content/runs/faster_rcnn/metrics.json \
               yolov8=/content/runs/yolov8/metrics.json \
               rtdetr=/content/runs/rtdetr/metrics.json \
        --out_dir /content/report_assets
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]


def load_runs(run_args: list[str]) -> dict:
    runs = {}
    for item in run_args:
        name, path = item.split("=", 1)
        with open(path) as f:
            runs[name] = json.load(f)
    return runs


def extract_map(run: dict) -> tuple[float, float]:
    """Best-effort extraction of mAP50 / mAP50-95, handling the differing key
    names torchmetrics (Faster R-CNN) vs ultralytics (YOLOv8 / RT-DETR) use."""
    tm = run.get("test_metrics", {})
    map50 = tm.get("map_50", tm.get("metrics/mAP50(B)", 0.0)) or 0.0
    map5095 = tm.get("map", tm.get("metrics/mAP50-95(B)", 0.0)) or 0.0
    return float(map50), float(map5095)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name=path/to/metrics.json pairs")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.runs)

    rows = []
    for name, run in runs.items():
        map50, map5095 = extract_map(run)
        rows.append({
            "name": name,
            "model_name": run.get("model_name", name),
            "params_M": run.get("num_params", 0) / 1e6,
            "train_time_min": run.get("train_time_min", 0.0),
            "map50": map50,
            "map50_95": map5095,
            "fps": run.get("fps") or 0.0,
            "latency_ms": run.get("latency_ms") or 0.0,
        })

    md_path = out_dir / "comparison_table.md"
    with open(md_path, "w") as f:
        f.write("| Model | Params (M) | Train time (min) | mAP@0.5 | mAP@[.5:.95] | FPS | Latency (ms) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['model_name']} | {r['params_M']:.1f} | {r['train_time_min']:.1f} | "
                f"{r['map50']:.3f} | {r['map50_95']:.3f} | {r['fps']:.1f} | {r['latency_ms']:.1f} |\n"
            )
    print(f"wrote {md_path}")

    with open(out_dir / "comparison_raw.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {out_dir / 'comparison_raw.json'}")

    names = [r["model_name"] for r in rows]
    for metric_key, title, ylabel, fname in [
        ("map50", "mAP@0.5 on held-out test set", "mAP@0.5", "compare_map50.png"),
        ("fps", "Inference speed (batch=1, same GPU)", "FPS", "compare_fps.png"),
        ("params_M", "Model size", "Params (millions)", "compare_params.png"),
        ("train_time_min", "Fine-tuning time (Colab T4)", "Minutes", "compare_train_time.png"),
    ]:
        vals = [r[metric_key] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(names, vals, color=COLORS[: len(names)])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        print(f"wrote {out_dir / fname}")


if __name__ == "__main__":
    main()
