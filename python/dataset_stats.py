"""
dataset_stats.py
Compute per-split statistics and save English-annotated plots, ready to paste
straight into the LaTeX/Word report (Requirement 1: "describe how the dataset
is divided into training, validation, and test sets").

Usage:
    python dataset_stats.py --splits_dir splits --out_dir report_assets
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xml_utils import parse_voc_xml, filter_to_selected_classes, VOC_CLASSES


def load_split(split_path: str | Path) -> list[dict]:
    entries = []
    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_id, year, img_path, xml_path = line.split(" ", 3)
            entries.append({"id": img_id, "year": year, "img_path": img_path, "xml_path": xml_path})
    return entries


def compute_split_stats(entries: list[dict]) -> dict:
    class_counts = defaultdict(int)
    n_objects = 0
    objects_per_image = []
    for e in entries:
        # only count our 6 target classes; images may still contain other
        # VOC objects in their XML (e.g. a "cat" alongside a "car") which
        # are simply not annotated/detected for this project's topic.
        ann = filter_to_selected_classes(parse_voc_xml(e["xml_path"]))
        objects_per_image.append(len(ann.objects))
        for obj in ann.objects:
            class_counts[obj.name] += 1
            n_objects += 1
    return {
        "n_images": len(entries),
        "n_objects": n_objects,
        "avg_objects_per_image": (n_objects / len(entries)) if entries else 0.0,
        "class_counts": dict(class_counts),
    }


def plot_split_sizes(stats: dict, out_path: Path):
    splits = list(stats.keys())
    n_images = [stats[s]["n_images"] for s in splits]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(splits, n_images, color=["#4C72B0", "#55A868", "#C44E52"][: len(splits)])
    ax.set_title("Number of images per split (PASCAL VOC 07+12)")
    ax.set_ylabel("Number of images")
    for b, v in zip(bars, n_images):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_class_distribution(stats: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    x = range(len(VOC_CLASSES))
    colors = {"train": "#4C72B0", "val": "#55A868", "test": "#C44E52"}
    for i, split in enumerate(stats.keys()):
        counts = [stats[split]["class_counts"].get(c, 0) for c in VOC_CLASSES]
        offset = (i - (len(stats) - 1) / 2) * width
        ax.bar([xi + offset for xi in x], counts, width=width, label=split, color=colors.get(split))
    ax.set_xticks(list(x))
    ax.set_xticklabels(VOC_CLASSES, rotation=60, ha="right")
    ax.set_ylabel("Number of object instances")
    ax.set_title("Class distribution across train / val / test splits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="splits")
    ap.add_argument("--out_dir", default="report_assets")
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split_name in ["train", "val", "test"]:
        split_path = splits_dir / f"{split_name}.txt"
        if not split_path.exists():
            print(f"  (skipping {split_name}: not found)")
            continue
        entries = load_split(split_path)
        all_stats[split_name] = compute_split_stats(entries)

    summary_path = out_dir / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"wrote {summary_path}")

    for split, s in all_stats.items():
        print(f"{split:6s}: {s['n_images']:5d} images, {s['n_objects']:6d} objects, "
              f"avg {s['avg_objects_per_image']:.2f} objects/image")

    if all_stats:
        plot_split_sizes(all_stats, out_dir / "split_sizes.png")
        plot_class_distribution(all_stats, out_dir / "class_distribution.png")


if __name__ == "__main__":
    main()
