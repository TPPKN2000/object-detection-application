"""
voc_to_yolo.py
Convert a split list (produced by download_and_split.py) into YOLO-format:
  <out_dir>/images/<split>/xxxx.jpg   (symlinked, no copy -> saves disk/time on Colab)
  <out_dir>/labels/<split>/xxxx.txt   (class_idx cx cy w h, all normalized 0-1)
Also writes <out_dir>/data.yaml ready for `ultralytics` training.

Usage:
    python voc_to_yolo.py --splits_dir splits --out_dir yolo_data
        (expects splits/train.txt, splits/val.txt, splits/test.txt to exist)
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import yaml  # PyYAML - available by default on Colab; falls back to manual dump if missing

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


def convert_split(split_path: str, out_dir: Path, split_name: str, exclude_difficult: bool = False):
    entries = load_split(split_path)
    img_out = out_dir / "images" / split_name
    lbl_out = out_dir / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    class_to_id = {name: i for i, name in enumerate(VOC_CLASSES)}
    n_boxes, n_skipped = 0, 0

    for e in entries:
        # defensive re-filter, see comment in voc_to_coco.py
        ann = filter_to_selected_classes(parse_voc_xml(e["xml_path"]))
        # Unique filename across years (07/12 share numeric ids)
        stem = f"{e['year']}_{e['id']}"
        img_src = Path(e["img_path"]).resolve()
        img_dst = img_out / f"{stem}.jpg"
        if not img_dst.exists():
            try:
                os.symlink(img_src, img_dst)
            except (OSError, FileExistsError):
                # symlinks unsupported (rare) -> fall back to copy
                import shutil
                shutil.copy(img_src, img_dst)

        lines = []
        for obj in ann.objects:
            if exclude_difficult and obj.difficult:
                n_skipped += 1
                continue
            cx = (obj.xmin + obj.xmax) / 2 / ann.width
            cy = (obj.ymin + obj.ymax) / 2 / ann.height
            w = (obj.xmax - obj.xmin) / ann.width
            h = (obj.ymax - obj.ymin) / ann.height
            cls_id = class_to_id[obj.name]
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_boxes += 1

        with open(lbl_out / f"{stem}.txt", "w") as f:
            f.write("\n".join(lines))

    print(f"[{split_name}] images={len(entries)} boxes={n_boxes} skipped_difficult={n_skipped}")


def write_data_yaml(out_dir: Path):
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(VOC_CLASSES),
        "names": VOC_CLASSES,
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"wrote {yaml_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", default="splits")
    ap.add_argument("--out_dir", default="yolo_data")
    ap.add_argument("--exclude_difficult", action="store_true")
    args = ap.parse_args()

    splits_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)

    for split_name in ["train", "val", "test"]:
        split_path = splits_dir / f"{split_name}.txt"
        if split_path.exists():
            convert_split(str(split_path), out_dir, split_name, exclude_difficult=args.exclude_difficult)
        else:
            print(f"  (skipping {split_name}: {split_path} not found)")

    write_data_yaml(out_dir)


if __name__ == "__main__":
    main()
