"""
voc_to_coco.py
Convert a split list (produced by download_and_split.py) into a single COCO-format
JSON file: {images: [...], annotations: [...], categories: [...]}.

Used by: Faster R-CNN (torchvision, via a small CocoDetection wrapper) and
RT-DETR (HuggingFace `RTDetrForObjectDetection`, which expects COCO-style targets).

Usage:
    python voc_to_coco.py --split splits/train.txt --out coco/train.json
    python voc_to_coco.py --split splits/val.txt   --out coco/val.json
    python voc_to_coco.py --split splits/test.txt  --out coco/test.json

By default `difficult` objects are INCLUDED (kept for train, since VOC training
conventionally uses all boxes) but you can exclude them with --exclude_difficult
(commonly done when scoring, following the original VOC devkit convention).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

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


def convert(split_path: str, out_path: str, exclude_difficult: bool = False):
    entries = load_split(split_path)

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(VOC_CLASSES)]
    name_to_catid = {c["name"]: c["id"] for c in categories}

    images, annotations = [], []
    ann_id = 1
    n_skipped_boxes = 0

    for img_idx, e in enumerate(entries, start=1):
        # entries already only contain images with >=1 selected-class object
        # (filtered upstream in download_and_split.py), but we filter again
        # here defensively so this script is safe to run standalone too.
        ann = filter_to_selected_classes(parse_voc_xml(e["xml_path"]))
        images.append({
            "id": img_idx,
            "file_name": Path(e["img_path"]).name,
            "width": ann.width,
            "height": ann.height,
            # keep the absolute path too so downstream loaders don't need to guess a root
            "abs_path": e["img_path"],
        })
        for obj in ann.objects:
            if exclude_difficult and obj.difficult:
                n_skipped_boxes += 1
                continue
            w = obj.xmax - obj.xmin
            h = obj.ymax - obj.ymin
            annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": name_to_catid[obj.name],
                "bbox": [round(obj.xmin, 2), round(obj.ymin, 2), round(w, 2), round(h, 2)],  # COCO: xywh
                "area": round(w * h, 2),
                "iscrowd": 0,
                "difficult": int(obj.difficult),
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco, f)

    print(f"[{split_path}] -> {out_path}")
    print(f"  images={len(images)}  annotations={len(annotations)}  "
          f"skipped_difficult={n_skipped_boxes}  categories={len(categories)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="path to train.txt/val.txt/test.txt")
    ap.add_argument("--out", required=True, help="output COCO json path")
    ap.add_argument("--exclude_difficult", action="store_true")
    args = ap.parse_args()
    convert(args.split, args.out, exclude_difficult=args.exclude_difficult)


if __name__ == "__main__":
    main()
