"""
common/coco_dataset.py
Minimal COCO-style torch Dataset for torchvision detection models.

Reads the custom COCO JSON produced by data_prep/voc_to_coco.py. That JSON
stores an absolute "abs_path" per image (see voc_to_coco.py), so this loader
needs no separate image-root argument -- it opens images directly.

Only used by the Faster R-CNN pipeline. YOLOv8 / RT-DETR read the YOLO-format
data produced by data_prep/voc_to_yolo.py directly via `ultralytics`.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class CocoStyleDetection(Dataset):
    def __init__(self, json_path: str, transforms=None):
        with open(json_path) as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images.keys())

        self.anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            self.anns_by_image[ann["image_id"]].append(ann)

        self.categories = sorted(coco["categories"], key=lambda c: c["id"])
        # torchvision detection models reserve label 0 for background, so we
        # remap the original (1-based) category ids to a contiguous 1..N range.
        self.catid_to_label = {c["id"]: i + 1 for i, c in enumerate(self.categories)}
        self.label_to_name = {i + 1: c["name"] for i, c in enumerate(self.categories)}

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img = Image.open(img_info["abs_path"]).convert("RGB")

        anns = self.anns_by_image[img_id]
        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue  # defensive: skip degenerate boxes
            boxes.append([x, y, x + w, y + h])
            labels.append(self.catid_to_label[ann["category_id"]])
            areas.append(ann["area"])
            iscrowd.append(ann.get("iscrowd", 0))

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        areas_t = torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32)
        iscrowd_t = torch.as_tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([img_id]),
            "area": areas_t,
            "iscrowd": iscrowd_t,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    @property
    def num_classes_with_background(self) -> int:
        return len(self.categories) + 1


def collate_fn(batch):
    """torchvision detection models expect a list of images + list of target
    dicts per batch (variable number of boxes per image), not a stacked tensor."""
    return tuple(zip(*batch))
