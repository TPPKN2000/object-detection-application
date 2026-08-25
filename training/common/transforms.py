"""
common/transforms.py
Minimal (image, target)-aware transform pipeline for the Faster R-CNN loop --
plain torchvision.transforms only operates on the image, but we also need to
flip/adjust the ground-truth boxes in `target` consistently.

This is the same small pattern used in torchvision's official object-detection
finetuning tutorial (PILToTensor + ConvertImageDtype + a target-aware
RandomHorizontalFlip), reimplemented here in ~30 lines so this repo has no
dependency on the (separate, not-pip-installable) torchvision `references/`
folder.
"""
from __future__ import annotations
import random

import torch
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        return image, target


class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            _, _, w = image.shape  # image is CHW at this point
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


def build_transforms(train: bool) -> Compose:
    ts = [PILToTensor(), ConvertImageDtype(torch.float32)]
    if train:
        ts.append(RandomHorizontalFlip(0.5))
    return Compose(ts)
