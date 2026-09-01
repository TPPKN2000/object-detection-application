"""
inference.py
Unified detector interface so app.py doesn't need to know the differences
between torchvision (Faster R-CNN) and ultralytics (YOLOv8 / RT-DETR) APIs.

Every detector's .predict(pil_image, conf_thresh, device) returns the SAME
format: [{"box": [x1, y1, x2, y2], "label": str, "score": float}, ...]

ZeroGPU note: models are constructed on CPU and stay there at rest -- HF
Spaces' ZeroGPU only attaches a real GPU for the duration of a function
decorated with @spaces.GPU (see app.py). `device` is therefore passed into
predict() per-call rather than fixed at construction time, since a different
physical GPU may be assigned on each call.

IMPORTANT: VOC_CLASSES order below MUST exactly match the order used during
training (data_prep/xml_utils.py::VOC_CLASSES) -- Faster R-CNN's output label
indices (1..6, 0=background) are positional and carry no class names in the
checkpoint itself, unlike YOLOv8/RT-DETR (via ultralytics) which embed class
names in the .pt file directly.
"""
from __future__ import annotations

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

VOC_CLASSES = ["person", "bicycle", "car", "motorbike", "bus", "train"]


class FasterRCNNDetector:
    def __init__(self, weights_path: str, min_size: int = 600, max_size: int = 1000):
        model = fasterrcnn_resnet50_fpn_v2(weights=None, min_size=min_size, max_size=max_size)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(VOC_CLASSES) + 1)  # +1 background
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        self.model = model  # stays on CPU until predict() moves it inside a GPU-allocated call

    @torch.no_grad()
    def predict(self, pil_image, conf_thresh: float = 0.5, device: str = "cpu") -> list[dict]:
        self.model.to(device)
        img_t = TF.pil_to_tensor(pil_image)
        img_t = TF.convert_image_dtype(img_t, torch.float32).to(device)
        output = self.model([img_t])[0]

        boxes = output["boxes"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()

        results = []
        for box, label, score in zip(boxes, labels, scores):
            if score < conf_thresh:
                continue
            results.append({
                "box": [float(v) for v in box],
                "label": VOC_CLASSES[label - 1],  # label 0 is background, never predicted as a box
                "score": float(score),
            })
        return results


class UltralyticsDetector:
    """Shared wrapper for YOLOv8 and RT-DETR -- both are exposed through the
    same ultralytics `Model` API (.predict()), just different constructor
    classes. Class names are read from the checkpoint itself (no need for
    VOC_CLASSES here), so this stays correct even if training class order
    ever changes independently of the Faster R-CNN pipeline.
    """

    def __init__(self, weights_path: str, family: str):
        if family == "yolo":
            from ultralytics import YOLO
            self.model = YOLO(weights_path)
        elif family == "rtdetr":
            from ultralytics import RTDETR
            self.model = RTDETR(weights_path)
        else:
            raise ValueError(f"unknown family: {family}")

    def predict(self, pil_image, conf_thresh: float = 0.5, device: str = "cpu") -> list[dict]:
        result = self.model.predict(pil_image, conf=conf_thresh, device=device, verbose=False)[0]
        names = result.names
        out = []
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            score = float(box.conf[0])
            out.append({"box": xyxy, "label": names[cls_id], "score": score})
        return out
