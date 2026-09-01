"""
app.py — Gradio demo (ZeroGPU) cho đồ án "Phát hiện người và phương tiện
giao thông phục vụ giám sát an ninh đô thị".

Cho phép chọn 1 trong 3 model (Faster R-CNN / YOLOv8 / RT-DETR) để chạy detect
trên ảnh do người dùng upload. Model nào chưa có file weight trong weights/
sẽ tự động bị loại khỏi dropdown (kèm cảnh báo), thay vì làm sập app.

ZeroGPU: models được load lên CPU lúc khởi động app (không cần GPU), chỉ
thực sự dùng GPU bên trong hàm `detect()` được đánh dấu @spaces.GPU -- đây
là hàm duy nhất HF Spaces cấp phát GPU thật (H200 slice) khi được gọi, và
thu hồi ngay sau khi hàm return. Xem: https://huggingface.co/docs/hub/spaces-zerogpu

Cách thêm weight: xem weights/README.md
"""
from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import spaces
import torch
from PIL import Image, ImageDraw, ImageFont

from inference import FasterRCNNDetector, UltralyticsDetector, VOC_CLASSES

WEIGHTS_DIR = Path(__file__).parent / "weights"

MODEL_CONFIGS = {
    "Faster R-CNN (ResNet50-FPN v2)": {
        "file": WEIGHTS_DIR / "faster_rcnn_best.pt",
        "kind": "faster_rcnn",
        "paper": "Ren et al., NeurIPS 2015",
    },
    "YOLOv8s": {
        "file": WEIGHTS_DIR / "yolov8_best.pt",
        "kind": "yolo",
        "paper": "Ultralytics (YOLO lineage, Redmon et al.)",
    },
    "RT-DETR-L": {
        "file": WEIGHTS_DIR / "rtdetr_best.pt",
        "kind": "rtdetr",
        "paper": "Lv et al., 2023 — DETRs Beat YOLOs on Real-Time Object Detection",
    },
}

# fixed color per class so the same class always looks the same across models
COLORS = {
    "person": "#e6194b",
    "bicycle": "#3cb44b",
    "car": "#4363d8",
    "motorbike": "#f58231",
    "bus": "#911eb4",
    "train": "#42d4f4",
}

_detector_cache: dict[str, object] = {}


def get_detector(model_name: str):
    """Lazily construct + cache each detector (on CPU) the first time it's
    selected, so switching models in the dropdown doesn't reload from disk
    every single click."""
    if model_name not in _detector_cache:
        cfg = MODEL_CONFIGS[model_name]
        if cfg["kind"] == "faster_rcnn":
            _detector_cache[model_name] = FasterRCNNDetector(str(cfg["file"]))
        else:
            _detector_cache[model_name] = UltralyticsDetector(str(cfg["file"]), family=cfg["kind"])
    return _detector_cache[model_name]


def draw_boxes(image: Image.Image, detections: list[dict]) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()  # older Pillow without `size` kwarg

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = COLORS.get(det["label"], "#ffffff")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f'{det["label"]} {det["score"]:.2f}'
        text_bbox = draw.textbbox((x1, y1), text, font=font)
        text_h = text_bbox[3] - text_bbox[1]
        draw.rectangle([x1, y1 - text_h - 4, text_bbox[2] + 4, y1], fill=color)
        draw.text((x1 + 2, y1 - text_h - 3), text, fill="white", font=font)
    return img


@spaces.GPU(duration=30)
def detect(model_name: str, image: Image.Image, conf_thresh: float):
    if image is None:
        return None, [], "⬆️ Hãy tải ảnh lên trước."
    if not model_name:
        return None, [], "⚠️ Chưa có model nào khả dụng (thiếu file weight)."

    detector = get_detector(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.time()
    detections = detector.predict(image, conf_thresh=conf_thresh, device=device)
    elapsed_ms = (time.time() - start) * 1000

    annotated = draw_boxes(image, detections)
    table = [
        [d["label"], round(d["score"], 3), ", ".join(f"{v:.0f}" for v in d["box"])]
        for d in sorted(detections, key=lambda d: -d["score"])
    ]
    info = f"⏱️ Thời gian inference: {elapsed_ms:.0f} ms  |  🎯 Số đối tượng phát hiện: {len(detections)}"
    if not detections:
        info += "\n\nKhông phát hiện đối tượng nào ở ngưỡng confidence hiện tại — thử giảm ngưỡng."
    return annotated, table, info


available_models = [name for name, cfg in MODEL_CONFIGS.items() if cfg["file"].exists()]
missing_models = [name for name, cfg in MODEL_CONFIGS.items() if not cfg["file"].exists()]

with gr.Blocks(title="Traffic & Person Detection") as demo:
    gr.Markdown("# 🚦 Phát hiện người & phương tiện giao thông đô thị")
    gr.Markdown(
        "Đồ án Statistical Machine Learning — so sánh Faster R-CNN / YOLOv8 / RT-DETR "
        "trên subset 6 lớp của PASCAL VOC (`person, bicycle, car, motorbike, bus, train`). "
        "Chạy trên ZeroGPU."
    )

    if not available_models:
        gr.Markdown(
            "### ⚠️ Chưa có file weight nào trong `weights/`.\n"
            "Xem `weights/README.md` để biết cách thêm model."
        )
    else:
        if missing_models:
            gr.Markdown("⚠️ Model chưa sẵn sàng (thiếu file weight): " + ", ".join(missing_models))

        with gr.Row():
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    choices=available_models, value=available_models[0], label="Chọn model"
                )
                paper_md = gr.Markdown(f"📄 {MODEL_CONFIGS[available_models[0]]['paper']}")
                conf_slider = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Ngưỡng confidence")
                image_in = gr.Image(type="pil", label="Ảnh đầu vào")
                btn = gr.Button("Detect", variant="primary")
                gr.Markdown("6 lớp: " + ", ".join(VOC_CLASSES))

            with gr.Column(scale=1):
                image_out = gr.Image(label="Kết quả")
                info_out = gr.Markdown()
                table_out = gr.Dataframe(
                    headers=["Lớp", "Confidence", "Box (x1, y1, x2, y2)"],
                    label="Chi tiết",
                )

        model_dd.change(
            fn=lambda name: f"📄 {MODEL_CONFIGS[name]['paper']}",
            inputs=model_dd,
            outputs=paper_md,
        )
        btn.click(fn=detect, inputs=[model_dd, image_in, conf_slider], outputs=[image_out, table_out, info_out])
        image_in.change(fn=detect, inputs=[model_dd, image_in, conf_slider], outputs=[image_out, table_out, info_out])

if __name__ == "__main__":
    demo.launch()
