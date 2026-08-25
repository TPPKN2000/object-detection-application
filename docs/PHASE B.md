# Pha A — Chuẩn bị dataset (PASCAL VOC 2007+2012, subset 6 lớp)

## Đề tài
**"Phát hiện người và phương tiện giao thông phục vụ giám sát an ninh đô thị"**

Dataset gốc: PASCAL VOC 2007+2012 (Everingham et al., IJCV 2010) — nhưng chỉ giữ lại
**6 lớp liên quan giao thông/an ninh đô thị**: `person, bicycle, car, motorbike, bus, train`
(đủ ≥5 lớp yêu cầu của đề bài). Ảnh nào **chỉ chứa** các lớp không liên quan (cat, sofa,
bottle, chair...) sẽ bị loại bỏ hoàn toàn khỏi dataset; ảnh có ít nhất 1 object thuộc
6 lớp trên thì được giữ lại, và chỉ annotation của 6 lớp đó được convert (annotation
của lớp khác trong cùng ảnh, nếu có, sẽ bị bỏ qua).

> Danh sách lớp là **single source of truth** đặt tại `VOC_CLASSES` trong `xml_utils.py`
> — mọi script khác import từ đây, nên muốn đổi/thêm lớp chỉ cần sửa 1 chỗ.

Bộ 5 script này đã được **validate end-to-end** bằng dữ liệu VOC giả (fake XML/JPEG
sinh tự động, đúng cấu trúc VOCdevkit, **cố tình trộn cả ảnh chỉ chứa lớp không liên
quan** để test đúng logic lọc subset) trước khi giao cho bạn — logic parse XML, lọc
subset, chia split, convert COCO/YOLO và vẽ thống kê đều đã kiểm chứng chạy đúng.

## Các file

| File | Vai trò |
|---|---|
| `xml_utils.py` | Parse file XML annotation của VOC → dict Python. Định nghĩa `VOC_CLASSES` (6 lớp subset, thứ tự cố định — **mọi script khác đều import từ đây**) và `filter_to_selected_classes()` để lọc annotation về đúng 6 lớp mục tiêu. |
| `download_and_split.py` | Tải VOC2007 (trainval+test) + VOC2012 (trainval) qua `torchvision`. **Lọc bỏ ảnh không chứa lớp nào thuộc subset** ngay khi gom pool. Gộp pool = VOC07-trainval + VOC12-trainval (đã lọc), tách **stratified 10%** làm `val` (stratify theo lớp chiếm đa số trong mỗi ảnh). VOC07-test (đã lọc) giữ nguyên làm `test`. Ghi ra 3 file `splits/{train,val,test}.txt`. |
| `voc_to_coco.py` | Convert 1 split → 1 file COCO JSON (chỉ 6 categories). Dùng cho **Faster R-CNN** và **RT-DETR**. |
| `voc_to_yolo.py` | Convert cả 3 split → format YOLO (`images/`, `labels/`, symlink ảnh gốc, class id 0-5) + sinh `data.yaml`. Dùng cho **YOLOv8**. |
| `dataset_stats.py` | Tính số ảnh/objects mỗi split (chỉ đếm 6 lớp mục tiêu), xuất `dataset_summary.json` + 2 hình PNG (English label) để paste thẳng vào report. |

## Cách chạy trong Google Colab

```python
# Cell 1: đưa 5 file .py này vào /content (upload hoặc git clone repo của bạn)

# Cell 2: cài đặt (Colab đã có sẵn hầu hết)
!pip install -q pyyaml matplotlib

# Cell 3: tải VOC + tạo split (mất ~5-10 phút tuỳ mạng, VOC ~2.7GB)
!python download_and_split.py --root /content/data --out_root /content/data/splits --val_frac 0.10

# Cell 4: convert sang COCO (cho Faster R-CNN & RT-DETR)
!python voc_to_coco.py --split /content/data/splits/train.txt --out /content/coco/train.json
!python voc_to_coco.py --split /content/data/splits/val.txt   --out /content/coco/val.json
!python voc_to_coco.py --split /content/data/splits/test.txt  --out /content/coco/test.json --exclude_difficult

# Cell 5: convert sang YOLO (cho YOLOv8)
!python voc_to_yolo.py --splits_dir /content/data/splits --out_dir /content/yolo_data

# Cell 6: thống kê + hình cho report
!python dataset_stats.py --splits_dir /content/data/splits --out_dir /content/report_assets
```

**Lưu ý về `--exclude_difficult`:** mặc định script COCO/YOLO **giữ cả object
`difficult=1`** khi convert cho **train/val** (đúng convention huấn luyện VOC gốc).
Khi convert **test** để đánh giá cuối, dùng `--exclude_difficult` để khớp cách
tính mAP chuẩn của VOC devkit (bỏ qua các box khó khi chấm điểm) — ví dụ ở Cell 4
trên đã áp dụng đúng cho `test.json`.

## Kiểm tra nhanh sau khi chạy thật trên Colab

```python
import json
d = json.load(open('/content/coco/train.json'))
print(len(d['images']), len(d['annotations']), len(d['categories']))  # kỳ vọng ~16500 ảnh, 20 categories
```

```python
import yaml
print(yaml.safe_load(open('/content/yolo_data/data.yaml')))
```

## Bước tiếp theo (Pha B)

Sau khi có `coco/{train,val,test}.json` và `yolo_data/` với `data.yaml`, bạn đã
sẵn sàng để viết script train cho 3 pipeline:
1. Faster R-CNN (`torchvision`, đọc từ `coco/*.json`)
2. YOLOv8 (`ultralytics`, đọc `yolo_data/data.yaml`)
3. RT-DETR (`ultralytics` hoặc HuggingFace, đọc từ `coco/*.json`)
