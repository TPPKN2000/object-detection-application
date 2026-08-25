# Pha B — Train & so sánh 3 kiến trúc object detection

## Cấu trúc repo cần push lên GitHub

Gộp chung cả Pha A và Pha B vào **1 repo** với cấu trúc sau (đúng như notebook
`notebooks/phase_B_train_and_compare.ipynb` mong đợi):

```
<your-repo>/
├── requirements.txt
├── data_prep/                      <- Pha A (đã giao ở lượt trước)
│   ├── xml_utils.py
│   ├── download_and_split.py
│   ├── voc_to_coco.py
│   ├── voc_to_yolo.py
│   └── dataset_stats.py
├── training/                       <- Pha B (gói này)
│   ├── common/
│   │   ├── coco_dataset.py
│   │   ├── transforms.py
│   │   └── metrics_utils.py
│   ├── faster_rcnn/
│   │   └── train_faster_rcnn.py
│   ├── yolov8/
│   │   └── train_yolov8.py
│   ├── rtdetr/
│   │   └── train_rtdetr.py
│   └── compare_results.py
└── notebooks/
    └── phase_B_train_and_compare.ipynb
```

> ⚠️ **Lưu ý quan trọng:** file `data_prep/download_and_split.py` trong gói Pha A
> **đã được sửa lại** (fix bug `abs_path` không phải absolute path khi `--root`
> truyền dạng tương đối). Nếu bạn đã push Pha A từ lượt trước, hãy **ghi đè lại
> file này** bằng bản mới nhất trong `/mnt/user-data/outputs/phaseA_voc_prep/`
> trước khi chạy Pha B — nếu không, `abs_path` trong COCO JSON sẽ sai và
> `train_faster_rcnn.py` sẽ báo lỗi "file not found" khi mở ảnh.

Tên các thư mục `data_prep/` và `training/` phải giữ nguyên (notebook gọi
đường dẫn tương đối `data_prep/...` và `training/...` từ root repo).

## Cách chạy

1. Push toàn bộ cấu trúc trên lên GitHub repo của bạn.
2. Mở `notebooks/phase_B_train_and_compare.ipynb` trên Google Colab
   (`File > Upload notebook`, hoặc mở thẳng từ GitHub qua `File > Open notebook > GitHub`).
3. `Runtime > Change runtime type > T4 GPU`.
4. Sửa biến `REPO_URL` ở cell đầu tiên (mục 1) thành link GitHub repo của bạn.
5. Chạy tuần tự từng cell từ trên xuống. Notebook đã có sẵn lệnh kiểm tra
   (`cat metrics.json`, đếm số ảnh/annotation, hiển thị biểu đồ so sánh) sau
   mỗi bước quan trọng để bạn xác nhận không có gì sai trước khi sang bước kế.

## Những gì đã được validate bằng chạy thật (không chỉ đọc code)

Do sandbox không có GPU, các script được test thật trên CPU với dữ liệu giả
(ảnh nhỏ, vài chục ảnh) để xác nhận **toàn bộ luồng chạy không lỗi**:

| Script | Đã test |
|---|---|
| `common/coco_dataset.py`, `transforms.py` | Load ảnh + annotation, augmentation, collate_fn — chạy đúng |
| `common/metrics_utils.py` | `benchmark_fps`, `count_params`, `save_metrics` — chạy đúng, output đúng schema |
| `faster_rcnn/train_faster_rcnn.py` | Training loop (loss giảm dần qua các iteration), `evaluate()` (torchmetrics+pycocotools), FPS benchmark — test riêng từng phần thành công. **2 bug được phát hiện & sửa trong lúc test:** (1) `abs_path` không absolute nếu `--root` tương đối, (2) OOM khi ảnh nhỏ + resize mặc định 800×1333 → thêm `--min_size/--max_size` |
| `yolov8/train_yolov8.py` | Chạy **trọn vẹn 1 epoch thật** qua `ultralytics`, `metrics.json` đúng format |
| `rtdetr/train_rtdetr.py` | Chạy **trọn vẹn 1 epoch thật**, không lỗi |
| `compare_results.py` | Test với 3 file `metrics.json` thật (từ 2 lần chạy trên) → bảng Markdown + 4 biểu đồ đúng |

**Chưa test được** (do sandbox không có GPU, không tải được pretrained weight
từ `download.pytorch.org`): train full epoch trên GPU thật với ảnh kích thước
thật (640×640 / 800×1333) và pretrained COCO weights. Đây là lý do notebook có
sẵn lệnh kiểm tra sau mỗi bước train — nếu `metrics.json` hiện mAP hợp lý
(không phải toàn 0) nghĩa là quá trình train trên Colab đã chạy đúng.

## Ghi chú thời gian ước tính trên T4 (Colab)

| Model | Epochs mặc định | Ước tính thời gian |
|---|---|---|
| Faster R-CNN | 15 | ~1.5–2.5h |
| YOLOv8s | 50 | ~45–90 phút |
| RT-DETR-L | 40 | ~1.5–2h |

Nếu hết giờ Colab free giữa chừng, chỉ cần chạy lại đúng cell của model đang
dở — không cần chạy lại từ đầu, vì mỗi script train độc lập, đọc dữ liệu đã
convert sẵn ở `/content/coco/` và `/content/yolo_data/`.

## Bước tiếp theo (Pha C)

Sau khi có đủ 3 `metrics.json` + `comparison_table.md`, bạn đã sẵn sàng viết
phần "So sánh & phân tích" trong report (Yêu cầu 1), và chọn model tốt nhất
(theo mAP, hoặc theo FPS nếu ưu tiên real-time) để triển khai Web App (Yêu cầu 2).
