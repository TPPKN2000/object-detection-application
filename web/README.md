---
title: Traffic & Person Detection Demo
emoji: 🚦
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.4.0"
app_file: app.py
pinned: false
---

# Phát hiện người & phương tiện giao thông đô thị (Gradio + ZeroGPU)

Demo web app cho đồ án Statistical Machine Learning — so sánh 3 kiến trúc
object detection (Faster R-CNN / YOLOv8 / RT-DETR) trên subset 6 lớp của
PASCAL VOC 07+12: `person, bicycle, car, motorbike, bus, train`.

Chọn model ở dropdown, upload ảnh, xem kết quả detect trực tiếp — chạy trên
**ZeroGPU** (HF Spaces cấp GPU thật, tạm thời, chỉ trong lúc gọi hàm
`detect()`).

## Cách deploy lên Hugging Face Spaces

1. Tạo Space mới tại https://huggingface.co/new-space:
   - SDK: **Gradio**
   - Hardware: **ZeroGPU** (mục "Space hardware" khi tạo, hoặc đổi sau ở
     Settings > Hardware — cần tài khoản đã bật ZeroGPU / là Pro hoặc tổ
     chức có quyền dùng ZeroGPU).

2. Clone Space repo về máy, copy **toàn bộ** nội dung thư mục này vào —
   chú ý copy cả file ẩn `.gitattributes` (dùng `cp -r source/. dest/`
   hoặc `cp -a`, **không** dùng `cp source/* dest/` vì dấu `*` không khớp
   file bắt đầu bằng dấu chấm, dễ bị thiếu file này).

3. **Thêm 3 file weight vào `weights/`** — xem hướng dẫn chi tiết trong
   `weights/README.md` (tên file + đường dẫn lấy từ Colab phải đúng chính
   xác, mỗi model có cấu trúc thư mục output khác nhau).

4. Kiểm tra `.gitattributes` đã có mặt và đúng nội dung TRƯỚC khi add file
   weight (bắt buộc để git-lfs nhận diện `.pt` là file lớn):
   ```bash
   cat .gitattributes
   # phải in ra: *.pt filter=lfs diff=lfs merge=lfs -text
   ```

5. Push lên Space:
   ```bash
   git lfs install          # nếu máy bạn chưa cài git-lfs
   git add .gitattributes   # add file này TRƯỚC hoặc CÙNG LÚC với các .pt
   git add .
   git commit -m "add model weights + gradio app"
   git push
   ```

6. Đợi Space build xong (vài phút, xem log ở tab "Logs"), sau đó có link
   public dạng `https://huggingface.co/spaces/<username>/<space-name>`.

## Chạy thử local trước khi deploy (khuyến khích)

```bash
pip install -r requirements.txt
python app.py
```
Mở link `http://127.0.0.1:7860` mà Gradio in ra. `@spaces.GPU` tự động
fallback về chạy bình thường (CPU, hoặc GPU nếu máy bạn có) khi không nằm
trong môi trường HF Spaces thật, nên test local không cần lo về ZeroGPU.

Nếu thiếu file weight nào, app vẫn chạy được bình thường — chỉ ẩn model đó
khỏi dropdown thay vì crash, nên bạn có thể test dần từng model một khi có
weight.

## Ghi chú thiết kế (ZeroGPU)

- Model được load lên **CPU** ngay khi app khởi động (`get_detector()`,
  cache lại để không load lại từ đĩa mỗi lần đổi model). GPU thật chỉ được
  HF Spaces cấp phát **bên trong** hàm `detect()` (đánh dấu
  `@spaces.GPU(duration=30)`) và thu hồi ngay sau khi hàm trả kết quả —
  đây là cơ chế ZeroGPU: chia sẻ GPU vật lý cho nhiều Space, mỗi lần gọi có
  thể rơi vào GPU khác nhau, nên không giữ model cố định trên 1 device cụ
  thể giữa các lần gọi.
- `requirements.txt` dùng torch bản CUDA đầy đủ (không dùng bản CPU-only
  như bản Streamlit trước) vì ZeroGPU cần torch có hỗ trợ CUDA thật khi GPU
  được cấp.
- Faster R-CNN dùng `min_size=600, max_size=1000` khi load lại (khớp với
  cấu hình train được khuyến nghị ở Pha B) — nếu bạn train Faster R-CNN với
  `--min_size/--max_size` khác, sửa lại 2 giá trị này trong
  `inference.py::FasterRCNNDetector.__init__` cho khớp.
- `duration=30` trong `@spaces.GPU(duration=30)` là số giây tối đa mỗi lần
  gọi được giữ GPU — 30s dư sức cho 1 ảnh, có thể giảm nếu muốn xếp hàng ít
  hơn khi nhiều người dùng cùng lúc.
