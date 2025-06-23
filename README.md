# SentenceSeg-VIHSD

Ví dụ tối giản dùng để thử nghiệm phân đoạn câu tiếng Việt trên tập dữ liệu
nhỏ. Mặc định công cụ sử dụng baseline tách câu bằng regex và mô hình phân loại
Logistic Regression đơn giản.

## Cài đặt

```bash
pip install -e .
python -m nltk.downloader punkt   # cho baseline Punkt
```

## Sử dụng

Sau khi chỉnh sửa đường dẫn dữ liệu trong `configs/default.yaml`, chạy thử nghiệm:
```bash
python -m sentseg.cli -c configs/default.yaml
```
Có thể thay `--baseline` thành `punkt` nếu muốn thử Punkt.


