# SentenceSeg-VIHSD

Bộ công cụ phân đoạn câu tiếng Việt trên tập **VIHSD** (gồm ba nhãn `clean=0`, `offensive=1`, `hate=2`) kèm 6 baseline:

| Baseline         | Nhóm        | Tham chiếu nghiên cứu         |
|------------------|--------------|-----------------------------------------|
| none             | Identity     | Không tách câu                         |
| regex            | RB           | Moses / SpaCy-SENT                      |
| crf              | SS           | Riley 1989, Splitta                     |
| punkt            | US           | Kiss & Strunk 2006                      |
| wtp              | SS (Self-sup)| Minixhofer et al 2023                   |


## Cài đặt

```bash
pip install -e .
python -m nltk.downloader punkt   # cho baseline Punkt
```

## Sử dụng

Sau khi chỉnh sửa đường dẫn dữ liệu trong `configs/default.yaml`, chạy toàn bộ pipeline phân đoạn câu và phân loại văn bản như sau:
```bash
python -m sentseg.cli -c configs/default.yaml --baseline regex --model textcnn
```

Thay `--baseline` bằng `none`, `punkt`, `wtp`, `crf` và `--model` bằng `bert` hoặc `gru` tùy nhu cầu. Lệnh sẽ in ra F1 và Accuracy trên tập dev và test.



