# SentenceSeg-VIHSD

Bộ công cụ phân đoạn câu tiếng Việt trên tập **VIHSD** (gồm ba nhãn `clean=1`, `offensive=2`, `hate=3`) kèm 7 baseline:

| Baseline         | Nhóm        | Tham chiếu nghiên cứu         |
|------------------|--------------|-----------------------------------------|
| regex            | RB           | Moses / SpaCy-SENT                      |
| crf              | SS           | Riley 1989, Splitta                     |
| phobert          | DL           | Ersatz (Transformer)                    |
| pysbd            | RB           | PySBD (Pragmatic SBD)                   |
| punkt            | US           | Kiss & Strunk 2006                      |
| wtp              | SS (Self-sup)| Minixhofer et al 2023                   |
| wtp_finetune     | Few-shot     | 〃 (64-256 câu)                        |

## Cài đặt

```bash
pip install -e .
python -m nltk.downloader punkt   # cho baseline Punkt
```

## Sử dụng

Sau khi chỉnh sửa đường dẫn dữ liệu trong `configs/default.yaml`, có thể chạy một baseline như sau:
```bash
python -m sentseg.cli -c configs/default.yaml --baseline regex
```

Thay tham số `--baseline` bằng `crf`, `phobert`, `pysbd`, `punkt`, `wtp` hoặc `wtp_finetune` để thử nghiệm các phương pháp khác. Lệnh sẽ in ra F1 và Accuracy của mô hình trên tập dev và test.

### Lưu ý

- Baseline `phobert` yêu cầu `transformers>=4.41.0`. Nếu cài phiên bản cũ hơn, lệnh có thể báo lỗi `TypeError` ở tham số `evaluation_strategy`.
- Khi dùng `phobert`, chương trình sẽ in F1 và Accuracy trên tập dev và test sau khi huấn luyện.
