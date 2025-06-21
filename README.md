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

Sau khi chỉnh sửa đường dẫn dữ liệu trong `configs/default.yaml`, chạy toàn bộ pipeline phân đoạn câu và phân loại văn bản như sau:
```bash
python -m sentseg.cli -c configs/default.yaml --baseline regex --model textcnn
```

Thay `--baseline` bằng `pysbd`, `punkt` hoặc `wtp` và `--model` bằng `bert` hoặc `gru` tùy nhu cầu. Lệnh sẽ in ra F1 và Accuracy trên tập dev và test.

### Lưu ý

- Baseline `phobert` yêu cầu `transformers>=4.41.0`. Nếu cài phiên bản cũ hơn, lệnh có thể báo lỗi `TypeError` ở tham số `evaluation_strategy`.
- Khi dùng `phobert`, chương trình sẽ in F1 và Accuracy trên tập dev và test sau khi huấn luyện.

## Phân loại văn bản

Pipeline phân loại dựa trên các bước: câu → tách câu (theo baseline) → mô hình (TextCNN, BERT, GRU). Có thể chạy bằng lệnh ở trên hoặc:
```bash
python -m sentseg.cli -c configs/default.yaml --baseline regex --model textcnn
```
Thay `--model` bằng `bert` hoặc `gru` để thử nghiệm các mô hình khác.
