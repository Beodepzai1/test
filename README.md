# SentenceSeg-VIHSD

Bộ công cụ phân đoạn câu tiếng Việt trên tập **VIHSD** kèm 7 baseline:

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

Thay tham số `--baseline` bằng `crf`, `phobert`, `pysbd`, `punkt`, `wtp` hoặc `wtp_finetune` để thử nghiệm các phương pháp khác.
