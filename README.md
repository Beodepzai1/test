# SentenceSeg-VIHSD

Bộ công cụ phân đoạn câu tiếng Việt trên tập **VIHSD** (gồm ba nhãn `clean=0`, `offensive=1`, `hate=2`) kèm 5 baseline:

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

Nếu muốn thử nghiệm trên bộ dữ liệu nhận dạng Vispam, chỉnh sửa đường dẫn trong
`configs/spam.yaml` (hoặc truyền file cấu hình tương tự) rồi chạy:

```bash
python -m sentseg.cli -c configs/spam.yaml --baseline regex --model textcnn
```

Tuỳ chọn `--task` cho phép chuyển đổi giữa hai bài toán:

```bash
# Task 1: phát hiện spam hay không spam
python -m sentseg.cli -c configs/spam.yaml --baseline regex --model textcnn --task 1
# Task 2: phân loại các kiểu spam (SPAM-1, SPAM-2, SPAM-3)
python -m sentseg.cli -c configs/spam.yaml --baseline regex --model textcnn --task 2
```

Hai file cấu hình `default.yaml` và `spam.yaml` có chung định dạng và chỉ khác ở
đường dẫn dữ liệu cũng như tên cột chứa nội dung (`text_column`) và nhãn
(`label_column`). Nhờ đó có thể chuyển đổi linh hoạt giữa các tập dữ liệu khi
thực hiện các thí nghiệm.



