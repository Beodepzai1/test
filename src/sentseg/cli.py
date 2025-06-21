#!/usr/bin/env python
# sentseg/cli.py
import argparse
import yaml
from pathlib import Path
from typing import Callable, List

from sentseg import dataset as ds, evaluator
from sentseg.baseline import split as regex_split
from sentseg.baselines import punkt_wrapper, wtp_wrapper
from sentseg.classifier_models import build_textcnn, build_gru, build_bert

LABEL_COL = "label_id"        


def apply_segmentation(df, split_func: Callable[[str], List[str]]):
    df = df.copy()
    df["segmented"] = df["free_text"].apply(lambda t: " ".join(split_func(str(t))))
    return df


def load_baseline(name: str) -> Callable[[str], List[str]]:
    if name == "regex":
        return regex_split
    if name == "punkt":
        return punkt_wrapper.PunktSplitter().split
    if name == "wtp":
        return wtp_wrapper.WtPSplitter().split
    raise ValueError("unknown baseline")


def pad_sequences(seqs: List[List[int]], pad_idx: int = 0) -> List[List[int]]:
    """Đệm tất cả chuỗi token về cùng chiều dài = max_len."""
    max_len = max(len(s) for s in seqs)
    return [s + [pad_idx] * (max_len - len(s)) for s in seqs]


def main():
    # ─── 1. Đọc tham số dòng lệnh ───────────────────────────────────────────
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--baseline", default="regex",
                    choices=["regex", "punkt", "wtp"])
    ap.add_argument("--model", required=True,
                    choices=["textcnn", "bert", "gru"],
                    help="classification model")
    args = ap.parse_args()

    # ─── 2. Load dữ liệu & tiền xử lý ───────────────────────────────────────
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    splitter = load_baseline(args.baseline)

    train_df, dev_df, test_df = ds.load(cfg)

    for df, name in [(train_df, "train"), (dev_df, "dev"), (test_df, "test")]:
        if LABEL_COL not in df.columns:
            raise KeyError(f"'{LABEL_COL}' not found in {name} dataframe")

    train_df = apply_segmentation(train_df, splitter)
    dev_df   = apply_segmentation(dev_df,   splitter)
    test_df  = apply_segmentation(test_df,  splitter)

    num_classes = train_df[LABEL_COL].nunique()

    # ─── 3. Xây mô hình & tokenizer ─────────────────────────────────────────
    if args.model == "bert":
        model, tk = build_bert(num_classes=num_classes)
        def encode_df(df):
            enc_out = tk(df["segmented"].tolist(),
                         truncation=True, padding=True, return_tensors="pt")
            return enc_out["input_ids"].tolist()
    else:
        # Tokenizer đơn giản: tách space
        tk = lambda x: x.split()
        vocab = {tok for text in train_df["segmented"] for tok in tk(text)}
        stoi = {tok: i + 2 for i, tok in enumerate(sorted(vocab))}
        stoi["<pad>"] = 0
        stoi["<unk>"] = 1

        def encode(text: str) -> List[int]:
            return [stoi.get(tok, 1) for tok in tk(text)]

        def encode_df(df):
            ids = [encode(t) for t in df["segmented"]]
            return pad_sequences(ids, pad_idx=0)

        if args.model == "textcnn":
            model = build_textcnn(len(stoi), num_classes)
        else:  # gru
            model = build_gru(len(stoi), num_classes)

    # ─── 4. Mã hoá & padding dữ liệu ────────────────────────────────────────
    train_ids = encode_df(train_df)
    dev_ids   = encode_df(dev_df)
    test_ids  = encode_df(test_df)

    train_ds = {"input_ids": train_ids, "labels": train_df[LABEL_COL].tolist()}
    dev_ds   = {"input_ids": dev_ids,   "labels": dev_df[LABEL_COL].tolist()}
    test_ds  = {"input_ids": test_ids,  "labels": test_df[LABEL_COL].tolist()}

    # ─── 5. Huấn luyện demo (2 epoch) ───────────────────────────────────────
    try:
        torch = __import__("importlib").import_module("torch")
        nn    = __import__("importlib").import_module("torch.nn")
    except Exception:
        raise ImportError("PyTorch required to train models")

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    X = torch.tensor(train_ds["input_ids"], dtype=torch.long)
    y = torch.tensor(train_ds["labels"],     dtype=torch.long)

    model.train()
    for _ in range(2):
        optim.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optim.step()

    # ─── 6. Dự đoán & đánh giá ─────────────────────────────────────────────
    def predict(dataset):
        model.eval()
        X = torch.tensor(dataset["input_ids"], dtype=torch.long)
        with torch.no_grad():
            return model(X).argmax(-1).tolist()

    dev_pred  = predict(dev_ds)
    test_pred = predict(test_ds)

    dev_res  = evaluator.evaluate_labels(dev_ds["labels"],  dev_pred)
    test_res = evaluator.evaluate_labels(test_ds["labels"], test_pred)
    print(f"Dev  - F1={dev_res['f1']:.4f}  Acc={dev_res['accuracy']:.4f}")
    print(f"Test - F1={test_res['f1']:.4f}  Acc={test_res['accuracy']:.4f}")


if __name__ == "__main__":
    main()
