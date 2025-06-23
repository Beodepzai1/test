#!/usr/bin/env python
# sentseg/cli.py
import sys
if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")
import argparse
import yaml
from pathlib import Path
from typing import Callable, List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sentseg import dataset as ds, evaluator
from sentseg.baseline import split as regex_split
from sentseg.classifier_models import build_textcnn, build_gru, build_bert

LABEL_COL = "label_id"        


def apply_segmentation(df, split_func: Callable[[str], List[str]]):
    df = df.copy()
    df["segmented"] = df["free_text"].apply(lambda t: " ".join(split_func(str(t))))
    return df


def load_baseline(name: str) -> Callable[[str], List[str]]:
    if name == "regex":
        return regex_split
    if name == "none":
        from sentseg.baseline import split_none
        return split_none
    if name == "punkt":
        from sentseg.baselines import punkt_wrapper
        return punkt_wrapper.PunktSplitter().split
    if name == "wtp":
        try:
            from sentseg.baselines import wtp_wrapper
            return wtp_wrapper.WtPSplitter().split
        except Exception as e:
            print(f"Warning: cannot load WtP baseline ({e}); falling back to regex", file=sys.stderr)
            return regex_split
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
                    choices=["regex", "none", "punkt", "wtp"])
    ap.add_argument("--model", required=True,
                    choices=["textcnn", "bert", "gru"],
                    help="classification model")
    ap.add_argument("--fasttext", help="Path to FastText .vec embeddings")
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
    use_torch = True
    try:
        import importlib
        torch = importlib.import_module("torch")
        nn = importlib.import_module("torch.nn")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        use_torch = False

    if use_torch and args.model != "bert":
        # Tokenizer đơn giản: tách space
        tk = lambda x: x.split()
        vocab = {tok for text in train_df["segmented"] for tok in tk(text)}
        stoi = {tok: i + 2 for i, tok in enumerate(sorted(vocab))}
        stoi["<pad>"] = 0
        stoi["<unk>"] = 1

        embed_dim = 128
        embeddings = None
        if args.fasttext:
            from gensim.models import KeyedVectors
            kv = KeyedVectors.load_word2vec_format(args.fasttext)
            embed_dim = kv.vector_size
            embeddings = np.random.normal(scale=0.6, size=(len(stoi), embed_dim))
            embeddings[0] = 0
            embeddings[1] = 0
            for tok, idx in stoi.items():
                if tok in kv:
                    embeddings[idx] = kv[tok]

        def encode(text: str) -> List[int]:
            return [stoi.get(tok, 1) for tok in tk(text)]

        def encode_df(df):
            ids = [encode(t) for t in df["segmented"]]
            return pad_sequences(ids, pad_idx=0)

        if args.model == "textcnn":
            model = build_textcnn(
                len(stoi),
                num_classes,
                embed_dim=embed_dim,
                pretrained_embeddings=embeddings,
            )
        else:  # gru
            model = build_gru(
                len(stoi),
                num_classes,
                embed_dim=embed_dim,
                pretrained_embeddings=embeddings,
            )

        # ─── 4. Mã hoá & padding dữ liệu ────────────────────────────────────
        train_ids = encode_df(train_df)
        dev_ids   = encode_df(dev_df)
        test_ids  = encode_df(test_df)

        train_ds = {"input_ids": train_ids, "labels": train_df[LABEL_COL].tolist()}
        dev_ds   = {"input_ids": dev_ids,   "labels": dev_df[LABEL_COL].tolist()}
        test_ds  = {"input_ids": test_ids,  "labels": test_df[LABEL_COL].tolist()}

        # ─── 5. Huấn luyện demo (2 epoch) ───────────────────────────────────
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        X = torch.tensor(train_ds["input_ids"], dtype=torch.long).to(device)
        y = torch.tensor(train_ds["labels"],     dtype=torch.long).to(device)
        model.to(device)

        model.train()
        for _ in range(2):
            optim.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optim.step()

        # ─── 6. Dự đoán & đánh giá ─────────────────────────────────────────
        def predict(dataset):
            model.eval()
            X = torch.tensor(dataset["input_ids"], dtype=torch.long).to(device)
            with torch.no_grad():
                return model(X).argmax(-1).cpu().tolist()

        dev_pred  = predict(dev_ds)
        test_pred = predict(test_ds)

        dev_res  = evaluator.evaluate_labels(dev_ds["labels"],  dev_pred)
        test_res = evaluator.evaluate_labels(test_ds["labels"], test_pred)
        print(f"Dev  - F1={dev_res['f1']:.4f}  Acc={dev_res['accuracy']:.4f}")
        print(f"Test - F1={test_res['f1']:.4f}  Acc={test_res['accuracy']:.4f}")
    elif use_torch and args.model == "bert":
        model, tk = build_bert(num_classes=num_classes)

        def encode_df(df):
            enc_out = tk(df["segmented"].tolist(),
                         truncation=True, padding=True, return_tensors="pt")
            return enc_out["input_ids"].tolist()

        train_ids = encode_df(train_df)
        dev_ids   = encode_df(dev_df)
        test_ids  = encode_df(test_df)

        train_ds = {"input_ids": train_ids, "labels": train_df[LABEL_COL].tolist()}
        dev_ds   = {"input_ids": dev_ids,   "labels": dev_df[LABEL_COL].tolist()}
        test_ds  = {"input_ids": test_ids,  "labels": test_df[LABEL_COL].tolist()}

        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        X = torch.tensor(train_ds["input_ids"], dtype=torch.long).to(device)
        y = torch.tensor(train_ds["labels"],     dtype=torch.long).to(device)
        model.to(device)

        model.train()
        for _ in range(2):
            optim.zero_grad()
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optim.step()

        def predict(dataset):
            model.eval()
            X = torch.tensor(dataset["input_ids"], dtype=torch.long).to(device)
            with torch.no_grad():
                return model(X).argmax(-1).cpu().tolist()

        dev_pred  = predict(dev_ds)
        test_pred = predict(test_ds)

        dev_res  = evaluator.evaluate_labels(dev_ds["labels"],  dev_pred)
        test_res = evaluator.evaluate_labels(test_ds["labels"], test_pred)
        print(f"Dev  - F1={dev_res['f1']:.4f}  Acc={dev_res['accuracy']:.4f}")
        print(f"Test - F1={test_res['f1']:.4f}  Acc={test_res['accuracy']:.4f}")
    else:
        # Fallback phân loại logistic regression nếu không cài được PyTorch
        tk = lambda x: x.split()
        vectorizer = CountVectorizer(tokenizer=tk)
        X_train = vectorizer.fit_transform(train_df["segmented"])
        X_dev   = vectorizer.transform(dev_df["segmented"])
        X_test  = vectorizer.transform(test_df["segmented"])

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, train_df[LABEL_COL])

        dev_pred  = clf.predict(X_dev)
        test_pred = clf.predict(X_test)

        dev_res  = evaluator.evaluate_labels(dev_df[LABEL_COL].tolist(),  dev_pred.tolist())
        test_res = evaluator.evaluate_labels(test_df[LABEL_COL].tolist(), test_pred.tolist())
        print(f"Dev  - F1={dev_res['f1']:.4f}  Acc={dev_res['accuracy']:.4f}")
        print(f"Test - F1={test_res['f1']:.4f}  Acc={test_res['accuracy']:.4f}")


if __name__ == "__main__":
    main()
