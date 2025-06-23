#!/usr/bin/env python
# sentseg/cli.py
import argparse
import yaml
from pathlib import Path
from typing import Callable, List

from sentseg import dataset as ds, evaluator
from sentseg.baseline import split as regex_split
from sentseg.baselines import punkt_wrapper, wtp_wrapper
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

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


def main():
    # ─── 1. Đọc tham số dòng lệnh ───────────────────────────────────────────
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--baseline", default="regex",
                    choices=["regex", "punkt", "wtp"])
    ap.add_argument("--model", default="logreg")
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

    # ─── 3. Vector hoá & huấn luyện đơn giản bằng Logistic Regression ───────
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_df["segmented"])
    X_dev   = vectorizer.transform(dev_df["segmented"])
    X_test  = vectorizer.transform(test_df["segmented"])

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, train_df[LABEL_COL])

    dev_pred  = model.predict(X_dev).tolist()
    test_pred = model.predict(X_test).tolist()

    dev_res  = evaluator.evaluate_labels(dev_df[LABEL_COL],  dev_pred)
    test_res = evaluator.evaluate_labels(test_df[LABEL_COL], test_pred)
    print(f"Dev  - F1={dev_res['f1']:.4f}  Acc={dev_res['accuracy']:.4f}")
    print(f"Test - F1={test_res['f1']:.4f}  Acc={test_res['accuracy']:.4f}")


if __name__ == "__main__":
    main()
