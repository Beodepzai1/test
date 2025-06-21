import argparse
import yaml
from pathlib import Path
from typing import Callable
from sentseg import dataset as ds, evaluator
from sentseg.baseline import split as regex_split
from sentseg.baselines import punkt_wrapper, wtp_wrapper
from sentseg.classifier_models import build_textcnn, build_gru, build_bert


def apply_segmentation(df, split_func: Callable[[str], list[str]]):
    df = df.copy()
    df["segmented"] = df["free_text"].apply(lambda t: " ".join(split_func(str(t))))
    return df


def load_baseline(name: str) -> Callable[[str], list[str]]:
    if name == "regex":
        return regex_split
    if name == "punkt":
        return punkt_wrapper.PunktSplitter().split
    if name == "wtp":
        return wtp_wrapper.WtPSplitter().split
    raise ValueError("unknown baseline")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--baseline", default="regex",
                    choices=["regex", "punkt", "wtp"])
    ap.add_argument("--model", default="textcnn",
                    choices=["textcnn", "bert", "gru"])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    splitter = load_baseline(args.baseline)

    train_df, dev_df, test_df = ds.load(cfg)
    train_df = apply_segmentation(train_df, splitter)
    dev_df = apply_segmentation(dev_df, splitter)
    test_df = apply_segmentation(test_df, splitter)

    if args.model == "bert":
        model, tk = build_bert(num_classes=3)
        enc = lambda batch: tk(batch["segmented"], truncation=True, padding=True)
    else:
        tk = lambda x: x.split()
        # Build vocabulary
        vocab = {tok for text in train_df["segmented"] for tok in tk(text)}
        stoi = {tok: i + 2 for i, tok in enumerate(sorted(vocab))}
        stoi["<pad>"] = 0
        stoi["<unk>"] = 1

        def encode(text):
            return [stoi.get(tok, 1) for tok in tk(text)]

        enc = lambda batch: {"input_ids": [encode(t) for t in batch["segmented"]]}
        if args.model == "textcnn":
            model = build_textcnn(len(stoi), 3)
        else:
            model = build_gru(len(stoi), 3)

    train_ds = {**enc(train_df), "labels": train_df["label"].tolist()}
    dev_ds = {**enc(dev_df), "labels": dev_df["label"].tolist()}
    test_ds = {**enc(test_df), "labels": test_df["label"].tolist()}

    # Simple training loop (placeholder)
    try:
        torch, nn, F = __import__("importlib").import_module("torch"), \
            __import__("importlib").import_module("torch.nn"), \
            __import__("importlib").import_module("torch.nn.functional")
    except Exception:
        raise ImportError("PyTorch required to train models")

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    X = torch.tensor(train_ds["input_ids"], dtype=torch.long)
    y = torch.tensor(train_ds["labels"], dtype=torch.long)
    model.train()
    for _ in range(2):  # few epochs for demo
        optim.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optim.step()

    def predict(dataset):
        model.eval()
        X = torch.tensor(dataset["input_ids"], dtype=torch.long)
        with torch.no_grad():
            out = model(X).argmax(-1).tolist()
        return out

    dev_pred = predict(dev_ds)
    test_pred = predict(test_ds)
    dev_res = evaluator.evaluate_labels(dev_ds["labels"], dev_pred)
    test_res = evaluator.evaluate_labels(test_ds["labels"], test_pred)
    print(f"Dev F1={dev_res['f1']:.4f} Acc={dev_res['accuracy']:.4f}")
    print(f"Test F1={test_res['f1']:.4f} Acc={test_res['accuracy']:.4f}")


if __name__ == "__main__":
    main()
