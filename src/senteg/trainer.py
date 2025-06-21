from pathlib import Path
from typing import Dict
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          Trainer, TrainingArguments)
from sentseg.models import crf_model
from sentseg.features import sent2features, sent2labels

# ------------ utilities -------------
def _read_conll(path: Path):
    sents, sent = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            if sent: sents.append(sent); sent = []
        else:
            tok, lab = line.split()
            sent.append((tok, lab))
    return sents

# ------------ CRF -------------------
def train_crf(cfg: Dict):
    sents = _read_conll(Path(cfg["data"]["train_conll"]))
    model = crf_model.train(
        sents,
        c1=cfg["models"]["crf"]["c1"],
        c2=cfg["models"]["crf"]["c2"],
        max_iter=cfg["models"]["crf"]["max_iter"]
    )
    out = Path(cfg["output"]["dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / "crf.pkl").write_bytes(__import__("pickle").dumps(model))
    return model

# ------------ PhoBERT ---------------
def train_transformer(cfg: Dict):
    tk = AutoTokenizer.from_pretrained(cfg["models"]["transformer"]["model_name"])

    def _encode(batch):
        enc = tk(batch["free_text"], truncation=True)
        enc["labels"] = [[0]*len(x) for x in enc["input_ids"]]  # dummy (B/I cần gắn nhãn thực nếu muốn)
        return enc

    train_df = pd.read_csv(cfg["data"]["train_path"])
    dev_df   = pd.read_csv(cfg["data"]["dev_path"])
    train_ds = Dataset.from_pandas(train_df).map(_encode, batched=True, remove_columns=train_df.columns)
    dev_ds   = Dataset.from_pandas(dev_df).map(_encode, batched=True, remove_columns=dev_df.columns)

    model = AutoModelForTokenClassification.from_pretrained(
        cfg["models"]["transformer"]["model_name"], num_labels=2
    )
    args = TrainingArguments(
        output_dir=cfg["output"]["dir"],
        evaluation_strategy="epoch",
        learning_rate=cfg["trainer"]["learning_rate"],
        per_device_train_batch_size=cfg["trainer"]["batch_size"],
        num_train_epochs=cfg["trainer"]["epochs"],
        save_strategy="no"
    )
    Trainer(model=model, args=args,
            train_dataset=train_ds, eval_dataset=dev_ds).train()
    model.save_pretrained(cfg["output"]["dir"])
    tk.save_pretrained(cfg["output"]["dir"])
    return model
