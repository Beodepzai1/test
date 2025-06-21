from __future__ import annotations
import pandas as pd, re
from pathlib import Path
from typing import List, Dict, Tuple

# ---------- tiny helpers ----------
def _split_by_punc(text: str) -> List[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]

def _sent2tokens(sent: str) -> List[str]:
    sent = re.sub(r"([.!?])", r" \1 ", sent)
    return sent.split()

# ---------- public API ------------
def load(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (pd.read_csv(cfg["data"]["train_path"]),
            pd.read_csv(cfg["data"]["dev_path"]),
            pd.read_csv(cfg["data"]["test_path"]))

def make_conll(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for txt in df["free_text"].astype(str):
        for sent in _split_by_punc(txt):
            toks = _sent2tokens(sent)
            for i, tok in enumerate(toks):
                lab = "B" if i == len(toks) - 1 else "I"
                rows.append(f"{tok}\t{lab}")
            rows.append("")          # blank line = new sentence
    out_path.write_text("\n".join(rows), encoding="utf-8")

def prepare(cfg: Dict):
    train, dev, test = load(cfg)
    make_conll(train, Path(cfg["data"]["train_conll"]))
    return train, dev, test

def df2sents(df: pd.DataFrame):
    sents = []
    for txt in df["free_text"].astype(str):
        for sent in _split_by_punc(txt):
            toks = _sent2tokens(sent)
            sents.append([
                (tok, "B" if i == len(toks) - 1 else "I")
                for i, tok in enumerate(toks)
            ])
    return sents
