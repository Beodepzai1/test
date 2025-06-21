from __future__ import annotations
import pandas as pd, re
from pathlib import Path
from typing import List, Dict, Tuple

"""Utilities for loading the VIHSD dataset.

The official VIHSD corpus labels each comment with one of three values::

    0 -> clean
    1 -> offensive
    2 -> hate

These labels are kept in the CSV files for reference.  Sentence
segmentation only relies on the ``free_text`` column but we expose a helper
to normalise the labels should older 1-indexed variants of the dataset be
encountered.
"""

# ---------- tiny helpers ----------
def _split_by_punc(text: str) -> List[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]

def _sent2tokens(sent: str) -> List[str]:
    sent = re.sub(r"([.!?])", r" \1 ", sent)
    return sent.split()

# convert labels to 0-based numbering
def _remap_labels(df: pd.DataFrame) -> pd.DataFrame:
    if "label" not in df.columns:
        return df
    labs = df["label"].astype(int)
    if labs.min() == 1 and labs.max() == 3:
        df = df.copy()
        df["label"] = labs - 1
    return df

# ---------- public API ------------
def load(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def read(path: str) -> pd.DataFrame:
        return _remap_labels(pd.read_csv(path))

    return (
        read(cfg["data"]["train_path"]),
        read(cfg["data"]["dev_path"]),
        read(cfg["data"]["test_path"]),
    )

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
