from __future__ import annotations
import pandas as pd, re
from pathlib import Path
from typing import List, Dict, Tuple

"""Utilities for loading the VIHSD dataset.

The original data assigns three labels:
    0 -> clean
    1 -> offensive
    2 -> hate
These labels are not used for sentence segmentation but remain in the CSV
files for reference.
"""

# ---------- tiny helpers ----------
def _split_by_punc(text: str) -> List[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]

def _sent2tokens(sent: str) -> List[str]:
    sent = re.sub(r"([.!?])", r" \1 ", sent)
    return sent.split()

# ---------- public API ------------
def _remap_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure class labels are 0-indexed (clean=0, offensive=1, hate=2)."""
    if "label" in df.columns:
        df = df.copy()
        df["label"] = df["label"].map({1: 0, 2: 1, 3: 2}).fillna(df["label"])
    return df


def load(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(cfg["data"]["train_path"])
    dev = pd.read_csv(cfg["data"]["dev_path"])
    test = pd.read_csv(cfg["data"]["test_path"])
    return (_remap_labels(train), _remap_labels(dev), _remap_labels(test))

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
