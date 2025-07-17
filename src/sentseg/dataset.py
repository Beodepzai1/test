from __future__ import annotations
import sys
if "/usr/lib/python3/dist-packages" not in sys.path:
    sys.path.append("/usr/lib/python3/dist-packages")
import pandas as pd, re
from pathlib import Path
from typing import List, Dict, Tuple

"""Utilities for loading the VIHSD dataset.

The classification portion of VIHSD may store labels either starting from 1
(`clean=1`, `offensive=2`, `hate=3`) or already zero-indexed
(`clean=0`, `offensive=1`, `hate=2`).

When loading the data we normalise them to the 0‑based scheme:
    0 -> clean
    1 -> offensive
    2 -> hate

These labels are not used for sentence segmentation but remain in the CSV
files for reference.
"""

def _split_by_punc(text: str) -> List[str]:
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]

def _sent2tokens(sent: str) -> List[str]:
    sent = re.sub(r"([.!?])", r" \1 ", sent)
    return sent.split()

def _remap_labels(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Ensure class labels are 0-indexed (clean=0, offensive=1, hate=2).

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the label column.
    col : str
        Name of the label column which will be normalised if appropriate.
    """

    if col not in df.columns:
        return df

    df = df.copy()
    vals = set(df[col].unique())
    if vals <= {1, 2, 3}:
        df[col] = df[col].map({1: 0, 2: 1, 3: 2})
    return df


def _standardise_columns(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Rename text/label columns to the canonical names used in the code."""

    text_col = cfg.get("text_column", "free_text")
    label_col = cfg.get("label_column", "label_id")

    df = df.copy()
    if text_col in df.columns and text_col != "free_text":
        df = df.rename(columns={text_col: "free_text"})
    if label_col in df.columns and label_col != "label_id":
        df = df.rename(columns={label_col: "label_id"})

    return _remap_labels(df, "label_id")


def load(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(cfg["data"]["train_path"])
    dev = pd.read_csv(cfg["data"]["dev_path"])
    test = pd.read_csv(cfg["data"]["test_path"])

    train = _standardise_columns(train, cfg["data"])
    dev = _standardise_columns(dev, cfg["data"])
    test = _standardise_columns(test, cfg["data"])

    return train, dev, test

def _split_row(row) -> List[str]:
    """Return the list of sentences for a DataFrame row.

    If the dataset provides a ``sentences`` column containing newline
    separated sentences this is used as the gold segmentation. Otherwise
    text is segmented with :func:`_split_by_punc` as a fallback.
    """

    if "sentences" in row and isinstance(row["sentences"], str):
        segs = [s for s in str(row["sentences"]).splitlines() if s.strip()]
        if segs:
            return segs
    return _split_by_punc(str(row["free_text"]))


def make_conll(df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for _, row in df.iterrows():
        for sent in _split_row(row):
            toks = _sent2tokens(sent)
            for i, tok in enumerate(toks):
                lab = "B" if i == len(toks) - 1 else "I"
                rows.append(f"{tok}\t{lab}")
            rows.append("")
    out_path.write_text("\n".join(rows), encoding="utf-8")

def prepare(cfg: Dict):
    train, dev, test = load(cfg)
    make_conll(train, Path(cfg["data"]["train_conll"]))
    return train, dev, test

def df2sents(df: pd.DataFrame):
    sents = []
    for _, row in df.iterrows():
        for sent in _split_row(row):
            toks = _sent2tokens(sent)
            sents.append([
                (tok, "B" if i == len(toks) - 1 else "I")
                for i, tok in enumerate(toks)
            ])
    return sents
