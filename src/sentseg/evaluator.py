"""Evaluation utilities for sentence segmentation baselines."""

from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def _flat(seq):
    if not seq:
        return []
    if isinstance(seq[0], list):
        return [y for x in seq for y in x]
    return list(seq)


def evaluate_labels(y_true, y_pred):
    y_true_f = _flat(y_true)
    y_pred_f = _flat(y_pred)

    labels = sorted(set(y_true_f) | set(y_pred_f))
    if set(labels) == {"B", "I"}:
        # Explicitly pass the label order to avoid ``pos_label`` errors on
        # some scikit-learn versions which expect numerical labels by default.
        p, r, f, _ = precision_recall_fscore_support(
            y_true_f,
            y_pred_f,
            labels=["B", "I"],
            average="binary",
            pos_label="B",
            zero_division=0,
        )
    else:
        p, r, f, _ = precision_recall_fscore_support(
            y_true_f,
            y_pred_f,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    acc = accuracy_score(y_true_f, y_pred_f)
    return {"precision": p, "recall": r, "f1": f, "accuracy": acc}


def evaluate_crf(model, X, y):
    y_pred = model.predict(X)
    return evaluate_labels(y, y_pred)


def evaluate_split(split_func, df):
    from sentseg import dataset as ds
    y_true, y_pred = [], []
    for txt in df["free_text"].astype(str):
        gold, pred = [], []
        for sent in ds._split_by_punc(txt):
            toks = ds._sent2tokens(sent)
            gold.extend(["I"] * (len(toks) - 1) + ["B"])
        for sent in split_func(txt):
            toks = ds._sent2tokens(sent)
            pred.extend(["I"] * (len(toks) - 1) + ["B"])
        y_true.append(gold)
        y_pred.append(pred)
    return evaluate_labels(y_true, y_pred)
