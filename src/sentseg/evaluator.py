from sklearn.metrics import precision_recall_fscore_support

def _flat(seq): return [y for x in seq for y in x]

def evaluate_crf(model, X_test, y_test):
    y_pred = model.predict(X_test)
    p, r, f, _ = precision_recall_fscore_support(
        _flat(y_test), _flat(y_pred),
        labels=["B"], average="binary"
    )
    return {"precision": p, "recall": r, "f1": f}
