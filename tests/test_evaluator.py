import pytest
from sentseg.evaluator import evaluate_labels


def test_evaluate_labels_binary():
    y_true = [["B", "I", "B"], ["I", "B"]]
    y_pred = [["B", "I", "B"], ["B", "B"]]
    metrics = evaluate_labels(y_true, y_pred)
    assert set(metrics) == {"precision", "recall", "f1", "accuracy"}
    for value in metrics.values():
        assert isinstance(value, float)


def test_evaluate_labels_with_extra_label():
    y_true = [["B", "X", "I"], ["I", "B"]]
    y_pred = [["B", "I", "X"], ["B", "B"]]
    metrics = evaluate_labels(y_true, y_pred)
    assert set(metrics) == {"precision", "recall", "f1", "accuracy"}
    for value in metrics.values():
        assert isinstance(value, float)
