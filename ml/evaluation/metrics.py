"""Dependency-light classification metrics for three-class security reporting."""

from __future__ import annotations

from dataclasses import dataclass

from ml.constants import ID_TO_LABEL, NUM_LABELS


@dataclass(frozen=True, slots=True)
class PerClassMetrics:
    """Precision, recall, and F1 for one class."""

    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Aggregate and per-class metrics, including phishing quality."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    per_class: dict[str, PerClassMetrics]
    confusion_matrix: tuple[tuple[int, ...], ...]


def calculate_metrics(expected: list[int], predicted: list[int]) -> ClassificationMetrics:
    """Calculate accuracy, macro metrics, per-class metrics, and confusion matrix."""

    if not expected or len(expected) != len(predicted):
        raise ValueError("expected and predicted labels must be non-empty and equal length")
    if any(label not in ID_TO_LABEL for label in [*expected, *predicted]):
        raise ValueError("labels must be in the configured three-class range")
    matrix = [[0 for _ in range(NUM_LABELS)] for _ in range(NUM_LABELS)]
    for actual, guess in zip(expected, predicted, strict=True):
        matrix[actual][guess] += 1
    per_class: dict[str, PerClassMetrics] = {}
    for label_id, label_name in ID_TO_LABEL.items():
        true_positive = matrix[label_id][label_id]
        false_positive = sum(matrix[row][label_id] for row in range(NUM_LABELS) if row != label_id)
        false_negative = sum(matrix[label_id][column] for column in range(NUM_LABELS) if column != label_id)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label_name] = PerClassMetrics(precision, recall, f1)
    macro_precision = sum(item.precision for item in per_class.values()) / NUM_LABELS
    macro_recall = sum(item.recall for item in per_class.values()) / NUM_LABELS
    macro_f1 = sum(item.f1 for item in per_class.values()) / NUM_LABELS
    return ClassificationMetrics(
        accuracy=sum(actual == guess for actual, guess in zip(expected, predicted, strict=True)) / len(expected),
        precision=macro_precision,
        recall=macro_recall,
        f1=macro_f1,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=tuple(tuple(row) for row in matrix),
    )
