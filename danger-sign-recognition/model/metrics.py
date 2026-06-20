from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch


@dataclass
class AverageMeter:
    """Track a weighted running average for losses and metrics."""

    name: str
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


@dataclass(frozen=True)
class ClassMetrics:
    class_id: str
    support: int
    predicted: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class EvaluationSummary:
    loss: float
    top1: float
    top3: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    support: int
    classes: list[str]
    confusion_matrix: list[list[int]]
    per_class: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def topk_correct(logits: torch.Tensor, labels: torch.Tensor, topk: Sequence[int] = (1, 3)) -> dict[int, int]:
    """Return the number of correct predictions for each requested top-k."""

    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape [batch, classes], got {tuple(logits.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"Expected labels with shape [batch], got {tuple(labels.shape)}")

    max_k = min(max(topk), logits.size(1))
    _, predictions = logits.topk(max_k, dim=1, largest=True, sorted=True)
    predictions = predictions.t()
    correct = predictions.eq(labels.view(1, -1).expand_as(predictions))

    scores: dict[int, int] = {}
    for k in topk:
        clipped_k = min(k, max_k)
        scores[k] = int(correct[:clipped_k].reshape(-1).float().sum().item())
    return scores


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for label, pred in zip(y_true, y_pred):
        if 0 <= label < num_classes and 0 <= pred < num_classes:
            matrix[int(label)][int(pred)] += 1
    return matrix


def per_class_metrics(matrix: Sequence[Sequence[int]], classes: Sequence[str]) -> list[ClassMetrics]:
    results: list[ClassMetrics] = []
    for index, class_id in enumerate(classes):
        true_positive = int(matrix[index][index])
        support = int(sum(matrix[index]))
        predicted = int(sum(row[index] for row in matrix))
        false_positive = predicted - true_positive
        false_negative = support - true_positive
        precision = safe_divide(true_positive, predicted)
        recall = safe_divide(true_positive, support)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        results.append(
            ClassMetrics(
                class_id=class_id,
                support=support,
                predicted=predicted,
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )
    return results


def summarize_classification(
    *,
    loss: float,
    top1: float,
    top3: float,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
) -> EvaluationSummary:
    matrix = confusion_matrix(y_true, y_pred, len(classes))
    class_rows = per_class_metrics(matrix, classes)
    support = sum(row.support for row in class_rows)
    macro_precision = safe_divide(sum(row.precision for row in class_rows), len(class_rows))
    macro_recall = safe_divide(sum(row.recall for row in class_rows), len(class_rows))
    macro_f1 = safe_divide(sum(row.f1 for row in class_rows), len(class_rows))
    weighted_f1 = safe_divide(sum(row.f1 * row.support for row in class_rows), support)

    return EvaluationSummary(
        loss=float(loss),
        top1=float(top1),
        top3=float(top3),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        support=support,
        classes=list(classes),
        confusion_matrix=matrix,
        per_class=[asdict(row) for row in class_rows],
    )
