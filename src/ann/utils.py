from __future__ import annotations

import numpy as np


VOWEL_LABELS = {0, 4, 8, 14, 20}  # A, E, I, O, U


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def labels_to_vowel_consonant(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, np.fromiter(VOWEL_LABELS, dtype=np.int64)).astype(np.int64)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> str:
    lines = ["class precision recall f1 support"]
    eps = 1e-12

    for cls in range(num_classes):
        true_pos = np.sum((y_true == cls) & (y_pred == cls))
        false_pos = np.sum((y_true != cls) & (y_pred == cls))
        false_neg = np.sum((y_true == cls) & (y_pred != cls))
        support = np.sum(y_true == cls)

        precision = true_pos / (true_pos + false_pos + eps)
        recall = true_pos / (true_pos + false_neg + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        lines.append(f"{cls:>5} {precision:>9.4f} {recall:>6.4f} {f1:>6.4f} {support:>7d}")

    lines.append(f"overall accuracy: {accuracy_score(y_true, y_pred):.4f}")
    return "\n".join(lines)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix
