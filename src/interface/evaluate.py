"""Offline evaluation utilities for sign-language recognition.

This module evaluates a trained classifier on a test set, prints common
classification metrics, and generates a confusion matrix figure.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns


import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


MODEL_PATH = "model.pth"
INPUT_SIZE = 63
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
BATCH_SIZE = 64
CONFUSION_MATRIX_PATH = "confusion_matrix.png"
REPORT_PATH = "evaluation_report.txt"


def build_model(input_size: int, num_classes: int) -> nn.Module:
    """Instantiate the project model.

    Args:
        input_size: Number of input features.
        num_classes: Number of output classes.

    Returns:
        A PyTorch model instance.
    """
    from model import SignModel  # type: ignore

    return SignModel(input_size=input_size, num_classes=num_classes)


def load_model(path: Path, num_classes: int) -> nn.Module:
    """Load trained weights from disk.

    Args:
        path: Path to the `.pth` file.
        num_classes: Number of output classes.

    Returns:
        Model configured for evaluation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = build_model(input_size=INPUT_SIZE, num_classes=num_classes)
    state_dict = torch.load(path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_test_dataloader(batch_size: int) -> DataLoader:
    """Load the team test dataloader.

    Args:
        batch_size: Batch size used by the dataloader.

    Returns:
        A dataloader yielding `(features, labels)`.
    """
    from dataset_loader import get_test_dataloader as team_loader  # type: ignore

    return team_loader(batch_size=batch_size)


def run_evaluation(
    model: nn.Module,
    dataloader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    class_names: Sequence[str],
) -> Tuple[List[int], List[int]]:
    """Run inference on the full test set.

    Args:
        model: PyTorch model in evaluation mode.
        dataloader: Test dataloader yielding feature and label batches.
        class_names: Ordered output labels.

    Returns:
        Tuple of `y_true` and `y_pred` integer label lists.
    """
    del class_names
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for features, labels in dataloader:
            logits = model(features.float())
            probabilities = functional.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    return y_true, y_pred


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    """Plot and save the confusion matrix heatmap.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.
        class_names: Ordered label names.
        output_path: Output image path.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix - Sign Language Recognition")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def top_misclassified_pairs(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str], top_k: int = 5
) -> List[Tuple[str, str, int]]:
    """Compute the most common error pairs.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.
        class_names: Ordered label names.
        top_k: Number of pairs to return.

    Returns:
        List of `(true_label, predicted_label, count)` tuples.
    """
    counts = Counter(
        (class_names[true_index], class_names[pred_index])
        for true_index, pred_index in zip(y_true, y_pred)
        if true_index != pred_index
    )
    return [
        (true_label, predicted_label, count)
        for (true_label, predicted_label), count in counts.most_common(top_k)
    ]


def write_report(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str], output_path: Path
) -> str:
    """Generate and persist the text classification report.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.
        class_names: Ordered label names.
        output_path: Destination file path.

    Returns:
        The generated report text.
    """
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        zero_division=0,
    )
    output_path.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    """Execute the offline evaluation workflow."""
    base_path = Path(__file__).resolve().parent
    model = load_model(base_path / MODEL_PATH, num_classes=len(CLASS_NAMES))
    dataloader = get_test_dataloader(batch_size=BATCH_SIZE)
    y_true, y_pred = run_evaluation(model, dataloader, CLASS_NAMES)

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    report_text = write_report(
        y_true=y_true,
        y_pred=y_pred,
        class_names=CLASS_NAMES,
        output_path=base_path / REPORT_PATH,
    )
    print(report_text)

    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=CLASS_NAMES,
        output_path=base_path / CONFUSION_MATRIX_PATH,
    )

    misclassifications = top_misclassified_pairs(y_true, y_pred, CLASS_NAMES)
    if misclassifications:
        print("Top misclassified pairs:")
        for true_label, predicted_label, count in misclassifications:
            print(f"- {true_label} -> {predicted_label}: {count}")
    else:
        print("No misclassified pairs found.")


if __name__ == "__main__":
    main()