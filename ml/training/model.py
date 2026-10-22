"""XLM-R three-class model and imbalance-aware loss utilities."""

from __future__ import annotations

from collections import Counter

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from ml.constants import ID_TO_LABEL, LABEL_TO_ID, NUM_LABELS


def build_model(model_name: str = "xlm-roberta-base", local_files_only: bool = False) -> PreTrainedModel:
    """Load XLM-R with three labels from a local cache or Hugging Face Hub."""

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        local_files_only=local_files_only,
    )


def compute_class_weights(labels: list[int], device: torch.device | str) -> Tensor:
    """Return inverse-frequency class weights for weighted cross-entropy."""

    if not labels:
        raise ValueError("labels cannot be empty")
    counts = Counter(labels)
    total = len(labels)
    return torch.tensor(
        [total / (NUM_LABELS * counts.get(label, 1)) for label in range(NUM_LABELS)],
        dtype=torch.float32,
        device=device,
    )


def weighted_loss(logits: Tensor, labels: Tensor, class_weights: Tensor) -> Tensor:
    """Calculate weighted loss so rare phishing examples have greater impact."""

    return CrossEntropyLoss(weight=class_weights)(logits, labels)
