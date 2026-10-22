"""Small explicit training loop for user-initiated local model training."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from ml.evaluation.metrics import ClassificationMetrics, calculate_metrics

from .config import TrainingConfig
from .data import EmailRecord
from .model import compute_class_weights, weighted_loss
from .tokenization import tokenize_record


def set_seed(seed: int) -> None:
    """Set reproducible random seeds for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested_device: str | None = None) -> torch.device:
    """Select GPU when available unless a specific device is requested."""

    return torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))


class TokenizedEmailDataset(Dataset[dict[str, torch.Tensor]]):
    """In-memory tokenized dataset for small-to-medium local training runs."""

    def __init__(self, records: list[EmailRecord], tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.items = []
        for record in records:
            item = tokenize_record(tokenizer, record, max_length)
            item["labels"] = torch.tensor(record.label, dtype=torch.long)
            self.items.append(item)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def train_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_records: list[EmailRecord],
    validation_records: list[EmailRecord],
    config: TrainingConfig,
) -> list[ClassificationMetrics]:
    """Train only when called explicitly; this function never fetches datasets."""

    if not train_records:
        raise ValueError("train_records cannot be empty")
    set_seed(config.seed)
    device = select_device(config.device)
    model.to(device)
    train_loader = DataLoader(TokenizedEmailDataset(train_records, tokenizer, config.max_sequence_length), batch_size=config.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    weights = compute_class_weights([record.label for record in train_records], device)
    history: list[ClassificationMetrics] = []
    for _ in range(config.num_epochs):
        model.train()
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            inputs = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad()
            logits = model(**inputs).logits
            weighted_loss(logits, labels, weights).backward()
            optimizer.step()
        if validation_records:
            history.append(evaluate_model(model, tokenizer, validation_records, config))
    return history


@torch.no_grad()
def evaluate_model(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, records: list[EmailRecord], config: TrainingConfig
) -> ClassificationMetrics:
    """Evaluate a local model and expose phishing-specific class metrics."""

    if not records:
        raise ValueError("records cannot be empty")
    device = select_device(config.device)
    model.to(device).eval()
    loader = DataLoader(TokenizedEmailDataset(records, tokenizer, config.max_sequence_length), batch_size=config.batch_size)
    expected: list[int] = []
    predicted: list[int] = []
    for batch in loader:
        labels = batch.pop("labels")
        logits = model(**{name: value.to(device) for name, value in batch.items()}).logits
        expected.extend(labels.tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
    return calculate_metrics(expected, predicted)
