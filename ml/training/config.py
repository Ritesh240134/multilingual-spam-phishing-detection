"""Configuration for reproducible local transformer training."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Explicit settings for a local XLM-R classification run."""

    model_name: str = "xlm-roberta-base"
    output_dir: str = "artifacts/xlm-roberta-email-classifier"
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_sequence_length: int = 256
    seed: int = 42
    device: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0 or self.num_epochs <= 0 or self.max_sequence_length <= 0:
            raise ValueError("batch_size, num_epochs, and max_sequence_length must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
