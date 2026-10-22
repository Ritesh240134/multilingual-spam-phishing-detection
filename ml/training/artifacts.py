"""Local model/tokenizer persistence helpers."""

from __future__ import annotations

from pathlib import Path

from transformers import PreTrainedModel, PreTrainedTokenizerBase


def save_artifacts(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, output_dir: str | Path) -> Path:
    """Save model and tokenizer locally; callers choose the storage location."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer.save_pretrained(destination)
    return destination
