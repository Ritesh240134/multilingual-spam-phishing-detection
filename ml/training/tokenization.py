"""XLM-R tokenizer creation and bounded email tokenization."""

from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .data import EmailRecord


def build_tokenizer(model_name: str = "xlm-roberta-base", local_files_only: bool = False) -> PreTrainedTokenizerBase:
    """Load the requested XLM-R tokenizer; no dataset is downloaded."""

    return AutoTokenizer.from_pretrained(model_name, use_fast=True, local_files_only=local_files_only)


def tokenize_record(tokenizer: PreTrainedTokenizerBase, record: EmailRecord, max_length: int) -> dict[str, Any]:
    """Tokenize a validated email with explicit truncation and fixed padding."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    encoded = tokenizer(
        record.combined_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    return {name: values.squeeze(0) for name, values in encoded.items()}
