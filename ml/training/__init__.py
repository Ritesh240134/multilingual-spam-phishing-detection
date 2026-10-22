"""XLM-R dataset, training, and artifact interfaces."""

from .config import TrainingConfig
from .data import EmailRecord, load_jsonl_dataset, validate_record
from .model import build_model
from .tokenization import build_tokenizer

__all__ = ["EmailRecord", "TrainingConfig", "build_model", "build_tokenizer", "load_jsonl_dataset", "validate_record"]
