"""Phase 3 unit tests using synthetic data and a tiny local model configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForSequenceClassification, XLMRobertaConfig

from ml.constants import NUM_LABELS
from ml.inference import EmailClassifier
from ml.training.data import EmailRecord, load_jsonl_dataset, validate_record
from ml.training.tokenization import tokenize_record


class FakeTokenizer:
    """Tokenizer substitute that avoids remote model downloads in unit tests."""

    def __call__(self, text: str, **_: object) -> dict[str, torch.Tensor]:
        del text
        return {"input_ids": torch.tensor([[0, 1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1, 1]])}


class FixedLogitModel(torch.nn.Module):
    """Small deterministic inference substitute returning phishing logits."""

    def to(self, *_: object) -> FixedLogitModel:
        return self

    def eval(self) -> FixedLogitModel:
        return self

    def forward(self, **_: torch.Tensor) -> object:
        return type("Output", (), {"logits": torch.tensor([[0.1, 0.2, 2.0]])})()


def _record() -> EmailRecord:
    return EmailRecord("synthetic-1", "Hello", "Please verify your account", "en", 2)


def test_dataset_validation_and_loading(tmp_path: Path) -> None:
    dataset = tmp_path / "synthetic.jsonl"
    dataset.write_text(json.dumps({"email_id": "synthetic-1", "subject": "Hola", "body": "Mensaje", "language": "es", "label": 1}) + "\n", encoding="utf-8")
    loaded = load_jsonl_dataset(dataset)
    assert loaded[0].language == "es"
    assert loaded[0].label == 1


@pytest.mark.parametrize("label", [-1, 3, "phishing", True])
def test_label_validation(label: object) -> None:
    with pytest.raises(ValueError):
        validate_record({"email_id": "x", "subject": "s", "body": "b", "language": "en", "label": label})


def test_tokenization_uses_bounded_sequence() -> None:
    encoded = tokenize_record(FakeTokenizer(), _record(), max_length=4)  # type: ignore[arg-type]
    assert encoded["input_ids"].shape == (4,)


def test_xlm_r_model_initialization_from_tiny_config() -> None:
    config = XLMRobertaConfig(vocab_size=250002, hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64, num_labels=NUM_LABELS)
    model = AutoModelForSequenceClassification.from_config(config)
    assert model.config.num_labels == 3


def test_single_email_inference_returns_three_probabilities() -> None:
    classifier = EmailClassifier(model=FixedLogitModel(), tokenizer=FakeTokenizer(), max_sequence_length=4)  # type: ignore[arg-type]
    result = classifier.predict("Verify", "Hola, कृपया verify ahora.", language="hi")
    assert result["label"] == "phishing"
    assert set(result["probabilities"]) == {"legitimate", "spam", "phishing"}


def test_empty_or_invalid_inference_input_is_rejected() -> None:
    classifier = EmailClassifier(model=FixedLogitModel(), tokenizer=FakeTokenizer())  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        classifier.predict("", "")
    with pytest.raises(TypeError):
        classifier.predict(123, "body")  # type: ignore[arg-type]


def test_multilingual_record_preserves_source_text() -> None:
    record = EmailRecord("synthetic-hi", "Aviso", "कृपया verify your account", "hi", 2)
    assert "कृपया" in record.combined_text
