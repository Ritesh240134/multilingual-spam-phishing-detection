"""Local single-email inference without a production API or network access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ml.constants import ID_TO_LABEL
from ml.preprocessing import preprocess_email
from ml.training.trainer import select_device


@dataclass(slots=True)
class EmailClassifier:
    """Run a locally supplied classification artifact on one untrusted email."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    max_sequence_length: int = 256
    max_input_chars: int = 200_000
    device: str | None = None

    @classmethod
    def load(cls, model_dir: str | Path, max_sequence_length: int = 256, device: str | None = None) -> EmailClassifier:
        """Load a local artifact directory; remote model downloads are disabled."""

        path = str(model_dir)
        return cls(
            model=AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True),
            tokenizer=AutoTokenizer.from_pretrained(path, local_files_only=True),
            max_sequence_length=max_sequence_length,
            device=device,
        )

    @torch.no_grad()
    def predict(self, subject: str | None, body: str | None, language: str | None = None) -> dict[str, object]:
        """Return a three-class result without logging, fetching, or executing content.

        ``language`` is accepted for future routing but does not translate or alter
        the original email text in this multilingual XLM-R pipeline.
        """

        del language
        if not isinstance(subject, (str, type(None))) or not isinstance(body, (str, type(None))):
            raise TypeError("subject and body must be strings or None")
        if not (subject or "").strip() and not (body or "").strip():
            raise ValueError("subject or body must contain text")
        if len(subject or "") + len(body or "") > self.max_input_chars:
            raise ValueError(f"email input exceeds {self.max_input_chars} characters")
        cleaned_text = preprocess_email(subject=subject, body=body, max_input_chars=self.max_input_chars).cleaned_text
        encoded = self.tokenizer(cleaned_text, truncation=True, max_length=self.max_sequence_length, padding=True, return_tensors="pt")
        device = select_device(self.device)
        self.model.to(device).eval()
        logits = self.model(**{name: value.to(device) for name, value in encoded.items()}).logits[0]
        probabilities = torch.softmax(logits, dim=0).cpu().tolist()
        predicted_id = int(torch.argmax(logits).item())
        return {
            "label": ID_TO_LABEL[predicted_id],
            "confidence": float(probabilities[predicted_id]),
            "probabilities": {ID_TO_LABEL[index]: float(score) for index, score in enumerate(probabilities)},
        }
