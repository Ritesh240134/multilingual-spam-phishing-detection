"""Local JSONL dataset loading and validation for email classification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.constants import LABEL_TO_ID


@dataclass(frozen=True, slots=True)
class EmailRecord:
    """Validated training example. Text remains in its original language."""

    email_id: str
    subject: str
    body: str
    language: str
    label: int

    @property
    def combined_text(self) -> str:
        """Combine fields with labels, preserving all original text."""

        return f"Subject: {self.subject}\nBody: {self.body}".strip()


def validate_record(raw: dict[str, Any]) -> EmailRecord:
    """Validate one JSON-compatible record without coercing unsafe values."""

    required = ("email_id", "subject", "body", "language", "label")
    missing = [field for field in required if field not in raw]
    if missing:
        raise ValueError(f"dataset record is missing fields: {', '.join(missing)}")
    for field in ("email_id", "subject", "body", "language"):
        if not isinstance(raw[field], str):
            raise TypeError(f"dataset field '{field}' must be a string")
    label = raw["label"]
    if not isinstance(label, int) or isinstance(label, bool) or label not in LABEL_TO_ID.values():
        raise ValueError("dataset label must be 0 (legitimate), 1 (spam), or 2 (phishing)")
    if not raw["email_id"].strip():
        raise ValueError("dataset field 'email_id' cannot be blank")
    if not (raw["subject"].strip() or raw["body"].strip()):
        raise ValueError("dataset record requires a non-empty subject or body")
    if not raw["language"].strip():
        raise ValueError("dataset field 'language' cannot be blank")
    return EmailRecord(**{field: raw[field] for field in required})


def load_jsonl_dataset(path: str | Path) -> list[EmailRecord]:
    """Load a user-supplied local JSONL file; this function never downloads data."""

    records: list[EmailRecord] = []
    with Path(path).open("r", encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}") from error
            if not isinstance(raw, dict):
                raise TypeError(f"dataset line {line_number} must be a JSON object")
            try:
                records.append(validate_record(raw))
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid dataset record on line {line_number}: {error}") from error
    if not records:
        raise ValueError("dataset contains no records")
    return records
