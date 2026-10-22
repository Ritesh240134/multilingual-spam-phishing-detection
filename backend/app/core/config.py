"""Runtime configuration for the API."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "xlm-roberta-email-classifier"


def get_model_dir() -> Path:
    """Resolve the configured local artifact path without loading it."""

    configured = os.getenv("EMAIL_CLASSIFIER_MODEL_DIR")
    return Path(configured).expanduser() if configured else DEFAULT_MODEL_DIR


MAX_INPUT_CHARS = 200_000
MAX_URLS = 100
