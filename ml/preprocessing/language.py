"""Language detection that preserves the original email text."""

from __future__ import annotations

import re

try:
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0
except ImportError:  # pragma: no cover - exercised only before dependency installation
    detect = None  # type: ignore[assignment]
    LangDetectException = Exception


_MEANINGFUL_TEXT = re.compile(r"[^\W\d_]", re.UNICODE)


def detect_language(text: str) -> str:
    """Detect an ISO 639-1 language code without translating the text.

    Returns ``"und"`` for empty, too-short, or unclassifiable input. A small
    script fallback keeps Devanagari text identifiable if detection is not
    available at runtime; installing ``ml/requirements.txt`` enables broader
    multilingual detection.
    """

    if len(_MEANINGFUL_TEXT.findall(text)) < 3:
        return "und"
    if detect is not None:
        try:
            return str(detect(text))
        except LangDetectException:
            pass
    if any("\u0900" <= char <= "\u097f" for char in text):
        return "hi"
    return "und"
