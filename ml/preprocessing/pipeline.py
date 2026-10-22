"""Reusable multilingual email preprocessing pipeline."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence

from .html_utils import extract_html_text_and_urls
from .language import detect_language
from .schemas import EmailMessage, MetadataFeatures, PreprocessingResult
from .url_features import build_url_features, extract_urls


class EmailPreprocessor:
    """Safely normalize untrusted email fields for downstream ML components.

    The preprocessor is entirely local and deterministic except for language
    detection. It does not render HTML, execute attachments, open URLs, make
    network requests, or log message content.
    """

    def __init__(self, max_input_chars: int = 200_000) -> None:
        if max_input_chars <= 0:
            raise ValueError("max_input_chars must be positive")
        self.max_input_chars = max_input_chars

    def preprocess(self, email: EmailMessage) -> PreprocessingResult:
        """Return cleaned text, language, metadata, and static URL features."""

        self._validate_email(email)
        subject = email.subject or self._header(email.headers, "subject") or ""
        plain_body = email.body or ""
        html = email.html or ""
        html_text, html_urls = extract_html_text_and_urls(html) if html else ("", ())
        source_text = "\n".join(part for part in (subject, plain_body, html_text) if part)
        cleaned_text = self._normalize_text(source_text)
        urls = extract_urls(f"{subject}\n{plain_body}\n{html_text}", html_urls)
        url_features = build_url_features(urls)
        metadata = MetadataFeatures(
            sender=self._header(email.headers, "from"),
            recipient=self._header(email.headers, "to"),
            subject_length=len(subject),
            body_length=len(plain_body) + len(html_text),
            attachment_count=len(email.attachments or ()),
            link_count=url_features.url_count,
            is_html=bool(html),
            has_plain_text=bool(plain_body),
        )
        return PreprocessingResult(
            cleaned_text=cleaned_text,
            detected_language=detect_language(cleaned_text),
            metadata_features=metadata,
            url_features=url_features,
            extracted_urls=urls,
        )

    def _validate_email(self, email: EmailMessage) -> None:
        if not isinstance(email, EmailMessage):
            raise TypeError("email must be an EmailMessage instance")
        for name in ("subject", "body", "html"):
            value = getattr(email, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"email.{name} must be a string or None")
        if email.headers is not None:
            if not isinstance(email.headers, Mapping):
                raise TypeError("email.headers must be a mapping or None")
            if any(not isinstance(key, str) or not isinstance(value, str) for key, value in email.headers.items()):
                raise TypeError("email.headers keys and values must be strings")
        if email.attachments is not None and (isinstance(email.attachments, (str, bytes)) or not isinstance(email.attachments, Sequence)):
            raise TypeError("email.attachments must be a sequence or None")
        total_length = sum(len(value or "") for value in (email.subject, email.body, email.html))
        total_length += sum(len(key) + len(value) for key, value in (email.headers or {}).items())
        if total_length > self.max_input_chars:
            raise ValueError(f"email input exceeds {self.max_input_chars} characters")

    @staticmethod
    def _header(headers: Mapping[str, str] | None, name: str) -> str | None:
        if not headers:
            return None
        expected = name.casefold()
        for key, value in headers.items():
            if key.casefold() == expected:
                return value
        return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize Unicode and whitespace while preserving multilingual signals."""

        normalized = unicodedata.normalize("NFC", text)
        normalized = "".join(
            char if char in "\n\t" or unicodedata.category(char) != "Cc" else " "
            for char in normalized
        )
        return re.sub(r"\s+", " ", normalized).strip()


def preprocess_email(
    subject: str | None = None,
    body: str | None = None,
    html: str | None = None,
    headers: Mapping[str, str] | None = None,
    attachments: Sequence[object] | None = None,
    *,
    max_input_chars: int = 200_000,
) -> PreprocessingResult:
    """Convenience wrapper for training and inference callers."""

    return EmailPreprocessor(max_input_chars=max_input_chars).preprocess(
        EmailMessage(subject=subject, body=body, html=html, headers=headers, attachments=attachments)
    )
