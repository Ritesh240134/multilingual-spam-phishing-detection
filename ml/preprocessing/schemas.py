"""Typed data contracts for safe, reusable email preprocessing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EmailMessage:
    """Untrusted email fields accepted by the preprocessing pipeline.

    This object holds only in-memory values. Attachments are represented by
    metadata placeholders and are never opened or executed by this phase.
    """

    subject: str | None = None
    body: str | None = None
    html: str | None = None
    headers: Mapping[str, str] | None = None
    attachments: Sequence[object] | None = None


@dataclass(frozen=True, slots=True)
class URLFeatures:
    """Non-network URL indicators extracted from an email."""

    url_count: int
    unique_url_count: int
    http_url_count: int
    https_url_count: int
    ip_address_url_count: int
    shortener_url_count: int
    suspicious_url_count: int
    longest_url_length: int
    average_url_length: float
    suspicious_urls: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MetadataFeatures:
    """Basic in-memory email metadata features for later model use."""

    sender: str | None
    recipient: str | None
    subject_length: int
    body_length: int
    attachment_count: int
    link_count: int
    is_html: bool
    has_plain_text: bool


@dataclass(frozen=True, slots=True)
class PreprocessingResult:
    """Structured output consumed later by training and inference code."""

    cleaned_text: str
    detected_language: str
    metadata_features: MetadataFeatures
    url_features: URLFeatures
    extracted_urls: tuple[str, ...] = field(default_factory=tuple)
