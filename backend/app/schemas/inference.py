"""Validated request and response contracts for email analysis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.core.config import MAX_INPUT_CHARS, MAX_URLS


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    subject: str | None = Field(default=None, max_length=20_000)
    body: str | None = Field(default=None, max_length=MAX_INPUT_CHARS)
    sender: str | None = Field(default=None, max_length=1_000)
    urls: list[str] | None = Field(default=None, max_length=MAX_URLS)

    @field_validator("subject", "body", "sender")
    @classmethod
    def reject_non_string_values(cls, value: str | None) -> str | None:
        return value

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and any(not url.strip() for url in value):
            raise ValueError("URLs must be non-empty strings")
        return value

    @model_validator(mode="after")
    def validate_email(self) -> AnalyzeRequest:
        if not (self.subject or "").strip() and not (self.body or "").strip():
            raise ValueError("subject or body must contain text")
        total = sum(len(part or "") for part in (self.subject, self.body, self.sender))
        total += sum(len(url) for url in (self.urls or []))
        if total > MAX_INPUT_CHARS:
            raise ValueError(f"combined input exceeds {MAX_INPUT_CHARS} characters")
        return self


class SecurityIndicators(BaseModel):
    url_count: int = 0
    suspicious_url_count: int = 0
    ip_address_url_count: int = 0
    shortener_url_count: int = 0
    suspicious_urls: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    label: Literal["legitimate", "spam", "phishing"] | None
    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    detected_language: str | None
    security_indicators: SecurityIndicators
    explanation: str
    model_status: Literal["available", "unavailable", "error"]
    inference_status: Literal["completed", "unavailable", "error"]
