"""Unit tests for Phase 2 multilingual email preprocessing."""

from __future__ import annotations

import pytest

from ml.preprocessing import EmailMessage, EmailPreprocessor, preprocess_email


def test_english_email_preserves_text_and_metadata() -> None:
    result = preprocess_email(
        subject="Account update",
        body="Hello, please review your account details.",
        headers={"From": "support@example.com", "To": "user@example.net"},
        attachments=[{"name": "notice.pdf"}],
    )
    assert "Account update" in result.cleaned_text
    assert result.detected_language == "en"
    assert result.metadata_features.sender == "support@example.com"
    assert result.metadata_features.recipient == "user@example.net"
    assert result.metadata_features.attachment_count == 1


def test_hindi_email_is_not_translated() -> None:
    body = "प्रिय ग्राहक, कृपया अपने खाते की जानकारी तुरंत सत्यापित करें।"
    result = preprocess_email(subject="खाता सत्यापन", body=body)
    assert body in result.cleaned_text
    assert result.detected_language == "hi"


def test_spanish_email_language_detection() -> None:
    result = preprocess_email(
        subject="Actualización importante",
        body="Hola cliente, revise la información de su cuenta para evitar problemas.",
    )
    assert result.detected_language == "es"


def test_mixed_language_email_retains_both_languages() -> None:
    result = preprocess_email(subject="Urgent aviso", body="Hola, कृपया verify your account ahora.")
    assert "Hola" in result.cleaned_text
    assert "कृपया" in result.cleaned_text
    assert result.detected_language != "und"


def test_html_is_extracted_without_scripts_or_execution() -> None:
    result = preprocess_email(
        html="<html><body><p>Verify your account</p><a href='https://example.com/login'>Open</a><script>alert(1)</script></body></html>"
    )
    assert "Verify your account" in result.cleaned_text
    assert "alert(1)" not in result.cleaned_text
    assert result.metadata_features.is_html is True
    assert result.metadata_features.has_plain_text is False
    assert result.url_features.https_url_count == 1


def test_suspicious_urls_are_detected_statically() -> None:
    result = preprocess_email(
        body="Click http://192.0.2.10/login and https://bit.ly/reset for immediate action."
    )
    assert result.url_features.url_count == 2
    assert result.url_features.ip_address_url_count == 1
    assert result.url_features.shortener_url_count == 1
    assert result.url_features.suspicious_url_count >= 1


def test_empty_email_has_safe_defaults() -> None:
    result = preprocess_email()
    assert result.cleaned_text == ""
    assert result.detected_language == "und"
    assert result.metadata_features.body_length == 0
    assert result.url_features.url_count == 0


@pytest.mark.parametrize(
    "email",
    [
        "not-an-email-object",
        EmailMessage(body=123),  # type: ignore[arg-type]
        EmailMessage(headers={"From": 123}),  # type: ignore[dict-item]
    ],
)
def test_malformed_input_is_rejected(email: object) -> None:
    with pytest.raises(TypeError):
        EmailPreprocessor().preprocess(email)  # type: ignore[arg-type]


def test_excessively_large_input_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        preprocess_email(body="x" * 101, max_input_chars=100)
