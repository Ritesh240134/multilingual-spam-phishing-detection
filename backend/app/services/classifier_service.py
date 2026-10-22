"""Adapter joining API validation, Phase 3 inference, and preprocessing."""

from __future__ import annotations

from pathlib import Path
import re
from threading import Lock

from backend.app.core.config import MAX_INPUT_CHARS, get_model_dir
from backend.app.schemas.inference import AnalyzeRequest, AnalyzeResponse, SecurityIndicators
from ml.inference import EmailClassifier
from ml.preprocessing import EmailMessage, preprocess_email


class ClassifierService:
    """Lazily load only configured local artifacts and never log email content."""

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else get_model_dir()
        self._classifier: EmailClassifier | None = None
        self._load_attempted = False
        self._lock = Lock()

    def _get_classifier(self) -> EmailClassifier | None:
        with self._lock:
            if self._load_attempted:
                return self._classifier
            self._load_attempted = True
            if not self.model_dir.is_dir():
                return None
            try:
                self._classifier = EmailClassifier.load(self.model_dir)
            except (OSError, ValueError, RuntimeError):
                self._classifier = None
            return self._classifier

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        supplied_urls = "\n".join(request.urls or [])
        html_body = request.body if request.body and re.search(r"</?[A-Za-z][^>]*>", request.body) else None
        plain_body = None if html_body else request.body
        prep = preprocess_email(
            subject=request.subject,
            body="\n".join(part for part in (plain_body, supplied_urls) if part),
            html=html_body,
            headers={"From": request.sender} if request.sender else None,
            max_input_chars=MAX_INPUT_CHARS,
        )
        url_features = prep.url_features
        indicators = SecurityIndicators(
            url_count=url_features.url_count,
            suspicious_url_count=url_features.suspicious_url_count,
            ip_address_url_count=url_features.ip_address_url_count,
            shortener_url_count=url_features.shortener_url_count,
            suspicious_urls=list(url_features.suspicious_urls),
        )
        classifier = self._get_classifier()
        if classifier is None:
            return AnalyzeResponse(
                label=None,
                risk_score=None,
                detected_language=prep.detected_language if prep.detected_language != "und" else None,
                security_indicators=indicators,
                explanation="No usable local classifier artifact is configured; no prediction was made.",
                model_status="unavailable",
                inference_status="unavailable",
            )
        try:
            inference_text = "\n".join((prep.cleaned_text, *prep.extracted_urls))
            result = classifier.predict(None, inference_text)
        except (OSError, ValueError, RuntimeError):
            return AnalyzeResponse(
                label=None,
                risk_score=None,
                detected_language=prep.detected_language if prep.detected_language != "und" else None,
                security_indicators=indicators,
                explanation="Local inference failed; no prediction was returned.",
                model_status="available",
                inference_status="error",
            )
        label = result.get("label")
        probabilities = result.get("probabilities")
        score = None
        if isinstance(probabilities, dict):
            spam_probability = probabilities.get("spam")
            phishing_probability = probabilities.get("phishing")
            if isinstance(spam_probability, (int, float)) and isinstance(phishing_probability, (int, float)):
                score = round(float(spam_probability) + float(phishing_probability), 6)
        return AnalyzeResponse(
            label=label,  # type: ignore[arg-type]
            risk_score=score,
            detected_language=prep.detected_language if prep.detected_language != "und" else None,
            security_indicators=indicators,
            explanation=f"The local classifier predicted {label}; malicious-class probability is {score:.3f}." if score is not None else f"The local classifier predicted {label}; class probabilities were unavailable.",
            model_status="available",
            inference_status="completed",
        )


classifier_service = ClassifierService()
