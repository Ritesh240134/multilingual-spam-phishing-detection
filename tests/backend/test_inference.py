"""API tests use a mocked classifier and never download model artifacts."""

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import classifier_service as service_module


client = TestClient(app)


class MockClassifier:
    def predict(self, subject: str | None, body: str | None) -> dict[str, object]:
        assert subject is None
        assert body is not None and "Aviso" in body and "Hola" in body
        return {"label": "spam", "confidence": 0.70, "probabilities": {"legitimate": 0.09, "spam": 0.70, "phishing": 0.21}}


def test_successful_inference_with_mocked_local_classifier(monkeypatch) -> None:
    monkeypatch.setattr(service_module.classifier_service, "_classifier", MockClassifier())
    monkeypatch.setattr(service_module.classifier_service, "_load_attempted", True)
    response = client.post("/api/v1/analyze", json={"subject": "Aviso", "body": "Hola, revise su cuenta"})
    assert response.status_code == 200
    result = response.json()
    assert result["label"] == "spam"
    assert result["risk_score"] == 0.91
    assert result["model_status"] == "available"
    assert result["inference_status"] == "completed"
    assert result["detected_language"] == "es"


def test_invalid_empty_and_oversized_input() -> None:
    assert client.post("/api/v1/analyze", json={}).status_code == 422
    assert client.post("/api/v1/analyze", json={"subject": "", "body": "  "}).status_code == 422
    assert client.post("/api/v1/analyze", json={"body": "x" * 200_001}).status_code == 422
    assert client.post("/api/v1/analyze", json={"body": "text", "unexpected": True}).status_code == 422


def test_response_schema_includes_required_fields() -> None:
    response = client.post("/api/v1/analyze", json={"body": "Visit https://192.0.2.2/login"})
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"label", "risk_score", "detected_language", "security_indicators", "explanation", "model_status", "inference_status"}
    assert result["security_indicators"]["ip_address_url_count"] == 1


def test_missing_model_artifacts_returns_no_fabricated_prediction(tmp_path, monkeypatch) -> None:
    from backend.app.services.classifier_service import ClassifierService

    service = ClassifierService(tmp_path / "missing")
    monkeypatch.setattr(service_module.classifier_service, "model_dir", tmp_path / "missing")
    monkeypatch.setattr(service_module.classifier_service, "_classifier", None)
    monkeypatch.setattr(service_module.classifier_service, "_load_attempted", False)
    response = client.post("/api/v1/analyze", json={"body": "Please review the account"})
    assert response.status_code == 200
    result = response.json()
    assert result["label"] is None
    assert result["risk_score"] is None
    assert result["model_status"] == "unavailable"
    assert result["inference_status"] == "unavailable"


def test_health_remains_operational() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_local_frontend_cors_is_allowed_but_unlisted_origin_is_not() -> None:
    allowed = client.options(
        "/api/v1/analyze",
        headers={
            "Origin": "http://127.0.0.1:5500",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "http://127.0.0.1:5500"

    denied = client.options(
        "/api/v1/analyze",
        headers={
            "Origin": "http://untrusted.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert "access-control-allow-origin" not in denied.headers


def test_no_remote_model_download_during_startup_or_inference(monkeypatch, tmp_path) -> None:
    from backend.app.services.classifier_service import ClassifierService

    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected model load")

    monkeypatch.setattr("ml.inference.classifier.AutoModelForSequenceClassification.from_pretrained", forbidden)
    monkeypatch.setattr("ml.inference.classifier.AutoTokenizer.from_pretrained", forbidden)
    service = ClassifierService(tmp_path / "absent")
    monkeypatch.setattr(service_module, "classifier_service", service)
    assert client.get("/health").status_code == 200
    result = client.post("/api/v1/analyze", json={"body": "Hello there"}).json()
    assert result["model_status"] == "unavailable"
