# Signalcheck frontend

This is a dependency-free HTML/CSS/JavaScript UI for the existing FastAPI `POST /api/v1/analyze` endpoint. It uses the backend request fields `subject`, `sender`, `body`, and optional `urls`, and handles `label`, `risk_score`, `detected_language`, `security_indicators`, `explanation`, `model_status`, and `inference_status` from the API response.

## Run locally

Open two terminals from the project root.

Terminal 1 — start the backend:

```powershell
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2 — serve the frontend:

```powershell
python -m http.server 5500 --directory frontend
```

Then open <http://127.0.0.1:5500>. The default API URL is `http://127.0.0.1:8000`. To use another API origin, edit `apiBaseUrl` in `config.js`; the backend must allow that frontend origin through CORS.

## Demonstrate the UI

1. Paste a subject or body (in any language); optionally add a sender and one URL per line.
2. Select **Analyze email** and wait for the API response.
3. When a trained local artifact is configured, the page displays the classifier label and only displays a risk score when the API supplies one.
4. Without a model artifact, the page explicitly says no prediction is available and can still show static URL indicators. It does not substitute a heuristic label.

The frontend does not save message contents. Submitted HTML is sent as text to the API's safe preprocessing path; the browser never inserts or executes it. Result values, including suspicious URLs, are inserted as text. URLs are not opened by the UI or API.

## Checks

Run the backend and API contract suite from the project root:

```powershell
python -m pytest tests -v
```

The frontend has no build step or third-party runtime dependencies. It can be served with Python's standard HTTP server. A Node.js test runner is not configured in this repository; if Node is installed, browser interaction can be checked manually using the demonstration steps above.

## Limits

The interface does not perform inference itself. Real classification requires a trained model saved locally and configured with `EMAIL_CLASSIFIER_MODEL_DIR` for the backend. It does not report model accuracy.
