# Multilingual Spam and Phishing Email Detection System

A cybersecurity platform for detecting spam and phishing emails across multiple languages. The planned system will combine transformer-based email-text classification with email metadata and URL-risk signals, then present clear explanations for its predictions.

## Current phase

**Phase 3 — multilingual transformer text-classification pipeline** is complete. It adds XLM-R dataset validation, deterministic splitting, tokenizer/model factories, weighted-loss training utilities, metrics, local artifact persistence, and local single-email inference. No dataset is downloaded and no model training starts automatically.

## Architecture

```text
frontend/       Future React application
backend/        FastAPI service and modular application packages
ml/             Future preprocessing, training, inference, and evaluation code
datasets/       Local, untracked dataset storage
tests/          Backend and ML test suites
config/         Non-secret configuration template
docs/           Project documentation
```

The backend is organized into API routes, core concerns, models, schemas, services, and utilities so future capabilities can be added without tightly coupling application layers.

## Phase 2 preprocessing architecture

`ml.preprocessing.EmailPreprocessor` accepts untrusted in-memory `EmailMessage` values and produces a typed `PreprocessingResult` containing:

- `cleaned_text`: subject, plain body, and visible HTML text normalized with Unicode NFC and whitespace cleanup. Original languages and lexical signals are preserved; messages are never translated.
- `detected_language`: an ISO 639-1 language code from `langdetect`, or `und` when language cannot be determined.
- `metadata_features`: sender/recipient headers, lengths, attachment count, link count, and HTML/plain-text indicators.
- `url_features`: static counts and lexical indicators for HTTP/HTTPS, IP-address URLs, shorteners, suspicious characteristics, and URL lengths.

HTML is tokenized with Python's standard library only. Script/style content is excluded; no HTML is rendered, no JavaScript executes, and URLs are never opened, resolved, downloaded, or logged. Inputs above the configured 200,000-character limit are rejected before processing.

The package is split into `pipeline.py`, `html_utils.py`, `language.py`, `url_features.py`, and `schemas.py` so later training and inference code can reuse the same contracts.

## Requirements

- Python 3.11 or newer
- A virtual environment manager (recommended)
- PostgreSQL and Redis will be required in later phases; neither is needed for the current health check
- Node.js will be required when the React frontend is introduced

## Setup and basic check

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r backend/requirements.txt
pip install -r ml/requirements.txt
uvicorn app.main:app --app-dir backend --reload
```

Then open `http://127.0.0.1:8000/health`. The endpoint responds with:

```json
{"status":"ok","message":"API is running"}
```

For development tools and the Phase 2 unit tests, install `pip install -r requirements-dev.txt` after the runtime requirements, then run:

```bash
python -m pytest tests/ml
```

Dataset schema, privacy rules, and future-source guidance are documented in `datasets/README.md`. No dataset is downloaded automatically.

## Phase 3 transformer pipeline

The classifier uses Hugging Face `AutoTokenizer` and `AutoModelForSequenceClassification` with `xlm-roberta-base` and three classes: `0` legitimate/ham, `1` spam, and `2` phishing. XLM-R handles multilingual input directly, so emails are not translated; translation can lose language-specific phishing signals and Unicode obfuscation.

Training data is a user-supplied local JSONL file with `email_id`, `subject`, `body`, `language`, and `label`. The pipeline validates records, performs seeded label-stratified splits, and uses class-weighted cross-entropy. Evaluation reports accuracy, macro precision/recall/F1, per-class metrics, confusion matrix, phishing recall, and phishing F1. See `docs/transformer-training.md` for the architecture, dataset format, limitations, and local training guidance.

The single-email classifier loads only a local saved model and accepts subject/body text. It does not expose a production API, retrieve URLs, execute content, download a model, or log email bodies. Synthetic test examples validate code paths only and must never be cited as real-world security performance.

## Configuration

Copy `config/.env.example` to an untracked local environment file when configuration is introduced. Do not commit passwords, API keys, database credentials, tokens, or other secrets.
