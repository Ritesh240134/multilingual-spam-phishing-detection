# Phase 6 data readiness (offline)

This guide prepares the existing Phase 3 pipeline for an explicitly initiated local training run. It does not download a dataset or model. Keep the dataset and model weights outside version control; `datasets/` and `artifacts/` are git-ignored.

## Current readiness

The local XLM-RoBERTa files are available at `C:\models\xlm-roberta-base`. SpaPhish provides 664 legitimate and 731 phishing records. The extracted Corpus 200 files do not include a reliable per-email ham/spam mapping, and no other spam corpus with verified message labels and reuse rights is currently approved. `datasets/training.jsonl` is therefore not ready, and training must wait for an authorized spam source. Do not infer Corpus 200 labels from file order or use a corpus whose email-text reuse terms are unclear.

Once the dataset author supplies a trustworthy Corpus 200 label manifest and confirms the stated CC BY 4.0 terms, audit it before preparing JSONL. Otherwise, obtain another spam dataset with both documented per-record labels and clear rights to use the message text. Do not train until all three project classes have verified examples and each has at least 10 records for the configured validation and test splits.

## Dataset format and validation

Use UTF-8 JSONL: one JSON object per nonblank line, with these loader fields:

```json
{"email_id":"mail-0001","subject":"Account notice","body":"Please review your account.","language":"en","label":0}
```

`email_id`, `subject`, `body`, and `language` must be strings. The ID and language must not be blank, and subject or body must contain text. `label` must be an integer (not a boolean): `0` = legitimate, `1` = spam, `2` = phishing. Do not merge spam and phishing. The loader rejects malformed JSON, missing required fields, empty datasets, and invalid labels. It does not enforce unique IDs, provenance, label correctness, or balanced classes; review those manually and remove duplicates/conflicting labels before training. Preserve source language and anonymize sensitive content. Extra object fields are ignored by the loader.

The default split is deterministic with seed `42`: approximately 80% train, 10% validation, 10% test, stratified by label. Counts are rounded down per class; classes smaller than 10 can have no validation or test example. Have at least 10 examples of **each** class for all three splits to contain every class; substantially larger, representative sets are needed for meaningful metrics. First inspect actual counts without training:

```powershell
python -c "from collections import Counter; from ml.training.data import load_jsonl_dataset; r=load_jsonl_dataset(r'datasets/training.jsonl'); c=Counter(x.label for x in r); print('total:',len(r)); print({name:c.get(i,0) for i,name in enumerate(('legitimate','spam','phishing'))})"
```

Put your own approved JSONL at `datasets/training.jsonl` (that path is ignored by Git). Never use test fixtures as training data.

## Make XLM-R weights available offline

Obtain/approve the XLM-R base snapshot separately, then place its complete Hugging Face model and tokenizer files in a local directory, for example `C:\\models\\xlm-roberta-base`. The directory must contain the model configuration and weights plus tokenizer files (such as `config.json`, `model.safetensors` or `pytorch_model.bin`, tokenizer configuration, and vocabulary/sentencepiece data). Do not point at an empty cache directory. Training below passes that directory to both existing builders with `local_files_only=True`; absent files cause a local error instead of a network request.

## Train, evaluate, and save

From the project root, set `$env:PHASE6_XLMR_DIR` to the local snapshot directory in this PowerShell command. It checks that each class can appear in validation and test, uses the existing seeded split and weighted-loss trainer, evaluates the held-out test split, and saves model plus tokenizer at `artifacts/xlm-roberta-email-classifier`:

```powershell
$env:PHASE6_XLMR_DIR = 'C:\models\xlm-roberta-base'
@'
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
from ml.training.artifacts import save_artifacts
from ml.training.config import TrainingConfig
from ml.training.data import load_jsonl_dataset
from ml.training.model import build_model
from ml.training.split import split_records
from ml.training.tokenization import build_tokenizer
from ml.training.trainer import evaluate_model, train_model

weights = Path(os.environ["PHASE6_XLMR_DIR"])
records = load_jsonl_dataset("datasets/training.jsonl")
counts = Counter(record.label for record in records)
print("class_counts:", {name: counts.get(i, 0) for i, name in enumerate(("legitimate", "spam", "phishing"))})
if any(counts.get(i, 0) < 10 for i in range(3)):
    raise SystemExit("Need at least 10 examples per class for all classes in validation and test splits.")
train, validation, test = split_records(records, validation_fraction=0.1, test_fraction=0.1, seed=42)
config = TrainingConfig(model_name=str(weights), output_dir="artifacts/xlm-roberta-email-classifier", seed=42)
tokenizer = build_tokenizer(str(weights), local_files_only=True)
model = build_model(str(weights), local_files_only=True)
history = train_model(model, tokenizer, train, validation, config)
test_metrics = evaluate_model(model, tokenizer, test, config)
artifact = save_artifacts(model, tokenizer, config.output_dir)
print("validation_by_epoch:", json.dumps([asdict(item) for item in history]))
print("test_metrics:", json.dumps(asdict(test_metrics)))
print("artifact:", artifact.resolve())
'@ | python -
```

The printed metrics are results from that run only. Review each class's precision, recall, F1, and the confusion matrix, especially phishing recall. Do not treat small or nonrepresentative validation/test sets as evidence of production quality.

## Verify through the existing inference API

The artifact directory above matches the backend default. To use another output directory, set `EMAIL_CLASSIFIER_MODEL_DIR` to its absolute path before starting the backend. For the default artifact and a local API:

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, send a sample request (replace with a suitable non-sensitive evaluation example):

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/analyze -Method Post -ContentType 'application/json' -Body '{"subject":"Account notice","sender":"sender@example.test","body":"Please review your account","urls":["https://example.test"]}'
```

A successful local load reports `model_status: available` and `inference_status: completed`. The API uses local-only artifact loading and does not fetch weights. If no usable artifact is found it returns an unavailable status and no prediction.
