# Phase 3: Multilingual Transformer Classification

## Model design

The classifier uses **XLM-RoBERTa base** through Hugging Face `AutoTokenizer` and `AutoModelForSequenceClassification`. XLM-R is chosen because it handles many writing systems in one model, allowing email text to remain in its original language. Translation is avoided because it can remove phishing-relevant wording, Unicode obfuscation, urgency cues, and language-specific scam patterns.

The model has three output classes: `0` legitimate/ham, `1` spam, and `2` phishing. Phishing remains separate because its recall and F1 are cybersecurity-critical metrics. Training uses inverse-frequency weighted cross-entropy to reduce majority-class bias.

```text
subject + body (original language)
        -> XLM-R tokenizer (bounded length)
        -> XLM-R encoder
        -> 3-logit classification head
        -> legitimate | spam | phishing probabilities
```

## Dataset format

Supply a local JSONL file; no dataset is downloaded by this project:

```json
{"email_id":"example-001","subject":"Verify account","body":"...","language":"en","label":2}
```

Validate examples with `ml.training.load_jsonl_dataset`, then split deterministically with `ml.training.split.split_records`. Use a real, licensed, anonymized dataset before reporting performance. Inline test data is synthetic and is useful only for code verification, not real-world accuracy claims.

## Training and evaluation

`TrainingConfig` exposes model name, batch size, learning rate, epochs, maximum sequence length, seed, output directory, and optional device. CPU is selected by default when no CUDA GPU is available.

The evaluator reports accuracy, macro precision/recall/F1, per-class precision/recall/F1, and a confusion matrix. Review `per_class["phishing"].recall` and `per_class["phishing"].f1` before considering a model for cybersecurity use.

Training is explicit: this repository does not start training, download a dataset, or claim real-world quality from synthetic test data.
