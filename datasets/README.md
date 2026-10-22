# Dataset Format and Handling

Phase 2 does not download, include, or process real email datasets. The `datasets/` directory is ignored by Git except for this documentation and `.gitkeep`.

## Recommended record schema

Store one normalized record per line in JSONL format after lawful collection and privacy review:

```json
{
  "email_id": "synthetic-or-anonymized-id",
  "label": 0,
  "language": "es",
  "subject": "Actualización de cuenta",
  "body": "Texto del correo...",
  "html": null,
  "headers": {"From": "redacted@example.test", "To": "redacted@example.test"},
  "attachments": [{"filename": "redacted.pdf", "content_type": "application/pdf"}],
  "source": "approved-source-name",
  "license": "license-identifier-or-reference",
  "is_synthetic": false
}
```

For the Phase 3 training loader, create a separate JSONL file with exactly these fields: `email_id`, `subject`, `body`, `language`, and `label`. The label is an integer: `0` = legitimate/ham, `1` = spam, `2` = phishing. Keep phishing distinct from spam. The source/license/synthetic fields above remain strongly recommended provenance metadata and can be managed alongside the loader-compatible records.

## Privacy and separation

- Do not place real credentials, access tokens, personal addresses, or unredacted sensitive email content in the repository.
- Store real or approved training data outside version control. Anonymize it before preprocessing or training.
- Keep synthetic fixtures in `tests/` only. Phase 2 tests use inline synthetic email strings and no real email corpus.

## Future public-data evaluation

Potential future sources include SpamAssassin public corpora, Enron mail for legitimate examples, and public phishing/URL feeds such as PhishTank or OpenPhish. Each source must be reviewed for licensing, redistribution rights, privacy risk, language coverage, and collection-date bias before it is added manually.
