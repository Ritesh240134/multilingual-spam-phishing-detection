"""Shared label definitions for the three-class email classifier."""

LABEL_TO_ID = {"legitimate": 0, "spam": 1, "phishing": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
NUM_LABELS = len(LABEL_TO_ID)
