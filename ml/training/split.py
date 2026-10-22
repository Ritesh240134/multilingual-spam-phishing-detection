"""Deterministic, label-stratified train/validation/test splitting."""

from __future__ import annotations

import random
from collections import defaultdict

from .data import EmailRecord


def split_records(
    records: list[EmailRecord], validation_fraction: float = 0.1, test_fraction: float = 0.1, seed: int = 42
) -> tuple[list[EmailRecord], list[EmailRecord], list[EmailRecord]]:
    """Split records by label without external dependencies.

    Small classes retain at least one training example; callers should use a
    sufficiently large real dataset for representative validation metrics.
    """

    if not records:
        raise ValueError("cannot split an empty dataset")
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be in [0, 1)")
    groups: dict[int, list[EmailRecord]] = defaultdict(list)
    for record in records:
        groups[record.label].append(record)
    rng = random.Random(seed)
    train: list[EmailRecord] = []
    validation: list[EmailRecord] = []
    test: list[EmailRecord] = []
    for group in groups.values():
        ordered = group.copy()
        rng.shuffle(ordered)
        validation_count = int(len(ordered) * validation_fraction)
        test_count = int(len(ordered) * test_fraction)
        while validation_count + test_count >= len(ordered) and (validation_count or test_count):
            if test_count:
                test_count -= 1
            else:
                validation_count -= 1
        validation.extend(ordered[:validation_count])
        test.extend(ordered[validation_count : validation_count + test_count])
        train.extend(ordered[validation_count + test_count :])
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test
