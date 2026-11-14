"""Measure the frozen contact rules on a local PII-Bench Parquet snapshot."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from pyarrow.parquet import read_table

from piimask.rules import Span, detect

LABELS = {"PHONE_NUMBER": "PHONE", "EMAIL": "EMAIL"}


def positions(spans: set[Span]) -> set[int]:
    return {i for start, end, _ in spans for i in range(start, end)}


def evaluate(path: Path) -> dict:
    rows = read_table(path, columns=["text", "entities"]).to_pylist()
    counts = {label: Counter() for label in LABELS.values()}
    coverage = Counter()
    clean_rows = 0
    for row in rows:
        text = row["text"]
        all_gold = set()
        for entity in row["entities"]:
            start, end = entity["start"], entity["end"]
            if not 0 <= start < end <= len(text) or text[start:end] != entity["text"]:
                raise ValueError("Invalid PII-Bench span")
            all_gold.add((start, end, entity["type"]))

        gold = {(start, end, LABELS[label]) for start, end, label in all_gold if label in LABELS}
        prediction = set(detect(text))
        masked = positions(prediction)
        target = positions(gold)
        outside_gold = masked - positions(all_gold)
        coverage["rows_with_missed_contact"] += bool(target - masked)
        coverage["rows_with_mask_outside_contact_gold"] += bool(masked - target)
        coverage["rows_with_mask_outside_any_gold_pii"] += bool(outside_gold)
        coverage["unmasked_contact_characters"] += len(target - masked)
        coverage["masked_characters_outside_contact_gold"] += len(masked - target)
        coverage["masked_characters_outside_any_gold_pii"] += len(outside_gold)
        clean_rows += not all_gold
        coverage["rows_without_any_gold_pii_with_mask"] += not all_gold and bool(masked)

        for start, end, label in gold:
            counts[label]["gold"] += 1
            counts[label]["gold_with_internal_whitespace"] += any(
                char.isspace() for char in text[start:end]
            )
            span = set(range(start, end))
            if span <= masked:
                counts[label]["fully_masked"] += 1
            elif span & masked:
                counts[label]["partly_masked"] += 1
            else:
                counts[label]["unmasked"] += 1
        for _, _, label in prediction:
            counts[label]["predicted"] += 1
        for _, _, label in gold & prediction:
            counts[label]["exact_matches"] += 1

    by_type = {}
    for label, count in counts.items():
        exact, predicted, gold = count["exact_matches"], count["predicted"], count["gold"]
        by_type[label] = {
            "exact_matches": exact,
            "predicted": predicted,
            "gold": gold,
            "gold_with_internal_whitespace": count["gold_with_internal_whitespace"],
            "fully_masked": count["fully_masked"],
            "partly_masked": count["partly_masked"],
            "unmasked": count["unmasked"],
            "exact_precision": exact / predicted if predicted else None,
            "exact_recall": exact / gold if gold else None,
        }
    return {
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(rows),
        "rows_without_any_gold_pii": clean_rows,
        "coverage": coverage,
        "by_type": by_type,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path, nargs="+")
    args = parser.parse_args()
    results = {path.name: evaluate(path) for path in args.parquet}
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
