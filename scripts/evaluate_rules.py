"""Evaluate contact rules on a deterministic, grouped validation split within train."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from audit_pii import normalize, parse_template
from audit_train_spans import target_spans

from piimask.rules import Span, detect

NAME_TYPES = {"FIRST_NAME", "LAST_NAME", "MIDDLE_NAME"}
SPLIT_VERSION = "piimask-v1"


def gold_spans(text: str, compact_tokens: list[str], tags: list[str]) -> list[Span]:
    offsets = [i for i, char in enumerate(text) if not char.isspace()]
    spans: list[Span] = []
    name_parts: set[str] = set()
    for start, end, label in target_spans(compact_tokens, tags):
        if start == end:
            raise ValueError("Empty target BIO span")
        left, right = offsets[start], offsets[end - 1] + 1
        if label in NAME_TYPES:
            if name_parts and label not in name_parts and text[spans[-1][1] : left].isspace():
                spans[-1] = (spans[-1][0], right, "PERSON")
                name_parts.add(label)
                continue
            name_parts = {label}
            spans.append((left, right, "PERSON"))
        else:
            name_parts.clear()
            spans.append((left, right, label))
    return spans


def find_group(parents, row_id):
    while parents[row_id] != row_id:
        parents[row_id] = parents[parents[row_id]]
        row_id = parents[row_id]
    return row_id


def add_group_keys(parents, owners, row_id, text, tokens, tags):
    template, _ = parse_template(tokens, tags)
    parents[row_id] = row_id
    for key in (("text", normalize(text)), ("template", template)):
        owner = owners.setdefault(key, row_id)
        first = find_group(parents, row_id)
        second = find_group(parents, owner)
        parents[max(first, second)] = min(first, second)


def evaluate(path):
    parents, owners, rows = {}, {}, {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row_id, row in enumerate(csv.DictReader(handle), start=1):
            text = row["text"]
            tokens, tags = json.loads(row["tokens"]), json.loads(row["ner_tags"])
            compact_tokens = ["".join(token.split()) for token in tokens]
            if "".join(compact_tokens) != "".join(text.split()):
                continue
            add_group_keys(parents, owners, row_id, text, tokens, tags)
            rows[row_id] = (text, gold_spans(text, compact_tokens, tags))

    groups = {row_id: find_group(parents, row_id) for row_id in rows}
    validation_groups = {
        group
        for group in set(groups.values())
        if hashlib.sha256(f"{SPLIT_VERSION}:{group}".encode()).digest()[0] < 51
    }
    validation_ids = [row_id for row_id, group in groups.items() if group in validation_groups]
    counts = {label: Counter() for label in ("PERSON", "PHONE", "EMAIL")}
    with_uncovered_gold = 0
    with_nonmatching_prediction = 0
    gold_email_with_whitespace = 0
    for row_id in validation_ids:
        text, gold = rows[row_id]
        prediction = set(detect(text))
        expected = set(gold)
        for start, end, label in expected:
            counts[label]["gold"] += 1
            if label == "EMAIL":
                gold_email_with_whitespace += any(char.isspace() for char in text[start:end])
        for _, _, label in prediction:
            counts[label]["predicted"] += 1
        for _, _, label in prediction & expected:
            counts[label]["exact_matches"] += 1
        with_uncovered_gold += any(
            not any(left <= start and end <= right for left, right, _ in prediction)
            for start, end, _ in expected
        )
        with_nonmatching_prediction += bool(prediction - expected)

    return {
        "train_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "split_version": SPLIT_VERSION,
        "aligned_rows": len(rows),
        "fit_rows": len(rows) - len(validation_ids),
        "validation_rows": len(validation_ids),
        "validation_groups": len(validation_groups),
        "validation_row_ids_sha256": hashlib.sha256(
            "\n".join(map(str, validation_ids)).encode()
        ).hexdigest(),
        "validation_rows_with_uncovered_gold": with_uncovered_gold,
        "validation_rows_with_nonmatching_prediction": with_nonmatching_prediction,
        "validation_gold_email_spans_with_whitespace": gold_email_with_whitespace,
        "by_type": {
            label: {
                "gold": count["gold"],
                "predicted": count["predicted"],
                "exact_matches": count["exact_matches"],
                "precision": count["exact_matches"] / count["predicted"]
                if count["predicted"]
                else None,
                "recall": count["exact_matches"] / count["gold"] if count["gold"] else None,
            }
            for label, count in counts.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.train), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
