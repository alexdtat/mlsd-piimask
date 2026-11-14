"""Audit the published BIO corpora without sampling or reassigning their splits."""

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from statistics import median

from audit_train_spans import target_spans

TARGET_TYPES = ("FIRST_NAME", "LAST_NAME", "MIDDLE_NAME", "PHONE", "EMAIL")
LENGTH_BINS = (64, 128, 256, 512)


def normalize(value):
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def parse_template(tokens, tags):
    parts = []
    previous = None
    entities = Counter()
    for token, tag in zip(tokens, tags, strict=True):
        if tag == "O":
            parts.append(("text", normalize(token)))
            previous = None
            continue
        prefix, _, entity = tag.partition("-")
        if not entity or prefix not in {"B", "I"} or (prefix == "I" and entity != previous):
            raise ValueError("Invalid BIO sequence")
        if prefix == "B":
            parts.append(("entity", entity))
            entities[entity] += 1
        previous = entity
    return tuple(parts), entities


def audit(path, *, inspect_email_spacing=False):
    csv.field_size_limit(2_000_000)
    raw, normalized, templates, annotations = Counter(), Counter(), Counter(), Counter()
    templates_with_context = Counter()
    errors, entities = Counter(), Counter()
    mismatched_entities = Counter()
    character_lengths = Counter()
    email_spacing = Counter()
    lengths = []
    labels_by_text = {}
    entity_free = 0
    rows = 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            value = row["text"]
            length_bin = next((str(limit) for limit in LENGTH_BINS if len(value) <= limit), ">512")
            character_lengths[length_bin] += 1
            raw[value] += 1
            normalized[normalize(value)] += 1
            try:
                tokens, tags = json.loads(row["tokens"]), json.loads(row["ner_tags"])
                if not all(
                    isinstance(seq, list) and all(isinstance(x, str) for x in seq)
                    for seq in (tokens, tags)
                ):
                    raise TypeError("Expected lists of strings")
            except (TypeError, ValueError):
                errors["malformed_lists"] += 1
                continue
            if len(tokens) != len(tags):
                errors["length_mismatch"] += 1
                continue
            lengths.append(len(tokens))
            mismatched = "".join(value.split()) != "".join("".join(tokens).split())
            if mismatched:
                errors["text_token_mismatch_ignoring_whitespace"] += 1
                if normalize(value) != normalize("".join(tokens)):
                    errors["text_token_mismatch_after_normalization"] += 1
            try:
                template, row_entities = parse_template(tokens, tags)
            except ValueError:
                errors["invalid_bio"] += 1
                continue
            templates[template] += 1
            if (
                sum(kind == "text" and any(c.isalpha() for c in text) for kind, text in template)
                >= 3
            ):
                templates_with_context[template] += 1
            annotations[(value, tuple(tokens), tuple(tags))] += 1
            labels_by_text.setdefault(value, set()).add(tuple(zip(tokens, tags, strict=True)))
            entities.update(row_entities)
            if mismatched:
                mismatched_entities.update(row_entities)
            elif inspect_email_spacing and row_entities["EMAIL"]:
                offsets = [i for i, char in enumerate(value) if not char.isspace()]
                compact_tokens = ["".join(token.split()) for token in tokens]
                for start, end, label in target_spans(compact_tokens, tags):
                    if label == "EMAIL":
                        surface = value[offsets[start] : offsets[end - 1] + 1]
                        key = (
                            "with_whitespace" if any(c.isspace() for c in surface) else "contiguous"
                        )
                        email_spacing[key] += 1
            entity_free += not row_entities
    if not lengths:
        raise ValueError(f"No parseable rows in {path.name}")
    stats = {
        "rows": rows,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "file_bytes": path.stat().st_size,
        "exact_duplicate_extra_rows": rows - len(raw),
        "normalized_duplicate_extra_rows": rows - len(normalized),
        "exact_annotated_duplicate_extra_rows": sum(annotations.values()) - len(annotations),
        "same_raw_text_with_different_token_tag_sequences": sum(
            len(x) > 1 for x in labels_by_text.values()
        ),
        "entity_free_rows_among_valid_bio": entity_free,
        "token_length": {"median": median(lengths), "max": max(lengths)},
        "character_length_histogram": {
            key: character_lengths[key] for key in (*map(str, LENGTH_BINS), ">512")
        },
        "target_bio_spans": {label: entities[label] for label in TARGET_TYPES},
        "target_bio_spans_in_mismatched_rows": {
            label: mismatched_entities[label] for label in TARGET_TYPES
        },
        "issues": dict(errors),
    }
    if inspect_email_spacing:
        stats["aligned_email_span_spacing"] = dict(email_spacing)
    return stats, entities, raw, normalized, templates, annotations, templates_with_context


def overlap(train, test):
    shared = train.keys() & test.keys()
    return {
        "shared_keys": len(shared),
        "train_rows": sum(train[x] for x in shared),
        "test_rows": sum(test[x] for x in shared),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    parser.add_argument("test", type=Path)
    args = parser.parse_args()
    train, test = audit(args.train, inspect_email_spacing=True), audit(args.test)
    result = {
        "train": {**train[0], "entity_spans": dict(train[1].most_common())},
        "published_test": test[0],
        "cross_split": {
            name: overlap(train[index], test[index])
            for index, name in enumerate(
                (
                    "exact_text",
                    "normalized_text",
                    "bio_masked_template",
                    "exact_annotated_row",
                    "bio_masked_template_at_least_three_unmasked_word_tokens",
                ),
                2,
            )
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
