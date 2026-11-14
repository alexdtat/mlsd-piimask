"""Check whether BIO spans can be mapped to the original text in mismatched train rows."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

TARGET_TYPES = {"FIRST_NAME", "LAST_NAME", "MIDDLE_NAME", "PHONE", "EMAIL"}


def target_spans(tokens, tags):
    """Yield target spans in the concatenated, whitespace-free token string."""
    position = 0
    start = 0
    active_type = ""
    for token, tag in zip(tokens, tags, strict=True):
        prefix, _, entity_type = tag.partition("-")
        if prefix != "I":
            if active_type in TARGET_TYPES:
                yield start, position, active_type
            start = position
            active_type = entity_type
        position += len(token)
    if active_type in TARGET_TYPES:
        yield start, position, active_type


def map_span(blocks, start, end):
    for token_start, text_start, size in blocks:
        if token_start <= start < end <= token_start + size:
            return text_start + start - token_start
    return None


def joins_word(text, boundary):
    return 0 < boundary < len(text) and text[boundary - 1].isalnum() and text[boundary].isalnum()


def classify_span(token_text, text, compact_text, offsets, forward, backward, start, end):
    surface = token_text[start:end]
    first_occurrence = compact_text.find(surface)
    if first_occurrence == -1:
        return "surface_absent"
    left = map_span(forward, start, end)
    right = map_span(backward, len(token_text) - end, len(token_text) - start)
    if left is None or right is None:
        return "span_changed_or_interrupted"
    if left != len(compact_text) - right - (end - start):
        return "alignments_disagree"

    if first_occurrence != compact_text.rfind(surface):
        return "surface_repeats"
    original_start = offsets[left]
    original_end = offsets[left + end - start - 1] + 1
    if joins_word(text, original_start) or joins_word(text, original_end):
        return "boundary_inside_word"
    return "exact_candidate"


def audit(path):
    rows = Counter()
    by_type = defaultdict(Counter)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows["train_total"] += 1
            text = row["text"]
            tokens = ["".join(token.split()) for token in json.loads(row["tokens"])]
            token_text = "".join(tokens)
            compact_text = "".join(text.split())
            if token_text == compact_text:
                continue
            rows["mismatch"] += 1
            spans = list(target_spans(tokens, json.loads(row["ner_tags"])))
            if not spans:
                rows["mismatch_without_target_spans"] += 1
                continue

            offsets = [i for i, character in enumerate(text) if not character.isspace()]
            forward = SequenceMatcher(None, token_text, compact_text, autojunk=False)
            backward = SequenceMatcher(None, token_text[::-1], compact_text[::-1], autojunk=False)
            forward_blocks = forward.get_matching_blocks()
            backward_blocks = backward.get_matching_blocks()
            recovered = 0
            for start, end, entity_type in spans:
                status = classify_span(
                    token_text,
                    text,
                    compact_text,
                    offsets,
                    forward_blocks,
                    backward_blocks,
                    start,
                    end,
                )
                by_type[entity_type][status] += 1
                recovered += status == "exact_candidate"

            if recovered == len(spans):
                rows["mismatch_all_target_spans_exact_candidates"] += 1
            elif recovered:
                rows["mismatch_some_target_spans_exact_candidates"] += 1
            else:
                rows["mismatch_no_target_spans_exact_candidates"] += 1
    return {
        "train_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": rows,
        "target_spans_in_mismatched_rows": sum(sum(counts.values()) for counts in by_type.values()),
        "target_spans_by_status": sum(by_type.values(), Counter()),
        "target_spans_by_type": by_type,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.train), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
