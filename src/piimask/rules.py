"""CPU baseline for masking standard phone numbers and email addresses."""

import re

Span = tuple[int, int, str]

EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9_+-]+(?:\.[A-Za-z0-9_+-]+)*"
    r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}(?![\w-])"
)
PHONE = re.compile(
    r"(?<![\w+])(?:\+7|8)[ \t-]*(?:\([0-9]{3}\)|[0-9]{3})"
    r"[ \t-]*[0-9]{3}[ \t-]*[0-9]{2}[ \t-]*[0-9]{2}(?!\w)"
)


def detect(text: str) -> list[Span]:
    emails = [(m.start(), m.end(), "EMAIL") for m in EMAIL.finditer(text)]
    phones = [
        (match.start(), match.end(), "PHONE")
        for match in PHONE.finditer(text)
        if all(match.end() <= start or end <= match.start() for start, end, _ in emails)
    ]
    return sorted(emails + phones)


def mask(text: str) -> tuple[str, list[Span]]:
    spans = detect(text)
    parts = []
    cursor = 0
    for start, end, label in spans:
        parts.extend((text[cursor:start], f"[{label}]"))
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), spans
