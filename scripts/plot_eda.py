"""Draw two aggregate EDA charts as a standalone SVG."""

import argparse
import json
from pathlib import Path


def plot(audit):
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="740" height="440" '
        'viewBox="0 0 740 440" role="img" aria-labelledby="title">',
        '<title id="title">Длина текстов и расхождения разметки</title>',
        '<rect width="740" height="440" fill="#fff"/>',
        '<g font-family="system-ui,sans-serif" font-size="14" fill="#182838">',
    ]

    def label(x, y, value):
        parts.append(f'<text x="{x}" y="{y}">{value}</text>')

    def bar(x, y, fraction, color):
        parts.append(f'<rect x="{x}" y="{y}" width="240" height="12" fill="#e7edf1"/>')
        parts.append(
            f'<rect x="{x}" y="{y}" width="{240 * fraction:.2f}" height="12" fill="{color}"/>'
        )

    label(24, 32, "Длина исходного текста")
    label(398, 32, "Разметка train в строках с расхождениями")
    label(24, 55, "Доля строк по числу символов")
    label(398, 55, "Доля BIO-спанов каждого типа")
    ranges = ("≤ 64", "65–128", "129–256", "257–512", "&gt; 512")
    for index, (key, length_range) in enumerate(
        zip(("64", "128", "256", "512", ">512"), ranges, strict=True)
    ):
        y = 88 + index * 61
        label(24, y, length_range)
        for shift, split, color in ((8, "train", "#145e81"), (25, "published_test", "#805432")):
            stats = audit[split]
            fraction = stats["character_length_histogram"][key] / stats["rows"]
            bar(24, y + shift, fraction, color)
            label(274, y + shift + 11, f"{fraction:.1%}")
    label(24, 404, "Train — синий · test — коричневый")

    train = audit["train"]
    for index, (key, title) in enumerate(
        (
            ("FIRST_NAME", "Имя"),
            ("LAST_NAME", "Фамилия"),
            ("MIDDLE_NAME", "Отчество"),
            ("PHONE", "Телефон"),
            ("EMAIL", "Email"),
        )
    ):
        y = 88 + index * 61
        fraction = (
            train["target_bio_spans_in_mismatched_rows"][key] / train["target_bio_spans"][key]
        )
        label(398, y, title)
        bar(398, y + 8, fraction, "#145e81")
        label(648, y + 19, f"{fraction:.1%}")
    label(398, 404, "Шкала каждой полосы от 0 до 100%")
    return "\n".join([*parts, "</g>", "</svg>", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(
        plot(json.loads(args.audit.read_text(encoding="utf-8"))), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
