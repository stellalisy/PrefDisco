#!/usr/bin/env python3
"""Validate the records in a generated or downloaded PrefDisco JSONL file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


REQUIRED_FIELDS = {
    "type",
    "problem_id",
    "original_problem",
    "persona",
    "persona_preferences",
    "evaluation_rubric",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Do not verify relative image paths on disk.",
    )
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    ids: set[str] = set()
    errors: list[str] = []
    with args.benchmark.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc})")
                continue
            row_type = row.get("type", "missing")
            counts[row_type] += 1
            if row_type == "metadata":
                continue
            missing = REQUIRED_FIELDS.difference(row)
            if missing:
                errors.append(f"line {line_number}: missing {sorted(missing)}")
            problem_id = row.get("problem_id")
            if problem_id in ids:
                errors.append(f"line {line_number}: duplicate problem_id {problem_id!r}")
            elif problem_id is not None:
                ids.add(problem_id)
            rubric = row.get("evaluation_rubric", {})
            if not rubric.get("evaluation_criteria"):
                errors.append(f"line {line_number}: empty evaluation rubric")
            image = row.get("parsed_problem", {}).get("image")
            if image in (None, ""):
                image = row.get("original_problem", {}).get("image")
            if image and isinstance(image, str) and not args.skip_images:
                parsed_image = urlparse(image)
                if not parsed_image.scheme:
                    image_path = args.benchmark.parent / image
                    if not image_path.is_file():
                        errors.append(
                            f"line {line_number}: missing image {image_path}"
                        )

    print(f"Records: {sum(counts.values())}")
    print("Types: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    if errors:
        for error in errors[:20]:
            print(f"ERROR: {error}")
        if len(errors) > 20:
            print(f"ERROR: ... and {len(errors) - 20} more")
        raise SystemExit(1)
    print("Validation passed")


if __name__ == "__main__":
    main()
