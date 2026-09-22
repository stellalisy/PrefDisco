#!/usr/bin/env python3
"""Download the released PrefDisco benchmark or training artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPOSITORIES = {
    "benchmark": "stellalisy/prefq-bench",
    "training": "stellalisy/prefq-training",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(REPOSITORIES))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination (default: data/<kind>).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. Defaults to HF_TOKEN/HUGGING_FACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Optional Hub path glob; repeat to select multiple subsets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_id = REPOSITORIES[args.kind]
    output_dir = args.output_dir or Path("data") / args.kind
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "local_dir": output_dir,
        "token": token,
    }
    if args.pattern:
        kwargs["allow_patterns"] = args.pattern
    path = snapshot_download(**kwargs)
    print(f"Downloaded {repo_id} to {path}")


if __name__ == "__main__":
    main()
