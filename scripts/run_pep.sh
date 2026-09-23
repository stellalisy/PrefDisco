#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DATASET="${1:-medqa}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$DATASET" in
  aime|commonsenseqa|medqa|socialiqa) ;;
  *)
    echo "Unknown PEP dataset: $DATASET" >&2
    echo "Choose one of: aime, commonsenseqa, medqa, socialiqa" >&2
    exit 2
    ;;
esac

DATA_ROOT="${PREFDISCO_DATA_DIR:-$ROOT_DIR/data/benchmark}"
DATA_PATH="$DATA_ROOT/benchmark_5k/$DATASET/full.jsonl"
OUTPUT_ROOT="${PEP_OUTPUT_DIR:-$ROOT_DIR/outputs/pep/$DATASET}"

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Downloading benchmark_5k/$DATASET/full.jsonl..."
  "$PYTHON_BIN" "$ROOT_DIR/scripts/download_data.py" benchmark \
    --pattern "benchmark_5k/$DATASET/full.jsonl" \
    --output-dir "$DATA_ROOT"
fi

export MPLBACKEND="${MPLBACKEND:-Agg}"
cd "$ROOT_DIR"
"$PYTHON_BIN" -m pep.run_pipeline "$DATA_PATH" \
  --output "$OUTPUT_ROOT" \
  "$@"
