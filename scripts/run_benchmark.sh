#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <benchmark.jsonl> [evaluator arguments...]" >&2
  exit 2
fi

benchmark="$1"
shift
mkdir -p results

python src/environment.py \
  --config src/config/evaluation_config.yaml \
  --benchmark_file "$benchmark" \
  --output "results/$(basename "${benchmark%.jsonl}")_results.jsonl" \
  "$@"
