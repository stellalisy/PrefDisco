#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset> [generator arguments...]" >&2
  echo "Datasets: aime commonsenseqa logiqa mascqa math medqa mmlu scienceqa simpleqa socialiqa" >&2
  exit 2
fi

dataset="$1"
shift
config="src/config/benchmark_generator_${dataset}.yaml"

if [[ ! -f "$config" ]]; then
  echo "Unknown dataset or missing config: $config" >&2
  exit 2
fi

python src/benchmark_generator.py --config "$config" "$@"
