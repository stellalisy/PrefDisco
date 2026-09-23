# PEP training

This directory contains the reproducibility pipeline for the preference
elicitation policy, **PEP (Preference Elicitation with Priors)**. Unlike the
neural RL baselines, PEP fits population-level preference statistics from the
training problems and uses those statistics to select informative questions
for held-out users.

The pipeline includes the complete training dependency chain:

- `data_utils.py`: JSONL loading, train/test preparation, and population
  statistics;
- `filter_utils.py`: per-problem criterion filtering;
- `models.py`: prior, Naive Bayes, collaborative-filtering, logistic, and
  Bayesian preference models;
- `strategies.py`: random, fixed, uncertainty, expected-value, and binary
  information-gain question policies;
- `run_pipeline.py`: training, held-out evaluation, trajectory aggregation,
  and plots.

## Run from the released data

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-pep.txt

# Downloads the public 5K MedQA release if it is not already present.
scripts/run_pep.sh medqa
```

The released 5K sets supported by the wrapper are `aime`, `commonsenseqa`,
`medqa`, and `socialiqa`. To run all four:

```bash
for dataset in aime commonsenseqa medqa socialiqa; do
  scripts/run_pep.sh "$dataset"
done
```

The paper configuration is a five-question budget, a 90/10 problem split,
criterion frequency below 10% within each problem, and at least four users per
criterion:

```bash
scripts/run_pep.sh medqa \
  --budget 5 \
  --train-ratio 0.9 \
  --per-problem-max 0.10 \
  --min-users 4 \
  --seed 42
```

Results are written under `outputs/pep/<dataset>/`. Set `PREFDISCO_DATA_DIR`
to reuse an existing benchmark download and `PEP_OUTPUT_DIR` to change the
output location.

The 1K benchmark has only ten personas per problem and is intended for model
benchmarking. PEP's default filtering thresholds require the 5K release, which
has fifty personas per problem.
