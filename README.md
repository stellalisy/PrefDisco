# PrefDisco

PrefDisco is a benchmark and data-generation pipeline for evaluating whether
language models discover a user's response preferences before answering. Each
benchmark record pairs an educational problem with a simulated persona,
problem-specific preferences, and a weighted evaluation rubric.

This repository currently contains the reproducibility code for:

- generating personalized benchmark records from public Hugging Face datasets;
- validating generated JSONL artifacts;
- running the three PrefDisco evaluation conditions (`no_prompt`,
  `persona_known`, and `infer_persona`);
- training and evaluating PEP (Preference Elicitation with Priors);
- downloading the released benchmark and training artifacts.

## Setup

```bash
git clone https://github.com/stellalisy/PrefDisco.git
cd PrefDisco
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Local Hugging Face inference additionally requires
`pip install -r requirements-local.txt`.

PEP training uses a small separate dependency set:

```bash
pip install -r requirements-pep.txt
```

Set credentials for the providers you plan to use. OpenAI-only runs need just:

```bash
export OPENAI_API_KEY=...
```

The checked-in `src/config/api_info.example.yaml` contains no secrets. The
default configs read it and fall back to `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or
the standard AWS credential environment variables. Never commit credentials.

## Download released data

The benchmark repository is public. The training repository is currently
private, so accepted training-data users must also set `HF_TOKEN`:

```bash
python scripts/download_data.py benchmark

export HF_TOKEN=...
python scripts/download_data.py training
```

Download one benchmark family when a full snapshot is unnecessary:

```bash
python scripts/download_data.py benchmark \
  --pattern 'benchmark_1k/math/*' \
  --output-dir data/benchmark
```

Released artifacts:

- `stellalisy/prefq-bench`: ten 1K benchmarks and four 5K benchmark variants.
- `stellalisy/prefq-training`: PrefAlign RL Parquet splits plus DPO, reward
  model, and SFT JSONL splits.

## Generate a benchmark

Dataset configs live in `src/config/`. A generation run is resumable: if its
output JSONL already exists, completed problem/persona pairs are skipped.

```bash
# Run MATH generation with one OpenAI model.
scripts/generate_benchmark.sh math \
  --model gpt-4o \
  --api-account openai \
  --parallel --workers 4

# Cheap smoke run using one source problem and one persona.
python src/benchmark_generator.py \
  --config src/config/benchmark_generator_math.yaml \
  --sample-size 1 \
  --personas-per-problem 1 \
  --initial-personas 1 \
  --output-dir outputs/smoke/math
```

The main output is `personalized_<dataset>.jsonl`. Generation also maintains a
shared persona library and preference history under `data/shared_personas/`.
Generation makes multiple model calls per record; estimate provider costs
before using the full 100-problem configs. Omitting `--model` uses the
multi-provider setup recorded in the YAML and therefore requires OpenAI,
Gemini, and AWS Bedrock credentials.

## Benchmark a model

Download a benchmark and run a small evaluation first:

```bash
python scripts/download_data.py benchmark \
  --pattern 'benchmark_1k/math/*' \
  --output-dir data/benchmark

scripts/run_benchmark.sh data/benchmark/benchmark_1k/math/train.jsonl \
  --max_problems 5 \
  --model_api_account openai \
  --evaluator_api_account openai
```

The evaluator supports:

- `no_prompt`: answer without persona information;
- `persona_known`: answer with the full persona profile;
- `infer_persona`: elicit preferences through conversation before answering.

Use `--mode infer_persona` to run only one condition, `--user_type passive` or
`--user_type passive_nostop` to choose a simulator prompt, and
`--fixed_num_questions K` for a fixed-question ablation.

Validate any generated or downloaded benchmark with:

```bash
python scripts/validate_benchmark.py path/to/benchmark.jsonl
```

Validation also checks that every relative image reference exists beside the
benchmark. ScienceQA images are stored under its `images/` directory and are
shared by all persona variants of the same visual question.

## Train the PEP policy

PEP learns population-level preference structure, then evaluates adaptive
question selection on held-out problems. The launcher downloads the requested
5K benchmark split automatically:

```bash
scripts/run_pep.sh medqa \
  --budget 5 \
  --train-ratio 0.9 \
  --per-problem-max 0.10 \
  --min-users 4 \
  --seed 42
```

The released 5K datasets are `aime`, `commonsenseqa`, `medqa`, and
`socialiqa`. See [`pep/README.md`](pep/README.md) for the pipeline components,
outputs, and configuration details.

## Source datasets

The generation configs cover MATH-500, AIME, CommonsenseQA, LogiQA, MaScQA,
MedQA, MMLU, ScienceQA, SimpleQA, and SocialIQA. Their exact Hugging Face IDs,
splits, and field mappings are recorded in each YAML config. Users are
responsible for complying with the source datasets' licenses and terms.

## Repository layout

```text
src/benchmark_generator.py   benchmark construction pipeline
src/environment.py           interactive benchmark evaluator
src/config/                  generation and evaluation configs
src/llm/                     model-provider adapters
src/prompts/                 simulated-user prompts
scripts/                     download, generation, evaluation, validation
pep/                         PEP training and policy evaluation
```

## Citation

BibTeX for both PrefDisco and PEP is available in
[`CITATIONS.bib`](CITATIONS.bib).

```bibtex
@article{li2025prefdisco,
  title   = {{PrefDisco}: Benchmarking Proactive Personalized Reasoning},
  author  = {Li, Shuyue Stella and Bose, Avinandan and Brahman, Faeze and Du, Simon Shaolei and Koh, Pang Wei and Fazel, Maryam and Tsvetkov, Yulia},
  journal = {arXiv preprint arXiv:2510.00177},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.00177}
}

@article{bose2026pep,
  title   = {Cold-Start Personalization via Training-Free Priors from Structured World Models},
  author  = {Bose, Avinandan and Li, Shuyue Stella and Brahman, Faeze and Koh, Pang Wei and Du, Simon Shaolei and Tsvetkov, Yulia and Fazel, Maryam and Xiao, Lin and Celikyilmaz, Asli},
  journal = {arXiv preprint arXiv:2602.15012},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.15012}
}
```
