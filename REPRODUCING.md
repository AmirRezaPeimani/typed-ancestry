# Reproducing the Study

Run all commands from the repository root. Generated data, model files, and
training runs are written to ignored directories.

## Environment

The lightweight analysis was verified with Python 3.10, NumPy 2.2.6,
pandas 2.3, Matplotlib 3.10, and pytest 9. The neural runs used Linux,
Python 3.10.12, CUDA, PyTorch 2.7.1, Transformers 4.57.1, PEFT 0.18.0,
and Accelerate 1.11.0.

Create the analysis environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements/analysis.txt
.venv/bin/python -m pip install --no-deps -e .
```

## Lightweight reproduction

```bash
make PYTHON=.venv/bin/python test
make PYTHON=.venv/bin/python reproduce-light
```

This path verifies the released prediction-level and aggregate results,
regenerates the figures and macros, validates evidence hashes, and refreshes
`manifests/artifact_manifest.json`. It does not refit models or rerun the
20,000-draw arrival analysis; those commands are documented below.

## Source data

Install the pinned neural dependencies and the package extras needed by the
download commands:

```bash
.venv/bin/python -m pip install -r requirements/neural.txt
```

Download the pinned datasets:

```bash
.venv/bin/hf download nvidia/HelpSteer3 \
  --repo-type dataset \
  --revision f6d145777bcbde96137596340fab89793acd1031 \
  --local-dir data/helpsteer3

.venv/bin/hf download nvidia/HelpSteer2 validation.jsonl.gz \
  --repo-type dataset \
  --revision 990b2711a36180dd19d9c94b8627844866f8982a \
  --local-dir data/helpsteer2
```

Verify the downloaded files:

```bash
.venv/bin/python scripts/verify_source_manifest.py \
  --dataset-dir data/helpsteer3 \
  --manifest manifests/helpsteer3_sources.json

.venv/bin/python scripts/verify_source_manifest.py \
  --dataset-dir data/helpsteer2 \
  --manifest manifests/helpsteer2_sources.json
```

## Typed-ancestry graph and controlled split

```bash
.venv/bin/ancestry-audit scan \
  --adapter helpsteer3 \
  --dataset-dir data/helpsteer3 \
  --revision f6d145777bcbde96137596340fab89793acd1031 \
  --output data/helpsteer3_audit \
  --seed 20260725 \
  --folds 20

PYTHONPATH=src .venv/bin/python scripts/prepare_factorial_data.py \
  --helpsteer3-dir data/helpsteer3 \
  --helpsteer2-validation data/helpsteer2/validation.jsonl.gz \
  --helpsteer2-contract manifests/helpsteer2_external_pairs.csv \
  --output data/controlled_factorial

.venv/bin/python scripts/verify_generated_data.py \
  --data-dir data/controlled_factorial \
  --manifest manifests/controlled_factorial_data.json
```

The verifier checks row counts and decompressed JSONL hashes, which are
independent of gzip container metadata.

Recover the construct-defining fields while preserving every pair, label,
assignment, and row position:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_construct_valid_data.py \
  --input-dir data/controlled_factorial \
  --helpsteer3-dir data/helpsteer3 \
  --output data/construct_complete_factorial

.venv/bin/python scripts/verify_generated_data.py \
  --data-dir data/construct_complete_factorial \
  --manifest manifests/construct_complete_factorial_data.json
```

## Component-disjoint training mixture

Construct the component-disjoint training and development files:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_neural_data.py \
  --edit-quality-train data/helpsteer3/edit_quality/train.jsonl.gz \
  --factorial-data data/construct_complete_factorial \
  --component-assignments \
    data/helpsteer3_audit/component_assignments.csv \
  --output data/controlled_neural

.venv/bin/python scripts/verify_generated_data.py \
  --data-dir data/controlled_neural \
  --manifest manifests/controlled_neural_data.json
```

## Sparse learner factorial

The sparse and neural learners use the same construct-complete files:

```bash
PYTHONPATH=src .venv/bin/python analysis/run_controlled_sparse_factorial.py \
  --data-dir data/controlled_neural \
  --output work/controlled_sparse

PYTHONPATH=src .venv/bin/python \
  analysis/analyze_controlled_sparse_factorial.py \
  --predictions work/controlled_sparse/predictions.csv.gz \
  --output work/controlled_sparse_analysis.json \
  --bootstrap-replicates 20000 \
  --bootstrap-seed 20261021 \
  --randomization-seed 20261022
```

The sparse checkpoints fit prefixes of exactly 902, 1,805, 2,707, 3,610,
and 4,512 pairs. Recompute the randomized ancestor-arrival analysis from the
saved prediction-level evidence and the reconstructed controlled data:

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_ancestor_arrival.py \
  --model \
    word_tfidf_bt=results/randomized_one_pass/word_predictions.csv.gz \
  --model \
    minilm_frozen_bt=results/randomized_one_pass/minilm_predictions.csv.gz \
  --data-dir data/controlled_factorial \
  --output-dir work/ancestor_arrival \
  --bootstrap-replicates 20000 \
  --randomization-replicates 20000
```

The reference release outputs are in `results/ancestor_arrival/`. The
assignment-randomization p-values use a plus-one correction.

## Neural learner factorial

Download the pinned base model:

```bash
.venv/bin/hf download Qwen/Qwen2.5-1.5B-Instruct \
  --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --local-dir data/models/qwen2.5-1.5b-instruct
```

Run the 64-pair overfit check and the separate 1,000-pair checkpoint-selector
run. Both use seed 20260801. The selector training set contains no
target-specific ancestors or boundary-specific fillers and is component-
disjoint from the exposed-target, ancestry-clean validation, untouched, and
development sets.

```bash
PYTHONPATH=analysis:src .venv/bin/python \
  analysis/run_controlled_qwen15_training.py overfit \
  --model data/models/qwen2.5-1.5b-instruct \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --model-revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --train data/controlled_neural/selector_train_1000.jsonl.gz \
  --output work/neural_overfit64 \
  --seed 20260801 \
  --learning-rate 0.0002 \
  --epochs 20 \
  --microbatch-pairs 4 \
  --gradient-accumulation 1 \
  --lora \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --device cuda

PYTHONPATH=analysis:src .venv/bin/python \
  analysis/run_controlled_qwen15_training.py train \
  --model data/models/qwen2.5-1.5b-instruct \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --model-revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --train data/controlled_neural/selector_train_1000.jsonl.gz \
  --eval \
    development=data/controlled_neural/selector_development_400.jsonl.gz \
  --output work/neural_selector \
  --seed 20260801 \
  --learning-rate 0.0002 \
  --epochs 1 \
  --microbatch-pairs 4 \
  --gradient-accumulation 8 \
  --lora \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --device cuda
```

Run the six fixed factorial cells:

```bash
for seed in 20260727 20260728 20260729; do
  for boundary in naive safe; do
    PYTHONPATH=analysis:src .venv/bin/python \
      analysis/run_controlled_qwen15_training.py train \
      --model data/models/qwen2.5-1.5b-instruct \
      --model-id Qwen/Qwen2.5-1.5B-Instruct \
      --model-revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
      --train \
        data/controlled_neural/train_${boundary}_seed${seed}.jsonl.gz \
      --output work/neural/runs/${boundary}_seed${seed} \
      --seed ${seed} \
      --learning-rate 0.0002 \
      --epochs 1 \
      --microbatch-pairs 4 \
      --gradient-accumulation 8 \
      --lora \
      --lora-rank 16 \
      --lora-alpha 32 \
      --lora-dropout 0.05 \
      --device cuda

    PYTHONPATH=analysis:src .venv/bin/python \
      analysis/evaluate_controlled_qwen15.py \
      --model data/models/qwen2.5-1.5b-instruct \
      --run work/neural/runs/${boundary}_seed${seed} \
      --output work/neural/evaluations/${boundary}_seed${seed} \
      --checkpoint-fraction 100 \
      --eval target=data/controlled_neural/target_512.jsonl.gz \
      --eval \
        clean_validation=data/controlled_neural/clean_validation_512.jsonl.gz \
      --eval \
        untouched_clean_test=data/controlled_neural/untouched_clean_test_512.jsonl.gz \
      --device cuda
  done
done
```

Analyze the fixed checkpoint and learner interactions:

```bash
PYTHONPATH=analysis:src .venv/bin/python \
  analysis/analyze_controlled_qwen15_factorial.py \
  --root work/neural \
  --checkpoint-selection configs/qwen_checkpoint_selection.json \
  --lexical-predictions work/controlled_sparse/predictions.csv.gz \
  --output work/controlled_neural_analysis.json \
  --bootstrap-replicates 20000 \
  --bootstrap-seed 20261011 \
  --randomization-seed 20261012
```

The saved public result files remain the reference artifact. GPU kernels and
library builds can introduce small score-level differences, so a full neural
rerun should be evaluated against the reported estimands and uncertainty
intervals rather than requiring byte-identical floating-point predictions.

## Final integrity check

```bash
make PYTHON=.venv/bin/python reproduce-light
.venv/bin/python -m pytest -q
```

The checksum manifest excludes itself, generated environments, downloaded
data, model files, and work directories.
