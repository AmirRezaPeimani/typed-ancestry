# Typed Ancestry for Post-Training Evaluation

This repository accompanies *Cross-View Contamination Inflates Reward-Model Validation Accuracy*, prepared for Language Resources and Evaluation. It provides an exact-ancestry audit library, controlled learning experiments, saved prediction-level evidence, and scripts for reproducing the reported tables and figures.

Field-level ancestry matching identifies cross-view training–validation links that complete-row checks miss. In the controlled experiments, **Ancestor included** and **Ancestor excluded** compare training sets with an exact ancestor block or a matched filler block. Evaluation records and training budgets remain fixed. The target-minus-clean effect measures the change on exposed targets minus the corresponding change on ancestry-clean validation. Its magnitude depends on the learner; none of the three controlled learners shows a clear corresponding gain on untouched evaluation.

The internal keys `naive` and `safe` retain their original meanings: ancestor included and ancestor excluded, respectively. They are preserved in code, configurations, and saved results for reproducibility.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements/analysis.txt
.venv/bin/python -m pip install --no-deps -e .
```

`requirements/analysis.txt` pins the verified lightweight environment.
`requirements/neural.txt` pins the optional GPU training stack.

## Lightweight reproduction

The lightweight path uses the saved predictions and does not download models
or third-party datasets.

```bash
make PYTHON=.venv/bin/python test
make PYTHON=.venv/bin/python reproduce-light
```

It verifies the saved predictions, regenerates all five paper figures and
macros, refreshes the claim-to-evidence registry, and rebuilds the checksum
manifest. The randomized-arrival quantities use the fitted sparse-prefix sizes
902, 1,805, 2,707, and 3,610.

## Full reproduction

The full path reconstructs the controlled data boundary from pinned public
datasets, refits the sparse learners, and optionally reruns the six neural
training cells. Exact source revisions, commands, expected output layout, and
hardware notes are in [REPRODUCING.md](REPRODUCING.md).

## Repository structure

- `src/ancestry_audit/`: typed-ancestry library and command-line interface.
- `analysis/`: estimators, training, evaluation, plotting, and verification;
  final figure entry points are under `analysis/figures/`.
- `scripts/`: source verification and controlled-data construction.
- `results/`: saved prediction-level and aggregate experimental evidence.
- `tables/`: table-ready numerical results.
- `figures/`: the five final publication figures in PDF and PNG formats.
- `macros/`: LaTeX value macros generated from saved results.
- `manifests/`: pinned sources, model revisions, split contracts, and hashes.
- `evidence/`: claim-to-file evidence registry.
- `docs/`: protocols, method details, and figure sources.
- `tests/`: unit and artifact-regression tests.

## Data and model restrictions

Third-party dataset rows, model weights, and training checkpoints are not
included. The repository contains derived predictions with hashed record
identifiers and a split contract containing indices, hashes, and labels but no
source text. Obtain source data and models from their original distributors
under their respective licenses. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff). The paper
citation should be used when the manuscript is available; the software entry
may be used for the code artifact.

## License

The repository's original code and documentation are licensed under the
Apache License 2.0. Third-party resources retain their own licenses.
