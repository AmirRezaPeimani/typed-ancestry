# Figure sources

`make figures` rebuilds every paper figure from the saved machine-readable
results. The plotting entry points live in `analysis/figures/`.

| Figure | Analytical question | Principal inputs |
|---|---|---|
| `figures/learning_dynamics_main.pdf` | Which exact typed atoms cross dataset views despite no complete-row match? | `results/typed_ancestry/exposure_summary.json`, `results/typed_ancestry/exposure_same_any_view.csv` |
| `figures/learning_dynamics_acquisition.pdf` | Does validation inflation grow across training prefixes and align with randomized ancestor arrival? | `results/learning_dynamics/`, `results/ancestor_arrival/` |
| `figures/controlled_cross_learner_factorial_main.pdf` | Do identical boundaries produce different target-minus-clean effects across learners? | `results/controlled_sparse_factorial/`, `results/controlled_qwen15_factorial/`, `results/controlled_cross_learner_factorial/` |
| `figures/candidate_selection_main.pdf` | Do exposed-target and ancestry-clean validation select different predefined candidates? | `results/model_selection/predictions.csv.gz`, `results/model_selection/summary.json` |
| `figures/candidate_selection_supp.pdf` | How sensitive is the validation weighting to its exposed-target weight? | `results/model_selection/predictions.csv.gz`, `results/model_selection/summary.json` |

The normalization sensitivity values remain available in
`results/ancestry_robustness/`; they are reported numerically rather than as a
separate release figure.

## Rendering and terminology

All five publication PDFs are vector output with embedded fonts; PNG files are convenience previews. Model identities and comparisons are recoverable through direct labels, marker shapes, line styles, and open/filled markers in grayscale.

- Learning dynamics overview A: Cross-view training–validation links.
- Acquisition A–D: Accuracy by training condition; Validation inflation across training prefixes; Controlled effects; Effect aligned to ancestor arrival.
- Cross-learner A–B: Accuracy by training condition; Cross-learner differences in target-minus-clean effect. Open markers mean Ancestor excluded; filled markers mean Ancestor included.
- Candidate selection A–B: Model rankings by validation set; Untouched evaluation.
- Candidate sensitivity A–C: Sensitivity to validation weighting; Bootstrap selection probability; Untouched evaluation.

Numerical arrays, uncertainty intervals, and saved predictions are unchanged by the terminology update. The supplementary probability matrix is rendered as vector cells.
