# Figure sources

`make figures` rebuilds every paper figure from the saved machine-readable
results. The plotting entry points live in `analysis/figures/`.

| Figure | Analytical question | Principal inputs |
|---|---|---|
| `figures/learning_dynamics_main.pdf` | Which exact typed atoms cross dataset views despite no complete-row match? | `results/typed_ancestry/exposure_summary.json`, `results/typed_ancestry/exposure_same_any_view.csv` |
| `figures/learning_dynamics_acquisition.pdf` | Does validation distortion grow during training and align with randomized ancestor arrival? | `results/learning_dynamics/`, `results/ancestor_arrival/` |
| `figures/controlled_cross_learner_factorial_main.pdf` | Do identical boundaries produce different target-minus-clean effects across learners? | `results/controlled_sparse_factorial/`, `results/controlled_qwen15_factorial/`, `results/controlled_cross_learner_factorial/` |
| `figures/candidate_selection_main.pdf` | Do exposed-target and ancestry-clean validation select different predefined candidates? | `results/model_selection/predictions.csv.gz`, `results/model_selection/summary.json` |
| `figures/candidate_selection_supp.pdf` | How sensitive is the post hoc validation objective to its exposed-target weight? | `results/model_selection/predictions.csv.gz`, `results/model_selection/summary.json` |

The normalization sensitivity values remain available in
`results/ancestry_robustness/`; they are reported numerically rather than as a
separate release figure.
