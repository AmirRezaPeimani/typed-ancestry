# Qwen Checkpoint-Selector Scoring

The separate selector run uses seed 20260801 and has one development
evaluation containing 400 binary preference pairs. It is not one of the six
factorial Qwen runs. The selected 100% checkpoint fraction is fixed for all
three seeds under both experimental boundaries.

| Outcome | Count | Credit |
|---|---:|---:|
| Correct | 317 | 1 |
| Incorrect | 82 | 0 |
| Exact score tie | 1 | 0.5 in the tie-adjusted metric |

Therefore:

```text
tie-adjusted accuracy = (317 + 0.5) / 400 = 79.375% -> 79.38%
strict accuracy       = 317 / 400         = 79.25%
strict Wilson 95% CI                         [75.01%, 82.94%]
```

The 79.38% value belongs only to the selector run; it is not an average across
the six experimental models, seeds, or repeated evaluations. The single exact
tie accounts for the apparent non-quarter-point increment.
