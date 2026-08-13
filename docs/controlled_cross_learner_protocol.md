# Controlled Cross-Learner Comparison

The cross-learner comparison uses only the fixed final estimators from the
controlled factorial. It adds two predefined contrasts:

```text
Qwen DiD - Word TF-IDF DiD
Qwen DiD - Character TF-IDF DiD
```

The learner interaction intervals reuse paired evaluation-context draws and
the seed-resampling structure of the primary factorial. No additional learner,
checkpoint, evaluation set, or post-outcome selection is introduced.

The interaction supports a bounded claim: under this dataset construction and
fixed protocol, different learner classes do not exploit the same ancestry
exposure equally. It is not a universal ranking of model classes.
