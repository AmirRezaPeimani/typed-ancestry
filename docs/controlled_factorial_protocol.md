# Controlled Factorial Protocol

## Fixed comparison

The controlled experiment compares two training boundaries with identical
training-set size:

- **Ancestor included (`naive`):** 4,000 shared training pairs plus 512 ancestor pairs tied
  to the target validation view.
- **Ancestor excluded (`safe`):** the same 4,000 shared pairs plus 512 size-matched,
  component-disjoint filler pairs.

Evaluation uses the same 512 target pairs, 512 ancestry-clean validation
pairs, and 512 untouched clean-test pairs for every learner, boundary, and
seed. The primary estimand is:

```text
(target ancestor-included-minus-ancestor-excluded accuracy gap)
-
(clean-validation ancestor-included-minus-ancestor-excluded accuracy gap)
```

The untouched estimand is the ancestor-included-minus-ancestor-excluded accuracy difference on the
untouched clean test.

## Learners

- Word TF-IDF over candidate edits with a Bradley-Terry logistic learner.
- Character TF-IDF over candidate edits with a Bradley-Terry logistic learner.
- Qwen2.5-1.5B-Instruct with one scalar output and LoRA.

All learners use the same 4,512-pair training files. The sparse controls
represent candidate edits only. The neural serialization includes the
context, original response, feedback, and candidate edit. The same
serialization and token budget are used within every neural comparison.

## Uncertainty

Intervals use paired context resampling and resample distinct training-seed
prediction vectors hierarchically. Paired sign randomization is used for the
reported two-sided point-estimate tests. Sparse nominal seeds that produce
identical prediction vectors count as one uncertainty unit; the neural grid
has three distinct vectors.

## Competence

A separate Qwen selector run is trained for one epoch with seed 20260801 on
the 1,000 `edit_quality` pairs that form the `edit_quality` portion of the
shared 4,000-pair training base. It uses neither experimental boundary and
contains no target-specific ancestors or boundary-specific fillers. Its
training components are disjoint from the exposed-target, ancestry-clean
validation, untouched, and 400-pair development sets. This selector run is
not one of the six factorial runs.

The selector chooses the 100% checkpoint fraction. Its single development
evaluation contains 317 correct outcomes, 82 incorrect outcomes, and one
exact tie. That fraction is then fixed for three seeds under each of the two
experimental boundaries. The competence report includes both:

- tie-adjusted accuracy: (317 + 0.5)/400 = 79.38%;
- strict accuracy: 317/400 = 79.25%, strict Wilson 95% interval
  [75.01%, 82.94%].

The 79.38% result belongs only to the selector run; it is not an average over
the six experimental models.
