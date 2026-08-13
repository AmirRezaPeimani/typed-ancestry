#!/usr/bin/env python3
"""Run frozen sparse baselines on the controlled ancestry boundaries."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


SEEDS = (20260727, 20260728, 20260729)
BOUNDARIES = ("naive", "safe")
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
MODELS = {
    "controlled_word": {
        "analyzer": "word",
        "ngram_range": (1, 2),
        "min_df": 3,
        "max_features": 120_000,
        "C": 0.3,
    },
    "controlled_character": {
        "analyzer": "char",
        "ngram_range": (3, 5),
        "min_df": 3,
        "max_features": 160_000,
        "C": 1.0,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    identifiers = [row["pair_id"] for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate pair identifier in {path}")
    return rows


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [int(float(row["overall_preference"]) > 0) for row in rows],
        dtype=np.int8,
    )


def fit_predict(
    train: list[dict[str, Any]],
    evaluation: dict[str, list[dict[str, Any]]],
    *,
    model_name: str,
    seed: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    spec = MODELS[model_name]
    vectorizer = TfidfVectorizer(
        analyzer=str(spec["analyzer"]),
        ngram_range=tuple(spec["ngram_range"]),
        min_df=int(spec["min_df"]),
        max_features=int(spec["max_features"]),
        sublinear_tf=True,
        dtype=np.float32,
    )
    vectorizer.fit(
        [str(row["response1"]) for row in train]
        + [str(row["response2"]) for row in train]
    )
    train_features = vectorizer.transform(
        [str(row["response2"]) for row in train]
    ) - vectorizer.transform([str(row["response1"]) for row in train])
    model = LogisticRegression(
        max_iter=1500,
        C=float(spec["C"]),
        random_state=seed,
        solver="liblinear",
    )
    model.fit(train_features, labels(train))

    outputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, rows in evaluation.items():
        features = vectorizer.transform(
            [str(row["response2"]) for row in rows]
        ) - vectorizer.transform([str(row["response1"]) for row in rows])
        margins = model.decision_function(features)
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(margins, -40, 40)))
        outputs[name] = (margins, probabilities)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    evaluation_paths = {
        "target": args.data_dir / "target_512.jsonl.gz",
        "clean_validation": args.data_dir / "clean_validation_512.jsonl.gz",
        "untouched_clean_test": (
            args.data_dir / "untouched_clean_test_512.jsonl.gz"
        ),
    }
    evaluation = {
        name: read_rows(path) for name, path in evaluation_paths.items()
    }
    if any(len(rows) != 512 for rows in evaluation.values()):
        raise ValueError("every evaluation set must contain 512 pairs")

    args.output.mkdir(parents=True)
    predictions_path = args.output / "predictions.csv.gz"
    fields = [
        "model",
        "seed",
        "boundary",
        "fraction",
        "evaluation_set",
        "pair_id",
        "label_response2_preferred",
        "margin_response2_minus_response1",
        "probability_response2_preferred",
        "correct",
    ]
    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    with gzip.open(
        predictions_path, "wt", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model_name in MODELS:
            for seed in SEEDS:
                for boundary in BOUNDARIES:
                    train_path = (
                        args.data_dir
                        / f"train_{boundary}_seed{seed}.jsonl.gz"
                    )
                    train_all = read_rows(train_path)
                    if len(train_all) != 4512:
                        raise ValueError(
                            f"expected 4512 pairs in {train_path}"
                        )
                    for fraction in FRACTIONS:
                        train = train_all[: round(len(train_all) * fraction)]
                        outputs = fit_predict(
                            train,
                            evaluation,
                            model_name=model_name,
                            seed=seed,
                        )
                        for name, rows in evaluation.items():
                            truth = labels(rows)
                            margins, probabilities = outputs[name]
                            predicted = probabilities >= 0.5
                            summaries.append(
                                {
                                    "model": model_name,
                                    "seed": seed,
                                    "boundary": boundary,
                                    "fraction": fraction,
                                    "training_pairs": len(train),
                                    "evaluation_set": name,
                                    "n": len(rows),
                                    "accuracy": float(
                                        accuracy_score(truth, predicted)
                                    ),
                                    "auroc": float(
                                        roc_auc_score(truth, probabilities)
                                    ),
                                    "margin_mean": float(np.mean(margins)),
                                    "margin_std": float(
                                        np.std(margins, ddof=1)
                                    ),
                                    "positive_margin_rate": float(
                                        np.mean(margins > 0)
                                    ),
                                }
                            )
                            for row, label, margin, probability in zip(
                                rows,
                                truth,
                                margins,
                                probabilities,
                                strict=True,
                            ):
                                writer.writerow(
                                    {
                                        "model": model_name,
                                        "seed": seed,
                                        "boundary": boundary,
                                        "fraction": fraction,
                                        "evaluation_set": name,
                                        "pair_id": row["pair_id"],
                                        "label_response2_preferred": int(
                                            label
                                        ),
                                        "margin_response2_minus_response1": (
                                            float(margin)
                                        ),
                                        "probability_response2_preferred": (
                                            float(probability)
                                        ),
                                        "correct": int(
                                            (probability >= 0.5) == label
                                        ),
                                    }
                                )

    completion = {
        "status": "complete",
        "protocol": "docs/controlled_factorial_protocol.md",
        "models": {
            name: {
                **spec,
                "ngram_range": list(spec["ngram_range"]),
                "sublinear_tf": True,
                "solver": "liblinear",
                "max_iter": 1500,
            }
            for name, spec in MODELS.items()
        },
        "seeds": list(SEEDS),
        "boundaries": list(BOUNDARIES),
        "fractions": list(FRACTIONS),
        "data_manifest_sha256": sha256(args.data_dir / "manifest.json"),
        "elapsed_seconds": time.monotonic() - started,
        "prediction_file": predictions_path.name,
        "prediction_file_sha256": sha256(predictions_path),
        "summary": summaries,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (args.output / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "elapsed_seconds": completion["elapsed_seconds"],
                "summary_rows": len(summaries),
                "predictions_sha256": completion[
                    "prediction_file_sha256"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
