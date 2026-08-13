#!/usr/bin/env python3
"""Evaluate saved construct-valid LoRA checkpoints on explicitly named sets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from controlled_serializer import ConstructSerializer
from run_controlled_qwen15_training import (
    choose_device,
    evaluate,
    load_jsonl_gz,
    tokenize_rows,
)


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("evaluation must be NAME=PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval", type=parse_named_path, action="append", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--checkpoint-fraction", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)

    completion = json.loads(
        (args.run / "completion.json").read_text(encoding="utf-8")
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    serializer = ConstructSerializer(tokenizer)
    evaluations = {
        name: tokenize_rows(load_jsonl_gz(path), serializer)
        for name, path in args.eval
    }
    device = choose_device(args.device)
    checkpoints = sorted(args.run.glob("checkpoint-*"))
    if args.checkpoint_fraction is not None:
        checkpoints = [
            path
            for path in checkpoints
            if path.name == f"checkpoint-{args.checkpoint_fraction:03d}"
        ]
    if not checkpoints:
        raise RuntimeError("no requested checkpoint found")

    summaries = []
    for checkpoint in checkpoints:
        fraction = int(checkpoint.name.rsplit("-", 1)[1])
        base = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=1,
            torch_dtype=(torch.bfloat16 if device.type == "cuda" else None),
            local_files_only=True,
            ignore_mismatched_sizes=True,
        )
        base.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(base.config, "use_cache"):
            base.config.use_cache = False
        model = PeftModel.from_pretrained(base, checkpoint, is_trainable=False)
        model.to(device)
        for name, rows in evaluations.items():
            metrics, predictions = evaluate(
                model, rows, tokenizer, device, batch_size=8
            )
            predictions_path = (
                args.output / f"checkpoint-{fraction:03d}_{name}.csv"
            )
            with predictions_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(predictions[0])
                )
                writer.writeheader()
                writer.writerows(predictions)
            summaries.append(
                {
                    "checkpoint_fraction": fraction,
                    "evaluation_set": name,
                    **metrics,
                    "predictions": predictions_path.name,
                }
            )
        del model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "status": "complete",
        "run": str(args.run.resolve()),
        "model": str(args.model.resolve()),
        "evaluation_sets": [name for name, _ in args.eval],
        "summaries": summaries,
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
