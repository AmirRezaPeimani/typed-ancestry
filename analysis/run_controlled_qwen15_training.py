#!/usr/bin/env python3
"""Run the frozen construct-valid compact-neural experiment."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from controlled_serializer import ConstructSerializer


CHECKPOINT_FRACTIONS = (20, 40, 60, 80, 100)


def load_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def wilson(successes: float, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return (centre - radius) / denominator, (centre + radius) / denominator


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("evaluation must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("evaluation must be NAME=PATH")
    return name, Path(path)


def tokenize_rows(
    rows: list[dict[str, Any]], serializer: ConstructSerializer
) -> list[dict[str, Any]]:
    encoded = []
    for row in rows:
        left = serializer.serialize(row, "response1")
        right = serializer.serialize(row, "response2")
        encoded.append(
            {
                "pair_id": row["pair_id"],
                "label": int(row["overall_preference"]),
                "left_ids": left.input_ids,
                "right_ids": right.input_ids,
            }
        )
    return encoded


def batches(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def collate_pair_batch(
    batch: list[dict[str, Any]], tokenizer: Any, device: torch.device
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    sequences = []
    for row in batch:
        sequences.append({"input_ids": row["left_ids"]})
        sequences.append({"input_ids": row["right_ids"]})
    model_inputs = tokenizer.pad(sequences, padding=True, return_tensors="pt")
    model_inputs = {key: value.to(device) for key, value in model_inputs.items()}
    labels = torch.tensor(
        [row["label"] for row in batch], dtype=torch.float32, device=device
    )
    return model_inputs, labels


def score_batch(
    model: torch.nn.Module,
    model_inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = model(**model_inputs).logits.squeeze(-1)
    return scores[0::2], scores[1::2]


def trainable_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode("utf-8"))
            digest.update(
                parameter.detach().float().cpu().contiguous().numpy().tobytes()
            )
    return digest.hexdigest()


def configure_model(
    args: argparse.Namespace,
    tokenizer: Any,
    device: torch.device,
) -> torch.nn.Module:
    dtype = torch.bfloat16 if device.type == "cuda" and args.lora else None
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        torch_dtype=dtype,
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.lora:
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = [
            value.strip()
            for value in args.lora_target_modules.split(",")
            if value.strip()
        ]
        if not target_modules:
            raise ValueError("LoRA target modules must be nonempty")
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_CLS,
                target_modules=target_modules,
                modules_to_save=[args.head_module],
            ),
        )
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model.to(device)


def evaluate(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in batches(rows, batch_size):
            inputs, labels = collate_pair_batch(batch, tokenizer, device)
            score1, score2 = score_batch(model, inputs)
            margins = labels * (score2 - score1)
            for row, label, left, right, margin in zip(
                batch,
                labels.cpu().tolist(),
                score1.cpu().tolist(),
                score2.cpu().tolist(),
                margins.cpu().tolist(),
                strict=True,
            ):
                correct = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
                predictions.append(
                    {
                        "pair_id": row["pair_id"],
                        "label": int(label),
                        "score1": float(left),
                        "score2": float(right),
                        "signed_margin": float(margin),
                        "correct": correct,
                    }
                )
    margins_array = np.asarray([row["signed_margin"] for row in predictions])
    successes = sum(row["correct"] for row in predictions)
    low, high = wilson(successes, len(predictions))
    distinct = len(set(np.round(margins_array, 4).tolist()))
    finite = bool(np.isfinite(margins_array).all())
    standard_deviation = float(np.std(margins_array, ddof=1))
    metrics = {
        "pairs": len(predictions),
        "accuracy": successes / len(predictions),
        "wilson_95": [low, high],
        "mean_signed_margin": float(np.mean(margins_array)),
        "median_signed_margin": float(np.median(margins_array)),
        "margin_std": standard_deviation,
        "margin_min": float(np.min(margins_array)),
        "margin_max": float(np.max(margins_array)),
        "distinct_margins_rounded_4dp": distinct,
        "positive_prediction_rate": float(
            np.mean(
                np.asarray(
                    [row["score2"] > row["score1"] for row in predictions],
                    dtype=float,
                )
            )
        ),
        "finite": finite,
        "noncollapsed": finite and standard_deviation >= 0.05 and distinct >= 10,
    }
    return metrics, predictions


def save_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment(device: torch.device) -> dict[str, Any]:
    packages = {}
    for name in (
        "torch",
        "transformers",
        "numpy",
        "scipy",
        "scikit-learn",
        "sentence-transformers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": str(device),
        "packages": packages,
        "torch_mps_available": torch.backends.mps.is_available(),
        "torch_cuda_available": torch.cuda.is_available(),
        "logical_cpus": os.cpu_count(),
    }


def overfit(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    seed_everything(args.seed)
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    serializer = ConstructSerializer(tokenizer)
    rows = load_jsonl_gz(args.train)[:64]
    encoded = tokenize_rows(rows, serializer)
    model = configure_model(args, tokenizer, device)
    initial_state = trainable_state_hash(model)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else (2e-4 if args.lora else 5e-5)
    )
    optimizer = AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    started = time.perf_counter()
    history = []
    passed_epoch = None
    maximum_epochs = args.epochs if args.epochs is not None else 30
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        losses = []
        for batch in batches(encoded, 8):
            inputs, labels = collate_pair_batch(batch, tokenizer, device)
            score1, score2 = score_batch(model, inputs)
            loss = F.softplus(-labels * (score2 - score1)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite overfit loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        metrics, _ = evaluate(model, encoded, tokenizer, device, batch_size=8)
        history.append(
            {
                "epoch": epoch,
                "mean_loss": float(np.mean(losses)),
                **metrics,
            }
        )
        if metrics["accuracy"] >= 0.95:
            passed_epoch = epoch
            break
    final_state = trainable_state_hash(model)
    payload = {
        "status": "complete",
        "mode": "overfit",
        "protocol": "docs/controlled_factorial_protocol.md",
        "fixture": "first 64 rows of the selector training set",
        "train_file": str(args.train.resolve()),
        "train_sha256": sha256(args.train),
        "model": str(args.model.resolve()),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "learning_rate": learning_rate,
        "maximum_epochs": maximum_epochs,
        "passed_epoch": passed_epoch,
        "passed": passed_epoch is not None,
        "trainable_state_changed": final_state != initial_state,
        "initial_trainable_state_sha256": initial_state,
        "final_trainable_state_sha256": final_state,
        "lora": bool(args.lora),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": environment(device),
        "history": history,
    }
    (args.output / "overfit_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    seed_everything(args.seed)
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(args.output / "tokenizer")
    serializer = ConstructSerializer(tokenizer)
    train_rows = tokenize_rows(load_jsonl_gz(args.train), serializer)
    evaluations = {
        name: tokenize_rows(load_jsonl_gz(path), serializer)
        for name, path in args.eval
    }
    model = configure_model(args, tokenizer, device)
    initial_state = trainable_state_hash(model)
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else (2e-4 if args.lora else 2e-5)
    )
    optimizer = AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    microbatch_size = args.microbatch_pairs
    accumulation = args.gradient_accumulation
    epochs = args.epochs if args.epochs is not None else (1 if args.lora else 3)
    updates_per_epoch = math.ceil(
        math.ceil(len(train_rows) / microbatch_size) / accumulation
    )
    total_updates = updates_per_epoch * epochs
    warmup_updates = round(total_updates * 0.06)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_updates, total_updates
    )
    checkpoint_updates = {
        max(1, round(total_updates * fraction / 100)): fraction
        for fraction in CHECKPOINT_FRACTIONS
    }
    if len(checkpoint_updates) != len(CHECKPOINT_FRACTIONS):
        raise AssertionError("checkpoint update collision")

    started = time.perf_counter()
    update = 0
    microstep = 0
    training_history = []
    checkpoint_results: dict[str, Any] = {}
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in batches(train_rows, microbatch_size):
            inputs, labels = collate_pair_batch(batch, tokenizer, device)
            score1, score2 = score_batch(model, inputs)
            raw_loss = F.softplus(-labels * (score2 - score1)).mean()
            if not torch.isfinite(raw_loss):
                raise FloatingPointError("nonfinite training loss")
            (raw_loss / accumulation).backward()
            microstep += 1
            end_epoch = microstep % math.ceil(
                len(train_rows) / microbatch_size
            ) == 0
            if microstep % accumulation == 0 or end_epoch:
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    .detach()
                    .cpu()
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                training_history.append(
                    {
                        "update": update,
                        "epoch": epoch,
                        "loss": float(raw_loss.detach().cpu()),
                        "gradient_norm_before_clip": gradient_norm,
                        "learning_rate": float(scheduler.get_last_lr()[0]),
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                if update in checkpoint_updates:
                    fraction = checkpoint_updates[update]
                    checkpoint = args.output / f"checkpoint-{fraction:03d}"
                    model.save_pretrained(checkpoint, safe_serialization=True)
                    results = {}
                    for name, eval_rows in evaluations.items():
                        metrics, predictions = evaluate(
                            model,
                            eval_rows,
                            tokenizer,
                            device,
                            batch_size=8,
                        )
                        results[name] = metrics
                        save_predictions(
                            checkpoint / f"predictions_{name}.csv",
                            predictions,
                        )
                    metadata = {
                        "fraction": fraction,
                        "update": update,
                        "epoch": epoch,
                        "evaluation": results,
                    }
                    (checkpoint / "metrics.json").write_text(
                        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    checkpoint_results[str(fraction)] = metadata
                    model.train()
    if update != total_updates:
        raise AssertionError(f"completed {update} updates, expected {total_updates}")
    final_state = trainable_state_hash(model)
    elapsed = time.perf_counter() - started
    payload = {
        "status": "complete",
        "mode": "train",
        "protocol": "docs/controlled_factorial_protocol.md",
        "train_file": str(args.train.resolve()),
        "train_sha256": sha256(args.train),
        "evaluation_files": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in args.eval
        },
        "model": str(args.model.resolve()),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "seed": args.seed,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": 0.01,
            "warmup_updates": warmup_updates,
            "gradient_clip": 1.0,
        },
        "epochs": epochs,
        "microbatch_pairs": microbatch_size,
        "gradient_accumulation": accumulation,
        "effective_pair_batch": microbatch_size * accumulation,
        "total_updates": total_updates,
        "checkpoint_updates": {
            str(fraction): update
            for update, fraction in sorted(checkpoint_updates.items())
        },
        "trainable_state_changed": final_state != initial_state,
        "initial_trainable_state_sha256": initial_state,
        "final_trainable_state_sha256": final_state,
        "lora": {
            "enabled": bool(args.lora),
            "rank": args.lora_rank if args.lora else None,
            "alpha": args.lora_alpha if args.lora else None,
            "dropout": args.lora_dropout if args.lora else None,
            "target_modules": (
                args.lora_target_modules.split(",") if args.lora else []
            ),
            "head_module": args.head_module if args.lora else None,
        },
        "elapsed_seconds": elapsed,
        "projected_six_run_grid_hours": elapsed * 6 / 3600,
        "environment": environment(device),
        "training_history": training_history,
        "checkpoints": checkpoint_results,
    }
    (args.output / "completion.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "training_history"}, indent=2, sort_keys=True))
    return payload


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command", required=True)
    for name in ("overfit", "train"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--model", type=Path, required=True)
        sub.add_argument("--model-id", default="unrecorded-local-model")
        sub.add_argument("--model-revision", default="unrecorded")
        sub.add_argument("--train", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--seed", type=int, default=20260727)
        sub.add_argument("--learning-rate", type=float)
        sub.add_argument("--epochs", type=int)
        sub.add_argument("--microbatch-pairs", type=int, default=4)
        sub.add_argument("--gradient-accumulation", type=int, default=8)
        sub.add_argument("--lora", action="store_true")
        sub.add_argument("--lora-rank", type=int, default=16)
        sub.add_argument("--lora-alpha", type=int, default=32)
        sub.add_argument("--lora-dropout", type=float, default=0.05)
        sub.add_argument(
            "--lora-target-modules",
            default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        )
        sub.add_argument("--head-module", default="score")
        sub.add_argument(
            "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
        )
        if name == "train":
            sub.add_argument(
                "--eval", type=parse_named_path, action="append", default=[]
            )
    return root


def main() -> int:
    args = parser().parse_args()
    payload = overfit(args) if args.command == "overfit" else train(args)
    if args.command == "overfit":
        return 0 if payload["passed"] else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
