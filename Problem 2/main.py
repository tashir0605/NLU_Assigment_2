
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from models.attention_rnn import ManualAttentionRNN
from models.blstm import ManualBLSTM
from models.vanilla_rnn import VanillaCharRNN
from training.train import (
    TrainConfig,
    build_dataloader,
    count_trainable_parameters,
    save_loss_plot,
    train_one_model,
    train_val_split,
)
from utils.dataset import load_names
from utils.vocab import NameVocab


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def qualitative_commentary(samples):


    repetition = 0
    short_names = 0

    for name in samples:
        if len(name) <= 2:
            short_names += 1
        if any(name[i] == name[i + 1] == name[i + 2] for i in range(max(0, len(name) - 2))):
            repetition += 1

    return {
        "very_short_fraction": short_names / max(len(samples), 1),
        "triple_repeat_fraction": repetition / max(len(samples), 1),
    }


def parse_float_csv(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_csv(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def build_model(
    model_name: str,
    vocab_size: int,
    embedding_size: int,
    hidden_size: int,
    num_layers: int,
    dropout_prob: float,
) -> torch.nn.Module:
    if model_name == "vanilla_rnn":
        return VanillaCharRNN(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
        )
    if model_name == "blstm":
        return ManualBLSTM(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
        )
    if model_name == "attention_rnn":
        return ManualAttentionRNN(
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
        )
    raise ValueError(f"Unknown model: {model_name}")


def find_hidden_size_for_budget(
    model_name: str,
    target_params: int,
    vocab_size: int,
    embedding_size: int,
    num_layers: int,
    dropout_prob: float,
    search_min: int,
    search_max: int,
) -> Tuple[int, int]:
    best_hidden = search_min
    best_params = None
    best_gap = float("inf")

    for hidden_size in range(search_min, search_max + 1):
        candidate = build_model(
            model_name=model_name,
            vocab_size=vocab_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
        )
        params = count_trainable_parameters(candidate)
        gap = abs(params - target_params)
        if gap < best_gap:
            best_gap = gap
            best_hidden = hidden_size
            best_params = params

    return best_hidden, best_params if best_params is not None else 0


def repetition_fraction(samples: List[str]) -> float:
    if not samples:
        return 0.0
    count = 0
    for name in samples:
        if any(name[i] == name[i + 1] == name[i + 2] for i in range(max(0, len(name) - 2))):
            count += 1
    return count / len(samples)


def sampling_score(novelty: float, diversity: float, repeat_rate: float) -> float:
    return 0.45 * novelty + 0.45 * diversity + 0.10 * (1.0 - repeat_rate)


def main():
    parser = argparse.ArgumentParser(description="Problem 2: Character-level Indian name generation")
    parser.add_argument("--data", type=str, default="dataset/TrainingNames.txt")
    parser.add_argument("--embedding_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--num_generated", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout_prob", type=float, default=0.15)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.001)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--tf_start", type=float, default=1.0)
    parser.add_argument("--tf_end", type=float, default=0.5)
    parser.add_argument("--temperature_sweep", type=str, default="0.75,0.9,1.05,1.2")
    parser.add_argument("--topk_sweep", type=str, default="0,5,10")
    parser.add_argument("--sweep_num_generated", type=int, default=80)
    parser.add_argument("--match_parameter_budget", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated list from {vanilla_rnn,blstm,attention_rnn} or 'all'",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    config = TrainConfig(
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_generation_length=args.max_length,
        temperature=args.temperature,
        num_generated=args.num_generated,
        dropout_prob=args.dropout_prob,
        val_fraction=args.val_fraction,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        grad_clip_norm=args.grad_clip_norm,
        tf_start=args.tf_start,
        tf_end=args.tf_end,
    )

    print("\n=== Hyperparameters ===")
    for key, value in vars(config).items():
        print(f"{key}: {value}")

    names = load_names(args.data)
    vocab = NameVocab.from_names(names)

    print("\n=== Dataset Stats ===")
    print(f"Total names: {len(names)}")
    print(f"Vocabulary size (with special tokens): {vocab.size}")

    train_names, val_names = train_val_split(names, val_fraction=config.val_fraction, seed=args.seed)
    train_loader = build_dataloader(names=train_names, vocab=vocab, batch_size=config.batch_size)
    val_loader = build_dataloader(names=val_names, vocab=vocab, batch_size=config.batch_size, shuffle=False)

    print(f"Train names: {len(train_names)} | Validation names: {len(val_names)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    all_model_names = ["vanilla_rnn", "blstm", "attention_rnn"]

    if args.models.strip().lower() == "all":
        selected_model_names = all_model_names
    else:
        selected_model_names = [name.strip() for name in args.models.split(",") if name.strip()]
        valid = set(all_model_names)
        invalid = [name for name in selected_model_names if name not in valid]
        if invalid:
            raise ValueError(f"Invalid model names: {invalid}. Valid options: {sorted(valid)}")

    model_hidden_sizes = {name: config.hidden_size for name in selected_model_names}

    if args.match_parameter_budget and selected_model_names:
        vanilla_probe = build_model(
            model_name="vanilla_rnn",
            vocab_size=vocab.size,
            embedding_size=config.embedding_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout_prob=config.dropout_prob,
        )
        target_params = count_trainable_parameters(vanilla_probe)
        print(f"\nParameter-budget matching enabled. Target params: {target_params} (from vanilla_rnn)")

        search_min = 8
        search_max = max(config.hidden_size * 3, config.hidden_size + 16)

        for name in selected_model_names:
            if name == "vanilla_rnn":
                continue
            best_hidden, best_params = find_hidden_size_for_budget(
                model_name=name,
                target_params=target_params,
                vocab_size=vocab.size,
                embedding_size=config.embedding_size,
                num_layers=config.num_layers,
                dropout_prob=config.dropout_prob,
                search_min=search_min,
                search_max=search_max,
            )
            model_hidden_sizes[name] = best_hidden
            print(f"  - {name}: hidden_size={best_hidden}, params~{best_params}")

    models = {
        name: build_model(
            model_name=name,
            vocab_size=vocab.size,
            embedding_size=config.embedding_size,
            hidden_size=model_hidden_sizes[name],
            num_layers=config.num_layers,
            dropout_prob=config.dropout_prob,
        )
        for name in selected_model_names
    }

    losses_by_model = {}
    metrics_by_model = {}

    output_dir = Path("outputs")
    model_dir = output_dir / "saved_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        print(f"\n=== Training: {model_name} ===")
        params = count_trainable_parameters(model)
        print(f"Trainable parameters: {params}")

        history = train_one_model(
            model=model,
            model_name=model_name,
            dataloader=train_loader,
            pad_idx=vocab.pad_idx,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            device=device,
            model_dir=str(model_dir),
            val_dataloader=val_loader,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_min_delta=config.early_stopping_min_delta,
            grad_clip_norm=config.grad_clip_norm,
            tf_start=config.tf_start,
            tf_end=config.tf_end,
        )
        losses_by_model[model_name] = history

        metrics_by_model[model_name] = {
            "parameter_count": params,
            "hidden_size_used": model_hidden_sizes[model_name],
            "final_train_loss": history["train_losses"][-1] if history["train_losses"] else None,
            "best_val_loss": min(history["val_losses"]) if history["val_losses"] else None,
            "best_epoch": history["best_epoch"],
        }

        print(f"Model: {model_name}")
        print(f"Final train loss: {metrics_by_model[model_name]['final_train_loss']:.4f}" if metrics_by_model[model_name]["final_train_loss"] is not None else "Final train loss: N/A")
        print(f"Best val loss: {metrics_by_model[model_name]['best_val_loss']:.4f}" if metrics_by_model[model_name]["best_val_loss"] is not None else "Best val loss: N/A")
        print(f"Best epoch: {metrics_by_model[model_name]['best_epoch']}")

    save_loss_plot(losses_by_model, str(output_dir / "training_loss_curves.png"))

    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as file:
        json.dump(metrics_by_model, file, indent=2)

    print("\nAll outputs saved under: outputs/")


if __name__ == "__main__":
    main()
