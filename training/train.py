from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import random

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from utils.dataset import NameSequenceDataset, collate_batch
from utils.vocab import NameVocab


@dataclass
class TrainConfig:
    embedding_size: int = 32
    hidden_size: int = 64
    num_layers: int = 1
    learning_rate: float = 0.003
    batch_size: int = 32
    epochs: int = 20
    max_generation_length: int = 20
    temperature: float = 0.9
    num_generated: int = 200
    dropout_prob: float = 0.15
    val_fraction: float = 0.1
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 0.001
    grad_clip_norm: float = 1.0
    tf_start: float = 1.0
    tf_end: float = 0.5


def build_dataloader(names: List[str], vocab: NameVocab, batch_size: int, shuffle: bool = True) -> DataLoader:
    dataset = NameSequenceDataset(names=names, vocab=vocab)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_batch(batch=batch, pad_idx=vocab.pad_idx),
    )


def train_val_split(names: List[str], val_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    """Random split for train/validation sets."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")

    shuffled = names[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_fraction))
    val_names = shuffled[:val_size]
    train_names = shuffled[val_size:]

    if len(train_names) == 0:
        raise ValueError("Validation split is too large; no training data left.")

    return train_names, val_names


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def train_one_model(
    model: torch.nn.Module,
    model_name: str,
    dataloader: DataLoader,
    pad_idx: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    model_dir: str,
    val_dataloader: Optional[DataLoader] = None,
    early_stopping_patience: int = 4,
    early_stopping_min_delta: float = 0.001,
    grad_clip_norm: float = 1.0,
    tf_start: float = 1.0,
    tf_end: float = 0.5,
) -> Dict[str, List[float] | int]:


    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    train_losses: List[float] = []
    val_losses: List[float] = []

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        if epochs == 1:
            teacher_forcing_ratio = tf_start
        else:
            progress = (epoch - 1) / (epochs - 1)
            teacher_forcing_ratio = tf_start + progress * (tf_end - tf_start)

        for inputs, targets, lengths in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)

            optimizer.zero_grad()
            logits, _ = model(
                inputs,
                lengths=lengths,
                target_ids=targets,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )

            vocab_size = logits.shape[-1]
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()

            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()

            total_loss += loss.item()
            steps += 1

        train_loss = total_loss / max(steps, 1)
        train_losses.append(train_loss)

        if val_dataloader is not None:
            model.eval()
            val_total = 0.0
            val_steps = 0
            with torch.no_grad():
                for val_inputs, val_targets, val_lengths in val_dataloader:
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)
                    val_lengths = val_lengths.to(device)

                    val_logits, _ = model(
                        val_inputs,
                        lengths=val_lengths,
                        target_ids=val_targets,
                        teacher_forcing_ratio=1.0,
                    )

                    vocab_size = val_logits.shape[-1]
                    val_loss = criterion(val_logits.reshape(-1, vocab_size), val_targets.reshape(-1))
                    val_total += val_loss.item()
                    val_steps += 1

            mean_val_loss = val_total / max(val_steps, 1)
            val_losses.append(mean_val_loss)
            print(
                f"[{model_name}] Epoch {epoch:02d}/{epochs} - "
                f"Train Loss: {train_loss:.4f} | Val Loss: {mean_val_loss:.4f} | TF: {teacher_forcing_ratio:.3f}"
            )

            if mean_val_loss < (best_val_loss - early_stopping_min_delta):
                best_val_loss = mean_val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(f"[{model_name}] Early stopping triggered at epoch {epoch}.")
                    break
        else:
            print(f"[{model_name}] Epoch {epoch:02d}/{epochs} - Loss: {train_loss:.4f} | TF: {teacher_forcing_ratio:.3f}")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(model_dir) / f"{model_name}.pt"

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), save_path)
    print(f"Saved model: {save_path}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
    }


def save_loss_plot(loss_map: Dict[str, Dict[str, List[float]]], output_path: str):
    """Saves training loss curves for all models in one figure."""

    plt.figure(figsize=(9, 5))
    for model_name, history in loss_map.items():
        train_losses = history.get("train_losses", [])
        val_losses = history.get("val_losses", [])

        if train_losses:
            plt.plot(range(1, len(train_losses) + 1), train_losses, label=f"{model_name} (train)")
        if val_losses:
            plt.plot(range(1, len(val_losses) + 1), val_losses, linestyle="--", label=f"{model_name} (val)")

    plt.title("Training Loss Curves - Character Name Generation")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved loss curves: {output_path}")
