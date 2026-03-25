"""Evaluation metrics for generated names."""

from typing import List, Set


def novelty_rate(generated_names: List[str], training_names: Set[str]) -> float:
    """
    Novelty = (# generated names not in training set) / (total generated names)
    """

    if not generated_names:
        return 0.0
    novel_count = sum(1 for name in generated_names if name not in training_names)
    return novel_count / len(generated_names)


def diversity_rate(generated_names: List[str]) -> float:
    """
    Diversity = (# unique generated names) / (total generated names)
    """

    if not generated_names:
        return 0.0
    return len(set(generated_names)) / len(generated_names)
