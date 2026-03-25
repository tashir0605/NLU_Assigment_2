import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from train_models import load_model, get_best_models


OUTPUT_DIR = os.path.join(".", "outputs")

# ── Word groups for coloured cluster visualisation ───────────────────────────
# Each key is a group label; value is a list of words we'd like to plot.
# Words not in the model's vocab will be silently skipped.
WORD_GROUPS: Dict[str, List[str]] = {
    "Programmes":   ["btech", "mtech", "phd", "msc",
                      "postgraduate", "ug", "pg", "doctoral", "degree"],
    "Research":     ["research", "publication", "journal", "conference",
                      "thesis", "patent", "innovation", "laboratory"],
    "Campus":       ["campus", "library", "sports", "facility",
                      "building", "infrastructure", "centre", "institute"],
    "Examination":  ["examination", "grading", "semester", "attendance",
                      "evaluation", "credits", "cgpa", "grade"],
    "People":       ["student", "professor", "faculty", "alumni", "dean",
                      "mentor", "scholar", "candidate"],
}

# Colour palette (one per group) — colourblind-friendly
GROUP_COLOURS = {
    "Programmes":  "#1f77b4",   # blue
    "Research":    "#ff7f0e",   # orange
    "Campus":      "#2ca02c",   # green
    "Examination": "#d62728",   # red
    "People":      "#9467bd",   # purple
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: EXTRACT VECTORS FOR SELECTED WORDS
# ══════════════════════════════════════════════════════════════════════════════

def _collect_vectors(
    model,
    word_groups: Dict[str, List[str]],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    For every word in every group, fetch its embedding vector from the model.

    Returns
    ───────
    vectors : (N, dim) numpy array
    labels  : list of words (length N)
    groups  : list of group names (length N) — used for colouring
    """
    vectors, labels, groups = [], [], []

    for group_name, words in word_groups.items():
        for w in words:
            if w in model.wv:
                vectors.append(model.wv[w])
                labels.append(w)
                groups.append(group_name)
            # Words missing from vocab are silently ignored; this is expected
            # for small corpora where some low-frequency words get pruned.

    if not vectors:
        raise ValueError("No words from WORD_GROUPS found in model vocabulary.")

    return np.array(vectors), labels, groups

def reduce_pca(vectors: np.ndarray) -> np.ndarray:

    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(vectors)


def reduce_tsne(vectors: np.ndarray) -> np.ndarray:

    n_samples = vectors.shape[0]
    perplexity = min(30, max(5, n_samples - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",            # PCA initialisation for stability
        random_state=42,
        max_iter=1000,         # renamed from n_iter in sklearn ≥ 1.6
    )
    return tsne.fit_transform(vectors)


def _scatter_plot(
    coords_2d: np.ndarray,
    labels: List[str],
    groups: List[str],
    title: str,
    save_path: str,
) -> None:

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot one group at a time so the legend is clean
    plotted_groups = set()
    for i, (x, y) in enumerate(coords_2d):
        grp = groups[i]
        colour = GROUP_COLOURS.get(grp, "#333333")

        # Only add a legend entry once per group
        lbl = grp if grp not in plotted_groups else None
        plotted_groups.add(grp)

        ax.scatter(x, y, c=colour, s=80, alpha=0.75, edgecolors="white",
                   linewidths=0.5, label=lbl, zorder=3)
        # Annotate each point with the word
        ax.annotate(
            labels[i],
            (x, y),
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="bottom",
            xytext=(0, 6),
            textcoords="offset points",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved → {save_path}")


def visualise_embeddings() -> List[str]:

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best = get_best_models()
    saved_files: List[str] = []

    print(f"\n{'='*60}")
    print("  TASK 4 — Embedding Visualisation")
    print(f"{'='*60}")

    for arch_name, model_tag in best.items():
        print(f"\n  Model: {model_tag}")
        model = load_model(model_tag)
        model_dim = getattr(model, "vector_size", "?")
        model_win = getattr(model, "window", "?")
        model_neg = getattr(model, "negative", "?")

        # Collect vectors for our chosen word groups
        vectors, labels, groups = _collect_vectors(model, WORD_GROUPS)
        print(f"    Plotting {len(labels)} words across {len(set(groups))} groups")

        # ── PCA plot ──────────────────────────────────────────────────────
        coords_pca = reduce_pca(vectors)
        pca_path = os.path.join(OUTPUT_DIR, f"viz_{arch_name}_pca.png")
        _scatter_plot(
            coords_pca, labels, groups,
            title=(
                f"PCA — {arch_name.upper()} "
                f"(dim={model_dim}, win={model_win}, neg={model_neg})"
            ),
            save_path=pca_path,
        )
        saved_files.append(pca_path)

        # ── t-SNE plot ────────────────────────────────────────────────────
        coords_tsne = reduce_tsne(vectors)
        tsne_path = os.path.join(OUTPUT_DIR, f"viz_{arch_name}_tsne.png")
        _scatter_plot(
            coords_tsne, labels, groups,
            title=(
                f"t-SNE — {arch_name.upper()} "
                f"(dim={model_dim}, win={model_win}, neg={model_neg})"
            ),
            save_path=tsne_path,
        )
        saved_files.append(tsne_path)

    print(f"\n  All visualisations saved to {OUTPUT_DIR}/")
    return saved_files


# ── standalone execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    visualise_embeddings()
