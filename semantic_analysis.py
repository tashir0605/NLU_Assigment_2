import os
import json
from typing import Dict, List, Tuple

import pandas as pd

# ─── project modules ─────────────────────────────────────────────────────────
from train_models import load_model, get_best_models, MODEL_DIR


RESULTS_DIR = os.path.join(".", "outputs")

# Probe words — high-frequency, domain-relevant terms for the IIT Jodhpur corpus
PROBE_WORDS = [
    "research", "student", "professor", "engineering",
    "examination", "semester", "department", "technology",
    "iitj", "jodhpur",
]

# Number of neighbours to retrieve for each probe word
TOP_K = 5

# Analogy experiments — each tuple is (A, B, C) for the test  A : B :: C : ?
# Interpretation: vec(B) - vec(A) + vec(C) ≈ vec(?)
ANALOGY_TESTS = [
    # student : examination :: professor : ?  (expected ~ teaching / evaluation)
    ("student", "examination", "professor"),
    # semester : grade :: thesis : ?  (expected ~ defence / evaluation)
    ("semester", "grade", "thesis"),
    # phd : research :: btech : ?  (expected ~ engineering / industry)
    ("phd", "research", "btech"),
    # engineering : technology :: science : ?  (expected ~ knowledge / research)
    ("engineering", "technology", "science"),
    # student : hostel :: professor : ?  (expected ~ office / department)
    ("student", "institute", "professor"),
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. NEAREST-NEIGHBOUR QUERIES
# ══════════════════════════════════════════════════════════════════════════════

def find_nearest_neighbours(
    model,
    words: List[str],
    top_k: int = TOP_K,
) -> Dict[str, List[Tuple[str, float]]]:

    neighbours: Dict[str, List[Tuple[str, float]]] = {}

    for w in words:
        if w not in model.wv:
            print(f"  ⚠ '{w}' not in vocabulary — skipping")
            continue
        # most_similar returns list of (word, cosine_similarity)
        sims = model.wv.most_similar(w, topn=top_k)
        neighbours[w] = [(word, round(score, 4)) for word, score in sims]

    return neighbours


def display_neighbours(
    neighbours: Dict[str, List[Tuple[str, float]]],
    arch_name: str,
) -> None:

    print(f"\n  ── Top-{TOP_K} Nearest Neighbours ({arch_name.upper()}) ──")
    for word, nbs in neighbours.items():
        nb_str = ", ".join(f"{w} ({s:.4f})" for w, s in nbs)
        print(f"    {word:<12s} → {nb_str}")




def run_analogies(
    model,
    tests: List[Tuple[str, str, str]],
    top_k: int = 3,
) -> List[dict]:

    results = []

    for a, b, c in tests:
        # Check all three words are in the vocab
        missing = [w for w in (a, b, c) if w not in model.wv]
        if missing:
            print(f"  ⚠ Analogy '{a}:{b}::{c}:?' — missing words: {missing}")
            results.append({
                "a": a, "b": b, "c": c,
                "predictions": [],
                "note": f"missing vocab: {missing}",
            })
            continue

        # Analogy formula: positive=[b, c], negative=[a]  →  b - a + c
        preds = model.wv.most_similar(positive=[b, c], negative=[a], topn=top_k)
        results.append({
            "a": a, "b": b, "c": c,
            "predictions": [(w, round(s, 4)) for w, s in preds],
        })

    return results


def display_analogies(analogy_results: List[dict], arch_name: str) -> None:
    """
    Pretty-print analogy experiment results.
    """
    print(f"\n  ── Analogy Tests ({arch_name.upper()}) ──")
    for r in analogy_results:
        if r["predictions"]:
            pred_str = ", ".join(f"{w} ({s:.4f})" for w, s in r["predictions"])
        else:
            pred_str = f"N/A ({r.get('note', '')})"
        print(f"    {r['a']} : {r['b']} :: {r['c']} : ?  →  {pred_str}")



def run_semantic_analysis() -> dict:

    best = get_best_models()           # {'cbow': tag, 'skipgram': tag}
    all_results: Dict[str, dict] = {}

    print(f"\n{'='*60}")
    print("  TASK 3 — Semantic Analysis")
    print(f"{'='*60}")

    for arch_name, model_tag in best.items():
        print(f"\n  Loading model: {model_tag}")
        model = load_model(model_tag)

        # ── Nearest neighbours ────────────────────────────────────────────
        neighbours = find_nearest_neighbours(model, PROBE_WORDS)
        display_neighbours(neighbours, arch_name)

        # ── Analogies ─────────────────────────────────────────────────────
        analogies = run_analogies(model, ANALOGY_TESTS)
        display_analogies(analogies, arch_name)

        # Pack into results
        all_results[arch_name] = {
            "model_tag": model_tag,
            "nearest_neighbours": {
                w: [(n, s) for n, s in nbs]
                for w, nbs in neighbours.items()
            },
            "analogies": analogies,
        }

    # ── Persist to disk ───────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "semantic_analysis.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n  Results saved to {out_path}")

    # ── Side-by-side comparison table ─────────────────────────────────────
    _print_comparison_table(all_results)

    return all_results


def _print_comparison_table(results: dict) -> None:
    """
    Print a compact side-by-side table comparing CBOW and Skip-gram
    nearest neighbours for each probe word.
    """
    print(f"\n{'='*70}")
    print("  CBOW vs Skip-gram — Nearest Neighbours Comparison")
    print(f"{'='*70}")
    print(f"  {'Word':<12s} | {'CBOW Top-1':<20s} | {'Skip-gram Top-1':<20s}")
    print(f"  {'-'*12}-+-{'-'*20}-+-{'-'*20}")

    for word in PROBE_WORDS:
        cbow_nb = "--"
        sg_nb   = "--"

        if "cbow" in results:
            nbs = results["cbow"]["nearest_neighbours"].get(word, [])
            if nbs:
                cbow_nb = f"{nbs[0][0]} ({nbs[0][1]:.4f})"

        if "skipgram" in results:
            nbs = results["skipgram"]["nearest_neighbours"].get(word, [])
            if nbs:
                sg_nb = f"{nbs[0][0]} ({nbs[0][1]:.4f})"

        print(f"  {word:<12s} | {cbow_nb:<20s} | {sg_nb:<20s}")
    print(f"{'='*70}\n")


# ── standalone execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    run_semantic_analysis()
