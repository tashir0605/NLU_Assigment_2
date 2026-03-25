import argparse
import time

# ── project modules (each handles one task) ──────────────────────────────────
from collect_data      import run_collection
from preprocess        import run_preprocessing
from train_models      import run_training
from semantic_analysis import run_semantic_analysis
from visualize         import visualise_embeddings


def main():
    # ── CLI argument: --live flag to scrape real websites ─────────────────
    parser = argparse.ArgumentParser(
        description="NLU Assignment 2 — Word2Vec on IIT Jodhpur data"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Scrape real IIT Jodhpur websites instead of using sample corpus",
    )
    args = parser.parse_args()

    t_start = time.time()

    print("\n" + "█" * 65)
    print("  NLU ASSIGNMENT 2 — FULL PIPELINE")
    print("█" * 65)

    # ── Step 1: Collect data ──────────────────────────────────────────────
    corpus_path = run_collection(live=args.live)

    # ── Step 2: Preprocess & generate stats / word cloud ──────────────────
    clean_corpus_path = run_preprocessing(corpus_path)

    # ── Step 3: Train Word2Vec models (CBOW + Skip-gram grid) ─────────────
    run_training(clean_corpus_path)

    # ── Step 4: Semantic analysis (neighbours + analogies) ────────────────
    run_semantic_analysis()

    # ── Step 5: Visualisation (PCA + t-SNE plots) ─────────────────────────
    visualise_embeddings()

    elapsed = time.time() - t_start

    print("\n" + "█" * 65)
    print(f"  ALL DONE — completed in {elapsed:.1f} seconds")
    print("█" * 65)
    print("\n  Output files:")
    print("    data/clean/clean_corpus.txt    — tokenized corpus")
    print("    data/clean/corpus_stats.json   — document / token stats")
    print("    outputs/wordcloud.png          — word cloud image")
    print("    outputs/training_results.json  — hyper-parameter grid results")
    print("    outputs/training_results.csv   — same, in CSV")
    print("    outputs/semantic_analysis.json — neighbours + analogies")
    print("    outputs/viz_*.png              — PCA & t-SNE plots")
    print("    models/*.model                 — all trained Word2Vec models")
    print()


if __name__ == "__main__":
    main()
