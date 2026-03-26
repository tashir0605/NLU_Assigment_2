# CSL 7640: Natural Language Understanding — Assignment 2

**Name:** Shivam Goyal  
**Roll No:** B23CM1036

This repository contains two end-to-end NLP implementations built from scratch:

1. **Problem 1:** Word2Vec on IIT Jodhpur corpus (`CBOW` + `Skip-gram with Negative Sampling`)  
2. **Problem 2:** Character-level Indian name generation (`Vanilla RNN`, `BLSTM`, `Attention-RNN`)

---

## Repository Structure

```text
NLP/
├── Problem 1/
│   ├── collect_data.py
│   ├── preprocess.py
│   ├── train_models.py
│   ├── semantic_analysis.py
│   ├── visualize.py
│   ├── run_all.py
│   ├── data/
│   │   ├── raw_corpus/
│   │   └── clean/
│   ├── models/
│   └── outputs/
│       ├── training_results.csv
│       ├── training_results.json
│       ├── semantic_analysis.json
│       ├── report_tables.json
│       ├── wordcloud.png
│       └── viz_*.png
├── Problem 2/
│   ├── main.py
│   ├── dataset/
│   │   └── TrainingNames.txt
│   ├── models/
│   │   ├── vanilla_rnn.py
│   │   ├── blstm.py
│   │   └── attention_rnn.py
│   ├── training/
│   │   └── train.py
│   ├── utils/
│   │   ├── dataset.py
│   │   └── vocab.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── outputs/
│       ├── metrics_summary.json
│       ├── generation_metrics.json
│       ├── generated_*.txt
│       └── training_loss_curves.png
└── Report.pdf
```

---

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas matplotlib scikit-learn wordcloud nltk requests beautifulsoup4 lxml pdfplumber pypdf PyPDF2
```

> `preprocess.py` already downloads required NLTK packages (`punkt`, `punkt_tab`, `stopwords`, `wordnet`, `omw-1.4`).

---

## Problem 1 — Word2Vec Pipeline

### Run full pipeline (recommended)

```bash
cd "Problem 1"
python run_all.py --live
```

This executes:
1. Live IITJ scraping (`collect_data.py`)
2. Cleaning/tokenization/stats/word cloud (`preprocess.py`)
3. CBOW + Skip-gram sweeps and tuning (`train_models.py`)
4. Nearest-neighbor + analogy evaluation (`semantic_analysis.py`)
5. PCA/t-SNE visualization (`visualize.py`)

### Main outputs
- `Problem 1/data/clean/corpus_stats.json`
- `Problem 1/outputs/training_results.csv`
- `Problem 1/outputs/semantic_analysis.json`
- `Problem 1/outputs/wordcloud.png`
- `Problem 1/outputs/viz_cbow_pca.png`, `viz_cbow_tsne.png`, `viz_skipgram_pca.png`, `viz_skipgram_tsne.png`

### Latest run snapshot
- Documents: **63**
- Tokens: **25,363**
- Vocabulary: **4,163**
- Best CBOW (semantic score): `cbow_d300_w3_n5_lr0p01_ep20_ss1em03` (**0.1577**)
- Best Skip-gram (semantic score): `skipgram_d100_w3_n5_lr0p002_ep20_ss5em04` (**0.0372**)

Top-10 frequent words: `student`, `iitj`, `institute`, `jodhpur`, `school`, `technology`, `professor`, `course`, `iit`, `academic`.

---

## Problem 2 — Character-level Name Generation

### Train all models

```bash
cd "Problem 2"
python main.py --data dataset/TrainingNames.txt
```

Models trained:
- `vanilla_rnn`
- `blstm`
- `attention_rnn`

### Main outputs
- `Problem 2/outputs/metrics_summary.json`
- `Problem 2/outputs/training_loss_curves.png`
- `Problem 2/outputs/generated_vanilla_rnn.txt`
- `Problem 2/outputs/generated_blstm.txt`
- `Problem 2/outputs/generated_attention_rnn.txt`
- `Problem 2/outputs/generation_metrics.json`

### Latest run snapshot
From `metrics_summary.json`:
- **Vanilla RNN:** params `9021`, best val loss `1.9199`
- **BLSTM:** params `8743`, best val loss `0.2188`
- **Attention-RNN:** params `9149`, best val loss `1.9790`

From `generation_metrics.json` (200 names/model):
- **Vanilla RNN:** novelty `0.925`, diversity `0.990`
- **BLSTM:** novelty `0.995`, diversity `0.935`
- **Attention-RNN:** novelty `0.955`, diversity `0.990`

---

## Key Observations

- In Problem 1, many configurations that reduced loss still showed embedding saturation, so semantic quality required geometry-aware model selection.
- In Problem 2, BLSTM achieved the lowest validation loss but produced more repetitive generations; attention/vanilla gave better diversity-quality balance.

---

## Report

Final report files:
- `Report.pdf`
