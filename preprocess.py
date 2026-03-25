"""
================================================================================
NLU Assignment 2 — Task 1: Text Preprocessing & Corpus Statistics
================================================================================
This module takes the raw collected text and produces a **clean tokenized
corpus** suitable for Word2Vec training.

Preprocessing pipeline (as required by the rubric)
───────────────────────────────────────────────────
1. Remove boilerplate / formatting artifacts (HTML entities, URLs, emails)
2. Remove non-English text (Hindi / Devanagari characters, CJK, etc.)
3. Tokenization  (NLTK word_tokenize for proper handling of contractions)
4. Lower-casing
5. Remove excessive punctuation & non-textual tokens (numbers-only, symbols)
6. Remove stopwords (optional flag — disabled by default so embeddings
   can learn function-word context, but can be toggled for the report)

It also computes corpus statistics:
 • total documents, total tokens, vocabulary size
and generates a **Word Cloud** of the most frequent terms.

Author  : <your‑name‑here>
Course  : Natural Language Understanding — Assignment 2
================================================================================
"""

# ─── stdlib ───────────────────────────────────────────────────────────────────
import os
import re
import json
from collections import Counter
from typing import List, Tuple

# ─── third‑party ──────────────────────────────────────────────────────────────
import nltk
from wordcloud import WordCloud
import matplotlib
matplotlib.use("Agg")                       # non-interactive back-end (no GUI)
import matplotlib.pyplot as plt

# Make sure required NLTK data packages are available
for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Shared lemmatiser instance
_lemmatizer = WordNetLemmatizer()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Where to read raw data and write processed output
RAW_CORPUS_PATH  = os.path.join(".", "data", "raw_corpus", "full_corpus.txt")
CLEAN_DIR        = os.path.join(".", "data", "clean")
CLEAN_CORPUS     = os.path.join(CLEAN_DIR, "clean_corpus.txt")
STATS_FILE       = os.path.join(CLEAN_DIR, "corpus_stats.json")
WORDCLOUD_FILE   = os.path.join(".", "outputs", "wordcloud.png")

# Tokens shorter than this are discarded (1-2 char fragments like 'dr', 'st',
# 'bb', 'ee' add noise — raising to 3 cleans them out)
MIN_TOKEN_LENGTH = 3

# English stopwords from NLTK — we'll optionally remove these
STOP_WORDS = set(stopwords.words("english"))

# PDF internal-markup tokens that leak through text extraction and must be
# blacklisted.  These are PDF operators / structure keywords, not real words.
PDF_ARTIFACT_TOKENS = {
    "obj", "endobj", "stream", "endstream", "xref", "startxref",
    "trailer", "eof", "pdf", "oo", "ooo", "rn", "tf", "td", "tm",
    "bdc", "emc", "bt", "et", "bm", "cm", "re", "sc", "cs", "gs",
    "rg", "rj", "tj", "font", "fl", "helvetica", "arial",
}

# Web-scrape boilerplate tokens that appear very frequently but carry no
# semantic meaning.  These come from email addresses split into tokens
# ("name dot surname at iitj dot ac dot in"), navigation elements,
# footer/header chrome, etc.
WEB_BOILERPLATE_TOKENS = {
    # email-address fragments
    "dot", "ac", "call", "email", "mailto", "fax", "tel", "phone",
    # navigation / page-furniture
    "click", "here", "read", "more", "home", "menu", "search",
    "login", "logout", "submit", "download", "upload", "view",
    "back", "next", "previous", "page", "skip", "close",
    # website boilerplate
    "portal", "links", "web", "intranet", "website", "site",
    "copyright", "reserved", "rights", "disclaimer", "privacy",
    "cookie", "cookies", "policy", "terms", "conditions",
    # common non-content fragments
    "nbsp", "amp", "quot", "lt", "gt", "img", "src", "href",
    "jpg", "png", "gif", "svg", "css", "js", "html", "http",
    "https", "www", "com", "org", "edu", "gov",
    # misc noise frequent in IIT Jodhpur scrapes
    "en", "lg", "id", "px", "sm", "md",
}


# ══════════════════════════════════════════════════════════════════════════════
# CLEANING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def strip_artifacts(text: str) -> str:
    """
    Remove non-prose artefacts that commonly appear in scraped web text.

    Targets
    ───────
    • URLs        (http://… or www.…)
    • Email addrs (user@domain)
    • HTML entities (&amp; &nbsp; &#123; etc.)
    • Residual HTML tags that slipped through BS4
    • Unicode dashes / bullets / special chars → space
    """
    # URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Emails
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # HTML entities
    text = re.sub(r"&[a-z]+;|&#\d+;", " ", text)
    # Any leftover HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # PDF stream operators (e.g. "0 0 0 rg", "12.5 Tf", "/F1 10 Tf")
    text = re.sub(r"/[A-Z][A-Za-z0-9]*\b", " ", text)
    text = re.sub(r"\b\d+\s+\d+\s+\d+\s+\d+\s+re\b", " ", text)
    text = re.sub(r"\b\d[\d.]*\s+(Tf|Td|Tm|cm|rg|RG|scn|SCN)\b", " ", text)
    # Non-ASCII dashes, bullets, fancy quotes → space
    text = re.sub(r"[–—•·''""…]", " ", text)
    return text


def remove_non_english(text: str) -> str:
    """
    Drop characters outside the Basic Latin + Latin-1 Supplement blocks.

    This catches Devanagari, CJK, Arabic, etc. — anything that isn't
    standard English letters, digits, or common punctuation.
    """
    # Keep only ASCII printable + a few safe Latin-1 chars (é, ñ, etc.)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text


def tokenize_and_clean(text: str, remove_stops: bool = False) -> List[str]:
    """
    Full cleaning pipeline for a single block of text.

    Steps (in order)
    ─────────────────
    1. Strip artefacts (URLs, emails, HTML residue)
    2. Remove non-English characters
    3. Lowercase
    4. Tokenize with NLTK (handles "don't" → ["do", "n't"] etc.)
    5. Keep only alphabetic tokens of length ≥ MIN_TOKEN_LENGTH
    6. Lemmatize (WordNet) — merges plurals & inflections
    7. Optionally remove stopwords

    Returns a list of clean tokens.
    """
    text = strip_artifacts(text)
    text = remove_non_english(text)
    text = text.lower()

    tokens = word_tokenize(text)

    # Filter: keep only pure-alpha tokens that are long enough
    clean_tokens = [
        tok for tok in tokens
        if tok.isalpha()
        and len(tok) >= MIN_TOKEN_LENGTH
        and tok not in PDF_ARTIFACT_TOKENS
        and tok not in WEB_BOILERPLATE_TOKENS
    ]

    # Lemmatize — reduces inflected forms to base form
    # e.g. students→student, examinations→examination, programmes→programme
    clean_tokens = [_lemmatizer.lemmatize(tok) for tok in clean_tokens]

    # Optionally strip stopwords
    if remove_stops:
        clean_tokens = [t for t in clean_tokens if t not in STOP_WORDS]

    return clean_tokens


# ══════════════════════════════════════════════════════════════════════════════
# CORPUS PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_documents(corpus_path: str) -> List[str]:
    """
    Read the combined corpus file and split it into individual documents.

    Documents in the combined file are separated by double newlines.
    We filter out empty chunks that result from trailing whitespace.
    """
    with open(corpus_path, "r", encoding="utf-8") as fh:
        raw = fh.read()

    # Split on blank lines (each doc was written with \n\n separator)
    docs = [d.strip() for d in raw.split("\n\n") if d.strip()]
    print(f"Loaded {len(docs)} raw documents from {corpus_path}")
    return docs


def process_corpus(
    corpus_path: str = RAW_CORPUS_PATH,
    remove_stops: bool = False,
) -> Tuple[List[List[str]], Counter]:
    """
    Run the full preprocessing pipeline over the raw corpus.

    Parameters
    ──────────
    corpus_path  : path to the combined raw text file
    remove_stops : whether to remove English stopwords

    Returns
    ───────
    tokenized_docs : list of documents, each a list of tokens
    vocab_counts   : Counter mapping each token to its total frequency
    """
    docs = load_raw_documents(corpus_path)
    tokenized_docs: List[List[str]] = []
    vocab_counts: Counter = Counter()

    for doc in docs:
        tokens = tokenize_and_clean(doc, remove_stops=remove_stops)
        if tokens:                         # skip docs that became empty
            tokenized_docs.append(tokens)
            vocab_counts.update(tokens)

    print(f"After cleaning: {len(tokenized_docs)} non-empty documents")
    return tokenized_docs, vocab_counts


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS & WORD CLOUD
# ══════════════════════════════════════════════════════════════════════════════

def compute_and_save_stats(
    tokenized_docs: List[List[str]],
    vocab_counts: Counter,
) -> dict:

    total_tokens = sum(len(d) for d in tokenized_docs)
    vocab_size   = len(vocab_counts)
    top_30       = vocab_counts.most_common(30)

    stats = {
        "num_documents": len(tokenized_docs),
        "total_tokens":  total_tokens,
        "vocabulary_size": vocab_size,
        "top_30_tokens": {w: c for w, c in top_30},
    }

    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    with open(STATS_FILE, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    # Print a nice summary to the console
    print("\n" + "=" * 55)
    print("             CORPUS STATISTICS")
    print("=" * 55)
    print(f"  Documents       : {stats['num_documents']}")
    print(f"  Total tokens    : {stats['total_tokens']:,}")
    print(f"  Vocabulary size : {stats['vocabulary_size']:,}")
    print("-" * 55)
    print("  Top 15 tokens:")
    for rank, (word, freq) in enumerate(top_30[:15], 1):
        print(f"    {rank:2d}. {word:<20s} {freq:>5d}")
    print("=" * 55)
    print(f"  Full stats saved to {STATS_FILE}")

    return stats


def generate_wordcloud(vocab_counts: Counter) -> str:

    os.makedirs(os.path.dirname(WORDCLOUD_FILE), exist_ok=True)

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        colormap="viridis",            # perceptually uniform colour map
        max_words=150,
        min_font_size=8,
        prefer_horizontal=0.7,
    )
    wc.generate_from_frequencies(vocab_counts)

    plt.figure(figsize=(14, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud — IIT Jodhpur Corpus", fontsize=16, pad=12)
    plt.tight_layout()
    plt.savefig(WORDCLOUD_FILE, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Word cloud saved to {WORDCLOUD_FILE}")
    return WORDCLOUD_FILE



# SAVE CLEAN CORPUS (for Word2Vec training)


def save_clean_corpus(tokenized_docs: List[List[str]]) -> str:
    """
    Write the tokenized documents to a text file — one sentence (document)
    per line, tokens space-separated.

    This format is consumed by the custom Word2Vec trainer in Task 2.
    """
    os.makedirs(CLEAN_DIR, exist_ok=True)

    with open(CLEAN_CORPUS, "w", encoding="utf-8") as fh:
        for doc_tokens in tokenized_docs:
            fh.write(" ".join(doc_tokens) + "\n")

    print(f"Clean corpus ({len(tokenized_docs)} lines) → {CLEAN_CORPUS}")
    return CLEAN_CORPUS



def run_preprocessing(corpus_path: str = RAW_CORPUS_PATH,
                      remove_stops: bool = True) -> str:
    """
    Execute the entire Task 1 pipeline:
      raw text → clean tokens → stats → word cloud → saved corpus

    Returns the path to the clean corpus file.
    """
    print("\n" + "─" * 55)
    print("  TASK 1 — Preprocessing & Corpus Statistics")
    print("─" * 55)

    # Step 1: tokenize & clean
    tokenized_docs, vocab_counts = process_corpus(corpus_path, remove_stops)

    # Step 2: compute & display statistics
    compute_and_save_stats(tokenized_docs, vocab_counts)

    # Step 3: generate word cloud visualisation
    generate_wordcloud(vocab_counts)

    # Step 4: persist the clean corpus for model training
    clean_path = save_clean_corpus(tokenized_docs)

    return clean_path


# ── standalone execution ─────────────────────────────────────────────────────
if __name__ == "__main__":
    run_preprocessing()
