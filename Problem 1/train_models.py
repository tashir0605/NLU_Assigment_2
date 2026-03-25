import json
import os
import pickle
import time
from collections import Counter
from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


CLEAN_CORPUS = os.path.join(".", "data", "clean", "clean_corpus.txt")
MODEL_DIR = os.path.join(".", "models")
RESULTS_DIR = os.path.join(".", "outputs")

EMBED_DIMS = [100, 200, 300]
WINDOW_SIZES = [3, 5, 7]
NEGATIVE_COUNTS = [5, 10, 15]

# Stage-1 (broad sweep)
BASE_EPOCHS = 20
BASE_SUBSAMPLE_T = 1e-3
BASE_START_LR = {0: 0.01, 1: 0.005}  # cbow, skipgram
END_LR = 0.0003

# Stage-2 (focused tuning on top candidates)
TOP_CANDIDATES_PER_ARCH = 3
TUNING_EPOCHS = [20, 40]
TUNING_SUBSAMPLE_T = [1e-3, 5e-4]
TUNING_START_LR = {
    0: [0.005, 0.01],
    1: [0.002, 0.005, 0.01],
}

GRAD_CLIP_NORM = 5.0
DEBUG_EPOCH_INTERVAL = 10

MIN_WORD_COUNT = 1
NUM_WORKERS = 4
SEED = 42

ARCH_LABELS = {0: "cbow", 1: "skipgram"}



class ScratchKeyedVectors:
    def __init__(self, vectors: np.ndarray, index_to_key: List[str]):
        self.vectors = vectors.astype(np.float32)
        self.index_to_key = list(index_to_key)
        self.key_to_index = {word: idx for idx, word in enumerate(self.index_to_key)}

    def __len__(self) -> int:
        return len(self.index_to_key)

    def __contains__(self, word: str) -> bool:
        return word in self.key_to_index

    def __getitem__(self, word: str) -> np.ndarray:
        return self.vectors[self.key_to_index[word]]

    def most_similar(
        self,
        positive,
        negative: Sequence[str] | None = None,
        topn: int = 10,
    ) -> List[Tuple[str, float]]:
        if isinstance(positive, str):
            positive = [positive]
        if negative is None:
            negative = []

        query = np.zeros(self.vectors.shape[1], dtype=np.float32)
        for word in positive:
            if word in self:
                query += self[word]
        for word in negative:
            if word in self:
                query -= self[word]

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm

        vec_norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        vec_norms[vec_norms == 0] = 1.0
        normalized = self.vectors / vec_norms

        sims = normalized @ query
        banned = set(positive) | set(negative)

        ranked = np.argsort(-sims)
        out: List[Tuple[str, float]] = []
        for idx in ranked:
            word = self.index_to_key[idx]
            if word in banned:
                continue
            out.append((word, float(sims[idx])))
            if len(out) >= topn:
                break
        return out


class ScratchWord2Vec:
    def __init__(
        self,
        wv: ScratchKeyedVectors,
        sg: int,
        vector_size: int,
        window: int,
        negative: int,
        min_count: int,
        epochs: int,
        start_lr: float,
        end_lr: float,
        subsample_t: float,
        loss_history: List[float] | None = None,
    ):
        self.wv = wv
        self.sg = sg
        self.vector_size = vector_size
        self.window = window
        self.negative = negative
        self.min_count = min_count
        self.epochs = epochs
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.subsample_t = subsample_t
        self.loss_history = loss_history or []

    def save(self, path: str) -> None:
        payload = {
            "vectors": self.wv.vectors,
            "index_to_key": self.wv.index_to_key,
            "sg": self.sg,
            "vector_size": self.vector_size,
            "window": self.window,
            "negative": self.negative,
            "min_count": self.min_count,
            "epochs": self.epochs,
            "start_lr": self.start_lr,
            "end_lr": self.end_lr,
            "subsample_t": self.subsample_t,
            "loss_history": self.loss_history,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "ScratchWord2Vec":
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        if not isinstance(payload, dict) or "vectors" not in payload:
            raise RuntimeError(
                "Model file is not in scratch Word2Vec format. "
                "Please retrain models by running run_all.py or train_models.py."
            )
        wv = ScratchKeyedVectors(payload["vectors"], payload["index_to_key"])
        return cls(
            wv=wv,
            sg=payload["sg"],
            vector_size=payload["vector_size"],
            window=payload["window"],
            negative=payload["negative"],
            min_count=payload["min_count"],
            epochs=payload["epochs"],
            start_lr=payload.get("start_lr", 0.02),
            end_lr=payload.get("end_lr", 0.0005),
            subsample_t=payload.get("subsample_t", 1e-3),
            loss_history=payload.get("loss_history", []),
        )



def _sigmoid_vec(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    return out


def _binary_cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # stable BCE: max(x, 0) - x*y + log1p(exp(-|x|))
    return np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))


def _read_sentences(corpus_path: str) -> List[List[str]]:
    sentences: List[List[str]] = []
    with open(corpus_path, "r", encoding="utf-8") as fh:
        for line in fh:
            toks = line.strip().split()
            if toks:
                sentences.append(toks)
    return sentences


def _build_vocab(
    sentences: List[List[str]],
    min_count: int,
) -> Tuple[List[str], np.ndarray, List[List[int]]]:
    counts = Counter(tok for sent in sentences for tok in sent)
    kept = [(w, c) for w, c in counts.items() if c >= min_count]
    kept.sort(key=lambda x: (-x[1], x[0]))

    index_to_key = [w for w, _ in kept]
    key_to_index = {w: i for i, w in enumerate(index_to_key)}
    freqs = np.array([c for _, c in kept], dtype=np.float64)

    corpus_ids: List[List[int]] = []
    for sent in sentences:
        ids = [key_to_index[w] for w in sent if w in key_to_index]
        if len(ids) >= 2:
            corpus_ids.append(ids)

    return index_to_key, freqs, corpus_ids


def _sample_negatives(
    rng: np.random.Generator,
    cumulative_probs: np.ndarray,
    k: int,
    forbidden: set[int],
) -> np.ndarray:
    negatives: List[int] = []
    used = set(forbidden)
    while len(negatives) < k:
        need = max(8, (k - len(negatives)) * 2)
        r = rng.random(need)
        idxs = np.searchsorted(cumulative_probs, r, side="left")
        idxs = np.clip(idxs, 0, len(cumulative_probs) - 1)
        for idx in idxs.tolist():
            if idx in used:
                continue
            negatives.append(idx)
            used.add(idx)
            if len(negatives) >= k:
                break
    return np.array(negatives, dtype=np.int64)


def _sgns_debug_epoch(
    epoch: int,
    epochs: int,
    logits_sum: float,
    logits_sq_sum: float,
    logits_count: int,
    logits_min: float,
    logits_max: float,
    W_in: np.ndarray,
    rng: np.random.Generator,
) -> None:
    if logits_count == 0:
        return

    logit_mean = logits_sum / logits_count
    logit_var = max(0.0, logits_sq_sum / logits_count - logit_mean**2)
    logit_std = float(np.sqrt(logit_var))

    norms = np.linalg.norm(W_in, axis=1)
    emb_var = float(np.var(W_in, axis=0).mean())

    pair_count = min(3000, W_in.shape[0] * 3)
    ids1 = rng.integers(0, W_in.shape[0], size=pair_count)
    ids2 = rng.integers(0, W_in.shape[0], size=pair_count)
    Wn = W_in / np.maximum(norms[:, None], 1e-8)
    cos = np.sum(Wn[ids1] * Wn[ids2], axis=1)

    print(
        f"    [sgns-debug e{epoch+1}/{epochs}] "
        f"norm(min/mean/max/std)=({norms.min():.4f}/{norms.mean():.4f}/{norms.max():.4f}/{norms.std():.4f}) "
        f"emb_var={emb_var:.6f} "
        f"logits(min/mean/max/std)=({logits_min:.4f}/{logit_mean:.4f}/{logits_max:.4f}/{logit_std:.4f}) "
        f"cos(min/mean/max/std)=({cos.min():.4f}/{cos.mean():.4f}/{cos.max():.4f}/{cos.std():.4f})"
    )

    if float(cos.mean()) > 0.95:
        print("      ⚠ collapse warning: mean cosine > 0.95")


def _subsample_sentence(
    sent: List[int],
    keep_probs: np.ndarray,
    rng: np.random.Generator,
) -> List[int]:
    sampled = [wid for wid in sent if rng.random() < keep_probs[wid]]
    return sampled if len(sampled) >= 2 else sent


def _train_skipgram_ns(
    corpus_ids: List[List[int]],
    W_in: np.ndarray,
    W_out: np.ndarray,
    window: int,
    negative: int,
    epochs: int,
    start_lr: float,
    end_lr: float,
    cumulative_probs: np.ndarray,
    keep_probs: np.ndarray,
    rng: np.random.Generator,
) -> List[float]:
    labels = np.concatenate(([1.0], np.zeros(negative, dtype=np.float32))).astype(
        np.float32
    )
    epoch_losses: List[float] = []

    total_pairs_est = 0
    for sent in corpus_ids:
        n = len(sent)
        if n < 2:
            continue
        total_pairs_est += n * (2 * window)
    total_pairs_est = max(1, total_pairs_est * epochs)
    processed_pairs = 0

    for epoch in range(epochs):
        order = rng.permutation(len(corpus_ids))
        total_loss = 0.0
        total_pairs = 0
        logits_sum = 0.0
        logits_sq_sum = 0.0
        logits_count = 0
        logits_min = float("inf")
        logits_max = float("-inf")

        for sent_idx in order:
            sent = _subsample_sentence(corpus_ids[sent_idx], keep_probs, rng)
            n = len(sent)
            for i, center in enumerate(sent):
                dynamic_window = int(rng.integers(1, window + 1))
                left = max(0, i - dynamic_window)
                right = min(n, i + dynamic_window + 1)

                for j in range(left, right):
                    if j == i:
                        continue
                    target = sent[j]

                    lr = start_lr - (start_lr - end_lr) * (
                        processed_pairs / total_pairs_est
                    )
                    lr = max(end_lr, lr)

                    v_in = W_in[center].copy()
                    negatives = _sample_negatives(
                        rng,
                        cumulative_probs,
                        negative,
                        forbidden={target, center},
                    )
                    targets = np.concatenate(
                        (np.array([target], dtype=np.int64), negatives)
                    )

                    v_out_old = W_out[targets].copy()
                    logits = v_out_old @ v_in
                    scores = _sigmoid_vec(logits)
                    errors = scores - labels

                    grad_out = np.outer(errors, v_in)
                    grad_in = errors @ v_out_old

                    grad_in_norm = float(np.linalg.norm(grad_in))
                    if grad_in_norm > GRAD_CLIP_NORM:
                        grad_in *= GRAD_CLIP_NORM / (grad_in_norm + 1e-12)

                    grad_out_norm = np.linalg.norm(grad_out, axis=1, keepdims=True)
                    grad_out_scale = np.minimum(
                        1.0, GRAD_CLIP_NORM / (grad_out_norm + 1e-12)
                    )
                    grad_out *= grad_out_scale

                    W_out[targets] -= lr * grad_out
                    W_in[center] -= lr * grad_in

                    batch_loss = float(np.sum(_binary_cross_entropy_from_logits(logits, labels)))
                    total_loss += float(batch_loss)
                    total_pairs += len(labels)
                    processed_pairs += 1

                    logits_sum += float(np.sum(logits))
                    logits_sq_sum += float(np.sum(logits * logits))
                    logits_count += int(logits.size)
                    logits_min = min(logits_min, float(np.min(logits)))
                    logits_max = max(logits_max, float(np.max(logits)))

        epoch_losses.append(total_loss / max(1, total_pairs))

        if (
            epoch == 0
            or (epoch + 1) % DEBUG_EPOCH_INTERVAL == 0
            or epoch == epochs - 1
        ):
            _sgns_debug_epoch(
                epoch=epoch,
                epochs=epochs,
                logits_sum=logits_sum,
                logits_sq_sum=logits_sq_sum,
                logits_count=logits_count,
                logits_min=logits_min,
                logits_max=logits_max,
                W_in=W_in,
                rng=rng,
            )

    return epoch_losses


def _train_cbow_ns(
    corpus_ids: List[List[int]],
    W_in: np.ndarray,
    W_out: np.ndarray,
    window: int,
    negative: int,
    epochs: int,
    start_lr: float,
    end_lr: float,
    cumulative_probs: np.ndarray,
    keep_probs: np.ndarray,
    rng: np.random.Generator,
) -> List[float]:
    labels = np.concatenate(([1.0], np.zeros(negative, dtype=np.float32))).astype(
        np.float32
    )
    eps = 1e-8
    epoch_losses: List[float] = []

    for epoch in range(epochs):
        lr = start_lr - (start_lr - end_lr) * (epoch / max(1, epochs - 1))
        order = rng.permutation(len(corpus_ids))
        total_loss = 0.0
        total_pairs = 0

        for sent_idx in order:
            sent = _subsample_sentence(corpus_ids[sent_idx], keep_probs, rng)
            n = len(sent)
            for i, target in enumerate(sent):
                dynamic_window = int(rng.integers(1, window + 1))
                left = max(0, i - dynamic_window)
                right = min(n, i + dynamic_window + 1)

                context_ids = [sent[j] for j in range(left, right) if j != i]
                if not context_ids:
                    continue

                hidden = np.mean(W_in[context_ids], axis=0)
                negatives = _sample_negatives(
                    rng, cumulative_probs, negative, forbidden={target}
                )
                targets = np.concatenate((np.array([target], dtype=np.int64), negatives))

                v_out_old = W_out[targets].copy()
                scores = _sigmoid_vec(v_out_old @ hidden)
                grads = (labels - scores) * lr

                W_out[targets] += grads[:, None] * hidden[None, :]
                context_update = (grads @ v_out_old) / len(context_ids)
                for c in context_ids:
                    W_in[c] += context_update

                batch_loss = -np.sum(
                    labels * np.log(scores + eps)
                    + (1.0 - labels) * np.log(1.0 - scores + eps)
                )
                total_loss += float(batch_loss)
                total_pairs += len(labels)

        epoch_losses.append(total_loss / max(1, total_pairs))

    return epoch_losses


def _train_from_scratch(
    sentences: List[List[str]],
    sg: int,
    vector_size: int,
    window: int,
    negative: int,
    min_count: int,
    epochs: int,
    start_lr: float,
    end_lr: float,
    subsample_t: float,
    seed: int,
) -> ScratchWord2Vec:
    index_to_key, freqs, corpus_ids = _build_vocab(sentences, min_count)
    vocab_size = len(index_to_key)

    if vocab_size == 0:
        raise RuntimeError("Vocabulary is empty after min_count filtering.")

    rng = np.random.default_rng(seed)
    W_in = rng.uniform(
        low=-0.5 / vector_size,
        high=0.5 / vector_size,
        size=(vocab_size, vector_size),
    ).astype(np.float32)
    W_out = np.zeros((vocab_size, vector_size), dtype=np.float32)

    # Negative-sampling distribution: unigram^(3/4)
    dist = np.power(freqs, 0.75)
    dist = dist / dist.sum()
    cumulative_probs = np.cumsum(dist)

    # Frequent-word subsampling keep probability
    rel_freqs = freqs / freqs.sum()
    keep_probs = (np.sqrt(subsample_t / rel_freqs) + 1.0) * (subsample_t / rel_freqs)
    keep_probs = np.clip(keep_probs, 0.05, 1.0).astype(np.float32)

    if sg == 1:
        loss_history = _train_skipgram_ns(
            corpus_ids=corpus_ids,
            W_in=W_in,
            W_out=W_out,
            window=window,
            negative=negative,
            epochs=epochs,
            start_lr=start_lr,
            end_lr=end_lr,
            cumulative_probs=cumulative_probs,
            keep_probs=keep_probs,
            rng=rng,
        )
    else:
        loss_history = _train_cbow_ns(
            corpus_ids=corpus_ids,
            W_in=W_in,
            W_out=W_out,
            window=window,
            negative=negative,
            epochs=epochs,
            start_lr=start_lr,
            end_lr=end_lr,
            cumulative_probs=cumulative_probs,
            keep_probs=keep_probs,
            rng=rng,
        )

    # Use combined embeddings for SGNS and center vectors for CBOW.
    # Combining input/output vectors improves stability and reduces anisotropy.
    if sg == 1:
        final_vectors = (W_in + W_out).astype(np.float32)
    else:
        final_vectors = W_in.astype(np.float32)

    final_vectors = final_vectors - np.mean(final_vectors, axis=0, keepdims=True)

    wv = ScratchKeyedVectors(final_vectors, index_to_key)
    return ScratchWord2Vec(
        wv=wv,
        sg=sg,
        vector_size=vector_size,
        window=window,
        negative=negative,
        min_count=min_count,
        epochs=epochs,
        start_lr=start_lr,
        end_lr=end_lr,
        subsample_t=subsample_t,
        loss_history=loss_history,
    )




PROBE_WORDS = [
    "research", "student", "professor", "engineering", "examination",
    "semester", "department", "technology", "jodhpur",
]

EXPECTED_NEIGHBOURS = {
    "research": {"phd", "thesis", "publication", "student", "professor", "innovation"},
    "student": {"professor", "course", "semester", "exam", "department", "phd"},
    "professor": {"faculty", "student", "department", "research", "course", "phd"},
    "engineering": {"technology", "science", "civil", "electrical", "mechanical", "chemical"},
    "examination": {"exam", "semester", "grade", "evaluation", "quiz", "course"},
    "semester": {"course", "exam", "grade", "student", "year", "attendance"},
    "department": {"engineering", "science", "faculty", "student", "research"},
    "technology": {"engineering", "innovation", "research", "science"},
    "jodhpur": {"iit", "institute", "student", "research", "campus"},
}

ANALOGY_TESTS = [
    ("ug", "btech", "pg", {"mtech", "msc", "postgraduate"}),
    ("student", "examination", "professor", {"evaluation", "assessment", "course", "faculty"}),
    ("phd", "research", "btech", {"engineering", "course", "technology", "mtech"}),
]


def _semantic_quality_score(model: ScratchWord2Vec) -> Dict[str, float]:
    wv = model.wv

    nn_scores: List[float] = []
    nn_top1_sims: List[float] = []
    for word in PROBE_WORDS:
        if word not in wv:
            continue
        predicted_with_scores = wv.most_similar(word, topn=5)
        predicted = [w for w, _ in predicted_with_scores]
        expected = EXPECTED_NEIGHBOURS.get(word, set())
        hits = sum(1 for w in predicted if w in expected)
        nn_scores.append(hits / 5.0)
        if predicted_with_scores:
            nn_top1_sims.append(float(predicted_with_scores[0][1]))

    analogy_scores: List[float] = []
    for a, b, c, expected in ANALOGY_TESTS:
        if any(w not in wv for w in (a, b, c)):
            continue
        predicted = [
            w for w, _ in wv.most_similar(positive=[b, c], negative=[a], topn=3)
        ]
        hit = 1.0 if any(w in expected for w in predicted) else 0.0
        analogy_scores.append(hit)

    nn_score = float(np.mean(nn_scores)) if nn_scores else 0.0
    analogy_score = float(np.mean(analogy_scores)) if analogy_scores else 0.0

    sat_ratio = (
        float(np.mean([s > 0.95 for s in nn_top1_sims])) if nn_top1_sims else 1.0
    )
    mean_top1 = float(np.mean(nn_top1_sims)) if nn_top1_sims else 1.0
    sim_std = float(np.std(nn_top1_sims)) if nn_top1_sims else 0.0

    saturation_penalty = max(0.0, sat_ratio - 0.20) * 0.8 + max(0.0, mean_top1 - 0.75) * 0.8

    total = max(0.0, 0.65 * nn_score + 0.35 * analogy_score - saturation_penalty)

    return {
        "semantic_score": round(total, 4),
        "nn_score": round(nn_score, 4),
        "analogy_score": round(analogy_score, 4),
        "sat_ratio": round(sat_ratio, 4),
        "mean_top1_sim": round(mean_top1, 4),
        "top1_sim_std": round(sim_std, 4),
        "saturation_penalty": round(float(saturation_penalty), 4),
    }



def _model_tag(
    arch: int,
    dim: int,
    win: int,
    neg: int,
    start_lr: float,
    epochs: int,
    subsample_t: float,
) -> str:
    lr_str = str(start_lr).replace(".", "p")
    ss_str = f"{subsample_t:.0e}".replace("-", "m").replace("+", "p")
    return f"{ARCH_LABELS[arch]}_d{dim}_w{win}_n{neg}_lr{lr_str}_ep{epochs}_ss{ss_str}"


def train_single_model(
    corpus_path: str,
    sg: int,
    vector_size: int,
    window: int,
    negative: int,
    start_lr: float,
    epochs: int,
    subsample_t: float,
) -> Tuple[ScratchWord2Vec, float]:
    tag = _model_tag(sg, vector_size, window, negative, start_lr, epochs, subsample_t)
    print(f"  Training {tag} …", end=" ", flush=True)

    sentences = _read_sentences(corpus_path)

    _ = NUM_WORKERS
    t0 = time.time()
    model = _train_from_scratch(
        sentences=sentences,
        sg=sg,
        vector_size=vector_size,
        window=window,
        negative=negative,
        min_count=MIN_WORD_COUNT,
        epochs=epochs,
        start_lr=start_lr,
        end_lr=END_LR,
        subsample_t=subsample_t,
        seed=SEED,
    )
    elapsed = time.time() - t0

    loss_start = model.loss_history[0] if model.loss_history else 0.0
    loss_end = model.loss_history[-1] if model.loss_history else 0.0
    trend = "↓" if loss_end < loss_start else "↑"
    print(
        f"done ({elapsed:.1f}s, vocab={len(model.wv):,}, loss {loss_start:.4f}->{loss_end:.4f} {trend})"
    )
    return model, elapsed


def _fit_and_record(
    results: List[dict],
    corpus_path: str,
    sg: int,
    dim: int,
    win: int,
    neg: int,
    start_lr: float,
    epochs: int,
    subsample_t: float,
    phase: str,
) -> dict:
    model, elapsed = train_single_model(
        corpus_path=corpus_path,
        sg=sg,
        vector_size=dim,
        window=win,
        negative=neg,
        start_lr=start_lr,
        epochs=epochs,
        subsample_t=subsample_t,
    )

    tag = _model_tag(sg, dim, win, neg, start_lr, epochs, subsample_t)
    model_path = os.path.join(MODEL_DIR, f"{tag}.model")
    model.save(model_path)

    sem_scores = _semantic_quality_score(model)
    loss_start = model.loss_history[0] if model.loss_history else 0.0
    loss_end = model.loss_history[-1] if model.loss_history else 0.0

    row = {
        "phase": phase,
        "architecture": ARCH_LABELS[sg],
        "dim": dim,
        "window": win,
        "negative": neg,
        "start_lr": start_lr,
        "epochs": epochs,
        "subsample_t": subsample_t,
        "vocab_size": len(model.wv),
        "train_time_s": round(elapsed, 2),
        "loss_start": round(loss_start, 6),
        "loss_end": round(loss_end, 6),
        "loss_drop": round(loss_start - loss_end, 6),
        **sem_scores,
        "model_tag": tag,
        "model_path": model_path,
    }
    results.append(row)
    return row


def run_hyperparameter_grid(corpus_path: str = CLEAN_CORPUS) -> List[dict]:
    os.makedirs(MODEL_DIR, exist_ok=True)
    results: List[dict] = []

    base_configs = list(product([0, 1], EMBED_DIMS, WINDOW_SIZES, NEGATIVE_COUNTS))
    print(f"\n{'='*78}")
    print("  TASK 2 — Stage 1: Broad sweep (dim/window/negative)")
    print(f"{'='*78}\n")

    best_by_arch: Dict[str, dict] = {}

    total = len(base_configs)
    for idx, (sg, dim, win, neg) in enumerate(base_configs, 1):
        print(f"[base {idx}/{total}]", end=" ")
        row = _fit_and_record(
            results=results,
            corpus_path=corpus_path,
            sg=sg,
            dim=dim,
            win=win,
            neg=neg,
            start_lr=BASE_START_LR[sg],
            epochs=BASE_EPOCHS,
            subsample_t=BASE_SUBSAMPLE_T,
            phase="base",
        )

        arch = row["architecture"]
        prev = best_by_arch.get(arch)
        if prev is None or row["semantic_score"] > prev["semantic_score"]:
            best_by_arch[arch] = row
            print(
                f"       ↳ new best {arch}: score={row['semantic_score']:.4f}, "
                f"nn={row['nn_score']:.4f}, analogy={row['analogy_score']:.4f}"
            )

    print(f"\n{'='*78}")
    print("  TASK 2 — Stage 2: Focused tuning (lr/epochs/subsampling)")
    print(f"{'='*78}\n")

    for arch_name in ("cbow", "skipgram"):
        arch_rows = [r for r in results if r["architecture"] == arch_name and r["phase"] == "base"]
        arch_rows = sorted(arch_rows, key=lambda r: r["semantic_score"], reverse=True)
        top_rows = arch_rows[:TOP_CANDIDATES_PER_ARCH]

        sg = 0 if arch_name == "cbow" else 1
        tune_space = list(
            product(
                TUNING_START_LR[sg],
                TUNING_EPOCHS,
                TUNING_SUBSAMPLE_T,
            )
        )

        for base_row in top_rows:
            for start_lr, epochs, subsample_t in tune_space:
                if (
                    start_lr == base_row["start_lr"]
                    and epochs == base_row["epochs"]
                    and subsample_t == base_row["subsample_t"]
                ):
                    continue

                print(
                    f"[tune {arch_name}] d={base_row['dim']} w={base_row['window']} "
                    f"n={base_row['negative']} lr={start_lr} ep={epochs} ss={subsample_t}"
                )

                row = _fit_and_record(
                    results=results,
                    corpus_path=corpus_path,
                    sg=sg,
                    dim=base_row["dim"],
                    win=base_row["window"],
                    neg=base_row["negative"],
                    start_lr=start_lr,
                    epochs=epochs,
                    subsample_t=subsample_t,
                    phase="tuned",
                )

                prev = best_by_arch.get(arch_name)
                if prev is None or row["semantic_score"] > prev["semantic_score"]:
                    best_by_arch[arch_name] = row
                    print(
                        f"       ↳ improved {arch_name}: score {prev['semantic_score'] if prev else 0:.4f} "
                        f"-> {row['semantic_score']:.4f} with "
                        f"(d={row['dim']}, w={row['window']}, n={row['negative']}, "
                        f"lr={row['start_lr']}, ep={row['epochs']}, ss={row['subsample_t']})"
                    )

    print("\nBest configuration by architecture:")
    for arch in ("cbow", "skipgram"):
        row = best_by_arch[arch]
        print(
            f"  - {arch:<8s} tag={row['model_tag']} score={row['semantic_score']:.4f} "
            f"(nn={row['nn_score']:.4f}, analogy={row['analogy_score']:.4f})"
        )

    return results


def save_training_results(results: List[dict]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    json_path = os.path.join(RESULTS_DIR, "training_results.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    df = pd.DataFrame(results)
    df = df.sort_values(by=["architecture", "semantic_score", "nn_score", "analogy_score"], ascending=[True, False, False, False])
    display_cols = [c for c in df.columns if c not in {"model_path"}]

    print(f"\n{'='*90}")
    print("  TRAINING RESULTS SUMMARY (sorted by semantic_score)")
    print(f"{'='*90}")
    print(df[display_cols].head(20).to_string(index=False))
    print(f"{'='*90}")
    print(f"  Results saved to {json_path}\n")

    csv_path = os.path.join(RESULTS_DIR, "training_results.csv")
    df[display_cols].to_csv(csv_path, index=False)
    print(f"  CSV table saved to {csv_path}")

    return json_path


def load_model(tag: str) -> ScratchWord2Vec:
    path = os.path.join(MODEL_DIR, f"{tag}.model")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}")
    return ScratchWord2Vec.load(path)


def get_best_models() -> Dict[str, str]:
    results_path = os.path.join(RESULTS_DIR, "training_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as fh:
            rows = json.load(fh)
        if isinstance(rows, list) and rows:
            out: Dict[str, str] = {}
            for arch in ("cbow", "skipgram"):
                arch_rows = [r for r in rows if r.get("architecture") == arch]
                if arch_rows:
                    best = max(arch_rows, key=lambda r: r.get("semantic_score", 0.0))
                    out[arch] = best["model_tag"]
            if "cbow" in out and "skipgram" in out:
                return out

    return {
        "cbow": _model_tag(0, 100, 5, 10, BASE_START_LR[0], BASE_EPOCHS, BASE_SUBSAMPLE_T),
        "skipgram": _model_tag(1, 100, 5, 10, BASE_START_LR[1], BASE_EPOCHS, BASE_SUBSAMPLE_T),
    }


def run_training(corpus_path: str = CLEAN_CORPUS) -> List[dict]:
    results = run_hyperparameter_grid(corpus_path)
    save_training_results(results)
    return results


if __name__ == "__main__":
    run_training()
