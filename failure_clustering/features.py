"""Text → vector features for clustering, dependency-free (numpy only).

Used as the clustering geometry when a backend has no native embedding model
(the mock and cursor-agent backends). Real embedding backends override
``embed`` directly.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")

# Generic English + audit boilerplate that carries no mechanism signal. Kept
# small and domain-neutral on purpose.
_STOP = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "is", "was", "were", "are", "be", "been", "it", "its", "that", "this", "as",
    "at", "by", "from", "not", "no", "so", "than", "then", "into", "onto", "out",
    "agent", "task", "verifier", "verdict", "rollout", "correct", "correctly",
    "produced", "submitted", "required", "instead", "despite", "because",
    "which", "while", "their", "they", "them", "all", "any", "one", "two",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


def tfidf_embed(texts: list[str], *, max_features: int = 4096) -> np.ndarray:
    """L2-normalized tf-idf vectors. Deterministic; rows align with ``texts``."""
    docs = [tokenize(t) for t in texts]
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    # cap vocabulary to the most frequent tokens for stability/speed
    vocab = {tok: i for i, (tok, _) in enumerate(df.most_common(max_features))}
    n = len(texts)
    idf = np.zeros(len(vocab))
    for tok, i in vocab.items():
        idf[i] = math.log((1 + n) / (1 + df[tok])) + 1.0
    X = np.zeros((n, len(vocab)), dtype=np.float64)
    for r, d in enumerate(docs):
        tf = Counter(d)
        for tok, c in tf.items():
            j = vocab.get(tok)
            if j is not None:
                X[r, j] = c * idf[j]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def first_clause(text: str, *, max_words: int = 22) -> str:
    """Cheap mechanism proxy: the first clause, truncated. Used by the mock
    backend and as a fallback when no LLM mechanism extraction is available."""
    head = re.split(r"[,.;:]| - | but | so | because ", text.strip(), maxsplit=1)[0]
    words = head.split()
    return " ".join(words[:max_words])


def cosine_dist(X: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance for L2-normalized rows (clipped to [0, 2])."""
    S = X @ X.T
    np.clip(S, -1.0, 1.0, out=S)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    return D
