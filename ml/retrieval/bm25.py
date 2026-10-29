"""Sparse-matrix BM25 (Okapi), numerically identical to ``rank_bm25.BM25Okapi``.

``rank_bm25`` is the reference implementation (a test asserts equal scores). This version
separates *fitting* (vocabulary, IDF, average document length) from *indexing*, so IDF can be fit
only on questions available before an evaluation cutoff, and it scores all documents with one
sparse matrix-vector product.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.+#_-]*[a-z0-9+#]|[a-z0-9]")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in ENGLISH_STOP_WORDS]


def question_text(title: str, body: str) -> str:
    """Title is repeated once to up-weight it; code blocks are excluded (logs are noisy)."""
    return f"{title}\n{title}\n{body}"


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.avgdl: float = 0.0
        self.matrix: sparse.csr_matrix | None = None

    def fit(self, docs: Iterable[list[str]]) -> BM25:
        df: dict[str, int] = {}
        n_docs, total_len = 0, 0
        for toks in docs:
            n_docs += 1
            total_len += len(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        dfa = np.array([df[t] for t in sorted(df)], dtype=float)
        idf = np.log(n_docs - dfa + 0.5) - np.log(dfa + 0.5)
        # rank_bm25 replaces negative idf with epsilon * mean idf
        idf = np.where(idf < 0, self.epsilon * idf.mean(), idf)
        self.idf = idf
        self.avgdl = total_len / max(n_docs, 1)
        return self

    def _tf(self, docs: list[list[str]]) -> tuple[sparse.csr_matrix, np.ndarray]:
        rows, cols, vals, lengths = [], [], [], []
        for i, toks in enumerate(docs):
            lengths.append(len(toks))
            counts: dict[int, int] = {}
            for t in toks:
                j = self.vocab.get(t)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            rows.extend([i] * len(counts))
            cols.extend(counts.keys())
            vals.extend(counts.values())
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)), dtype=float)
        return tf, np.asarray(lengths, dtype=float)

    def index(self, docs: list[list[str]]) -> BM25:
        """Precompute per-document term weights idf * tf(k1+1) / (tf + k1(1-b+b dl/avgdl))."""
        tf, dl = self._tf(docs)
        norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
        tf = tf.tocoo()
        w = self.idf[tf.col] * tf.data * (self.k1 + 1) / (tf.data + norm[tf.row])
        self.matrix = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        return self

    def query_vector(self, tokens: list[str]) -> np.ndarray:
        q = np.zeros(len(self.vocab))
        np.add.at(q, [self.vocab[t] for t in tokens if t in self.vocab], 1.0)
        return q

    def scores(self, tokens: list[str]) -> np.ndarray:
        """BM25 score of every indexed document for the query tokens (repeats count, as in rank_bm25)."""
        return self.matrix @ self.query_vector(tokens)

    def term_contributions(self, tokens: list[str], doc_index: int, top: int = 5) -> list[tuple[str, float]]:
        """Which query terms contributed most to one document's score (for explanations)."""
        inv = {i: t for t, i in self.vocab.items()}
        row = self.matrix.getrow(doc_index)
        q = {}
        for t in tokens:
            if t in self.vocab:
                q[self.vocab[t]] = q.get(self.vocab[t], 0) + 1
        contrib = [(inv[j], float(w * q[j])) for j, w in zip(row.indices, row.data, strict=True) if j in q]
        return sorted(contrib, key=lambda kv: -kv[1])[:top]
