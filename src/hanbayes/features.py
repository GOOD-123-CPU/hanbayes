"""Character N-gram vectorizer producing a single sparse count matrix.

The vectorizer is the **only** place where raw text is scanned. All models
downstream consume the returned count matrix, so training the full model
family costs exactly one tokenization pass over the corpus.

Vocabulary is built from training texts only, ordered deterministically by
``(document frequency desc, total frequency desc, lexicographic asc)``.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse

from .text import COMMON_PUNCTUATION


def extract_char_ngrams(text: str, ngram_range: tuple[int, int] = (1, 2)) -> list[str]:
    """Character N-grams of *text*; pure-punctuation grams are skipped."""

    if not isinstance(text, str):
        text = str(text)

    min_n, max_n = ngram_range
    if min_n < 1:
        raise ValueError("min_n必须大于或等于1")
    if max_n < min_n:
        raise ValueError("max_n不能小于min_n")

    features: list[str] = []
    length = len(text)
    for n in range(min_n, max_n + 1):
        if length < n:
            continue
        for start in range(length - n + 1):
            gram = text[start : start + n]
            if all(ch in COMMON_PUNCTUATION for ch in gram):
                continue
            features.append(gram)
    return features


class CharNgramVectorizer:
    """Fit a character N-gram vocabulary and transform texts to count matrices.

    Parameters
    ----------
    ngram_range:
        ``(min_n, max_n)`` character N-gram sizes.
    min_df:
        Minimum document frequency for a feature to enter the vocabulary.
    max_features:
        Vocabulary cap applied after the ``(df, tf, lex)`` ordering.
    dtype:
        Count-matrix dtype; ``np.int64`` by default.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (2, 2),
        min_df: int = 2,
        max_features: int | None = 50000,
        dtype: np.dtype = np.int64,
    ):
        self.ngram_range = tuple(ngram_range)
        self.min_df = int(min_df)
        self.max_features = max_features
        self.dtype = np.dtype(dtype)

        self.feature_to_index_: dict[str, int] = {}
        self.index_to_feature_: dict[int, str] = {}
        self.document_frequency_: Counter | None = None
        self.total_frequency_: Counter | None = None

    # ------------------------------------------------------------------
    def fit(self, texts) -> CharNgramVectorizer:
        texts = list(texts)
        document_frequency: Counter = Counter()
        total_frequency: Counter = Counter()

        for text in texts:
            features = extract_char_ngrams(text, self.ngram_range)
            total_frequency.update(features)
            document_frequency.update(set(features))

        candidates = [
            feature
            for feature, df_value in document_frequency.items()
            if df_value >= self.min_df
        ]
        candidates.sort(
            key=lambda f: (-document_frequency[f], -total_frequency[f], f)
        )
        if self.max_features is not None:
            candidates = candidates[: self.max_features]

        self.feature_to_index_ = {f: i for i, f in enumerate(candidates)}
        self.index_to_feature_ = {i: f for f, i in self.feature_to_index_.items()}
        self.document_frequency_ = document_frequency
        self.total_frequency_ = total_frequency
        return self

    # ------------------------------------------------------------------
    def transform(self, texts) -> sparse.csr_matrix:
        """Transform texts into a CSR count matrix (one scan per text)."""

        self._check_fitted()
        texts = [str(t) if not isinstance(t, str) else t for t in texts]

        indptr = np.zeros(len(texts) + 1, dtype=np.int64)
        indices: list[int] = []
        data: list[int] = []

        feature_to_index = self.feature_to_index_
        for row, text in enumerate(texts):
            counts: Counter = Counter()
            for feature in extract_char_ngrams(text, self.ngram_range):
                index = feature_to_index.get(feature)
                if index is not None:
                    counts[index] += 1

            # canonical order: sort row indices for deterministic CSR
            for index in sorted(counts):
                indices.append(index)
                data.append(counts[index])
            indptr[row + 1] = len(indices)

        matrix = sparse.csr_matrix(
            (
                np.asarray(data, dtype=self.dtype),
                np.asarray(indices, dtype=np.int32),
                indptr,
            ),
            shape=(len(texts), len(feature_to_index)),
        )
        return matrix

    def fit_transform(self, texts) -> sparse.csr_matrix:
        return self.fit(texts).transform(texts)

    # ------------------------------------------------------------------
    @property
    def vocabulary_size_(self) -> int:
        return len(self.feature_to_index_)

    def _check_fitted(self):
        if not self.feature_to_index_:
            raise RuntimeError("词表尚未建立，请先调用fit")
