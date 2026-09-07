"""Model family: StandardNB, MIWeightedNB, DFWNB-v2 weighting, DependencyNB.

All models operate on a **single sparse count matrix** produced by
:class:`hanbayes.features.CharNgramVectorizer`. Scoring is fully vectorized:

    scores = log_prior + W @ (log_prob.T)      # W: n_docs x vocab (csr)
    scores = scores + (W @ correction.T)       # DependencyNB sparse extra

Weighted variants simply multiply each column of ``log_prob`` by the feature
weight vector, so FWNB / DFWNB-v2 / SDFWNB share the same linear algebra.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ..text import PLACEHOLDERS, have_shared_character, is_pure_chinese_bigram

DEPENDENCY_PLACEHOLDERS = PLACEHOLDERS

MODEL_NAMES = ("StandardNB", "FWNB", "DFWNB-v2", "SDFWNB")


# ======================================================================
# Feature statistics (computed once from the count matrix)
# ======================================================================
def class_feature_document_frequency(
    counts: sparse.csr_matrix, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class document-frequency matrix from a binary presence matrix.

    Returns ``(class_df [2 x V], class_doc_counts [2], classes)``.
    """

    classes = np.array(sorted(np.unique(labels)), dtype=int)
    if not np.array_equal(classes, np.array([0, 1])):
        raise ValueError("当前实现要求二分类标签为0和1")

    presence = counts.copy()
    presence.data = np.ones_like(presence.data, dtype=np.int64)

    class_df = np.zeros((2, counts.shape[1]), dtype=np.int64)
    class_doc_counts = np.zeros(2, dtype=np.int64)

    for label in (0, 1):
        mask = labels == label
        rows = presence[np.asarray(np.nonzero(mask)[0])]
        class_df[label] = np.asarray(rows.sum(axis=0)).ravel()
        class_doc_counts[label] = int(mask.sum())

    return class_df, class_doc_counts, classes


def feature_mutual_information(
    class_df: np.ndarray, class_doc_counts: np.ndarray
) -> np.ndarray:
    """Binary-presence MI between each feature and the label (nat)."""

    class_df = np.asarray(class_df, dtype=np.float64)
    class_doc_counts = np.asarray(class_doc_counts, dtype=np.float64)

    if class_df.shape[0] != 2:
        raise ValueError("当前实现只支持二分类")

    n_neg, n_pos = class_doc_counts
    total = n_neg + n_pos

    n_10 = class_df[0]  # present, negative
    n_11 = class_df[1]  # present, positive
    n_00 = n_neg - n_10
    n_01 = n_pos - n_11

    present = n_10 + n_11
    absent = n_00 + n_01

    def _term(joint, x_count, y_count):
        term = np.zeros_like(joint, dtype=np.float64)
        valid = joint > 0
        term[valid] = (
            joint[valid] / total
            * np.log((joint[valid] * total) / (x_count[valid] * y_count))
        )
        return term

    mi = (
        _term(n_10, present, n_neg)
        + _term(n_11, present, n_pos)
        + _term(n_00, absent, n_neg)
        + _term(n_01, absent, n_pos)
    )
    return np.maximum(mi, 0.0)


def feature_direction_scores(
    class_df: np.ndarray, class_doc_counts: np.ndarray, smoothing: float = 0.5
) -> np.ndarray:
    """Log-odds direction score; positive leans to class 1 (positive)."""

    class_df = np.asarray(class_df, dtype=np.float64)
    class_doc_counts = np.asarray(class_doc_counts, dtype=np.float64)

    p_neg = (class_df[0] + smoothing) / (class_doc_counts[0] + 2.0 * smoothing)
    p_pos = (class_df[1] + smoothing) / (class_doc_counts[1] + 2.0 * smoothing)
    return np.log(p_pos / (1.0 - p_pos)) - np.log(p_neg / (1.0 - p_neg))


def mutual_information_weights(
    mi: np.ndarray,
    minimum_weight: float = 0.5,
    maximum_boost: float = 0.0,
    gamma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map MI to weights in ``[minimum_weight, 1 + maximum_boost]``."""

    mi = np.asarray(mi, dtype=np.float64)
    if not (0 < minimum_weight <= 1):
        raise ValueError("minimum_weight必须位于(0,1]内")
    if maximum_boost < 0:
        raise ValueError("maximum_boost不能小于0")
    if gamma <= 0:
        raise ValueError("gamma必须大于0")

    max_mi = float(mi.max())
    normalized = np.zeros_like(mi) if max_mi <= 0 else mi / max_mi

    weights = minimum_weight + (1.0 + maximum_boost - minimum_weight) * np.power(
        normalized, gamma
    )
    return weights, normalized


# ======================================================================
# DFWNB-v2 redundancy machinery
# ======================================================================
def select_redundancy_candidates(
    index_to_feature: dict[int, str], mi: np.ndarray, candidate_count: int
) -> list[int]:
    """Top-N pure-Chinese bigram feature indices by MI."""

    candidates: list[int] = []
    for feature_index in np.argsort(-np.asarray(mi)):
        feature_index = int(feature_index)
        if not is_pure_chinese_bigram(index_to_feature[feature_index]):
            continue
        candidates.append(feature_index)
        if len(candidates) >= candidate_count:
            break
    return candidates


def document_pair_frequency(
    presence_rows: sparse.csr_matrix,
    candidate_index_set: set,
    relevance: np.ndarray,
    max_features_per_document: int = 80,
) -> Counter:
    """Document-level co-occurrence counts of candidate feature pairs.

    ``presence_rows`` is a binary CSR matrix; for each document the top
    ``max_features_per_document`` candidates by MI are kept, then all pairs
    among them are counted once.
    """

    pair_df: Counter = Counter()

    col_indices = presence_rows.tocoo().col

    # group column indices by row (CSR rows are already sorted)
    row_starts = presence_rows.indptr
    row_ends = presence_rows.indptr[1:]

    for row in range(presence_rows.shape[0]):
        doc_features = [
            int(c)
            for c in col_indices[row_starts[row] : row_ends[row]]
            if c in candidate_index_set
        ]
        if len(doc_features) < 2:
            continue

        doc_features.sort(key=lambda i: (-relevance[i], i))
        doc_features = doc_features[:max_features_per_document]

        n = len(doc_features)
        for left in range(n):
            for right in range(left + 1, n):
                a, b = doc_features[left], doc_features[right]
                pair_df[(a, b) if a < b else (b, a)] += 1

    return pair_df


def corrected_redundancy_scores(
    pair_df: Counter,
    feature_df: np.ndarray,
    mi: np.ndarray,
    direction: np.ndarray,
    index_to_feature: dict[int, str],
    vocab_size: int,
    minimum_pair_df: int = 20,
    minimum_jaccard: float = 0.03,
    top_k: int = 3,
    support_tau: float = 30.0,
) -> tuple[np.ndarray, int]:
    """Asymmetric redundancy scores; returns ``(scores, relation_count)``."""

    feature_df = np.asarray(feature_df, dtype=np.float64)
    neighbour_scores: defaultdict = defaultdict(list)
    relation_count = 0

    for (left, right), pair_count in pair_df.items():
        if pair_count < minimum_pair_df:
            continue

        if have_shared_character(
            index_to_feature[left], index_to_feature[right]
        ):
            continue

        d_left, d_right = direction[left], direction[right]
        if d_left * d_right <= 0:
            continue

        df_left, df_right = feature_df[left], feature_df[right]
        union = df_left + df_right - pair_count
        if union <= 0:
            continue

        jaccard = pair_count / union
        if jaccard < minimum_jaccard:
            continue

        a_left, a_right = abs(d_left), abs(d_right)
        direction_similarity = min(a_left, a_right) / (max(a_left, a_right) + 1e-12)
        support_factor = pair_count / (pair_count + support_tau)
        relation_score = jaccard * direction_similarity * support_factor

        if mi[left] < mi[right]:
            penalized = left
        elif mi[right] < mi[left]:
            penalized = right
        elif df_left < df_right:
            penalized = left
        elif df_right < df_left:
            penalized = right
        else:
            penalized = max(left, right)

        neighbour_scores[penalized].append(relation_score)
        relation_count += 1

    raw = np.zeros(vocab_size, dtype=np.float64)
    for feature_index, scores in neighbour_scores.items():
        raw[feature_index] = float(np.mean(sorted(scores, reverse=True)[:top_k]))

    nonzero = raw[raw > 0]
    scale = float(np.quantile(nonzero, 0.95)) if len(nonzero) > 0 else 1.0
    if scale <= 0:
        scale = 1.0

    return np.clip(raw / scale, 0.0, 1.0), relation_count


def redundancy_aware_weights(
    base_weights: np.ndarray,
    redundancy_scores: np.ndarray,
    strength: float = 0.5,
    gamma: float = 1.0,
    preserve_mean: bool = True,
) -> np.ndarray:
    """Combine FWNB weights with redundancy suppression."""

    base_weights = np.asarray(base_weights, dtype=np.float64)
    redundancy_scores = np.asarray(redundancy_scores, dtype=np.float64)

    if len(base_weights) != len(redundancy_scores):
        raise ValueError("基础权重与冗余得分长度不一致")
    if strength < 0:
        raise ValueError("redundancy_strength不能为负数")
    if gamma <= 0:
        raise ValueError("redundancy_gamma必须大于0")

    raw = base_weights / (1.0 + strength * np.power(redundancy_scores, gamma))

    if preserve_mean:
        raw_mean = float(raw.mean())
        base_mean = float(base_weights.mean())
        if raw_mean > 0:
            raw = raw * base_mean / raw_mean
    return raw


# ======================================================================
# SDFWNB dependency machinery
# ======================================================================
def build_pair_matrix(
    texts,
    feature_to_index: dict[str, int],
):
    """Sparse occurrence matrix of ordered adjacent bigram pairs.

    Each trigram window ``(c1, c2, c3)`` yields the ordered pair
    ``(c1c2, c2c3)`` when both slices are pure Chinese bigrams in the
    vocabulary and the window contains no normalization placeholder.
    Each pair counts at most once per document.

    Returns ``(csr_matrix, DependencyPairRegistry)`` where the matrix rows
    are documents and columns are pairs (see the registry for mappings).
    """

    pair_to_column: dict[tuple[int, int], int] = {}
    index_to_pair: dict[int, tuple[int, int]] = {}

    texts = [str(t) if not isinstance(t, str) else t for t in texts]
    indptr = np.zeros(len(texts) + 1, dtype=np.int64)
    indices: list[int] = []
    data: list[int] = []

    placeholders = DEPENDENCY_PLACEHOLDERS

    for row, text in enumerate(texts):
        doc_pairs: set[tuple[int, int]] = set()
        if len(text) >= 3:
            for start in range(len(text) - 2):
                trigram = text[start : start + 3]
                left_feature, right_feature = trigram[:2], trigram[1:]

                if not is_pure_chinese_bigram(left_feature):
                    continue
                if not is_pure_chinese_bigram(right_feature):
                    continue
                if any(p in trigram for p in placeholders):
                    continue

                left_index = feature_to_index.get(left_feature)
                right_index = feature_to_index.get(right_feature)
                if left_index is None or right_index is None:
                    continue
                if left_index == right_index:
                    continue

                doc_pairs.add((int(left_index), int(right_index)))

        for pair in sorted(doc_pairs):
            column = pair_to_column.get(pair)
            if column is None:
                column = len(pair_to_column)
                pair_to_column[pair] = column
                index_to_pair[column] = pair
            indices.append(column)
            data.append(1)
        indptr[row + 1] = len(indices)

    matrix = sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.int64),
            np.asarray(indices, dtype=np.int32),
            indptr,
        ),
        shape=(len(texts), max(len(pair_to_column), 1)),
    )
    registry = DependencyPairRegistry(pair_to_column, index_to_pair)
    return matrix, registry


class DependencyPairRegistry:
    """Bidirectional mapping between ordered pairs and matrix columns."""

    def __init__(self, pair_to_column: dict, index_to_pair: dict):
        self.pair_to_column = pair_to_column
        self.index_to_pair = index_to_pair


def pair_presence_mutual_information(
    pair_df_0: int, pair_df_1: int, n_neg: int, n_pos: int
) -> float:
    """MI between pair presence and class label (nat)."""

    counts = np.array(
        [
            [n_neg - pair_df_0, n_pos - pair_df_1],
            [pair_df_0, pair_df_1],
        ],
        dtype=np.float64,
    )
    total = counts.sum()
    x_counts = counts.sum(axis=1)
    y_counts = counts.sum(axis=0)

    mi = 0.0
    for x_state in range(2):
        for y_state in range(2):
            joint = counts[x_state, y_state]
            if joint <= 0:
                continue
            mi += (
                joint
                / total
                * math.log((joint * total) / (x_counts[x_state] * y_counts[y_state]))
            )
    return float(max(mi, 0.0))


@dataclass
class DependencyTable:
    """Class-conditional dependency correction table (DataFrame wrapper)."""

    frame: object  # pandas.DataFrame
    minimum_pair_df: int
    support_tau: float
    smoothing: float
    delta_clip: float


def build_dependency_table(
    pair_matrix: sparse.csr_matrix,
    labels: np.ndarray,
    class_df: np.ndarray,
    class_doc_counts: np.ndarray,
    index_to_feature: dict[int, str],
    minimum_pair_df: int = 8,
    support_tau: float = 30.0,
    smoothing: float = 0.5,
    delta_clip: float = 4.0,
    pair_registry=None,
):
    """Build the class-conditional dependency correction table.

    ``pair_matrix`` rows are documents, columns are ordered adjacent pairs;
    ``pair_registry`` is the DependencyPairRegistry returned by
    :func:`build_pair_matrix`.
    """

    import pandas as pd

    registry = pair_registry.index_to_pair

    pair_counts_by_class: dict[int, Counter] = {0: Counter(), 1: Counter()}
    presence = pair_matrix.copy()
    presence.data = np.ones_like(presence.data, dtype=np.int64)

    for label in (0, 1):
        row_indices = np.nonzero(labels == label)[0]
        rows_matrix = presence[row_indices]
        counts = np.asarray(rows_matrix.sum(axis=0)).ravel()
        for column, count in enumerate(counts):
            if count > 0:
                pair = registry[column]
                pair_counts_by_class[label][pair] = int(count)

    all_pairs = set(pair_counts_by_class[0]) | set(pair_counts_by_class[1])

    n_neg, n_pos = int(class_doc_counts[0]), int(class_doc_counts[1])
    rows: list[dict] = []

    for left, right in all_pairs:
        pair_df_0 = pair_counts_by_class[0].get((left, right), 0)
        pair_df_1 = pair_counts_by_class[1].get((left, right), 0)
        total_df = pair_df_0 + pair_df_1
        if total_df < minimum_pair_df:
            continue

        raw_deltas = np.zeros(2, dtype=np.float64)
        for position, (class_count, pair_df) in enumerate(
            ((n_neg, pair_df_0), (n_pos, pair_df_1))
        ):
            left_df = class_df[position, left]
            right_df = class_df[position, right]

            p_pair = (pair_df + smoothing) / (class_count + 2.0 * smoothing)
            p_left = (left_df + smoothing) / (class_count + 2.0 * smoothing)
            p_right = (right_df + smoothing) / (class_count + 2.0 * smoothing)

            raw_deltas[position] = np.clip(
                math.log(p_pair / (p_left * p_right)), -delta_clip, delta_clip
            )

        support_factor = total_df / (total_df + support_tau)
        shrunk = raw_deltas * support_factor
        centered = shrunk - shrunk.mean()

        contrast = float(abs(centered[1] - centered[0]))
        pair_mi = pair_presence_mutual_information(pair_df_0, pair_df_1, n_neg, n_pos)

        left_feature = index_to_feature[left]
        right_feature = index_to_feature[right]

        rows.append(
            {
                "left_index": int(left),
                "right_index": int(right),
                "left_feature": left_feature,
                "right_feature": right_feature,
                "trigram": left_feature + right_feature[-1],
                "neg_doc_freq": pair_df_0,
                "pos_doc_freq": pair_df_1,
                "total_doc_freq": total_df,
                "support_factor": float(support_factor),
                "delta_raw_0": float(raw_deltas[0]),
                "delta_raw_1": float(raw_deltas[1]),
                "delta_0": float(centered[0]),
                "delta_1": float(centered[1]),
                "dependency_contrast": contrast,
                "pair_label_mi": pair_mi,
                "preferred_class": "positive" if centered[1] > 0 else "negative",
                "selection_score": contrast,
            }
        )

    table = pd.DataFrame(rows)
    if len(table) > 0:
        table = table.sort_values(
            by=["selection_score", "pair_label_mi", "total_doc_freq"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    return table


def select_dependency_structure(
    table,
    top_pair_count: int,
):
    """Pick top pairs and build ``(corrections, relevance)`` dictionaries."""

    if top_pair_count < 0:
        raise ValueError("top_pair_count不能为负数")

    selected = table.head(top_pair_count)
    corrections: dict[tuple[int, int], np.ndarray] = {}
    relevance: dict[tuple[int, int], float] = {}

    for _, row in selected.iterrows():
        pair = (int(row["left_index"]), int(row["right_index"]))
        corrections[pair] = np.array(
            [float(row["delta_0"]), float(row["delta_1"])], dtype=np.float64
        )
        relevance[pair] = float(row["selection_score"])

    return corrections, relevance


# ======================================================================
# Model classes
# ======================================================================
class _BaseNB:
    """Shared fit/score logic for the count-matrix model family."""

    def __init__(self, alpha: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha必须大于0")
        self.alpha = float(alpha)
        self.is_fitted_ = False

    def _fit_core(self, counts: sparse.csr_matrix, labels: np.ndarray):
        counts = sparse.csr_matrix(counts)
        labels = np.asarray(labels, dtype=int)

        if counts.shape[0] != len(labels):
            raise ValueError("counts与labels行数不一致")
        if counts.shape[0] == 0:
            raise ValueError("训练数据不能为空")

        self.classes_ = np.array(sorted(np.unique(labels)), dtype=int)
        self.class_to_position_ = {
            int(c): i for i, c in enumerate(self.classes_)
        }

        n_classes = len(self.classes_)
        self.class_document_count_ = np.zeros(n_classes, dtype=np.int64)
        self.feature_count_ = np.zeros((n_classes, counts.shape[1]), dtype=np.int64)

        for label in np.unique(labels):
            position = self.class_to_position_[int(label)]
            row_indices = np.nonzero(labels == label)[0]
            rows = counts[row_indices]
            self.class_document_count_[position] = int(row_indices.size)
            self.feature_count_[position] = np.asarray(rows.sum(axis=0)).ravel()

        self.log_class_prior_ = np.log(
            self.class_document_count_ / self.class_document_count_.sum()
        )
        self.class_token_count_ = self.feature_count_.sum(axis=1)

        numerator = self.feature_count_ + self.alpha
        denominator = (
            self.class_token_count_[:, None] + self.alpha * counts.shape[1]
        )
        self.log_feature_probability_ = np.log(numerator / denominator)

        self.is_fitted_ = True
        return self

    def _weighted_log_prob(self) -> np.ndarray:
        """``log_feature_probability_`` scaled by feature weights (if any)."""

        weights = getattr(self, "feature_weights_", None)
        if weights is None:
            return self.log_feature_probability_
        return self.log_feature_probability_ * weights[np.newaxis, :]

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("模型尚未训练，请先调用fit")

    def decision_function(self, counts: sparse.csr_matrix) -> np.ndarray:
        """Class log-scores for a count matrix (vectorized)."""

        self._check_fitted()
        counts = sparse.csr_matrix(counts)
        weighted = self._weighted_log_prob()
        scores = counts @ weighted.T + self.log_class_prior_
        return np.asarray(scores)

    def predict(self, counts: sparse.csr_matrix) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(counts), axis=1)]

    def predict_proba(self, counts: sparse.csr_matrix) -> np.ndarray:
        scores = self.decision_function(counts)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)


class StandardNB(_BaseNB):
    """Multinomial Naive Bayes with Laplace smoothing (baseline)."""

    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha=alpha)

    def fit(self, counts: sparse.csr_matrix, labels) -> StandardNB:
        self._fit_core(counts, labels)
        return self


class MIWeightedNB(_BaseNB):
    """FWNB: per-feature log evidence scaled by MI-derived weights."""

    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha=alpha)

    def fit(
        self,
        counts: sparse.csr_matrix,
        labels,
        feature_weights: np.ndarray | None = None,
    ) -> MIWeightedNB:
        self._fit_core(counts, labels)

        if feature_weights is not None:
            weights = np.asarray(feature_weights, dtype=np.float64)
            if len(weights) != counts.shape[1]:
                raise ValueError("特征权重数量与词表大小不一致")
            if not np.all(np.isfinite(weights)):
                raise ValueError("特征权重中存在非有限值")
            if np.any(weights < 0):
                raise ValueError("特征权重不能为负数")
            self.feature_weights_ = weights.copy()
        return self


class DependencyNB(_BaseNB):
    """SDFWNB: weighted NB plus sparse local dependency corrections.

    ``pair_matrix`` is the ordered-adjacent-pair occurrence matrix produced by
    :func:`build_pair_matrix` over the **same** texts used for ``counts``.
    """

    def __init__(self, alpha: float = 1.0, ngram_range: tuple[int, int] = (2, 2)):
        super().__init__(alpha=alpha)
        self.ngram_range = tuple(ngram_range)

    def fit(
        self,
        counts: sparse.csr_matrix,
        labels,
        feature_weights: np.ndarray,
        pair_matrix: sparse.csr_matrix,
        pair_registry,
        dependency_corrections: dict,
        dependency_pair_scores: dict | None = None,
        dependency_strength: float = 1.0,
        max_pairs_per_document: int | None = 30,
    ) -> DependencyNB:
        self._fit_core(counts, labels)

        weights = np.asarray(feature_weights, dtype=np.float64)
        if len(weights) != counts.shape[1]:
            raise ValueError("特征权重数量与词表大小不一致")
        self.feature_weights_ = weights.copy()

        if dependency_strength < 0:
            raise ValueError("dependency_strength不能为负数")
        if max_pairs_per_document is not None and max_pairs_per_document < 1:
            raise ValueError("最大依赖词对数量必须大于0")

        self.dependency_strength_ = float(dependency_strength)
        self.max_pairs_per_document_ = max_pairs_per_document

        self.pair_index_to_pair_: dict[int, tuple[int, int]] = dict(
            pair_registry.index_to_pair
        )
        self.pair_to_index_: dict[tuple[int, int], int] = dict(
            pair_registry.pair_to_column
        )

        self.dependency_corrections_: dict[int, np.ndarray] = {}
        for pair, correction in dependency_corrections.items():
            column = self.pair_to_index_.get(pair)
            if column is None:
                continue  # pair absent from this corpus's pair matrix
            correction = np.asarray(correction, dtype=np.float64)
            if correction.shape != (len(self.classes_),):
                raise ValueError("依赖修正向量长度与类别数不一致")
            if not np.all(np.isfinite(correction)):
                raise ValueError("依赖修正向量存在非有限值")
            self.dependency_corrections_[column] = correction.copy()

        if dependency_pair_scores is None:
            self.dependency_pair_scores_ = {
                column: float(np.max(np.abs(corr)))
                for column, corr in self.dependency_corrections_.items()
            }
        else:
            self.dependency_pair_scores_ = {
                self.pair_to_index_[pair]: float(score)
                for pair, score in dependency_pair_scores.items()
                if pair in self.pair_to_index_
            }

        # column-aligned dense correction matrix for vectorized scoring
        n_pairs = pair_matrix.shape[1]
        self.correction_matrix_ = np.zeros((n_pairs, len(self.classes_)))
        for column, correction in self.dependency_corrections_.items():
            self.correction_matrix_[column] = correction

        return self

    # ------------------------------------------------------------------
    def decision_function(
        self, counts: sparse.csr_matrix, pair_matrix: sparse.csr_matrix | None = None
    ) -> np.ndarray:
        self._check_fitted()
        scores = super().decision_function(counts)

        if (
            self.dependency_strength_ > 0
            and len(self.dependency_corrections_) > 0
        ):
            if pair_matrix is None:
                raise ValueError(
                    "DependencyNB需要pair_matrix参数进行依赖修正评分"
                )
            pair_matrix = sparse.csr_matrix(pair_matrix)

            if pair_matrix.shape[1] != self.correction_matrix_.shape[0]:
                raise ValueError(
                    "pair_matrix列数与训练时不一致，请使用同一registry重建"
                )

            active_columns = sorted(self.dependency_corrections_.keys())
            active = np.zeros(pair_matrix.shape[1], dtype=bool)
            active[active_columns] = True

            effective = self.dependency_strength_ * self.correction_matrix_

            if self.max_pairs_per_document_ is not None:
                # cap corrections per document: keep only the strongest pairs.
                # NOTE: after eval-time column remapping the row indices are
                # NOT sorted, so positions are tracked explicitly (no
                # searchsorted on an unsorted row).
                indptr, indices = pair_matrix.indptr, pair_matrix.indices
                relevance = np.zeros(pair_matrix.shape[1])
                for column, score in self.dependency_pair_scores_.items():
                    relevance[column] = score

                keep_data = np.zeros(len(indices), dtype=np.float64)
                for row in range(pair_matrix.shape[0]):
                    row_start, row_end = indptr[row], indptr[row + 1]
                    row_cols = indices[row_start:row_end]
                    if len(row_cols) == 0:
                        continue
                    active_positions = np.nonzero(active[row_cols])[0]
                    if active_positions.size == 0:
                        continue
                    if active_positions.size > self.max_pairs_per_document_:
                        active_positions = np.array(
                            sorted(
                                active_positions.tolist(),
                                key=lambda pos: (
                                    -relevance[row_cols[pos]],
                                    int(row_cols[pos]),
                                ),
                            )[: self.max_pairs_per_document_]
                        )
                    for pos in active_positions:
                        keep_data[row_start + pos] = 1.0

                pair_effect = sparse.csr_matrix(
                    (keep_data, indices.copy(), indptr.copy()),
                    shape=pair_matrix.shape,
                )
                pair_effect.eliminate_zeros()
                scores = scores + pair_effect @ effective
            else:
                scores = scores + pair_matrix @ effective

        return np.asarray(scores)
