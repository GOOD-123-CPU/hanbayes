"""High-level API: train the full HanBayes model family in one call.

``ChineseSentimentAnalyzer`` is the recommended entry point::

    analyzer = ChineseSentimentAnalyzer.from_config()
    analyzer.fit(train_texts, train_labels)
    report = analyzer.evaluate(dev_texts, dev_labels)
    print(report.frame)

Design notes
------------
- Texts are tokenized into count matrices **once** (train + eval);
- All four models share the same vocabulary and the same base statistics;
- ``fit`` also stores an :class:`ExplanationEngine` for per-prediction
  attributions (see ``hanbayes.interpretability``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .configs import load_frozen_config
from .features import CharNgramVectorizer
from .interpretability import ExplanationEngine
from .models.core import (
    MODEL_NAMES,
    DependencyNB,
    MIWeightedNB,
    StandardNB,
    build_dependency_table,
    build_pair_matrix,
    class_feature_document_frequency,
    corrected_redundancy_scores,
    document_pair_frequency,
    feature_direction_scores,
    feature_mutual_information,
    mutual_information_weights,
    redundancy_aware_weights,
    select_dependency_structure,
    select_redundancy_candidates,
)


@dataclass
class EvaluationReport:
    """Container for one evaluation pass over all four models."""

    frame: object  # pandas.DataFrame
    predictions: dict
    probabilities: dict
    confusion: dict

    def __getitem__(self, key):
        return self.frame[key]

    def to_string(self) -> str:
        return self.frame.to_string(index=False)


class ChineseSentimentAnalyzer:
    """One-fit-trains-all facade over the four-model HanBayes family.

    Parameters
    ----------
    config:
        Frozen configuration dict (see ``configs/frozen.json``).
    normalizer:
        Optional text normalizer override.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or load_frozen_config()
        self.vectorizer: CharNgramVectorizer | None = None
        self.models: dict = {}
        self.explainer: ExplanationEngine | None = None
        self.metadata_: dict = {}

        text_feature = self.config["text_feature"]
        self.ngram_range = tuple(text_feature["ngram_range"])
        self.alpha = self.config["naive_bayes"]["alpha"]

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config_path=None) -> ChineseSentimentAnalyzer:
        return cls(config=load_frozen_config(config_path))

    # ------------------------------------------------------------------
    def fit(
        self,
        texts,
        labels,
        show_progress: bool = False,
    ) -> ChineseSentimentAnalyzer:
        """Train StandardNB / FWNB / DFWNB-v2 / SDFWNB on *texts*."""

        texts = list(texts)
        labels = np.asarray(labels, dtype=int)

        if len(texts) != len(labels):
            raise ValueError("texts与labels长度不一致")
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("训练数据必须同时包含标签0和1")

        # 1. vectorize once -------------------------------------------
        self.vectorizer = CharNgramVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.config["text_feature"]["min_df"],
            max_features=self.config["text_feature"]["max_features"],
        )
        counts = self.vectorizer.fit_transform(texts)

        # 2. base statistics ------------------------------------------
        class_df, class_doc_counts, _ = class_feature_document_frequency(
            counts, labels
        )
        mi = feature_mutual_information(class_df, class_doc_counts)

        fwnb_config = self.config["fwnb"]
        fwnb_weights, normalized_mi = mutual_information_weights(
            mi,
            minimum_weight=fwnb_config["minimum_weight"],
            maximum_boost=fwnb_config["maximum_boost"],
            gamma=fwnb_config["gamma"],
        )

        # 3. StandardNB ------------------------------------------------
        standard = StandardNB(alpha=self.alpha).fit(counts, labels)

        # 4. FWNB -------------------------------------------------------
        fwnb = MIWeightedNB(alpha=self.alpha).fit(counts, labels, fwnb_weights)

        # 5. DFWNB-v2 ---------------------------------------------------
        dfwnb_config = self.config["dfwnb_v2"]
        candidates = select_redundancy_candidates(
            self.vectorizer.index_to_feature_,
            mi,
            dfwnb_config["candidate_feature_count"],
        )
        candidate_set = set(candidates)

        presence = counts.copy()
        presence.data = np.ones_like(presence.data, dtype=np.int64)
        pair_df = document_pair_frequency(
            presence,
            candidate_set,
            mi,
            max_features_per_document=dfwnb_config["maximum_features_per_document"],
        )

        direction = feature_direction_scores(
            class_df,
            class_doc_counts,
            smoothing=dfwnb_config["direction_smoothing"],
        )
        redundancy_scores, relation_count = corrected_redundancy_scores(
            pair_df,
            class_df.sum(axis=0),
            mi,
            direction,
            self.vectorizer.index_to_feature_,
            self.vectorizer.vocabulary_size_,
            minimum_pair_df=dfwnb_config["minimum_pair_df"],
            minimum_jaccard=dfwnb_config["minimum_jaccard"],
            top_k=dfwnb_config["top_k"],
            support_tau=dfwnb_config["support_tau"],
        )
        dfwnb_weights = redundancy_aware_weights(
            fwnb_weights,
            redundancy_scores,
            strength=dfwnb_config["redundancy_strength"],
            gamma=dfwnb_config["redundancy_gamma"],
            preserve_mean=True,
        )
        dfwnb = MIWeightedNB(alpha=self.alpha).fit(counts, labels, dfwnb_weights)

        # 6. SDFWNB -----------------------------------------------------
        sdfwnb_config = self.config["sdfwnb"]
        pair_matrix, pair_registry = build_pair_matrix(
            texts, self.vectorizer.feature_to_index_
        )
        dependency_table = build_dependency_table(
            pair_matrix,
            labels,
            class_df,
            class_doc_counts,
            self.vectorizer.index_to_feature_,
            minimum_pair_df=sdfwnb_config["minimum_pair_df"],
            support_tau=sdfwnb_config["support_tau"],
            smoothing=sdfwnb_config["smoothing"],
            delta_clip=sdfwnb_config["delta_clip"],
            pair_registry=pair_registry,
        )
        selected_pair_count = min(
            sdfwnb_config["dependency_pair_count"], len(dependency_table)
        )
        corrections, relevance = select_dependency_structure(
            dependency_table, selected_pair_count
        )
        sdfwnb = DependencyNB(alpha=self.alpha, ngram_range=self.ngram_range).fit(
            counts,
            labels,
            dfwnb_weights,
            pair_matrix,
            pair_registry,
            corrections,
            relevance,
            dependency_strength=sdfwnb_config["dependency_strength"],
            max_pairs_per_document=sdfwnb_config["maximum_pairs_per_document"],
        )

        self.models = {
            "StandardNB": standard,
            "FWNB": fwnb,
            "DFWNB-v2": dfwnb,
            "SDFWNB": sdfwnb,
        }
        self._train_texts = texts
        self._pair_matrix_train = pair_matrix

        self.metadata_ = {
            "training_sample_count": len(texts),
            "vocabulary_size": self.vectorizer.vocabulary_size_,
            "redundancy_candidate_count": len(candidates),
            "redundancy_relation_count": relation_count,
            "dependency_candidate_count": len(dependency_table),
            "selected_dependency_pair_count": selected_pair_count,
        }

        self.explainer = ExplanationEngine(
            analyzer=self,
            mutual_information=mi,
            normalized_mi=normalized_mi,
            direction_scores=direction,
            redundancy_scores=redundancy_scores,
            fwnb_weights=fwnb_weights,
            dfwnb_weights=dfwnb_weights,
            dependency_table=dependency_table,
        )
        return self

    # ------------------------------------------------------------------
    def _prepare_eval(self, texts):
        texts = list(texts)
        counts = self.vectorizer.transform(texts)

        # rebuild the pair matrix over the training pair vocabulary so
        # correction columns align with what DependencyNB learned
        pair_matrix, pair_registry = build_pair_matrix(
            texts, self.vectorizer.feature_to_index_
        )
        # remap: DependencyNB stores corrections keyed by the TRAIN pair
        # columns; here the eval matrix may have fewer columns, so re-key
        # corrections through feature-index pairs. Pairs that never appear
        # in the training pair vocabulary are DROPPED (they carry no learned
        # correction) — this exactly matches the reference behaviour.
        sdfwnb = self.models["SDFWNB"]
        eval_pair_to_col = pair_registry.pair_to_column
        train_pair_to_col = sdfwnb.pair_to_index_

        col_map = {
            eval_col: train_pair_to_col[pair]
            for pair, eval_col in eval_pair_to_col.items()
            if pair in train_pair_to_col
        }

        keep = np.array(
            [col_map.get(c, -1) for c in pair_matrix.indices], dtype=np.int32
        )
        valid = keep >= 0
        keep[~valid] = 0  # placeholder; these entries are removed below

        from scipy import sparse as _sparse

        remapped = _sparse.csr_matrix(
            (
                pair_matrix.data.astype(np.float64),
                keep,
                pair_matrix.indptr.copy(),
            ),
            shape=(pair_matrix.shape[0], sdfwnb.correction_matrix_.shape[0]),
        )
        # zero-out entries whose pair had no training-column mapping
        if len(valid) > 0:
            remapped.data[~valid] = 0.0
        remapped = remapped.tocsr()
        remapped.eliminate_zeros()
        return texts, counts, remapped

    def evaluate(self, texts, labels, model_names=None) -> EvaluationReport:
        """Evaluate (a subset of) the model family on *texts*."""

        from .evaluation import evaluate_binary_classification

        labels = np.asarray(labels, dtype=int)
        texts, counts, pair_matrix = self._prepare_eval(texts)

        names = model_names or MODEL_NAMES
        rows, predictions, probabilities, confusions = [], {}, {}, {}

        for name in names:
            model = self.models[name]

            if isinstance(model, DependencyNB):
                scores = model.decision_function(counts, pair_matrix)
            else:
                scores = model.decision_function(counts)

            proba = np.exp(scores - scores.max(axis=1, keepdims=True))
            proba = proba / proba.sum(axis=1, keepdims=True)
            preds = model.classes_[np.argmax(scores, axis=1)]
            positive_proba = proba[:, model.class_to_position_[1]]

            metrics, confusion = evaluate_binary_classification(
                y_true=labels, y_pred=preds, y_score=positive_proba
            )

            rows.append({"model": name, **metrics.iloc[0].to_dict()})
            predictions[name] = preds
            probabilities[name] = positive_proba
            confusions[name] = confusion

        import pandas as pd

        return EvaluationReport(
            frame=pd.DataFrame(rows),
            predictions=predictions,
            probabilities=probabilities,
            confusion=confusions,
        )

    # ------------------------------------------------------------------
    def predict(self, texts, model_name: str = "SDFWNB") -> np.ndarray:
        """Predict 0/1 labels for raw texts using *model_name*."""

        report = self.evaluate(texts, np.zeros(max(len(texts), 1), dtype=int),
                               model_names=[model_name])
        return report.predictions[model_name]

    def predict_proba(self, texts, model_name: str = "SDFWNB") -> np.ndarray:
        """Positive-class probabilities for raw texts."""

        report = self.evaluate(texts, np.zeros(max(len(texts), 1), dtype=int),
                               model_names=[model_name])
        return report.probabilities[model_name]
