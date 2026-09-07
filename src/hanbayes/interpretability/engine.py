"""Interpretability: per-prediction attributions and global feature tables.

The family is linear in feature evidence, so attributions decompose exactly:

    logit_c(text) = log_prior[c]
                  + Σ_f count(f) · w(f) · log P(f|c)
                  + Σ_pair strength · Δ_c(pair)          (SDFWNB only)

``ExplanationEngine`` exposes this decomposition both globally (top
discriminative features, strongest dependencies, redundancy rankings) and
locally (per-prediction evidence for one text).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.core import DependencyNB, build_pair_matrix


class ExplanationEngine:
    """Global + local explanations for the fitted HanBayes family."""

    def __init__(
        self,
        analyzer,
        mutual_information: np.ndarray,
        normalized_mi: np.ndarray,
        direction_scores: np.ndarray,
        redundancy_scores: np.ndarray,
        fwnb_weights: np.ndarray,
        dfwnb_weights: np.ndarray,
        dependency_table,
    ):
        self._analyzer = analyzer
        self.mutual_information = mutual_information
        self.normalized_mi = normalized_mi
        self.direction_scores = direction_scores
        self.redundancy_scores = redundancy_scores
        self.fwnb_weights = fwnb_weights
        self.dfwnb_weights = dfwnb_weights
        self.dependency_table = dependency_table

    # ------------------------------------------------------------------
    # Global explanations
    # ------------------------------------------------------------------
    def top_features(
        self,
        top_n: int = 20,
        by: str = "mutual_information",
        class_filter: int | None = None,
    ) -> pd.DataFrame:
        """Most discriminative features with direction and weights."""

        vectorizer = self._analyzer.vectorizer
        rows = []
        for feature, index in vectorizer.feature_to_index_.items():
            direction = self.direction_scores[index]
            if class_filter is not None:
                if class_filter == 1 and direction <= 0:
                    continue
                if class_filter == 0 and direction >= 0:
                    continue

            rows.append(
                {
                    "feature": feature,
                    "mutual_information": float(self.mutual_information[index]),
                    "direction_score": float(direction),
                    "preferred_class": (
                        "positive" if direction > 0 else "negative"
                    ),
                    "fwnb_weight": float(self.fwnb_weights[index]),
                    "dfwnb_weight": float(self.dfwnb_weights[index]),
                    "redundancy_score": float(self.redundancy_scores[index]),
                }
            )

        frame = pd.DataFrame(rows)
        if by == "mutual_information":
            frame = frame.sort_values("mutual_information", ascending=False)
        elif by == "redundancy":
            frame = frame.sort_values("redundancy_score", ascending=False)
        else:
            raise ValueError("by必须是'mutual_information'或'redundancy'")
        return frame.head(top_n).reset_index(drop=True)

    def top_dependencies(self, top_n: int = 20) -> pd.DataFrame:
        """Strongest class-conditional dependency trigrams."""

        columns = [
            "trigram",
            "left_feature",
            "right_feature",
            "neg_doc_freq",
            "pos_doc_freq",
            "delta_0",
            "delta_1",
            "dependency_contrast",
            "preferred_class",
        ]
        available = [c for c in columns if c in self.dependency_table.columns]
        return self.dependency_table[available].head(top_n)

    def redundancy_relations(self, top_n: int = 20):
        """Top suppressed feature pairs (if the relation table was kept)."""

        return self.top_features(top_n=top_n, by="redundancy")

    # ------------------------------------------------------------------
    # Local explanations (per prediction)
    # ------------------------------------------------------------------
    def explain_prediction(
        self,
        text: str,
        model_name: str = "SDFWNB",
        top_n: int = 10,
    ) -> dict:
        """Exact evidence decomposition for one text under *model_name*.

        Returns a dict with per-feature contributions toward each class, the
        dependency corrections (SDFWNB), and the final scores/probabilities.
        """

        analyzer = self._analyzer
        model = analyzer.models[model_name]

        normalized = analyzer.vectorizer  # vocabulary source
        feature_to_index = normalized.feature_to_index_

        counts = normalized.transform([text])
        row = counts.getrow(0)

        contributions: list[dict] = []
        weighted_log_prob = model._weighted_log_prob()

        for column_index, count in zip(row.indices, row.data):
            evidence = weighted_log_prob[:, column_index].ravel()
            contributions.append(
                {
                    "feature": analyzer.vectorizer.index_to_feature_[int(column_index)],
                    "count": int(count),
                    "weight": float(
                        getattr(model, "feature_weights_", np.ones(1))[0]
                        if not hasattr(model, "feature_weights_")
                        else model.feature_weights_[column_index]
                    ),
                    "evidence_negative": float(count * evidence[0]),
                    "evidence_positive": float(count * evidence[1]),
                }
            )

        if isinstance(model, DependencyNB):
            _, _, pair_matrix = analyzer._prepare_eval([str(text)])
            scores = model.decision_function(counts, pair_matrix)
        else:
            scores = model.decision_function(counts)
        proba = np.exp(scores - scores.max(axis=1, keepdims=True))
        proba = proba / proba.sum(axis=1, keepdims=True)

        dependency_effects = []
        if model_name == "SDFWNB" and model.dependency_strength_ > 0:
            pair_matrix, _ = build_pair_matrix([str(text)], feature_to_index)
            pm = pair_matrix.getrow(0)
            for column in pm.indices:
                pair = model.pair_index_to_pair_.get(int(column))
                if pair is None or column not in model.dependency_corrections_:
                    continue
                correction = model.dependency_corrections_[column]
                left_feature = analyzer.vectorizer.index_to_feature_[pair[0]]
                right_feature = analyzer.vectorizer.index_to_feature_[pair[1]]
                dependency_effects.append(
                    {
                        "trigram": left_feature + right_feature[-1],
                        "effect_negative": float(
                            model.dependency_strength_ * correction[0]
                        ),
                        "effect_positive": float(
                            model.dependency_strength_ * correction[1]
                        ),
                    }
                )

            dependency_effects.sort(
                key=lambda item: -abs(item["effect_positive"] - item["effect_negative"])
            )

        contributions.sort(
            key=lambda item: -(
                abs(item["evidence_positive"] - item["evidence_negative"])
            )
        )

        return {
            "text": str(text),
            "model": model_name,
            "scores": scores[0].tolist(),
            "probabilities": proba[0].tolist(),
            "predicted_class": int(model.classes_[np.argmax(scores[0])]),
            "log_prior": model.log_class_prior_.tolist(),
            "feature_evidence": contributions[:top_n],
            "dependency_effects": dependency_effects[:top_n],
        }
