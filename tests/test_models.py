"""Tests for model correctness: degradation properties and basic behavior."""

from __future__ import annotations

import numpy as np
import pytest

from hanbayes.features import CharNgramVectorizer
from hanbayes.models.core import (
    DependencyNB,
    MIWeightedNB,
    StandardNB,
    build_dependency_table,
    build_pair_matrix,
    class_feature_document_frequency,
    feature_mutual_information,
    mutual_information_weights,
    redundancy_aware_weights,
    select_dependency_structure,
    select_redundancy_candidates,
)


@pytest.fixture
def vectorized(tiny_dataset):
    texts, labels = tiny_dataset
    vectorizer = CharNgramVectorizer((2, 2), min_df=2, max_features=5000)
    counts = vectorizer.fit_transform(texts)
    return texts, labels, vectorizer, counts


class TestDegradation:
    def test_fwnb_unit_weights_degrade_to_nb(self, vectorized):
        texts, labels, vectorizer, counts = vectorized

        nb = StandardNB(alpha=0.05).fit(counts, labels)
        unit = MIWeightedNB(alpha=0.05).fit(
            counts, labels, np.ones(counts.shape[1])
        )

        probe = vectorizer.transform(
            ["房间干净服务好", "房间差态度恶劣", "价格贵体验糟糕"]
        )
        np.testing.assert_allclose(
            nb.decision_function(probe),
            unit.decision_function(probe),
            atol=1e-10,
        )
        assert np.array_equal(nb.predict(probe), unit.predict(probe))

    def test_sdfwnb_zero_strength_degrades_to_dfwnb(self, vectorized):
        texts, labels, vectorizer, counts = vectorized

        class_df, class_doc_counts, _ = class_feature_document_frequency(
            counts, labels
        )
        mi = feature_mutual_information(class_df, class_doc_counts)
        weights, _ = mutual_information_weights(mi, 0.25, 0.5, 1.0)

        pair_matrix, pair_registry = build_pair_matrix(
            texts, vectorizer.feature_to_index_
        )
        table = build_dependency_table(
            pair_matrix,
            labels,
            class_df,
            class_doc_counts,
            vectorizer.index_to_feature_,
            minimum_pair_df=2,
            support_tau=5.0,
            pair_registry=pair_registry,
        )
        corrections, relevance = select_dependency_structure(table, 20)

        sdfwnb_zero = DependencyNB(alpha=0.05).fit(
            counts,
            labels,
            weights,
            pair_matrix,
            pair_registry,
            corrections,
            relevance,
            dependency_strength=0.0,
        )

        fwnb = MIWeightedNB(alpha=0.05).fit(counts, labels, weights)

        probe_texts = ["房间干净服务好", "价格贵体验糟糕"]
        probe_counts = vectorizer.transform(probe_texts)
        probe_pairs, _ = build_pair_matrix(
            probe_texts, vectorizer.feature_to_index_
        )
        np.testing.assert_allclose(
            fwnb.decision_function(probe_counts),
            sdfwnb_zero.decision_function(probe_counts, probe_pairs),
            atol=1e-10,
        )

    def test_dfwnb_zero_redundancy_degrades_to_fwnb(self, vectorized):
        texts, labels, vectorizer, counts = vectorized
        class_df, class_doc_counts, _ = class_feature_document_frequency(
            counts, labels
        )
        mi = feature_mutual_information(class_df, class_doc_counts)
        fwnb_weights, _ = mutual_information_weights(mi, 0.25, 0.5, 1.0)

        dfwnb_weights = redundancy_aware_weights(
            fwnb_weights,
            np.zeros_like(mi),
            strength=1.0,
            gamma=0.5,
            preserve_mean=True,
        )
        np.testing.assert_allclose(dfwnb_weights, fwnb_weights, atol=1e-12)


class TestMIWeights:
    def test_weight_range(self, vectorized):
        _, labels, _, counts = vectorized
        class_df, class_doc_counts, _ = class_feature_document_frequency(
            counts, labels
        )
        mi = feature_mutual_information(class_df, class_doc_counts)
        weights, normalized = mutual_information_weights(
            mi, minimum_weight=0.25, maximum_boost=0.5, gamma=1.0
        )

        assert weights.min() >= 0.25 - 1e-12
        assert weights.max() <= 1.75 + 1e-12
        assert np.all(mi >= 0)
        assert normalized.max() <= 1.0 + 1e-12

    def test_redundancy_weights_preserve_mean(self):
        base = np.array([1.0, 0.5, 0.8, 1.2, 0.3])
        scores = np.array([0.0, 1.0, 0.5, 0.0, 0.2])
        result = redundancy_aware_weights(
            base, scores, strength=1.0, gamma=0.5, preserve_mean=True
        )
        np.testing.assert_allclose(result.mean(), base.mean(), rtol=1e-12)
        # penalized features must be lower than base (before mean rescale
        # the ordering is preserved; here just check the penalty exists)
        assert result[1] < result[0]


class TestPairMatrix:
    def test_registry_roundtrip(self, vectorized):
        texts, _, vectorizer, _ = vectorized
        pair_matrix, registry = build_pair_matrix(
            texts, vectorizer.feature_to_index_
        )
        for column, pair in registry.index_to_pair.items():
            assert registry.pair_to_column[pair] == column

    def test_pair_matrix_binary(self, vectorized):
        texts, _, vectorizer, _ = vectorized
        pair_matrix, _ = build_pair_matrix(texts, vectorizer.feature_to_index_)
        assert set(np.unique(pair_matrix.data)) <= {1}

    def test_pair_extraction_matches_trigram_semantics(self):

        texts = ["不满意这家酒店"]
        feature_to_index = {"不满": 0, "满意": 1, "意这": 2, "这家": 3, "家酒": 4}
        pair_matrix, registry = build_pair_matrix(texts, feature_to_index)
        pairs = {registry.index_to_pair[c] for c in pair_matrix.indices}
        # trigram "不满意" -> (不满, 满意); the placeholder-adjacent windows skip
        assert (0, 1) in pairs


class TestDependencyTable:
    def test_selection_deterministic(self, vectorized):
        texts, labels, vectorizer, counts = vectorized
        class_df, class_doc_counts, _ = class_feature_document_frequency(
            counts, labels
        )
        pair_matrix, registry = build_pair_matrix(
            texts, vectorizer.feature_to_index_
        )
        t1 = build_dependency_table(
            pair_matrix, labels, class_df, class_doc_counts,
            vectorizer.index_to_feature_, minimum_pair_df=2, support_tau=5.0,
            pair_registry=registry,
        )
        t2 = build_dependency_table(
            pair_matrix, labels, class_df, class_doc_counts,
            vectorizer.index_to_feature_, minimum_pair_df=2, support_tau=5.0,
            pair_registry=registry,
        )
        assert t1.equals(t2)

    def test_centered_deltas_sum_zero(self, vectorized):
        texts, labels, vectorizer, counts = vectorized
        class_df, class_doc_counts, _ = class_feature_document_frequency(
            counts, labels
        )
        pair_matrix, registry = build_pair_matrix(
            texts, vectorizer.feature_to_index_
        )
        table = build_dependency_table(
            pair_matrix, labels, class_df, class_doc_counts,
            vectorizer.index_to_feature_, minimum_pair_df=2, support_tau=5.0,
            pair_registry=registry,
        )
        if len(table) > 0:
            total = table["delta_0"] + table["delta_1"]
            np.testing.assert_allclose(total, 0.0, atol=1e-12)


class TestRedundancy:
    def test_candidate_selection_pure_chinese(self, vectorized):
        _, labels, vectorizer, counts = vectorized
        class_df, class_doc_counts, _ = class_feature_document_frequency(
            counts, labels
        )
        mi = feature_mutual_information(class_df, class_doc_counts)
        candidates = select_redundancy_candidates(
            vectorizer.index_to_feature_, mi, 10
        )
        assert len(candidates) <= 10
        for index in candidates:
            assert vectorizer.index_to_feature_[index].isascii() is False


class TestEvaluation:
    def test_perfect_predictions(self):
        from hanbayes.evaluation import evaluate_binary_classification

        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        metrics, confusion = evaluate_binary_classification(
            y_true, y_pred, y_score=[0.1, 0.9, 0.2, 0.8]
        )
        assert metrics.iloc[0]["Accuracy"] == 1.0
        assert metrics.iloc[0]["Macro_F1"] == 1.0
        assert metrics.iloc[0]["AUC"] == 1.0

    def test_mcnemar_identical_predictions(self):
        from hanbayes.evaluation import mcnemar_exact_test

        y_true = [0, 1, 0, 1, 0, 1]
        preds = [0, 1, 1, 1, 0, 0]
        result = mcnemar_exact_test(y_true, preds, preds, "A", "B")
        assert result["p_value_exact_two_sided"] == 1.0
        assert result["discordant_pairs"] == 0


class TestAnalyzer:
    def test_fit_evaluates_all_models(self, fitted_analyzer, tiny_dataset):
        texts, labels = tiny_dataset
        report = fitted_analyzer.evaluate(texts, labels)
        assert list(report.frame["model"]) == [
            "StandardNB", "FWNB", "DFWNB-v2", "SDFWNB",
        ]
        assert report.frame["Accuracy"].max() > 0.9

    def test_predict_and_proba_consistent(self, fitted_analyzer):
        texts = ["房间干净服务好", "房间太差态度恶劣"]
        preds = fitted_analyzer.predict(texts)
        proba = fitted_analyzer.predict_proba(texts)
        assert preds[0] == 1
        assert preds[1] == 0
        assert proba[0] > proba[1]

    def test_explanation_decomposition_consistent(self, fitted_analyzer):
        expl = fitted_analyzer.explainer.explain_prediction(
            "房间干净服务好", top_n=10
        )
        # scores field already is the full log-score; check keys and types
        assert set(expl) >= {
            "feature_evidence", "dependency_effects", "probabilities",
            "predicted_class", "scores",
        }
        assert len(expl["probabilities"]) == 2
        assert expl["predicted_class"] in (0, 1)
        # feature evidence sums (plus dependency effects and prior) must
        # equal the final score for the linear family
        feature_sum = np.array(expl["log_prior"])
        for item in expl["feature_evidence"]:
            feature_sum = feature_sum + np.array(
                [item["evidence_negative"], item["evidence_positive"]]
            )
        # NOTE: feature_evidence is truncated to top_n; with the tiny corpus
        # few features exist so the top-10 covers everything.
        dependency_sum = np.zeros(2)
        for item in expl["dependency_effects"]:
            dependency_sum = dependency_sum + np.array(
                [item["effect_negative"], item["effect_positive"]]
            )
        counts_row = fitted_analyzer.vectorizer.transform(["房间干净服务好"])
        assert counts_row.nnz <= 10  # top_n covers all features
        np.testing.assert_allclose(
            feature_sum + dependency_sum,
            np.array(expl["scores"]),
            atol=1e-9,
        )

    def test_persistence_roundtrip(self, fitted_analyzer, tmp_path):
        from hanbayes.persistence import load_analyzer, save_analyzer

        path = tmp_path / "model.pkl"
        save_analyzer(fitted_analyzer, path)
        restored = load_analyzer(path)
        texts = ["房间干净服务好", "价格贵体验糟糕"]
        np.testing.assert_allclose(
            fitted_analyzer.predict_proba(texts),
            restored.predict_proba(texts),
        )
