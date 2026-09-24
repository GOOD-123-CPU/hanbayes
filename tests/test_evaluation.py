import math

import numpy as np
import pytest

from hanbayes.evaluation import (
    calculate_binary_auc,
    mcnemar_exact_test,
    paired_bootstrap_metric_difference,
)


def test_auc_is_half_when_all_scores_are_tied():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    assert calculate_binary_auc(y_true, y_score) == pytest.approx(0.5)


def test_auc_is_one_for_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert calculate_binary_auc(y_true, y_score) == pytest.approx(1.0)


def test_auc_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        calculate_binary_auc([0, 1], [0.2])


def test_auc_is_nan_for_single_class_input():
    assert math.isnan(calculate_binary_auc([1, 1, 1], [0.1, 0.2, 0.3]))


def test_mcnemar_no_discordant_pairs_returns_p_one():
    result = mcnemar_exact_test(
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        "a",
        "b",
    )
    assert result["discordant_pairs"] == 0
    assert result["p_value_exact_two_sided"] == pytest.approx(1.0)
    assert result["significant_at_0.05"] is False


def test_mcnemar_known_exact_probability():
    # A right/B wrong once; A wrong/B right three times -> n=4, min=1.
    # Two-sided exact p = 2 * P[Binom(4, .5) <= 1] = 0.625.
    y_true = np.array([1, 1, 1, 1])
    pred_a = np.array([1, 0, 0, 0])
    pred_b = np.array([0, 1, 1, 1])
    result = mcnemar_exact_test(y_true, pred_a, pred_b, "a", "b")
    assert result["a_correct_b_wrong"] == 1
    assert result["a_wrong_b_correct"] == 3
    assert result["discordant_pairs"] == 4
    assert result["p_value_exact_two_sided"] == pytest.approx(0.625)


def test_paired_bootstrap_is_reproducible_for_same_seed():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    old = np.array([0, 1, 1, 0, 1, 0, 0, 0])
    new = np.array([0, 0, 1, 1, 1, 0, 0, 0])

    first = paired_bootstrap_metric_difference(
        y_true, old, new, bootstrap_count=250, random_seed=123
    )
    second = paired_bootstrap_metric_difference(
        y_true, old, new, bootstrap_count=250, random_seed=123
    )

    assert first == second
    assert first["bootstrap_resamples"] == 250
    assert first["random_seed"] == 123
    assert first["ci_95_lower"] <= first["mean_macro_f1_difference"] <= first["ci_95_upper"]
