"""Binary classification metrics and significance tests, from scratch."""

from __future__ import annotations

import numpy as np
import pandas as pd


def safe_divide(numerator, denominator) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_binary_auc(y_true, y_score) -> float:
    """Rank-sum AUC with average ranks for tied scores."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(y_true) != len(y_score):
        raise ValueError("y_true与y_score长度不一致")

    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))

    if positive_count == 0 or negative_count == 0:
        return float("nan")

    sorted_indices = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[sorted_indices]

    ranks = np.zeros(len(y_score), dtype=float)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while (
            end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]
        ):
            end += 1
        average_rank = ((start + 1) + end) / 2.0  # 1-based average rank
        ranks[sorted_indices[start:end]] = average_rank
        start = end

    positive_rank_sum = ranks[y_true == 1].sum()
    auc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)

    return float(auc)


def evaluate_binary_classification(y_true, y_pred, y_score=None):
    """Compute a metrics DataFrame and a confusion-matrix DataFrame."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true与y_pred长度不一致")

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = safe_divide(tp + tn, tp + tn + fp + fn)

    precision_1 = safe_divide(tp, tp + fp)
    recall_1 = safe_divide(tp, tp + fn)
    f1_1 = safe_divide(2 * precision_1 * recall_1, precision_1 + recall_1)

    precision_0 = safe_divide(tn, tn + fn)
    recall_0 = safe_divide(tn, tn + fp)
    f1_0 = safe_divide(2 * precision_0 * recall_0, precision_0 + recall_0)

    macro_precision = (precision_0 + precision_1) / 2.0
    macro_recall = (recall_0 + recall_1) / 2.0
    macro_f1 = (f1_0 + f1_1) / 2.0
    balanced_accuracy = (recall_0 + recall_1) / 2.0

    auc = float("nan")
    if y_score is not None:
        auc = calculate_binary_auc(y_true, y_score)

    metrics = pd.DataFrame(
        [
            {
                "Accuracy": accuracy,
                "Precision_positive": precision_1,
                "Recall_positive": recall_1,
                "F1_positive": f1_1,
                "Precision_negative": precision_0,
                "Recall_negative": recall_0,
                "F1_negative": f1_0,
                "Macro_Precision": macro_precision,
                "Macro_Recall": macro_recall,
                "Macro_F1": macro_f1,
                "Balanced_Accuracy": balanced_accuracy,
                "AUC": auc,
            }
        ]
    )

    confusion_matrix = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["true_negative_0", "true_positive_1"],
        columns=["pred_negative_0", "pred_positive_1"],
    )

    return metrics, confusion_matrix


def stable_softmax(log_scores) -> np.ndarray:
    log_scores = np.asarray(log_scores, dtype=np.float64)
    maximum_scores = np.max(log_scores, axis=1, keepdims=True)
    exp_scores = np.exp(log_scores - maximum_scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def exact_binomial_lower_tail(
    number_of_successes: int,
    number_of_trials: int,
    success_probability: float = 0.5,
) -> float:
    """P(X <= number_of_successes) for a binomial distribution (no scipy)."""

    if number_of_trials < 0:
        raise ValueError("试验次数不能为负数")
    if not (0 <= number_of_successes <= number_of_trials):
        raise ValueError("成功次数范围错误")
    if not (0 < success_probability < 1):
        raise ValueError("成功概率必须位于0和1之间")

    probability_of_zero = (1.0 - success_probability) ** number_of_trials
    cumulative_probability = probability_of_zero
    current_probability = probability_of_zero

    for k in range(1, number_of_successes + 1):
        current_probability *= (
            (number_of_trials - k + 1)
            / k
            * success_probability
            / (1.0 - success_probability)
        )
        cumulative_probability += current_probability

    return float(cumulative_probability)


def mcnemar_exact_test(
    y_true,
    predictions_a,
    predictions_b,
    model_a_name: str,
    model_b_name: str,
) -> dict:
    """Two-sided exact McNemar test comparing two models' predictions."""

    y_true = np.asarray(y_true, dtype=int)
    predictions_a = np.asarray(predictions_a, dtype=int)
    predictions_b = np.asarray(predictions_b, dtype=int)

    correct_a = predictions_a == y_true
    correct_b = predictions_b == y_true

    b_count = int(np.sum(correct_a & (~correct_b)))  # A right, B wrong
    c_count = int(np.sum((~correct_a) & correct_b))  # A wrong, B right

    discordant_count = b_count + c_count

    if discordant_count == 0:
        p_value = 1.0
    else:
        smaller_count = min(b_count, c_count)
        one_side_probability = exact_binomial_lower_tail(
            number_of_successes=smaller_count,
            number_of_trials=discordant_count,
            success_probability=0.5,
        )
        p_value = min(1.0, 2.0 * one_side_probability)

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "a_correct_b_wrong": b_count,
        "a_wrong_b_correct": c_count,
        "discordant_pairs": discordant_count,
        "p_value_exact_two_sided": p_value,
        "significant_at_0.05": p_value < 0.05,
    }


def calculate_macro_f1_only(y_true, y_pred) -> float:
    """Macro-F1 only (used inside bootstrap resampling for speed)."""

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision_positive = safe_divide(tp, tp + fp)
    recall_positive = safe_divide(tp, tp + fn)
    f1_positive = safe_divide(
        2 * precision_positive * recall_positive,
        precision_positive + recall_positive,
    )

    precision_negative = safe_divide(tn, tn + fn)
    recall_negative = safe_divide(tn, tn + fp)
    f1_negative = safe_divide(
        2 * precision_negative * recall_negative,
        precision_negative + recall_negative,
    )

    return (f1_positive + f1_negative) / 2.0


def paired_bootstrap_metric_difference(
    y_true,
    predictions_old,
    predictions_new,
    bootstrap_count: int = 3000,
    random_seed: int = 42,
) -> dict:
    """Paired bootstrap 95% CI of Macro-F1(new) - Macro-F1(old)."""

    y_true = np.asarray(y_true, dtype=int)
    predictions_old = np.asarray(predictions_old, dtype=int)
    predictions_new = np.asarray(predictions_new, dtype=int)

    sample_count = len(y_true)
    random_generator = np.random.default_rng(random_seed)

    differences = np.zeros(bootstrap_count, dtype=np.float64)

    for bootstrap_index in range(bootstrap_count):
        sampled_indices = random_generator.integers(
            low=0, high=sample_count, size=sample_count
        )
        sampled_true = y_true[sampled_indices]
        sampled_old = predictions_old[sampled_indices]
        sampled_new = predictions_new[sampled_indices]

        differences[bootstrap_index] = calculate_macro_f1_only(
            sampled_true, sampled_new
        ) - calculate_macro_f1_only(sampled_true, sampled_old)

    lower_bound = float(np.quantile(differences, 0.025))
    upper_bound = float(np.quantile(differences, 0.975))

    return {
        "mean_macro_f1_difference": float(differences.mean()),
        "ci_95_lower": lower_bound,
        "ci_95_upper": upper_bound,
        "ci_excludes_zero": (lower_bound > 0) or (upper_bound < 0),
        "bootstrap_resamples": bootstrap_count,
        "random_seed": random_seed,
    }
