"""HanBayes pipeline: frozen-configuration training and evaluation.

``run_frozen_pipeline`` reproduces the paper protocol exactly: it trains the
full model family on the given training texts and evaluates all four models
on the evaluation split, returning per-model metrics plus artifacts.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .analyzer import ChineseSentimentAnalyzer, EvaluationReport
from .configs import load_frozen_config
from .evaluation import (
    mcnemar_exact_test,
    paired_bootstrap_metric_difference,
)
from .persistence import save_analyzer

__all__ = [
    "run_frozen_pipeline",
    "significance_tests",
    "ChineseSentimentAnalyzer",
    "EvaluationReport",
    "load_frozen_config",
    "save_analyzer",
]


def run_frozen_pipeline(
    training_texts,
    training_labels,
    evaluation_texts,
    evaluation_labels,
    config: dict | None = None,
    model_names=None,
    save_models_to=None,
) -> dict:
    """Train the full family and evaluate under the frozen configuration."""

    config = config or load_frozen_config()
    analyzer = ChineseSentimentAnalyzer(config=config)

    start_time = time.perf_counter()
    analyzer.fit(training_texts, training_labels)
    fit_time = time.perf_counter() - start_time

    report = analyzer.evaluate(evaluation_texts, evaluation_labels,
                               model_names=model_names)

    output = {
        "results": report.frame,
        "report": report,
        "analyzer": analyzer,
        "fit_time_seconds": fit_time,
        "metadata": dict(analyzer.metadata_,
                         evaluation_sample_count=len(evaluation_labels)),
    }

    if save_models_to is not None:
        output["model_path"] = save_analyzer(analyzer, save_models_to, config)

    return output


def significance_tests(
    y_true,
    predictions: dict,
    bootstrap_count: int = 5000,
    random_seed: int = 142,
) -> dict:
    """McNemar exact tests + paired bootstrap CIs for model pairs."""

    y_true = np.asarray(y_true, dtype=int)

    mcnemar_rows = [
        mcnemar_exact_test(y_true, predictions["StandardNB"],
                           predictions["SDFWNB"], "StandardNB", "SDFWNB"),
        mcnemar_exact_test(y_true, predictions["FWNB"],
                           predictions["DFWNB-v2"], "FWNB", "DFWNB-v2"),
        mcnemar_exact_test(y_true, predictions["DFWNB-v2"],
                           predictions["SDFWNB"], "DFWNB-v2", "SDFWNB"),
    ]

    bootstrap_rows = []
    for old, new in (("StandardNB", "FWNB"), ("FWNB", "DFWNB-v2"),
                     ("DFWNB-v2", "SDFWNB"), ("StandardNB", "SDFWNB")):
        result = paired_bootstrap_metric_difference(
            y_true=y_true,
            predictions_old=predictions[old],
            predictions_new=predictions[new],
            bootstrap_count=bootstrap_count,
            random_seed=random_seed,
        )
        result["model_old"] = old
        result["model_new"] = new
        bootstrap_rows.append(result)

    return {
        "mcnemar": pd.DataFrame(mcnemar_rows),
        "bootstrap": pd.DataFrame(bootstrap_rows),
    }
