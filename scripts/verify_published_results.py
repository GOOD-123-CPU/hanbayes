"""Verify consistency of committed HanBayes result artifacts.

This script does not retrain models. It checks that published CSV evidence is
internally consistent with the frozen experiment configuration and with basic
metric invariants. A fresh training run remains the stronger reproduction test.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "frozen.json"
PACKAGED_CONFIG_PATH = ROOT / "src" / "hanbayes" / "frozen.json"
METRICS_PATH = ROOT / "results" / "final_test_metrics.csv"
MCNEMAR_PATH = ROOT / "results" / "final_test_mcnemar.csv"
BOOTSTRAP_PATH = ROOT / "results" / "final_test_bootstrap.csv"

EXPECTED_MODELS = ["StandardNB", "FWNB", "DFWNB-v2", "SDFWNB"]
UNIT_INTERVAL_COLUMNS = [
    "Accuracy",
    "Precision_positive",
    "Recall_positive",
    "F1_positive",
    "Precision_negative",
    "Recall_negative",
    "F1_negative",
    "Macro_Precision",
    "Macro_Recall",
    "Macro_F1",
    "Balanced_Accuracy",
    "AUC",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise AssertionError(f"invalid boolean value: {value!r}")


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    packaged = json.loads(PACKAGED_CONFIG_PATH.read_text(encoding="utf-8"))
    assert config == packaged, "configs/frozen.json and packaged frozen.json differ"

    expected_seed = (
        int(config["random_seed"])
        + int(config["final_test"]["bootstrap_seed_offset"])
    )
    expected_resamples = int(config["final_test"]["bootstrap_resamples"])
    assert config["final_test"]["train_dev_merge"] is True

    metrics = read_csv(METRICS_PATH)
    models = [row["model"] for row in metrics]
    assert models == EXPECTED_MODELS, f"unexpected model order/set: {models}"
    for row in metrics:
        for column in UNIT_INTERVAL_COLUMNS:
            value = float(row[column])
            assert 0.0 <= value <= 1.0, f"{row['model']} {column} out of [0,1]: {value}"
        # Balanced Accuracy and Macro Recall are algebraically identical here.
        assert abs(float(row["Balanced_Accuracy"]) - float(row["Macro_Recall"])) < 1e-12

    mcnemar = read_csv(MCNEMAR_PATH)
    assert mcnemar, "McNemar artifact must not be empty"
    for row in mcnemar:
        b = int(row["a_correct_b_wrong"])
        c = int(row["a_wrong_b_correct"])
        discordant = int(row["discordant_pairs"])
        p_value = float(row["p_value_exact_two_sided"])
        significant = parse_bool(row["significant_at_0.05"])
        assert discordant == b + c
        assert 0.0 <= p_value <= 1.0
        assert significant == (p_value < 0.05)

    bootstrap = read_csv(BOOTSTRAP_PATH)
    assert bootstrap, "bootstrap artifact must not be empty"
    for row in bootstrap:
        lower = float(row["ci_95_lower"])
        upper = float(row["ci_95_upper"])
        mean = float(row["mean_macro_f1_difference"])
        excludes_zero = parse_bool(row["ci_excludes_zero"])
        assert int(row["bootstrap_resamples"]) == expected_resamples
        assert int(row["random_seed"]) == expected_seed
        assert lower <= upper
        assert lower <= mean <= upper
        assert excludes_zero == ((lower > 0.0) or (upper < 0.0))
        assert row["model_old"] in EXPECTED_MODELS
        assert row["model_new"] in EXPECTED_MODELS

    print(
        "published-result contract OK: "
        f"{len(metrics)} metric rows, {len(mcnemar)} McNemar rows, "
        f"{len(bootstrap)} bootstrap rows; seed={expected_seed}, "
        f"resamples={expected_resamples}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
