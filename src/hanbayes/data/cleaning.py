"""Two-stage data cleaning, faithfully reproducing the notebook protocol.

Stage 1 (raw text, via ``normalize_text_for_duplicate_check`` keys):

1. remove within-split label-conflict texts (whole groups);
2. remove within-split duplicates (keep first occurrence);
3. remove dev texts that also appear in test (test has priority);
4. remove train texts that also appear in dev or test.

Stage 2 (model text, via ``normalize_text_for_model`` output):

1. drop rows whose model text is empty;
2. remove model-text label conflicts (whole groups);
3. remove model-text duplicates (keep first);
4. remove dev model texts overlapping test; then train model texts
   overlapping the (cleaned) dev or test sets.

The result contains ``label``, ``text``, ``model_text`` columns per split.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from ..text import TextNormalizer

_NORMALIZER = TextNormalizer()


def normalize_text_for_model(text) -> str:
    """Model-input normalization (thin wrapper over TextNormalizer)."""

    return _NORMALIZER.normalize(text)


def normalize_text_for_duplicate_check(text) -> str:
    """NFKC + lowercase + whitespace strip, for duplicate detection only."""

    if pd.isna(text):
        return ""
    normalized = str(text)
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = normalized.lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _add_text_key(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["text_key"] = data["text"].map(normalize_text_for_duplicate_check)
    return data


def _conflict_keys(data: pd.DataFrame) -> set:
    label_counts = data.groupby("text_key")["label"].nunique(dropna=True)
    return set(label_counts[label_counts > 1].index)


def _remove_keys(
    data: pd.DataFrame, keys_to_remove: set, removal_reason: str
):
    keys_to_remove = set(keys_to_remove)
    if len(keys_to_remove) == 0:
        remove_mask = pd.Series(False, index=data.index)
    else:
        remove_mask = data["text_key"].isin(keys_to_remove)

    removed = data.loc[remove_mask].copy()
    removed["removal_reason"] = removal_reason
    kept = data.loc[~remove_mask].copy()
    return kept, removed


def _remove_internal_duplicates(data: pd.DataFrame, removal_reason: str):
    duplicate_mask = data.duplicated(subset=["text_key"], keep="first")
    removed = data.loc[duplicate_mask].copy()
    removed["removal_reason"] = removal_reason
    kept = data.loc[~duplicate_mask].copy()
    return kept, removed


def _clean_raw_split(data: pd.DataFrame, split_name: str):
    data = _add_text_key(data)

    data, removed_conflicts = _remove_keys(
        data, _conflict_keys(data), f"{split_name}内部同一文本对应不同标签"
    )
    data, removed_duplicates = _remove_internal_duplicates(
        data, f"{split_name}内部重复文本"
    )
    return data, removed_conflicts, removed_duplicates


def clean_and_prepare(train_data, dev_data, test_data):
    """Run the full two-stage cleaning protocol.

    Parameters are the standardized DataFrames (``label`` / ``text`` columns)
    for the three splits. Returns a dict with ``train`` / ``dev`` / ``test``
    ready DataFrames and a summary DataFrame of both cleaning stages.
    """

    splits = {"train": train_data, "dev": dev_data, "test": test_data}

    # ---------------------------------------------------------------
    # Stage 1: raw-text level cleaning
    # ---------------------------------------------------------------
    raw_clean: dict = {}
    removed_raw: dict = {}
    for split_name, data in splits.items():
        kept, removed_conflicts, removed_duplicates = _clean_raw_split(
            data, split_name
        )
        raw_clean[split_name] = kept
        removed_raw[split_name] = (removed_conflicts, removed_duplicates)

    # dev vs test overlap (test priority)
    test_keys = set(raw_clean["test"]["text_key"])
    dev_overlap_keys = set(raw_clean["dev"]["text_key"]) & test_keys
    raw_clean["dev"], removed_dev_test_overlap = _remove_keys(
        raw_clean["dev"], dev_overlap_keys, "验证集文本同时出现在测试集中"
    )

    # train vs holdout overlap
    holdout_keys = set(raw_clean["dev"]["text_key"]) | set(
        raw_clean["test"]["text_key"]
    )
    train_overlap_keys = set(raw_clean["train"]["text_key"]) & holdout_keys
    raw_clean["train"], removed_train_holdout_overlap = _remove_keys(
        raw_clean["train"], train_overlap_keys,
        "训练集文本同时出现在验证集或测试集中",
    )

    raw_summary = pd.DataFrame(
        [
            {
                "数据集": "训练集",
                "原始记录数": len(splits["train"]),
                "删除内部冲突记录数": len(removed_raw["train"][0]),
                "删除内部重复记录数": len(removed_raw["train"][1]),
                "删除跨集合重叠记录数": len(removed_train_holdout_overlap),
                "清洗后记录数": len(raw_clean["train"]),
            },
            {
                "数据集": "验证集",
                "原始记录数": len(splits["dev"]),
                "删除内部冲突记录数": len(removed_raw["dev"][0]),
                "删除内部重复记录数": len(removed_raw["dev"][1]),
                "删除跨集合重叠记录数": len(removed_dev_test_overlap),
                "清洗后记录数": len(raw_clean["dev"]),
            },
            {
                "数据集": "测试集",
                "原始记录数": len(splits["test"]),
                "删除内部冲突记录数": len(removed_raw["test"][0]),
                "删除内部重复记录数": len(removed_raw["test"][1]),
                "删除跨集合重叠记录数": 0,
                "清洗后记录数": len(raw_clean["test"]),
            },
        ]
    )
    raw_summary["总删除记录数"] = (
        raw_summary["原始记录数"] - raw_summary["清洗后记录数"]
    )

    # ---------------------------------------------------------------
    # Stage 2: model-text level cleaning
    # ---------------------------------------------------------------
    model_data: dict = {}
    for split_name, data in raw_clean.items():
        data = data.copy().reset_index(drop=True)
        data["model_text"] = data["text"].map(normalize_text_for_model)
        model_data[split_name] = data

    def clean_internal_model_text(data: pd.DataFrame, split_name: str):
        working = data.copy()

        empty_mask = working["model_text"] == ""
        removed_empty = working.loc[empty_mask].copy()
        removed_empty["model_removal_reason"] = (
            f"{split_name}规范化后为空文本"
        )
        working = working.loc[~empty_mask].copy()

        label_counts = working.groupby("model_text")["label"].nunique()
        conflict_texts = set(label_counts[label_counts > 1].index)
        conflict_mask = working["model_text"].isin(conflict_texts)
        removed_conflicts = working.loc[conflict_mask].copy()
        removed_conflicts["model_removal_reason"] = (
            f"{split_name}规范化后同一文本对应不同标签"
        )
        working = working.loc[~conflict_mask].copy()

        duplicate_mask = working.duplicated(subset=["model_text"], keep="first")
        removed_duplicates = working.loc[duplicate_mask].copy()
        removed_duplicates["model_removal_reason"] = (
            f"{split_name}规范化后重复文本"
        )
        working = working.loc[~duplicate_mask].copy()

        return working.reset_index(drop=True), {
            "empty": len(removed_empty),
            "conflict": len(removed_conflicts),
            "duplicate": len(removed_duplicates),
        }

    model_clean: dict = {}
    removed_model: dict = {}
    for split_name, data in model_data.items():
        model_clean[split_name], removed_model[split_name] = (
            clean_internal_model_text(data, split_name)
        )

    # dev vs test model-text overlap (test priority)
    test_model_texts = set(model_clean["test"]["model_text"])
    dev_model_overlap_mask = model_clean["dev"]["model_text"].isin(test_model_texts)
    removed_dev_model_overlap = int(dev_model_overlap_mask.sum())
    model_clean["dev"] = model_clean["dev"].loc[~dev_model_overlap_mask].copy()
    model_clean["dev"] = model_clean["dev"].reset_index(drop=True)

    # train vs holdout model-text overlap
    protected_model_texts = set(model_clean["dev"]["model_text"]) | set(
        model_clean["test"]["model_text"]
    )
    train_overlap_mask = model_clean["train"]["model_text"].isin(
        protected_model_texts
    )
    removed_train_model_overlap = int(train_overlap_mask.sum())
    model_clean["train"] = model_clean["train"].loc[~train_overlap_mask].copy()
    model_clean["train"] = model_clean["train"].reset_index(drop=True)

    model_summary = pd.DataFrame(
        [
            {
                "数据集": "训练集",
                "进入模型级清洗前记录数": len(model_data["train"]),
                "删除空文本数": removed_model["train"]["empty"],
                "删除模型级标签冲突数": removed_model["train"]["conflict"],
                "删除模型级重复数": removed_model["train"]["duplicate"],
                "删除模型级跨集合重叠数": removed_train_model_overlap,
                "最终记录数": len(model_clean["train"]),
            },
            {
                "数据集": "验证集",
                "进入模型级清洗前记录数": len(model_data["dev"]),
                "删除空文本数": removed_model["dev"]["empty"],
                "删除模型级标签冲突数": removed_model["dev"]["conflict"],
                "删除模型级重复数": removed_model["dev"]["duplicate"],
                "删除模型级跨集合重叠数": removed_dev_model_overlap,
                "最终记录数": len(model_clean["dev"]),
            },
            {
                "数据集": "测试集",
                "进入模型级清洗前记录数": len(model_data["test"]),
                "删除空文本数": removed_model["test"]["empty"],
                "删除模型级标签冲突数": removed_model["test"]["conflict"],
                "删除模型级重复数": removed_model["test"]["duplicate"],
                "删除模型级跨集合重叠数": 0,
                "最终记录数": len(model_clean["test"]),
            },
        ]
    )
    model_summary["模型级总删除数"] = (
        model_summary["进入模型级清洗前记录数"] - model_summary["最终记录数"]
    )

    ready = {
        split_name: data[["label", "text", "model_text"]].reset_index(drop=True)
        for split_name, data in model_clean.items()
    }

    return {
        "ready": ready,
        "raw_summary": raw_summary,
        "model_summary": model_summary,
    }


def prepare_model_ready_datasets(train_data, dev_data, test_data):
    """Backwards-compatible alias for :func:`clean_and_prepare`."""

    return clean_and_prepare(train_data, dev_data, test_data)
