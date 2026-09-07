"""Tests for the data cleaning protocol."""

from __future__ import annotations

import pandas as pd

from hanbayes.data.cleaning import (
    clean_and_prepare,
    normalize_text_for_duplicate_check,
)
from hanbayes.data.io import standardize_dataset_columns


def _make_df(rows, split="train"):
    data = pd.DataFrame(rows, columns=["label", "text_a"])
    return standardize_dataset_columns(data, split)


class TestStandardization:
    def test_column_mapping(self):
        data = _make_df([(1, "很好"), (0, "很差")], "train")
        assert list(data.columns) == ["label", "text", "source_split"]
        assert data["label"].tolist() == [1, 0]

    def test_invalid_label_raises(self):
        import pytest

        with pytest.raises(ValueError):
            _make_df([(2, "很好")], "train")


class TestCleaning:
    def test_duplicate_removed_keep_first(self):
        train = _make_df([(1, "很好"), (1, "很好"), (0, "很差")])
        dev = _make_df([(0, "还行")], "dev")
        test = _make_df([(1, "不错")], "test")

        result = clean_and_prepare(train, dev, test)
        assert len(result["ready"]["train"]) == 2

    def test_label_conflict_group_removed(self):
        train = _make_df([(1, "很好"), (0, "很好"), (0, "很差")])
        dev = _make_df([(0, "还行")], "dev")
        test = _make_df([(1, "不错")], "test")

        result = clean_and_prepare(train, dev, test)
        ready_train = result["ready"]["train"]
        assert "很好" not in ready_train["text"].tolist()
        assert ready_train["text"].tolist() == ["很差"]

    def test_cross_split_overlap_removed_from_lower_priority(self):
        train = _make_df([(1, "很好"), (0, "很差")])
        dev = _make_df([(0, "很差"), (0, "还行")], "dev")   # overlaps test
        test = _make_df([(1, "很差"), (1, "不错")], "test")  # priority

        result = clean_and_prepare(train, dev, test)
        assert "很差" not in result["ready"]["dev"]["text"].tolist()
        assert "很差" in result["ready"]["test"]["text"].tolist()
        # train must not overlap the holdout either
        assert "很差" not in result["ready"]["train"]["text"].tolist()

    def test_model_text_collision_dedup(self):
        # different raw texts -> same model text after digit normalization
        train = _make_df([(1, "好评100分"), (1, "好评200分"), (0, "差评")])
        dev = _make_df([(0, "还行")], "dev")
        test = _make_df([(1, "不错")], "test")

        result = clean_and_prepare(train, dev, test)
        ready_train = result["ready"]["train"]
        assert len(ready_train) == 2
        assert ready_train["model_text"].nunique() == 2

    def test_empty_model_text_removed(self):
        train = _make_df([(1, "。。。"), (0, "很差")])
        dev = _make_df([(0, "还行")], "dev")
        test = _make_df([(1, "不错")], "test")

        result = clean_and_prepare(train, dev, test)
        assert "。。。" not in result["ready"]["train"]["text"].tolist()


class TestDupKey:
    def test_case_and_whitespace_insensitive(self):
        assert (
            normalize_text_for_duplicate_check("Good  Hotel")
            == normalize_text_for_duplicate_check("good hotel")
        )
