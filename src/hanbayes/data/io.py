"""Robust TSV reading and column standardization for ChnSentiCorp splits."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd


def read_dataset_file(file_path) -> pd.DataFrame:
    """Read a ``label<TAB>text_a`` TSV file robustly.

    Tries utf-8-sig / utf-8 / gb18030 encodings, uses tab separators with
    quoting disabled (review text may contain quote characters), and validates
    that the required columns ``label`` and ``text_a`` exist.
    """

    last_error: Exception | None = None

    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            data = pd.read_csv(
                file_path,
                sep="\t",
                encoding=encoding,
                quoting=csv.QUOTE_NONE,
                dtype={"label": "int64", "text_a": "string"},
            )
            break
        except (UnicodeDecodeError, ValueError) as error:
            last_error = error
    else:
        raise ValueError(f"无法读取数据文件 {file_path}：{last_error}")

    required_columns = {"label", "text_a"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(
            f"数据文件 {file_path} 缺少必需列：{sorted(missing)}"
        )

    return data


def standardize_dataset_columns(data: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Normalize columns to ``label`` / ``text`` / ``source_split``.

    - column names are lowercased and stripped;
    - candidate label columns: label / labels / sentiment / y;
    - candidate text columns: text_a / text / review / content;
    - labels must be exactly 0 or 1;
    - rows with missing labels or texts are dropped.
    """

    working = data.copy()
    working.columns = [str(column).strip().lower() for column in working.columns]

    label_candidates = ["label", "labels", "sentiment", "y"]
    text_candidates = ["text_a", "text", "review", "content"]

    label_column = next(
        (column for column in label_candidates if column in working.columns), None
    )
    text_column = next(
        (column for column in text_candidates if column in working.columns), None
    )

    if label_column is None or text_column is None:
        raise ValueError(
            f"{split_name}：找不到标签列或文本列。现有列：{list(working.columns)}"
        )

    working = working[[label_column, text_column]].copy()
    working.columns = ["label", "text"]

    working["label"] = pd.to_numeric(working["label"], errors="coerce")
    working = working.dropna(subset=["label", "text"]).copy()
    working["label"] = working["label"].astype("int64")
    working["text"] = working["text"].astype("string")

    invalid_labels = set(np.unique(working["label"].to_numpy())) - {0, 1}
    if invalid_labels:
        raise ValueError(f"{split_name}：发现非法标签值 {sorted(invalid_labels)}，只允许0和1")

    working["source_split"] = split_name

    return working.reset_index(drop=True)
