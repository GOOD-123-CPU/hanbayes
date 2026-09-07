"""Shared fixtures: tiny synthetic Chinese sentiment data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

POSITIVE_TEXTS = [
    "这家酒店非常好服务热情房间干净",
    "位置很好交通方便早餐丰富",
    "环境优雅安静舒适值得推荐",
    "性价比超高下次还会再来",
    "房间宽敞明亮员工态度好",
]

NEGATIVE_TEXTS = [
    "房间太差态度恶劣不会再来了",
    "价格贵体验糟糕非常失望",
    "设施老旧噪音很大睡眠不好",
    "卫生条件差前台态度冷淡",
    "位置偏僻服务慢早餐难吃",
]


@pytest.fixture
def tiny_dataset():
    texts = POSITIVE_TEXTS + NEGATIVE_TEXTS
    labels = [1] * len(POSITIVE_TEXTS) + [0] * len(NEGATIVE_TEXTS)
    # repeat to give the vocabulary enough document frequency support
    texts = texts * 4
    labels = labels * 4
    rng = np.random.default_rng(42)
    order = rng.permutation(len(texts))
    texts = [texts[i] for i in order]
    labels = [labels[i] for i in order]
    return texts, np.asarray(labels, dtype=int)


@pytest.fixture
def fitted_analyzer(tiny_dataset):
    """A ChineseSentimentAnalyzer fitted on the tiny synthetic data."""

    from hanbayes.analyzer import ChineseSentimentAnalyzer

    texts, labels = tiny_dataset
    analyzer = ChineseSentimentAnalyzer.from_config()
    analyzer.fit(texts, labels)
    return analyzer
