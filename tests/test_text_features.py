"""Tests for text normalization, vectorizer and feature statistics."""

from __future__ import annotations

import numpy as np

from hanbayes.features import CharNgramVectorizer, extract_char_ngrams
from hanbayes.text import TextNormalizer, is_pure_chinese_bigram

_NORMALIZER = TextNormalizer()


def normalize_text_for_model(text):
    return _NORMALIZER.normalize(text)


class TestNormalization:
    def test_repeated_exclamation_collapsed(self):
        result = normalize_text_for_model("酒店位置不错！！！但是房间太小了……")
        assert result == "酒店位置不错！但是房间太小了"

    def test_digits_and_lowercase(self):
        assert normalize_text_for_model("CPU是60nm的，价格550.00元。") == (
            "cpu是数字nm的，价格数字元"
        )

    def test_html_stripped(self):
        assert normalize_text_for_model("<p>服务很好</p>，下次还会再来。") == (
            "服务很好，下次还会再来"
        )

    def test_url_placeholder(self):
        assert normalize_text_for_model(
            "详情请访问 https://example.com/product?id=123 看看"
        ) == "详情请访问网址看看"

    def test_nan_returns_empty(self):
        import pandas as pd

        assert normalize_text_for_model(pd.NA) == ""
        assert normalize_text_for_model(float("nan")) == ""

    def test_leading_trailing_punctuation_stripped(self):
        # faithful to the original notebook: only ，。；：、 are stripped
        assert normalize_text_for_model("。好的，") == "好的"
        assert normalize_text_for_model("、好的。") == "好的"

    def test_transform_matches_normalize(self):
        texts = ["很好！！！", "很差。。。", "一般般"]
        assert list(_NORMALIZER.transform(texts)) == [
            normalize_text_for_model(t) for t in texts
        ]


class TestPureChineseBigram:
    def test_true(self):
        assert is_pure_chinese_bigram("不满")

    def test_false_ascii(self):
        assert not is_pure_chinese_bigram("a满")

    def test_false_wrong_length(self):
        assert not is_pure_chinese_bigram("不满 ")
        assert not is_pure_chinese_bigram("满")


class TestCharNgrams:
    def test_bigrams(self):
        assert extract_char_ngrams("服务不好", (2, 2)) == ["服务", "务不", "不好"]

    def test_unigram_plus_bigram(self):
        result = extract_char_ngrams("服务", (1, 2))
        assert result == ["服", "务", "服务"]

    def test_pure_punctuation_grams_excluded(self):
        # "，。" produces only punctuation grams which must be skipped
        assert extract_char_ngrams("，。", (1, 2)) == []

    def test_short_text(self):
        assert extract_char_ngrams("好", (2, 2)) == []


class TestVectorizer:
    def test_min_df_filter_and_order(self):
        vectorizer = CharNgramVectorizer((2, 2), min_df=2, max_features=10)
        vectorizer.fit(["好的好的", "好的很好", "很差很差"])
        # "好的" appears in 2 docs, "很差" in 1 doc -> filtered by min_df
        assert "好的" in vectorizer.feature_to_index_
        assert "很差" not in vectorizer.feature_to_index_
        assert vectorizer.feature_to_index_["好的"] == 0

    def test_max_features(self):
        vectorizer = CharNgramVectorizer((2, 2), min_df=1, max_features=5)
        vectorizer.fit(["甲乙丙丁", "乙丙丁戊", "丙丁戊己", "丁戊己庚"])
        assert vectorizer.vocabulary_size_ == 5

    def test_deterministic(self):
        texts = ["房间干净服务好", "房间太差服务慢", "位置好交通方便"]
        v1 = CharNgramVectorizer((2, 2), 1, 100).fit(texts)
        v2 = CharNgramVectorizer((2, 2), 1, 100).fit(texts)
        assert v1.feature_to_index_ == v2.feature_to_index_

    def test_transform_count_matrix(self):
        vectorizer = CharNgramVectorizer((2, 2), min_df=1, max_features=None)
        texts = ["服务好服务好", "服务差"]
        counts = vectorizer.fit_transform(texts)
        row0 = counts.getrow(0)
        feature = vectorizer.feature_to_index_["服务"]
        assert row0[0, feature] == 2  # term frequency preserved

    def test_transform_unseen_features_ignored(self):
        vectorizer = CharNgramVectorizer((2, 2), min_df=1, max_features=None)
        vectorizer.fit(["服务好"])
        counts = vectorizer.transform(["未知词汇完全"])
        assert counts.nnz == 0

    def test_rows_sorted_canonical(self):
        vectorizer = CharNgramVectorizer((2, 2), min_df=1, max_features=None)
        texts = ["房间干净服务好态度热情", "价格贵体验糟糕"]
        counts = vectorizer.fit_transform(texts)
        for row in range(counts.shape[0]):
            cols = counts.indices[counts.indptr[row]:counts.indptr[row + 1]]
            assert np.all(np.diff(cols) > 0), "row indices must be sorted"
