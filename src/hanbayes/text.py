"""Text normalization utilities.

``TextNormalizer`` reproduces the normalization protocol validated in the
original research:

1. Unicode NFKC normalization;
2. lowercase ASCII;
3. strip HTML tags;
4. unify URLs / e-mails / digit runs into placeholders (网址/邮箱/数字);
5. unify ASCII punctuation into Chinese punctuation;
6. collapse repeated punctuation;
7. remove whitespace;
8. replace unrecognized characters with "，";
9. strip leading/trailing "，。；：、" (faithful to the original protocol).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

COMMON_PUNCTUATION = frozenset("，。！？；：、")

PLACEHOLDERS = ("数字", "网址", "邮箱")

_CHINESE_RANGE = r"\u3400-\u4dbf\u4e00-\u9fff"

_HTML_TAG_RE = re.compile(r"<[^>]*>")
_URL_RE = re.compile(r"https?://[^\s]+|www\.[^\s]+")
_EMAIL_RE = re.compile(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}")
_DIGIT_RE = re.compile(r"\d+(?:[.,]\d+)*")
_WHITESPACE_RE = re.compile(r"\s+")
_SPECIAL_RE = re.compile(r"[^a-z0-9" + _CHINESE_RANGE + r"，。！？；：、]+")
_PURE_CHINESE_BIGRAM_RE = re.compile(r"[" + _CHINESE_RANGE + r"]{2}")

_ASCII_TO_CHINESE: tuple[tuple[str, str], ...] = (
    (",", "，"),
    (".", "。"),
    ("!", "！"),
    ("?", "？"),
    (";", "；"),
    (":", "："),
)

_STRIP_CHARS = "，。；：、"


@dataclass
class TextNormalizer:
    """Stateless normalizer; call :meth:`normalize` or :meth:`transform`."""

    def normalize(self, text) -> str:
        if pd.isna(text):
            return ""
        out = str(text)

        out = unicodedata.normalize("NFKC", out)
        out = out.lower()
        out = _HTML_TAG_RE.sub("", out)
        out = _URL_RE.sub("网址", out)
        out = _EMAIL_RE.sub("邮箱", out)
        out = _DIGIT_RE.sub("数字", out)

        for ascii_punct, chinese_punct in _ASCII_TO_CHINESE:
            out = out.replace(ascii_punct, chinese_punct)

        for punct in ("，", "。", "！", "？", "；", "："):
            out = re.sub(re.escape(punct) + "+", punct, out)

        out = _WHITESPACE_RE.sub("", out)
        out = _SPECIAL_RE.sub("，", out)
        out = re.sub("，+", "，", out)
        out = out.strip(_STRIP_CHARS)
        return out

    def transform(self, texts) -> list[str]:
        return [self.normalize(t) for t in texts]


def is_pure_chinese_bigram(feature: object) -> bool:
    """True when *feature* is exactly two Chinese characters."""

    if not isinstance(feature, str):
        return False
    return bool(_PURE_CHINESE_BIGRAM_RE.fullmatch(feature))


def have_shared_character(feature_a: str, feature_b: str) -> bool:
    """True when two bigrams share at least one character.

    Shared-character bigrams are usually complementary slices of the same
    longer word; the redundancy model must not treat them as redundant.
    """

    return not set(feature_a).isdisjoint(set(feature_b))
