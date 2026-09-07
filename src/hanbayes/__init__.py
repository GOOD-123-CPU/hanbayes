"""HanBayes: Interpretable Bayesian models for Chinese sentiment analysis.

A from-scratch (numpy/pandas only) toolkit implementing a family of
interpretability-first Naive Bayes variants:

- ``StandardNB``   multinomial Naive Bayes with Laplace smoothing;
- ``MIWeightedNB`` FWNB  - mutual-information discriminative weighting;
- DFWNB-v2 weights - document-level Jaccard redundancy suppression;
- ``DependencyNB`` SDFWNB - sparse class-conditional local dependency
  correction for adjacent character-bigram pairs.

Design principle: every text is converted to a **sparse count matrix exactly
once** per model; training, weighting and prediction all consume that matrix.
A single fit produces all four models for comparison.
"""

__version__ = "2.0.0"

from hanbayes.analyzer import ChineseSentimentAnalyzer
from hanbayes.features import CharNgramVectorizer
from hanbayes.models import DependencyNB, MIWeightedNB, StandardNB
from hanbayes.text import TextNormalizer

__all__ = [
    "StandardNB",
    "MIWeightedNB",
    "DependencyNB",
    "ChineseSentimentAnalyzer",
    "TextNormalizer",
    "CharNgramVectorizer",
]
