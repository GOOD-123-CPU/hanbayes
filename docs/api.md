# API Reference

All public objects are importable from the top-level `hanbayes` namespace.
The package ships a `py.typed` marker; every public API is fully type-annotated.

```python
import hanbayes
hanbayes.__version__   # "2.0.0"
```

---

## High-level API

### `ChineseSentimentAnalyzer`

The recommended entry point. One `fit()` trains the full model family.

```python
from hanbayes import ChineseSentimentAnalyzer

analyzer = ChineseSentimentAnalyzer.from_config()  # paper-frozen config
analyzer.fit(train_texts, train_labels)
report = analyzer.evaluate(dev_texts, dev_labels)
```

| Member | Description |
|---|---|
| `from_config(config_path=None)` | Build an analyzer from a frozen JSON config (defaults to the bundled one). |
| `fit(texts, labels)` | Train StandardNB / FWNB / DFWNB-v2 / SDFWNB. Texts are scanned **once**; all models share the resulting CSR matrix. |
| `evaluate(texts, labels, model_names=None)` | Evaluate (a subset of) the family; returns an `EvaluationReport`. |
| `predict(texts, model_name="SDFWNB")` | 0/1 labels for raw texts. |
| `predict_proba(texts, model_name="SDFWNB")` | Positive-class probability for raw texts. |
| `models` | Dict of the four fitted model objects, keyed by name. |
| `vectorizer` | The fitted `CharNgramVectorizer` (vocabulary source of truth). |
| `explainer` | The `ExplanationEngine` for attributions. |
| `metadata_` | Dict: sample counts, vocabulary size, redundancy/dependency statistics. |

### `EvaluationReport`

Container returned by `evaluate()`:

| Attribute | Description |
|---|---|
| `frame` | `pandas.DataFrame` with per-model metrics (Accuracy, per-class P/R/F1, Macro-*, Balanced_Accuracy, AUC). |
| `predictions` | Dict: model name → predicted labels. |
| `probabilities` | Dict: model name → positive-class probabilities. |
| `confusion` | Dict: model name → confusion matrix. |
| `to_string()` | Pretty-printed metrics table. |

---

## Models

All models operate on sparse CSR count matrices.

```python
from hanbayes import StandardNB, MIWeightedNB, DependencyNB
```

| Class | Signature | Notes |
|---|---|---|
| `StandardNB` | `fit(counts, labels)` | Multinomial NB + Laplace smoothing. |
| `MIWeightedNB` | `fit(counts, labels, feature_weights=None)` | FWNB: per-feature log evidence scaled by weights. Unit weights degrade **exactly** to StandardNB (tested). |
| `DependencyNB` | `fit(counts, labels, feature_weights, pair_matrix, pair_registry, dependency_corrections, ...)`; `decision_function(counts, pair_matrix=None)` | SDFWNB. Zero strength degrades **exactly** to the weighted base (tested). |

Shared model API: `predict(counts)`, `predict_proba(counts)`,
`decision_function(...)`, attributes `classes_`, `log_class_prior_`,
`feature_weights_` (weighted variants).

---

## Feature extraction

```python
from hanbayes import CharNgramVectorizer, TextNormalizer
from hanbayes.features import extract_char_ngrams
```

| Class / Function | Description |
|---|---|
| `TextNormalizer` | `.normalize(text)` / `.transform(texts)`: HTML strip, NFKC, lowercase, digit/URL/email placeholders, punctuation folding, edge-punctuation strip. |
| `CharNgramVectorizer(ngram_range, min_df, max_features)` | `.fit(texts)`, `.transform(texts)`, `.fit_transform(texts)` → CSR count matrix. Vocabulary ordered by `(df↓, tf↓, lex↑)` — deterministic across machines. |
| `extract_char_ngrams(text, ngram_range)` | Raw character N-gram list (pure-punctuation grams skipped). |

---

## Interpretability

```python
from hanbayes import ExplanationEngine
```

Access it via `analyzer.explainer` after `fit()`.

| Method | Description |
|---|---|
| `explain_prediction(text, model_name="SDFWNB", top_n=10)` | Exact decomposition: log-prior + per-feature evidence + per-dependency effects → final scores/probabilities. Returns a JSON-serializable dict. |
| `top_features(top_n=20, by="mutual_information", class_filter=None)` | Global discriminative-feature table with MI, direction, FWNB/DFWNB weights, redundancy score. |
| `top_dependencies(top_n=20)` | Strongest class-conditional dependency trigrams with per-class deltas. |

**Guarantee**: explanations and predictions come from the same code path —
the decomposition sums exactly to the model's log-scores (unit-tested).

---

## Pipeline & persistence

```python
from hanbayes.pipeline import run_frozen_pipeline, significance_tests
from hanbayes.persistence import save_analyzer, load_analyzer
from hanbayes.configs import load_frozen_config
```

| Function | Description |
|---|---|
| `run_frozen_pipeline(training_texts, training_labels, evaluation_texts, evaluation_labels, config=None, model_names=None, save_models_to=None)` | Train + evaluate the full family under a frozen config. Returns `results` (DataFrame), `report`, `analyzer`, `metadata`, timing. |
| `significance_tests(y_true, predictions, bootstrap_count=5000, random_seed=142)` | McNemar exact tests + paired bootstrap CIs for the canonical model pairs. |
| `save_analyzer(analyzer, path)` / `load_analyzer(path)` | Pickle round-trip of a fitted analyzer (verified by tests). |
| `load_frozen_config(path=None)` | Load a frozen configuration JSON. |

---

## Data

```python
from hanbayes.data.download import download_dataset, verify_checksum
from hanbayes.data.cleaning import clean_and_prepare
from hanbayes.data.io import read_dataset_file, standardize_dataset_columns
```

| Function | Description |
|---|---|
| `download_dataset(data_dir)` | Download the three ChnSentiCorp TSV splits with SHA-256 verification, timeout, UA header and retry/back-off. The dataset itself is **not** redistributed. |
| `verify_checksum(path, expected)` | Streaming SHA-256 check. |
| `standardize_dataset_columns(df, split)` | Normalize column layout (`label` / `text` / `source_split`). |
| `clean_and_prepare(train, dev, test)` | Two-stage cleaning protocol (raw-text level + model-text level): removes label conflicts, duplicates, cross-split leakage with test > dev > train priority. |

---

## Error handling

- `ValueError` — invalid configuration or malformed inputs (messages in English/Chinese).
- `RuntimeError` — calling `predict*` before `fit`, or download failures after retries.
- `FileNotFoundError` — missing data files (CLI catches and prints a hint).
