# Changelog

All notable changes to HanBayes are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/);
versioning follows [Semantic Versioning](https://semver.org/lang/zh-CN/).

## [2.0.0] - 2026-09-07

HanBayes debut: the package formerly released as `dfwnb` was redesigned and
renamed. The algorithm family and the paper-frozen numbers are unchanged and
remain bit-exact reproducible.

### Added

- **Single-scan sparse architecture**: `CharNgramVectorizer` converts the
  corpus to one CSR count matrix shared by all four models; scoring is fully
  vectorized (`counts @ weighted_log_prob.T`).
- **`ChineseSentimentAnalyzer`** high-level API: one `fit()` trains
  StandardNB / FWNB / DFWNB-v2 / SDFWNB; `evaluate()` returns an
  `EvaluationReport`; `predict` / `predict_proba` for raw texts.
- **`ExplanationEngine`**: exact, approximation-free per-prediction
  attributions (feature evidence + dependency effects) plus global tables
  (`top_features`, `top_dependencies`).
- **Model persistence**: `save_analyzer` / `load_analyzer` (pickle envelope).
- **`hanbayes explain` CLI** subcommand for per-prediction JSON explanations.
- **DependencyPairRegistry**: bidirectional pair↔column mapping so
  evaluation-time pair matrices are remapped onto the training column space;
  pairs absent from training are dropped (never wrapped around).
- English-only output column names across `evaluation` metrics, McNemar and
  bootstrap tables (international-friendly CSVs).
- Project hardening: GitHub Actions CI (Python 3.9–3.13 × 3 OS, lint + tests
  + build), ruff configuration, `py.typed`, CONTRIBUTING.md, issue/PR
  templates, Dependabot, EditorConfig, MIT license under HanBayes
  Contributors, bilingual README with benchmark/architecture/explanation
  figures, API reference, changelog.
- Hardened dataset downloader: request timeout, User-Agent, retry with
  exponential back-off, corrupt-file cleanup on checksum failure.

### Changed

- Package renamed `dfwnb` → `hanbayes`; CLI entry point `dfwnb` → `hanbayes`.
- `DependencyNB.decision_function` now takes an optional pair matrix and
  applies the per-document top-k cap with position-exact bookkeeping
  (correct on unsorted CSR rows after column remapping).
- scipy added as a dependency (sparse linear algebra); the models themselves
  remain from-scratch (no sklearn anywhere).
- Full final-test pipeline now runs in ~40 s end-to-end (fit + evaluate ~11 s),
  vs minutes for the 1.0 pure-Python loop implementation.

### Fixed

- **Negative-index wraparound** in evaluation-time pair-matrix remapping
  (scipy CSR treats −1 as "last column", silently injecting corrections for
  pairs the model never learned).
- **Top-k cap on unsorted rows**: the per-document pair cap previously
  mis-selected via `searchsorted` when remapped rows were not sorted.
- `explain_prediction` for SDFWNB now routes through the same remapped pair
  matrix as scoring, so explanations and predictions are strictly consistent.

### Migration from dfwnb 1.x

```python
# old
from dfwnb.pipeline import run_frozen_pipeline
# new
from hanbayes.pipeline import run_frozen_pipeline          # same signature
# or the higher-level API
from hanbayes import ChineseSentimentAnalyzer
```

Model-name keys in evaluation outputs are unchanged
(`StandardNB` / `FWNB` / `DFWNB-v2` / `SDFWNB`); only the statistical-table
column names became English (see above).

## [1.0.0] - 2026-09-07 (as `dfwnb`)

- Initial from-scratch release: StandardNB / FWNB / DFWNB-v2 / SDFWNB,
  two-stage cleaning protocol, frozen-config pipeline, CLI, 33 tests.
  Reproduced the paper numbers exactly (SDFWNB test: Acc=0.8073,
  Macro-F1=0.8065, AUC=0.8867).
