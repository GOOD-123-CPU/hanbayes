<!--
  Keep the PR focused on a single change. CI runs ruff + pytest on
  Python 3.9-3.13 across Linux/macOS/Windows.
-->

## What does this PR do? / 这个 PR 做了什么

<!-- One or two sentences. -->

## Type of change / 改动类型

- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (fix or feature that changes existing behavior/API)
- [ ] Documentation only
- [ ] Refactor / tooling

## Checklist / 检查清单

- [ ] Tests added/updated; `pytest tests/ -v` passes locally
- [ ] `ruff check src/ tests/ scripts/` passes
- [ ] Docstrings updated for public API changes
- [ ] **Frozen-numbers guarantee**: `hanbayes final-test` still reproduces
      `results/final_test_metrics.csv` exactly (required for anything that
      touches modeling, features, evaluation or the pipeline)
- [ ] No dataset contents committed (ChnSentiCorp stays out of git)
