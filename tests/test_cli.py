"""CLI smoke tests (download-free)."""

from __future__ import annotations

import pytest

from hanbayes.cli import build_parser


def test_top_level_imports_match_readme_examples():
    from hanbayes import (
        CharNgramVectorizer,
        ChineseSentimentAnalyzer,
        DependencyNB,
        MIWeightedNB,
        StandardNB,
        TextNormalizer,
    )

    assert ChineseSentimentAnalyzer.__name__ == "ChineseSentimentAnalyzer"
    assert StandardNB.__name__ == "StandardNB"
    assert MIWeightedNB.__name__ == "MIWeightedNB"
    assert DependencyNB.__name__ == "DependencyNB"
    assert TextNormalizer.__name__ == "TextNormalizer"
    assert CharNgramVectorizer.__name__ == "CharNgramVectorizer"


class TestParser:
    def test_all_commands_present(self):
        parser = build_parser()
        # parsing each subcommand with --help raises SystemExit(0)
        for command in ("download", "run", "final-test", "explain", "info"):
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args([command, "--help"])
            assert exc_info.value.code == 0

    def test_run_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["run"])
        assert args.data_dir == "data/raw"
        assert args.out_dir == "results"

    def test_final_test_skip_bootstrap(self):
        parser = build_parser()
        args = parser.parse_args(["final-test", "--skip-bootstrap"])
        assert args.skip_bootstrap is True

    def test_explain_requires_model_path(self):
        parser = build_parser()
        args = parser.parse_args(["explain", "--model-path", "x.pkl"])
        assert args.model == "SDFWNB"
        assert args.top == 10
