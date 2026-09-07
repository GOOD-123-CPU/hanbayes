"""HanBayes command-line interface.

Commands
--------
``hanbayes download``    download and checksum-verify the dataset;
``hanbayes run``         train/dev protocol over the full model family;
``hanbayes final-test``  paper protocol: merge train+dev, evaluate on test;
``hanbayes explain``     per-prediction attribution for raw texts;
``hanbayes info``        show the frozen configuration.

Examples
--------
    hanbayes download --data-dir data/raw
    hanbayes run --data-dir data/raw --out-dir results
    hanbayes final-test --data-dir data/raw --out-dir results
    hanbayes explain --model-path results/hanbayes_model.pkl --text "房间干净服务好"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .configs import default_config_path, load_frozen_config


def command_download(args) -> int:
    from .data.download import download_dataset

    data_dir = Path(args.data_dir)
    download_dataset(data_dir)
    print(f"数据集就绪：{data_dir}")
    return 0


def _prepare_ready_datasets(args):
    """Load raw TSVs, standardize columns and run the cleaning protocol."""

    from .data.cleaning import clean_and_prepare
    from .data.io import read_dataset_file, standardize_dataset_columns

    data_dir = Path(args.data_dir)

    raw = {}
    for split_name, filename in (
        ("train", "train.tsv"),
        ("dev", "dev.tsv"),
        ("test", "test.tsv"),
    ):
        file_path = data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"找不到 {file_path}。请先运行 hanbayes download 或手动放置数据文件。"
            )
        raw[split_name] = standardize_dataset_columns(
            read_dataset_file(file_path), split_name
        )

    prepared = clean_and_prepare(raw["train"], raw["dev"], raw["test"])

    if args.verbose:
        print("== 原始文本级清洗 ==")
        print(prepared["raw_summary"].to_string(index=False))
        print()
        print("== 模型文本级清洗 ==")
        print(prepared["model_summary"].to_string(index=False))
        print()

    print(
        "清洗后规模：train={}/{}, dev={}/{}, test={}/{}".format(
            len(prepared["ready"]["train"]), len(raw["train"]),
            len(prepared["ready"]["dev"]), len(raw["dev"]),
            len(prepared["ready"]["test"]), len(raw["test"]),
        )
    )
    return prepared


def _save_outputs(out_dir: Path, output: dict, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    output["results"].to_csv(
        out_dir / f"{prefix}_metrics.csv", index=False, encoding="utf-8-sig"
    )
    with open(out_dir / f"{prefix}_meta.json", "w", encoding="utf-8") as file:
        json.dump(output["metadata"], file, ensure_ascii=False, indent=4,
                  default=str)


def command_run(args) -> int:
    from .pipeline import run_frozen_pipeline

    config = load_frozen_config(args.config)
    prepared = _prepare_ready_datasets(args)
    ready = prepared["ready"]

    print("Training model family (train/dev protocol) ...")
    output = run_frozen_pipeline(
        training_texts=ready["train"]["model_text"].tolist(),
        training_labels=ready["train"]["label"].astype(int).to_numpy(),
        evaluation_texts=ready["dev"]["model_text"].tolist(),
        evaluation_labels=ready["dev"]["label"].astype(int).to_numpy(),
        config=config,
    )

    print()
    print(output["results"].to_string(index=False))
    _save_outputs(Path(args.out_dir), output, "validation")
    print(f"\nResults saved to {args.out_dir}")
    return 0


def command_final_test(args) -> int:
    import pandas as pd

    from .pipeline import run_frozen_pipeline, significance_tests

    config = load_frozen_config(args.config)
    prepared = _prepare_ready_datasets(args)
    ready = prepared["ready"]

    final_config = config.get("final_test", {})
    bootstrap_count = int(final_config.get("bootstrap_resamples", 5000))
    bootstrap_seed = config.get("random_seed", 42) + int(
        final_config.get("bootstrap_seed_offset", 100)
    )

    merged_train = pd.concat([ready["train"], ready["dev"]], ignore_index=True)

    print(
        f"Final test protocol: train({len(ready['train'])}) + dev({len(ready['dev'])})"
        f" = {len(merged_train)} merged for training; "
        f"single-shot evaluation on test({len(ready['test'])})."
    )

    output = run_frozen_pipeline(
        training_texts=merged_train["model_text"].tolist(),
        training_labels=merged_train["label"].astype(int).to_numpy(),
        evaluation_texts=ready["test"]["model_text"].tolist(),
        evaluation_labels=ready["test"]["label"].astype(int).to_numpy(),
        config=config,
        save_models_to=Path(args.out_dir) / "hanbayes_model.pkl"
        if args.save_model else None,
    )

    print()
    print("== Final test results ==")
    print(output["results"].to_string(index=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_outputs(out_dir, output, "final_test")

    y_true = ready["test"]["label"].astype(int).to_numpy()
    tests = significance_tests(
        y_true,
        output["report"].predictions,
        bootstrap_count=bootstrap_count if not args.skip_bootstrap else 2,
        random_seed=bootstrap_seed,
    )

    print()
    print("== McNemar exact tests ==")
    print(tests["mcnemar"].to_string(index=False))
    tests["mcnemar"].to_csv(
        out_dir / "final_test_mcnemar.csv", index=False, encoding="utf-8-sig"
    )

    if not args.skip_bootstrap:
        print()
        print(f"== Paired bootstrap ({bootstrap_count} resamples, seed {bootstrap_seed}) ==")
        print(tests["bootstrap"].to_string(index=False))
        tests["bootstrap"].to_csv(
            out_dir / "final_test_bootstrap.csv", index=False, encoding="utf-8-sig"
        )

    print(f"\nResults saved to {out_dir}")
    return 0


def command_explain(args) -> int:
    from .persistence import load_analyzer

    analyzer = load_analyzer(args.model_path)

    texts = list(args.text) if args.text else [sys.stdin.read().strip()]
    for text in texts:
        explanation = analyzer.explainer.explain_prediction(
            text, model_name=args.model, top_n=args.top
        )
        print(json.dumps(explanation, ensure_ascii=False, indent=2))
    return 0


def command_info(args) -> int:
    config = load_frozen_config(args.config)
    print(json.dumps(config, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hanbayes",
        description=(
            "HanBayes: interpretable Bayesian models for Chinese sentiment "
            "(StandardNB / FWNB / DFWNB-v2 / SDFWNB)"
        ),
    )
    parser.add_argument(
        "--config",
        default=str(default_config_path()),
        help="frozen configuration JSON (defaults to the bundled file)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub):
        sub.add_argument("--data-dir", default="data/raw", help="raw TSV directory")
        sub.add_argument("--out-dir", default="results", help="output directory")
        sub.add_argument("-v", "--verbose", action="store_true", help="verbose output")

    sub_download = subparsers.add_parser("download", help="download dataset + verify")
    sub_download.add_argument("--data-dir", default="data/raw")
    sub_download.set_defaults(func=command_download)

    sub_run = subparsers.add_parser("run", help="train/dev protocol")
    add_common(sub_run)
    sub_run.set_defaults(func=command_run)

    sub_final = subparsers.add_parser(
        "final-test", help="final protocol (train+dev merged, test evaluation)"
    )
    add_common(sub_final)
    sub_final.add_argument("--skip-bootstrap", action="store_true")
    sub_final.add_argument("--save-model", action="store_true",
                           help="persist the fitted analyzer to out-dir")
    sub_final.set_defaults(func=command_final_test)

    sub_explain = subparsers.add_parser(
        "explain", help="explain predictions for raw texts"
    )
    sub_explain.add_argument("--model-path", required=True,
                             help="path to a saved hanbayes_model.pkl")
    sub_explain.add_argument("--text", nargs="+", help="texts to explain")
    sub_explain.add_argument("--model", default="SDFWNB",
                             choices=["StandardNB", "FWNB", "DFWNB-v2", "SDFWNB"])
    sub_explain.add_argument("--top", type=int, default=10,
                             help="top features/effects to show")
    sub_explain.set_defaults(func=command_explain)

    sub_info = subparsers.add_parser("info", help="show frozen configuration")
    sub_info.set_defaults(func=command_info)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2
    except Exception as error:  # pragma: no cover - defensive CLI guard
        print(f"错误：{error}", file=sys.stderr)
        if getattr(args, "verbose", False):
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
