"""One-click reproducible pipeline script.

Usage (from the repository root)::

    python scripts/run_pipeline.py              # train/dev protocol
    python scripts/run_pipeline.py --final      # final test protocol
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a source checkout without installation.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from hanbayes.cli import main as cli_main  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final", action="store_true",
        help="run the final test protocol (train+dev merged, evaluate on test)",
    )
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results"))
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args()

    command = "final-test" if args.final else "run"
    argv = [command, "--data-dir", args.data_dir, "--out-dir", args.out_dir]
    if args.progress:
        argv.append("--progress")
    if args.skip_bootstrap:
        argv.append("--skip-bootstrap")

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
