"""Generate README figure assets from the reproduction results.

Run from the repository root with the project venv:

    python scripts/generate_assets.py

Outputs into docs/assets/ (English labels only, to avoid CJK font issues).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ASSET_DIR = REPO_ROOT / "docs" / "assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

ACCENT = "#2563eb"
ACCENT_LIGHT = "#93c5fd"
GREY = "#9ca3af"
DARK = "#111827"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.sans-serif": [
            "DejaVu Sans", "Microsoft YaHei", "SimHei", "PingFang SC",
            "Noto Sans CJK SC", "WenQuanYi Zen Hei",
        ],
        "axes.unicode_minus": False,
        "axes.edgecolor": "#d1d5db",
        "axes.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def benchmark_figure(metrics: pd.DataFrame) -> None:
    models = metrics["model"].tolist()
    accuracy = metrics["Accuracy"].to_numpy()
    macro_f1 = metrics["Macro_F1"].to_numpy()
    auc = metrics["AUC"].to_numpy()

    x = np.arange(len(models))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=200)

    colors = [GREY, ACCENT_LIGHT, ACCENT_LIGHT, ACCENT]
    bars_auc = ax.bar(x + width, auc, width, label="AUC", color=colors)
    bars_f1 = ax.bar(x, macro_f1, width, label="Macro-F1", color=colors)
    bars_acc = ax.bar(x - width, accuracy, width, label="Accuracy", color=colors)

    for bars in (bars_acc, bars_f1, bars_auc):
        for rect in bars:
            height = rect.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.2,
                color=DARK,
            )

    ax.set_ylim(0.70, 0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score (ChnSentiCorp test, n=1178)")
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "benchmark.png", bbox_inches="tight")
    plt.close(fig)


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(10.2, 4.6), dpi=200)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge="#1f2937", fs=9, tc="white"):
        rect = plt.Rectangle((x, y), w, h, facecolor=face, edgecolor=edge,
                             linewidth=1.1, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=tc, zorder=3, linespacing=1.5)

    def arrow(x1, y1, x2, y2, text=None, color="#374151"):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.3),
            zorder=1,
        )
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.06, text, ha="center",
                    fontsize=7.5, color="#4b5563")

    # input layer
    box(0.0, 0.52, 0.15, 0.30, "Raw Chinese\ntexts", "#f3f4f6", tc=DARK)
    # normalize + vectorize
    box(0.22, 0.52, 0.19, 0.30,
        "TextNormalizer\n+\nCharNgramVectorizer\n(one scan)", "#1e3a8a")
    # single matrix
    box(0.48, 0.56, 0.16, 0.22, "CSR count\nmatrix C", "#2563eb")
    # models
    box(0.71, 0.90, 0.28, 0.14, "StandardNB   (baseline)", "#4b5563", fs=8.5)
    box(0.71, 0.72, 0.28, 0.14, "FWNB   (MI weights)", "#4b5563", fs=8.5)
    box(0.71, 0.54, 0.28, 0.14, "DFWNB-v2   (+ redundancy\nsuppression)", "#4b5563", fs=8.5)
    box(0.71, 0.33, 0.28, 0.16,
        "SDFWNB   (+ sparse local\ndependency correction)", "#111827", fs=8.5)
    # outputs
    box(0.30, 0.10, 0.40, 0.14,
        "Metrics / McNemar / Bootstrap   |   Exact per-prediction explanations",
        "#f3f4f6", tc=DARK, fs=8)

    arrow(0.15, 0.67, 0.22, 0.67)
    arrow(0.41, 0.67, 0.48, 0.67)
    arrow(0.64, 0.67, 0.71, 0.97)
    arrow(0.64, 0.67, 0.71, 0.79)
    arrow(0.64, 0.67, 0.71, 0.61)
    arrow(0.64, 0.67, 0.71, 0.41)
    arrow(0.85, 0.90, 0.85, 0.33, color="#9ca3af")
    arrow(0.85, 0.24, 0.85, 0.10)
    ax.plot([0.71, 0.50], [0.24, 0.24], color="#9ca3af", linewidth=1.3)
    ax.plot([0.50, 0.50], [0.24, 0.17], color="#9ca3af", linewidth=1.3)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.12)

    fig.tight_layout()
    fig.savefig(ASSET_DIR / "architecture.png", bbox_inches="tight")
    plt.close(fig)


def _cjk_capable_font() -> str | None:
    """Return a font family name that can render CJK on this machine."""

    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in (
        "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
    ):
        if candidate in installed:
            return candidate
    return None


def explanation_figure() -> None:
    """Waterfall-style exact attribution for one example prediction."""

    from hanbayes.persistence import load_analyzer

    model_path = REPO_ROOT / "results" / "hanbayes_model.pkl"
    if not model_path.exists():
        print("skip explanation figure (no saved model; run final-test --save-model first)")
        return

    analyzer = load_analyzer(model_path)
    expl = analyzer.explainer.explain_prediction(
        "房间干净服务好但隔音太差", model_name="SDFWNB", top_n=6
    )

    labels = [it["feature"] for it in expl["feature_evidence"][:5]] + [
        "dep: " + it["trigram"] for it in expl["dependency_effects"][:3]
    ]
    # evidence toward positive minus toward negative = net push
    values = [
        it["evidence_positive"] - it["evidence_negative"] for it in expl["feature_evidence"][:5]
    ] + [
        it["effect_positive"] - it["effect_negative"] for it in expl["dependency_effects"][:3]
    ]

    order = np.argsort([abs(v) for v in values])[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    cjk_font = _cjk_capable_font()
    title_text = (
        'Exact attribution: "房间干净服务好但隔音太差" -> positive (SDFWNB)'
        if cjk_font
        else "Exact attribution of a mixed-review example (SDFWNB)"
    )

    fig, ax = plt.subplots(figsize=(8.4, 3.6), dpi=200)
    colors = ["#dc2626" if v < 0 else "#2563eb" for v in values]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62)
    for yi, v in zip(y, values):
        ax.text(v + (0.05 if v >= 0 else -0.05), yi, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontfamily=cjk_font or "DejaVu Sans")
    ax.axvline(0, color="#9ca3af", linewidth=0.8)
    ax.set_xlabel("Net evidence toward positive class (log-scale, SDFWNB)")
    ax.set_title(title_text, fontsize=10,
                 fontfamily=cjk_font or "DejaVu Sans")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "explanation.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    metrics_path = REPO_ROOT / "results" / "final_test_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    benchmark_figure(metrics)
    architecture_figure()
    explanation_figure()
    print(f"assets written to {ASSET_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
