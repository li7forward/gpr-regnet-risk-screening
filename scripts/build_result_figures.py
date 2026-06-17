#!/usr/bin/env python
"""Build common manuscript-ready figures from experiment CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build result figures.")
    parser.add_argument("--comparison-csv", type=Path, default=Path("outputs/source_data/run_comparison.csv"))
    parser.add_argument("--history", type=Path, nargs="*", default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/figures"))
    return parser.parse_args()


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def plot_comparison(csv_path: Path, out_dir: Path) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    metric_cols = ["test_balanced_accuracy", "test_average_precision", "test_auc", "test_threshold_recall", "test_threshold_fpr"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    if not metric_cols:
        return
    methods = df["model"].astype(str).tolist()
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(3.0 * len(metric_cols), 3.2), sharey=False)
    if len(metric_cols) == 1:
        axes = [axes]
    colors = ["#2f6f9f" if "gafr" not in m.lower() else "#c04b37" for m in methods]
    for ax, metric in zip(axes, metric_cols):
        ax.barh(methods, df[metric], color=colors, edgecolor="black", linewidth=0.4)
        ax.set_title(metric.replace("test_", "").replace("_", " "))
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout()
    savefig(out_dir / "strong_baseline_metrics.png")


def plot_histories(history_paths: list[Path], out_dir: Path) -> None:
    if not history_paths:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    metrics = [("train_loss", "training loss"), ("val_min_expected_cost", "validation cost"), ("val_average_precision", "validation AP")]
    palette = plt.cm.tab10.colors
    for idx, path in enumerate(history_paths):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        label = path.parent.name
        for ax, (metric, title) in zip(axes, metrics):
            if metric in df.columns:
                ax.plot(df["epoch"], df[metric], label=label, color=palette[idx % len(palette)], linewidth=1.8)
                ax.set_title(title)
                ax.set_xlabel("epoch")
                ax.grid(color="#e6e6e6", linewidth=0.6)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    savefig(out_dir / "training_histories.png")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(args.comparison_csv, args.out_dir)
    plot_histories(args.history, args.out_dir)


if __name__ == "__main__":
    main()
