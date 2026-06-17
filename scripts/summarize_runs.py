#!/usr/bin/env python
"""Collect train_summary.json files into a compact comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GPR risk model runs.")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="Run directories containing train_summary.json.")
    parser.add_argument("--out", type=Path, default=Path("outputs/source_data/run_comparison.csv"))
    return parser.parse_args()


def pick(summary: dict, split: str, metric: str):
    obj = summary.get(split, {})
    if metric in obj:
        return obj[metric]
    if metric.startswith("threshold_"):
        return obj.get("threshold_metrics", {}).get(metric.replace("threshold_", ""))
    if metric.startswith("cost_"):
        return obj.get("min_expected_cost", {}).get(metric.replace("cost_", ""))
    return None


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "model",
        "method",
        "best_epoch",
        "selected_threshold_from_val",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_average_precision",
        "test_auc",
        "test_threshold_precision",
        "test_threshold_recall",
        "test_threshold_fpr",
        "test_threshold_f1",
        "test_threshold_fp",
        "test_threshold_fn",
        "test_cost_expected_cost_per_sample",
    ]
    rows = []
    for run in args.runs:
        summary_path = run / "train_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "run": str(run),
            "model": summary.get("model"),
            "method": summary.get("method"),
            "best_epoch": summary.get("best_epoch"),
            "selected_threshold_from_val": summary.get("selected_threshold_from_val"),
        }
        for field in fields[5:]:
            split, metric = field.split("_", 1)
            row[field] = pick(summary, split, metric)
        rows.append(row)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out": str(args.out), "n": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
