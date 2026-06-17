#!/usr/bin/env python
"""Create a mixed-source all-source ImageFolder view."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List

from create_lodo_views import IMAGE_EXTS, LABELS, class_map, image_files, link_or_copy, safe_name


def parse_args():
    parser = argparse.ArgumentParser(description="Create an all-source GPR risk view.")
    parser.add_argument("--tigpr-view", type=Path, default=Path("datasets/tigpr_binary_damage"))
    parser.add_argument("--utility-view", type=Path, default=Path("datasets/utility_void_binary"))
    parser.add_argument("--urdd-view", type=Path, default=Path("datasets/urdd_multiclass"))
    parser.add_argument("--out-root", type=Path, default=Path("datasets/gpr_multisource_risk_allsource"))
    parser.add_argument("--copy", action="store_true")
    return parser.parse_args()


def add_split(view_root: Path, rows: List[Dict], modes: Counter, source: str, source_root: Path, split: str, copy: bool) -> None:
    split_root = source_root / split
    for cls_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        label = class_map(source, cls_dir.name)
        for src in image_files(cls_dir):
            dst = view_root / split / label / f"{safe_name(source)}__{safe_name(split)}__{safe_name(cls_dir.name)}__{safe_name(src.name)}"
            mode = link_or_copy(src, dst, copy)
            modes[mode] += 1
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "source": source,
                    "source_class": cls_dir.name,
                    "source_path": str(src.resolve()),
                    "dataset_path": str(dst.relative_to(view_root)),
                }
            )


def main() -> None:
    args = parse_args()
    if args.out_root.exists():
        shutil.rmtree(args.out_root)
    args.out_root.mkdir(parents=True, exist_ok=True)
    roots = {"tigpr": args.tigpr_view, "utility": args.utility_view, "urdd": args.urdd_view}
    rows: List[Dict] = []
    modes: Counter = Counter()
    for source, root in roots.items():
        for split in ("train", "val", "test"):
            add_split(args.out_root, rows, modes, source, root, split, args.copy)

    with (args.out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "label", "source", "source_class", "source_path", "dataset_path"])
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter((r["split"], r["label"]) for r in rows)
    summary = {
        "protocol": "mixed-source target-present risk recognition",
        "classes": LABELS,
        "num_images": len(rows),
        "counts": {split: {label: counts[(split, label)] for label in LABELS} for split in ("train", "val", "test")},
        "copy_modes": dict(modes),
    }
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
