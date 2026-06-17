#!/usr/bin/env python
"""Create leave-one-domain-out GPR image-level risk views."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LABELS = ["damage", "no_damage"]


def parse_args():
    parser = argparse.ArgumentParser(description="Create GPR multisource LODO risk views.")
    parser.add_argument("--tigpr-view", type=Path, default=Path("datasets/tigpr_binary_damage"))
    parser.add_argument("--utility-view", type=Path, default=Path("datasets/utility_void_binary"))
    parser.add_argument("--urdd-view", type=Path, default=Path("datasets/urdd_multiclass"))
    parser.add_argument("--out-root", type=Path, default=Path("datasets/gpr_multisource_risk_lodo"))
    parser.add_argument("--seed", type=int, default=2037)
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    return parser.parse_args()


def image_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))


def link_or_copy(src: Path, dst: Path, copy: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return "copy"
    try:
        dst.symlink_to(src.resolve())
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def class_map(source: str, cls_name: str) -> str:
    if source in {"tigpr", "utility"}:
        if cls_name in LABELS:
            return cls_name
        raise ValueError(f"Unexpected {source} class: {cls_name}")
    if source == "urdd":
        return "no_damage" if cls_name.lower() == "background" else "damage"
    raise ValueError(source)


def add_source_split(
    view_root: Path,
    rows: List[Dict],
    modes: Counter,
    source: str,
    source_root: Path,
    source_split: str,
    out_split: str,
    copy: bool,
) -> None:
    split_root = source_root / source_split
    for cls_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        label = class_map(source, cls_dir.name)
        for src in image_files(cls_dir):
            dst = view_root / out_split / label / (
                f"{safe_name(source)}__{safe_name(source_split)}__{safe_name(cls_dir.name)}__{safe_name(src.name)}"
            )
            mode = link_or_copy(src, dst, copy)
            modes[mode] += 1
            rows.append(
                {
                    "split": out_split,
                    "label": label,
                    "source": source,
                    "source_split": source_split,
                    "source_class": cls_dir.name,
                    "source_path": str(src.resolve()),
                    "dataset_path": str(dst.relative_to(view_root)),
                }
            )


def write_view(view_root: Path, rows: List[Dict], modes: Counter, heldout: str, train_sources: List[str]) -> Dict:
    counts = Counter((r["split"], r["label"]) for r in rows)
    source_counts = Counter((r["split"], r["source"]) for r in rows)
    class_source_counts = Counter((r["split"], r["label"], r["source"], r["source_class"]) for r in rows)
    with (view_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "label", "source", "source_split", "source_class", "source_path", "dataset_path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "view_root": str(view_root.resolve()),
        "protocol": "leave-one-domain-out target-present risk recognition",
        "heldout_test_domain": heldout,
        "train_val_domains": train_sources,
        "classes": LABELS,
        "num_images": len(rows),
        "counts": {split: {label: counts[(split, label)] for label in LABELS} for split in ("train", "val", "test")},
        "source_counts": {"|".join(k): v for k, v in sorted(source_counts.items())},
        "class_source_counts": {"|".join(k): v for k, v in sorted(class_source_counts.items())},
        "copy_modes": dict(modes),
        "strict_note": f"No {heldout} samples are linked into train/val.",
    }
    (view_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def create_view(out_root: Path, roots: Dict[str, Path], heldout: str, seed: int, copy: bool) -> Dict:
    view_root = out_root / f"risk_lodo_to_{heldout}_seed{seed}"
    if view_root.exists():
        shutil.rmtree(view_root)
    rows: List[Dict] = []
    modes: Counter = Counter()
    train_sources = [src for src in roots if src != heldout]
    for source in train_sources:
        for split in ("train", "val"):
            add_source_split(view_root, rows, modes, source, roots[source], split, split, copy)
    add_source_split(view_root, rows, modes, heldout, roots[heldout], "test", "test", copy)
    return write_view(view_root, rows, modes, heldout, train_sources)


def main() -> None:
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    roots = {"tigpr": args.tigpr_view, "utility": args.utility_view, "urdd": args.urdd_view}
    payload = {"views": []}
    for heldout in roots:
        payload["views"].append(create_view(args.out_root, roots, heldout, args.seed, args.copy))
    (args.out_root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
