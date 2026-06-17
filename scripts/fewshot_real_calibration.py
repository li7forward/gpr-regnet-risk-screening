#!/usr/bin/env python
"""Line-heldout few-shot calibration for real concrete specimens."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from predict_real_specimens import SAMPLES, contrast_normalize, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-shot line-heldout real-specimen calibration.")
    parser.add_argument("--input-root", type=Path, default=Path("examples/real_specimens"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/gafr_regnet_allsource"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/source_data"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--aug-per-image", type=int, default=96)
    parser.add_argument("--seed", type=int, default=2037)
    return parser.parse_args()


def jitter_image(image: Image.Image, rng: random.Random) -> Image.Image:
    im = image.copy()
    if rng.random() < 0.5:
        im = ImageOps.mirror(im)
    w, h = im.size
    pad_x = max(2, int(round(0.04 * w)))
    pad_y = max(2, int(round(0.04 * h)))
    im_pad = ImageOps.expand(im, border=(pad_x, pad_y), fill=0)
    dx = rng.randint(0, 2 * pad_x)
    dy = rng.randint(0, 2 * pad_y)
    im = im_pad.crop((dx, dy, dx + w, dy + h))
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.75, 1.35))
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.85, 1.15))
    return im


@torch.no_grad()
def feature(model, transform, image: Image.Image, device: torch.device) -> np.ndarray:
    x = transform(image).unsqueeze(0).to(device)
    feat, _, _ = model.encoder_features_logits(x)
    return feat.detach().cpu().numpy()[0]


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_pred, dtype=int)
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    accuracy = (tp + tn) / max(len(y), 1)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "sensitivity": recall, "specificity": specificity, "precision": precision, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, transform, _, _, _ = load_model(args.run_dir, device)

    base_images = {}
    for sample in SAMPLES:
        image = contrast_normalize(Image.open(args.input_root / sample["file"]).convert("RGB"))
        base_images[(sample["specimen"], sample["line"])] = image

    pred_rows = []
    summary_rows = []
    for train_line, test_line in [(1, 2), (2, 1)]:
        train_samples = [s for s in SAMPLES if s["line"] == train_line]
        test_samples = [s for s in SAMPLES if s["line"] == test_line]
        x_train = []
        y_train = []
        for sample in train_samples:
            base = base_images[(sample["specimen"], sample["line"])]
            for _ in range(args.aug_per_image):
                x_train.append(feature(model, transform, jitter_image(base, rng), device))
                y_train.append(int(sample["gt_target_present"]))
        scaler = StandardScaler()
        x_train = scaler.fit_transform(np.vstack(x_train))
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, solver="liblinear", random_state=args.seed + train_line)
        clf.fit(x_train, y_train)

        fold_true = []
        fold_pred = []
        for sample in test_samples:
            base = base_images[(sample["specimen"], sample["line"])]
            fwd = scaler.transform(feature(model, transform, base, device).reshape(1, -1))
            rev = scaler.transform(feature(model, transform, ImageOps.mirror(base), device).reshape(1, -1))
            score_fwd = float(clf.predict_proba(fwd)[0, 1])
            score_rev = float(clf.predict_proba(rev)[0, 1])
            score = 0.5 * (score_fwd + score_rev)
            pred_fwd = int(score_fwd >= 0.5)
            pred_rev = int(score_rev >= 0.5)
            pred = int(score >= 0.5)
            gt = int(sample["gt_target_present"])
            fold_true.append(gt)
            fold_pred.append(pred)
            pred_rows.append(
                {
                    **sample,
                    "preprocess": "contrast_norm_fewshot",
                    "fold": f"train_line{train_line}_test_line{test_line}",
                    "image_path": str(args.input_root / sample["file"]),
                    "threshold": 0.5,
                    "score_forward": score_fwd,
                    "score_reverse": score_rev,
                    "score_mean": score,
                    "pred_forward": pred_fwd,
                    "pred_reverse": pred_rev,
                    "pred_target_present": pred,
                    "reverse_consistent": int(pred_fwd == pred_rev),
                    "correct": int(pred == gt),
                }
            )
        summary_rows.append({"fold": f"train_line{train_line}_test_line{test_line}", **binary_metrics(fold_true, fold_pred)})

    pooled = binary_metrics([int(r["gt_target_present"]) for r in pred_rows], [int(r["pred_target_present"]) for r in pred_rows])
    summary_rows.append({"fold": "pooled_two_direction_line_heldout", **pooled})

    pred_csv = args.out_dir / "real_specimen_fewshot_predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)
    summary_csv = args.out_dir / "real_specimen_fewshot_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({"predictions": str(pred_csv), "summary": str(summary_csv), "pooled": pooled}, indent=2))


if __name__ == "__main__":
    main()
