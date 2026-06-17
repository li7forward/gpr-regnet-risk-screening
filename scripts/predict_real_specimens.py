#!/usr/bin/env python
"""Evaluate trained risk model on four real concrete specimen types."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpr_risk.train import DomainAdaptedRiskNet
from gpr_risk.transforms import build_transforms


SAMPLES = [
    {"specimen": "Plastic pipe", "risk_surrogate": "non-metallic pipe or PVC-like inclusion", "gt_target_present": 1, "line": 1, "file": "plastic_line1.png"},
    {"specimen": "Plastic pipe", "risk_surrogate": "non-metallic pipe or PVC-like inclusion", "gt_target_present": 1, "line": 2, "file": "plastic_line2.png"},
    {"specimen": "Foam surrogate", "risk_surrogate": "low-density void surrogate", "gt_target_present": 1, "line": 1, "file": "foam_line1.png"},
    {"specimen": "Foam surrogate", "risk_surrogate": "low-density void surrogate", "gt_target_present": 1, "line": 2, "file": "foam_line2.png"},
    {"specimen": "Steel pipe", "risk_surrogate": "metallic pipe or strong reflector", "gt_target_present": 1, "line": 1, "file": "steel_line1.png"},
    {"specimen": "Steel pipe", "risk_surrogate": "metallic pipe or strong reflector", "gt_target_present": 1, "line": 2, "file": "steel_line2.png"},
    {"specimen": "Plain concrete", "risk_surrogate": "negative control", "gt_target_present": 0, "line": 1, "file": "plain_line1.png"},
    {"specimen": "Plain concrete", "risk_surrogate": "negative control", "gt_target_present": 0, "line": 2, "file": "plain_line2.png"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict target-present risk on real specimens.")
    parser.add_argument("--input-root", type=Path, default=Path("examples/real_specimens"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/gafr_regnet_allsource"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/source_data"))
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def contrast_normalize(image: Image.Image) -> Image.Image:
    gray = image.convert("L")
    arr = np.asarray(gray).astype(np.float32)
    lo, hi = np.percentile(arr, [1.0, 99.0])
    if hi <= lo:
        norm = np.clip(arr, 0, 255)
    else:
        norm = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255)
    out = Image.fromarray(norm.astype(np.uint8), mode="L")
    return Image.merge("RGB", (out, out, out))


def load_model(run_dir: Path, device: torch.device):
    ckpt_path = run_dir / "best_model.pt"
    summary_path = run_dir / "train_summary.json"
    state = torch.load(ckpt_path, map_location=device)
    ckpt_args = state.get("args", {})
    classes = list(state.get("classes", ["damage", "no_damage"]))
    domains = state.get("domains", {"source0": 0})
    model = DomainAdaptedRiskNet(
        ckpt_args.get("model", "gafr_regnet_y_8gf"),
        num_classes=len(classes),
        num_domains=len(domains),
        pretrained=False,
        domain_hidden=int(ckpt_args.get("domain_hidden", 256)),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    threshold = 0.5
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        threshold = float(summary.get("selected_threshold_from_val", threshold))
    positive_class = ckpt_args.get("positive_class", "damage")
    positive_idx = classes.index(positive_class) if positive_class in classes else 0
    _, eval_tf = build_transforms(int(ckpt_args.get("imgsz", 224)), input_mode="rgb")
    return model, eval_tf, positive_idx, threshold, classes


@torch.no_grad()
def score_image(model, transform, image: Image.Image, positive_idx: int, device: torch.device) -> float:
    x = transform(image).unsqueeze(0).to(device)
    logits, _, _ = model(x, grl_coeff=0.0)
    return float(torch.softmax(logits, dim=1)[0, positive_idx].detach().cpu())


def infer_one(model, transform, path: Path, positive_idx: int, device: torch.device, preprocess: str) -> dict:
    original = Image.open(path).convert("RGB")
    image = contrast_normalize(original) if preprocess == "contrast_norm" else original
    score_forward = score_image(model, transform, image, positive_idx, device)
    score_reverse = score_image(model, transform, ImageOps.mirror(image), positive_idx, device)
    return {
        "score_forward": score_forward,
        "score_reverse": score_reverse,
        "score_mean": 0.5 * (score_forward + score_reverse),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, transform, positive_idx, learned_threshold, classes = load_model(args.run_dir, device)
    threshold = float(args.threshold) if args.threshold is not None else float(learned_threshold)

    rows = []
    for preprocess in ["raw", "contrast_norm"]:
        for sample in SAMPLES:
            path = args.input_root / sample["file"]
            scores = infer_one(model, transform, path, positive_idx, device, preprocess)
            pred_forward = int(scores["score_forward"] >= threshold)
            pred_reverse = int(scores["score_reverse"] >= threshold)
            pred = int(scores["score_mean"] >= threshold)
            gt = int(sample["gt_target_present"])
            rows.append(
                {
                    **sample,
                    "preprocess": preprocess,
                    "image_path": str(path),
                    "positive_class": classes[positive_idx],
                    "threshold": threshold,
                    **scores,
                    "pred_forward": pred_forward,
                    "pred_reverse": pred_reverse,
                    "pred_target_present": pred,
                    "reverse_consistent": int(pred_forward == pred_reverse),
                    "correct": int(pred == gt),
                }
            )

    pred_csv = args.out_dir / "real_specimen_predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"predictions": str(pred_csv)}, indent=2))


if __name__ == "__main__":
    main()
