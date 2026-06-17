#!/usr/bin/env python
"""Run image-level risk prediction for one image or a folder of B-scan images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpr_risk.train import DomainAdaptedRiskNet
from gpr_risk.transforms import build_transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict GPR image-level risk scores.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Training run directory with best_model.pt.")
    parser.add_argument("--out", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def iter_images(path: Path):
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTS:
            yield path
        return
    for image_path in sorted(path.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
            yield image_path


def load_model(run_dir: Path, device: torch.device):
    state = torch.load(run_dir / "best_model.pt", map_location=device)
    train_args = state.get("args", {})
    classes = list(state.get("classes", ["damage", "no_damage"]))
    domains = state.get("domains", {"source0": 0})
    model = DomainAdaptedRiskNet(
        train_args.get("model", "gafr_regnet_y_8gf"),
        num_classes=len(classes),
        num_domains=len(domains),
        pretrained=False,
        domain_hidden=int(train_args.get("domain_hidden", 256)),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    threshold = 0.5
    summary_path = run_dir / "train_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        threshold = float(summary.get("selected_threshold_from_val", threshold))
    positive_class = train_args.get("positive_class", "damage")
    positive_idx = classes.index(positive_class) if positive_class in classes else 0
    _, transform = build_transforms(int(train_args.get("imgsz", 224)), input_mode="rgb")
    return model, transform, classes, positive_idx, threshold


@torch.no_grad()
def predict_one(model, transform, image_path: Path, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)
    logits, _, _ = model(x, grl_coeff=0.0)
    prob = torch.softmax(logits, dim=1)[0].detach().cpu()
    pred_idx = int(prob.argmax().item())
    return pred_idx, prob.tolist()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, transform, classes, positive_idx, learned_threshold = load_model(args.run_dir, device)
    threshold = float(args.threshold) if args.threshold is not None else learned_threshold

    rows = []
    for image_path in iter_images(args.input):
        pred_idx, probs = predict_one(model, transform, image_path, device)
        risk_score = float(probs[positive_idx])
        rows.append(
            {
                "image": str(image_path),
                "pred_class": classes[pred_idx],
                "risk_score": risk_score,
                "threshold": threshold,
                "pred_risk_present": int(risk_score >= threshold),
                **{f"prob_{cls}": float(probs[idx]) for idx, cls in enumerate(classes)},
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else ["image", "pred_class", "risk_score", "threshold", "pred_risk_present"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"out": str(args.out), "num_images": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
