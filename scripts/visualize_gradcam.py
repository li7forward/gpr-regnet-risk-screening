#!/usr/bin/env python
"""Generate Grad-CAM overlays for trained GPR risk models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpr_risk.train import DomainAdaptedRiskNet
from gpr_risk.transforms import build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Grad-CAM visualizations.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--images", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/gradcam"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-class", type=int, default=None)
    return parser.parse_args()


def load_model(run_dir: Path, device: torch.device):
    state = torch.load(run_dir / "best_model.pt", map_location=device)
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
    _, eval_tf = build_transforms(int(ckpt_args.get("imgsz", 224)), input_mode="rgb")
    return model, eval_tf, classes


def find_last_conv(module: nn.Module) -> nn.Module:
    convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return convs[-1]


def colorize(cam: np.ndarray) -> np.ndarray:
    cam = np.clip(cam, 0.0, 1.0)
    red = np.clip(1.5 * cam, 0, 1)
    green = np.clip(1.5 * (1.0 - np.abs(cam - 0.5) * 2.0), 0, 1)
    blue = np.clip(1.5 * (1.0 - cam), 0, 1)
    return np.stack([red, green, blue], axis=-1)


def overlay(original: Image.Image, cam: np.ndarray, alpha: float = 0.42) -> Image.Image:
    original = original.convert("RGB")
    heat = Image.fromarray((colorize(cam) * 255).astype(np.uint8)).resize(original.size, Image.BILINEAR)
    base = np.asarray(original).astype(np.float32) / 255.0
    hmap = np.asarray(heat).astype(np.float32) / 255.0
    out = np.clip((1.0 - alpha) * base + alpha * hmap, 0.0, 1.0)
    return Image.fromarray((out * 255).astype(np.uint8))


def gradcam(model: DomainAdaptedRiskNet, x: torch.Tensor, target_class: int | None) -> tuple[np.ndarray, int, float]:
    layer = find_last_conv(model.encoder)
    activations = {}
    gradients = {}

    def fwd_hook(_, __, output):
        activations["value"] = output.detach()

    def bwd_hook(_, __, grad_output):
        gradients["value"] = grad_output[0].detach()

    h1 = layer.register_forward_hook(fwd_hook)
    h2 = layer.register_full_backward_hook(bwd_hook)
    try:
        logits, _, _ = model(x, grl_coeff=0.0)
        prob = torch.softmax(logits, dim=1)
        cls = int(prob.argmax(dim=1).item()) if target_class is None else int(target_class)
        score = logits[:, cls].sum()
        model.zero_grad(set_to_none=True)
        score.backward()
        act = activations["value"]
        grad = gradients["value"]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), cls, float(prob[0, cls].detach().cpu())
    finally:
        h1.remove()
        h2.remove()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, transform, classes = load_model(args.run_dir, device)
    rows = []
    for image_path in args.images:
        image = Image.open(image_path).convert("RGB")
        x = transform(image).unsqueeze(0).to(device)
        cam, cls, prob = gradcam(model, x, args.target_class)
        out_path = args.out_dir / f"{image_path.stem}_gradcam.png"
        overlay(image, cam).save(out_path, dpi=(600, 600))
        rows.append({"image": str(image_path), "gradcam": str(out_path), "class_index": cls, "class_name": classes[cls], "probability": prob})
    (args.out_dir / "gradcam_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "n": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
