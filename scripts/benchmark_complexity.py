#!/usr/bin/env python
"""Benchmark parameter count, optional GMACs and inference speed."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpr_risk.train import DomainAdaptedRiskNet


DEFAULT_MODELS = [
    ("GAFR-RegNet", "gafr_regnet_y_8gf"),
    ("RegNetY-8GF", "regnet_y_8gf"),
    ("ConvNeXt-Small", "convnext_small"),
    ("MaxViT-T", "maxvit_t"),
    ("Swin-T", "swin_t"),
    ("EfficientNetV2-S", "efficientnet_v2_s"),
    ("ViT-B/16", "vit_b_16"),
    ("DenseNet-121", "densenet121"),
    ("ResNet-34", "resnet34"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark GPR risk recognition models.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/source_data"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--reps", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--num-domains", type=int, default=3)
    parser.add_argument("--no-macs", action="store_true")
    return parser.parse_args()


def count_params(model: torch.nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def try_macs(model: torch.nn.Module, x: torch.Tensor) -> float:
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(x,), verbose=False)
        return float(macs) / 1e9
    except Exception as exc:
        print(f"MAC profiling failed: {exc}", file=sys.stderr)
        return float("nan")


@torch.no_grad()
def benchmark(model: torch.nn.Module, x: torch.Tensor, reps: int, warmup: int) -> tuple[float, float]:
    device = x.device
    model.eval()
    for _ in range(warmup):
        _ = model(x, grl_coeff=0.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = model(x, grl_coeff=0.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    return elapsed * 1000.0 / (reps * x.shape[0]), (reps * x.shape[0]) / elapsed


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    x = torch.randn(args.batch, 3, 224, 224, device=device)
    x_cpu = torch.randn(1, 3, 224, 224)
    rows = []
    for display, name in DEFAULT_MODELS:
        print(f"Benchmarking {display} ({name})", flush=True)
        model = DomainAdaptedRiskNet(
            name,
            num_classes=args.num_classes,
            num_domains=args.num_domains,
            pretrained=False,
            domain_hidden=256,
        ).to(device)
        params_m = count_params(model)
        gmacs = float("nan")
        if not args.no_macs:
            mac_model = DomainAdaptedRiskNet(
                name,
                num_classes=args.num_classes,
                num_domains=args.num_domains,
                pretrained=False,
                domain_hidden=256,
            ).eval()
            gmacs = try_macs(mac_model, x_cpu)
            del mac_model
        ms, ips = benchmark(model, x, args.reps, args.warmup)
        rows.append(
            {
                "Method": display,
                "Backbone": name,
                "Params_M": params_m,
                "GMACs": gmacs,
                "batch": args.batch,
                "device": str(device),
                "mean_ms_per_image": ms,
                "images_per_second": ips,
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    csv_path = args.out_dir / "model_complexity_speed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "model_complexity_speed.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(csv_path), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
