#!/usr/bin/env python
"""Train GAFR-RegNet for image-level GPR risk recognition.

The script keeps the B-scan-specific gated axial-frequency evidence design used
in the public benchmark experiments and adds two multi-source domain-adaptation
constraints:

1. a gradient-reversal domain head, which encourages domain-invariant features;
2. class-conditional CORAL, which aligns same-risk features across source domains.

It expects ImageFolder-style splits and derives a domain id from filenames like
``tigpr__train__xxx.jpg`` and ``utility__val__xxx.jpg``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets

from .models import build_model
from .transforms import build_transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GAFR-RegNet multi-source GPR risk recognition.")
    parser.add_argument("--data", type=Path, required=True, help="ImageFolder root with train/val/test.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--model", default="gafr_regnet_y_8gf", help="Model name; use GAFR-RegNet by default or a public backbone baseline.")
    parser.add_argument("--positive-class", default="damage", help="Risk/positive class used for threshold metrics.")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2037)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--loss", choices=["ce", "weighted_ce"], default="weighted_ce")
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--domain-adv-weight", type=float, default=0.15)
    parser.add_argument("--branch-domain-adv-weight", type=float, default=0.0, help="Extra GRL weight for each branch feature in multi-branch encoders.")
    parser.add_argument("--ccoral-weight", type=float, default=0.05)
    parser.add_argument("--domain-hidden", type=int, default=256)
    parser.add_argument(
        "--selection-metric",
        choices=["balanced_accuracy", "best_f1", "min_expected_cost", "source_worst_min_cost"],
        default="min_expected_cost",
    )
    parser.add_argument("--fn-cost", type=float, default=10.0)
    parser.add_argument("--fp-cost", type=float, default=1.0)
    parser.add_argument("--gpr-physics-aug", action="store_true")
    parser.add_argument("--gpr-aug-p", type=float, default=0.60)
    parser.add_argument("--gpr-max-time-shift", type=float, default=0.04)
    parser.add_argument("--gpr-max-trace-shift", type=float, default=0.03)
    parser.add_argument("--gpr-noise-std", type=float, default=0.025)
    parser.add_argument("--gpr-gain-min", type=float, default=0.85)
    parser.add_argument("--gpr-gain-max", type=float, default=1.15)
    parser.add_argument("--gpr-clutter-strength", type=float, default=0.18)
    parser.add_argument("--gpr-dielectric-jitter", type=float, default=0.10)
    parser.add_argument("--gpr-phase-jitter", type=float, default=0.35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional smoke-test cap.")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Optional smoke-test cap.")
    parser.add_argument("--max-test-samples", type=int, default=0, help="Optional smoke-test cap.")
    parser.add_argument(
        "--extra-test-data",
        action="append",
        default=[],
        help="Optional additional ImageFolder root to test after training; use name=path or path.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def infer_domain(path: str) -> str:
    name = Path(path).name
    if "__" in name:
        return name.split("__", 1)[0]
    return "unknown"


class DomainImageFolder(Dataset):
    def __init__(self, root: Path, transform=None, domain_to_idx: Optional[Dict[str, int]] = None, max_samples: int = 0):
        self.inner = datasets.ImageFolder(root, transform=transform)
        self.classes = self.inner.classes
        self.class_to_idx = self.inner.class_to_idx
        domains = [infer_domain(path) for path, _ in self.inner.samples]
        if domain_to_idx is None:
            domain_to_idx = {d: i for i, d in enumerate(sorted(set(domains)))}
        self.domain_to_idx = dict(domain_to_idx)
        self.samples = list(self.inner.samples)
        if max_samples and max_samples > 0 and len(self.samples) > max_samples:
            rng = random.Random(2037)
            by_bucket: Dict[Tuple[int, int], List[Tuple[str, int]]] = defaultdict(list)
            for path, target in self.samples:
                dom = self.domain_to_idx.get(infer_domain(path), len(self.domain_to_idx))
                by_bucket[(target, dom)].append((path, target))
            capped: List[Tuple[str, int]] = []
            buckets = list(by_bucket.values())
            while len(capped) < max_samples and any(buckets):
                for bucket in buckets:
                    if bucket and len(capped) < max_samples:
                        capped.append(bucket.pop(rng.randrange(len(bucket))))
            self.samples = capped
        self.targets = [target for _, target in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, target = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.inner.transform is not None:
            image = self.inner.transform(image)
        domain_name = infer_domain(path)
        domain = self.domain_to_idx.get(domain_name, len(self.domain_to_idx))
        return image, target, domain


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, coeff: float) -> torch.Tensor:
        ctx.coeff = float(coeff)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coeff * grad_output, None


def grad_reverse(x: torch.Tensor, coeff: float) -> torch.Tensor:
    return GradReverse.apply(x, coeff)


def split_features(features):
    if torch.is_tensor(features):
        return [features.flatten(1)]
    if isinstance(features, (list, tuple)):
        parts = [x for x in features if torch.is_tensor(x)]
        if not parts:
            raise RuntimeError("forward_features returned no tensor features.")
        return [x.flatten(1) for x in parts]
    raise TypeError(f"Unsupported feature type: {type(features)!r}")


def flatten_features(features):
    return torch.cat(split_features(features), dim=1)


def make_domain_head(feat_dim: int, domain_hidden: int, num_domains: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(feat_dim),
        nn.Linear(feat_dim, domain_hidden),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(domain_hidden, num_domains),
    )


class DomainAdaptedRiskNet(nn.Module):
    def __init__(self, model_name: str, num_classes: int, num_domains: int, pretrained: bool, domain_hidden: int):
        super().__init__()
        self.model_name = model_name
        self.encoder = build_model(model_name, num_classes=num_classes, pretrained=pretrained)
        if hasattr(self.encoder, "feature_dim"):
            feat_dim = int(self.encoder.feature_dim)
        else:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                features, _, branch_features = self.encoder_features_logits(dummy)
                feat_dim = int(features.shape[1])
        self.feature_dim = feat_dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            _, _, branch_features = self.encoder_features_logits(dummy)
        self.branch_dims = [int(f.shape[1]) for f in branch_features]
        self.domain_head = make_domain_head(feat_dim, domain_hidden, num_domains)
        self.branch_domain_heads = nn.ModuleList(
            make_domain_head(dim, domain_hidden, num_domains) for dim in self.branch_dims if dim != feat_dim or len(self.branch_dims) > 1
        )

    def encoder_features_logits(self, x: torch.Tensor):
        if hasattr(self.encoder, "forward_features"):
            raw_features = self.encoder.forward_features(x)
            branch_features = split_features(raw_features)
            features = torch.cat(branch_features, dim=1)
            if hasattr(self.encoder, "logits_from_features"):
                logits = self.encoder.logits_from_features(raw_features)
            elif hasattr(self.encoder, "classifier"):
                logits = self.encoder.classifier(features)
            elif hasattr(self.encoder, "fc"):
                logits = self.encoder.fc(features)
            else:
                logits = self.encoder(x)
            return features, logits, branch_features

        if all(hasattr(self.encoder, attr) for attr in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc")):
            z = self.encoder.conv1(x)
            z = self.encoder.bn1(z)
            z = self.encoder.relu(z)
            z = self.encoder.maxpool(z)
            z = self.encoder.layer1(z)
            z = self.encoder.layer2(z)
            z = self.encoder.layer3(z)
            z = self.encoder.layer4(z)
            features = torch.flatten(self.encoder.avgpool(z), 1)
            return features, self.encoder.fc(features), [features]

        if self.model_name.startswith("darc_convnext_tiny") and hasattr(self.encoder, "features"):
            z = x
            if getattr(self.encoder, "input_gate", None) is not None:
                z = z * (1.0 + 0.10 * torch.tanh(self.encoder.input_gate))
            for idx, layer in enumerate(self.encoder.features):
                z = layer(z)
                key = str(idx)
                if hasattr(self.encoder, "defect_blocks") and key in self.encoder.defect_blocks:
                    z = self.encoder.defect_blocks[key](z)
            features = torch.flatten(self.encoder.avgpool(z), 1)
            if getattr(self.encoder, "risk_head", None) is not None:
                logits = self.encoder.risk_head(z)
            else:
                logits = self.encoder.classifier(self.encoder.avgpool(z))
            return features, logits, [features]

        if hasattr(self.encoder, "features") and hasattr(self.encoder, "classifier"):
            z = self.encoder.features(x)
            if self.model_name.startswith("densenet"):
                z = F.relu(z, inplace=True)
                features = torch.flatten(F.adaptive_avg_pool2d(z, (1, 1)), 1)
                return features, self.encoder.classifier(features), [features]
            if hasattr(self.encoder, "avgpool"):
                pooled = self.encoder.avgpool(z)
            else:
                pooled = F.adaptive_avg_pool2d(z, (1, 1))
            features = torch.flatten(pooled, 1)
            if self.model_name.startswith("convnext"):
                logits = self.encoder.classifier(pooled)
            else:
                logits = self.encoder.classifier(features)
            return features, logits, [features]

        if self.model_name.startswith("swin") and hasattr(self.encoder, "features") and hasattr(self.encoder, "head"):
            z = self.encoder.features(x)
            z = self.encoder.norm(z)
            z = self.encoder.permute(z)
            pooled = self.encoder.avgpool(z)
            features = self.encoder.flatten(pooled)
            return features, self.encoder.head(features), [features]

        if self.model_name.startswith("vit") and hasattr(self.encoder, "_process_input") and hasattr(self.encoder, "heads"):
            z = self.encoder._process_input(x)
            n = z.shape[0]
            batch_class_token = self.encoder.class_token.expand(n, -1, -1)
            z = torch.cat([batch_class_token, z], dim=1)
            z = self.encoder.encoder(z)
            features = z[:, 0]
            return features, self.encoder.heads(features), [features]

        if self.model_name.startswith("maxvit") and hasattr(self.encoder, "stem") and hasattr(self.encoder, "blocks"):
            z = self.encoder.stem(x)
            for block in self.encoder.blocks:
                z = block(z)
            features = self.encoder.classifier[:-1](z)
            return features, self.encoder.classifier[-1](features), [features]

        if self.model_name.startswith("regnet") and hasattr(self.encoder, "stem") and hasattr(self.encoder, "trunk_output"):
            z = self.encoder.stem(x)
            z = self.encoder.trunk_output(z)
            features = torch.flatten(self.encoder.avgpool(z), 1)
            return features, self.encoder.fc(features), [features]

        logits = self.encoder(x)
        features = logits.detach()
        return features, logits, [features]

    def forward(self, x: torch.Tensor, grl_coeff: float = 0.0):
        features, risk_logits, branch_features = self.encoder_features_logits(x)
        domain_logits = self.domain_head(grad_reverse(features, grl_coeff))
        branch_domain_logits = [
            head(grad_reverse(feat, grl_coeff))
            for head, feat in zip(self.branch_domain_heads, branch_features)
        ]
        return risk_logits, (domain_logits, branch_domain_logits), features


def class_counts(dataset: DomainImageFolder) -> np.ndarray:
    counts = np.zeros(len(dataset.classes), dtype=np.float64)
    for target in dataset.targets:
        counts[int(target)] += 1.0
    return counts


def make_loaders(args: argparse.Namespace):
    aug_args = argparse.Namespace(
        input_mode="rgb",
        gpr_physics_aug=args.gpr_physics_aug,
        gpr_aug_p=args.gpr_aug_p,
        gpr_max_time_shift=args.gpr_max_time_shift,
        gpr_max_trace_shift=args.gpr_max_trace_shift,
        gpr_noise_std=args.gpr_noise_std,
        gpr_gain_range=(args.gpr_gain_min, args.gpr_gain_max),
        gpr_clutter_strength=args.gpr_clutter_strength,
        gpr_dielectric_jitter=args.gpr_dielectric_jitter,
        gpr_phase_jitter=args.gpr_phase_jitter,
    )
    train_tf, eval_tf = build_transforms(
        args.imgsz,
        input_mode=aug_args.input_mode,
        gpr_physics_aug=aug_args.gpr_physics_aug,
        gpr_aug_p=aug_args.gpr_aug_p,
        gpr_max_time_shift=aug_args.gpr_max_time_shift,
        gpr_max_trace_shift=aug_args.gpr_max_trace_shift,
        gpr_noise_std=aug_args.gpr_noise_std,
        gpr_gain_range=aug_args.gpr_gain_range,
        gpr_clutter_strength=aug_args.gpr_clutter_strength,
        gpr_dielectric_jitter=aug_args.gpr_dielectric_jitter,
        gpr_phase_jitter=aug_args.gpr_phase_jitter,
    )
    train_probe = DomainImageFolder(args.data / "train", transform=train_tf, max_samples=args.max_train_samples)
    domain_to_idx = train_probe.domain_to_idx
    train_ds = train_probe
    val_ds = DomainImageFolder(args.data / "val", transform=eval_tf, domain_to_idx=domain_to_idx, max_samples=args.max_val_samples)
    test_ds = DomainImageFolder(args.data / "test", transform=eval_tf, domain_to_idx=domain_to_idx, max_samples=args.max_test_samples)
    sampler = None
    shuffle = True
    if args.weighted_sampler:
        counts = class_counts(train_ds)
        weights = 1.0 / np.maximum(counts[np.asarray(train_ds.targets)], 1.0)
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        shuffle = False
    loaders = {
        "train": DataLoader(train_ds, batch_size=args.batch, shuffle=shuffle, sampler=sampler, num_workers=args.workers, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True),
    }
    return train_ds, val_ds, test_ds, loaders


def covariance(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    denom = max(x.shape[0] - 1, 1)
    return x.t().mm(x) / float(denom)


def coral_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mean_loss = F.mse_loss(a.mean(dim=0), b.mean(dim=0))
    cov_loss = F.mse_loss(covariance(a), covariance(b))
    return mean_loss + cov_loss


def class_conditional_coral(features: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for cls in labels.unique():
        cls_mask = labels == cls
        cls_domains = domains[cls_mask].unique()
        domain_feats = []
        for dom in cls_domains:
            f = features[cls_mask & (domains == dom)]
            if f.shape[0] >= 2:
                domain_feats.append(f)
        for i in range(len(domain_feats)):
            for j in range(i + 1, len(domain_feats)):
                losses.append(coral_pair(domain_feats[i], domain_feats[j]))
    if not losses:
        return features.new_tensor(0.0)
    return torch.stack(losses).mean()


def grl_schedule(epoch: int, batch_idx: int, batches_per_epoch: int, epochs: int) -> float:
    p = (epoch + batch_idx / max(batches_per_epoch, 1)) / max(epochs, 1)
    return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def threshold_metrics(y: np.ndarray, scores: np.ndarray, threshold: float, fp_cost: float, fn_cost: float) -> Dict[str, float]:
    pred_pos = scores >= threshold
    y_pos = y.astype(bool)
    tp = int(np.logical_and(y_pos, pred_pos).sum())
    tn = int(np.logical_and(~y_pos, ~pred_pos).sum())
    fp = int(np.logical_and(~y_pos, pred_pos).sum())
    fn = int(np.logical_and(y_pos, ~pred_pos).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    cost = (float(fp_cost) * fp + float(fn_cost) * fn) / max(len(y), 1)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "expected_cost_per_sample": float(cost),
    }


def select_operating_point(y_bin: np.ndarray, scores: np.ndarray, fp_cost: float, fn_cost: float) -> Dict[str, Dict[str, float]]:
    if len(np.unique(y_bin)) < 2:
        base = threshold_metrics(y_bin, scores, 0.5, fp_cost, fn_cost)
        return {"best_f1": base, "min_expected_cost": base}
    unique = np.unique(scores)
    mids = (unique[:-1] + unique[1:]) / 2.0 if len(unique) > 1 else unique
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], unique, mids)))
    rows = [threshold_metrics(y_bin, scores, float(th), fp_cost, fn_cost) for th in thresholds]
    return {
        "best_f1": max(rows, key=lambda r: (r["f1"], r["recall"], -r["fpr"])),
        "min_expected_cost": min(rows, key=lambda r: (r["expected_cost_per_sample"], -r["recall"], r["fpr"])),
    }


def select_source_worst_operating_point(
    y_bin: np.ndarray,
    scores: np.ndarray,
    domains: np.ndarray,
    fp_cost: float,
    fn_cost: float,
) -> Dict[str, float]:
    if len(np.unique(y_bin)) < 2:
        base = threshold_metrics(y_bin, scores, 0.5, fp_cost, fn_cost)
        base.update({"worst_domain_cost": base["expected_cost_per_sample"], "worst_domain_fpr": base["fpr"], "min_domain_recall": base["recall"]})
        return base
    unique = np.unique(scores)
    mids = (unique[:-1] + unique[1:]) / 2.0 if len(unique) > 1 else unique
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], unique, mids)))
    best = None
    for threshold in thresholds:
        pooled = threshold_metrics(y_bin, scores, float(threshold), fp_cost, fn_cost)
        per_domain = []
        for domain in np.unique(domains):
            mask = domains == domain
            per_domain.append(threshold_metrics(y_bin[mask], scores[mask], float(threshold), fp_cost, fn_cost))
        pooled["worst_domain_cost"] = float(max(row["expected_cost_per_sample"] for row in per_domain))
        pooled["worst_domain_fpr"] = float(max(row["fpr"] for row in per_domain))
        pooled["min_domain_recall"] = float(min(row["recall"] for row in per_domain))
        key = (
            pooled["worst_domain_cost"],
            pooled["expected_cost_per_sample"],
            -pooled["min_domain_recall"],
            pooled["worst_domain_fpr"],
            -pooled["f1"],
        )
        if best is None or key < best[0]:
            best = (key, pooled)
    return best[1]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, positive_idx: int, fp_cost: float, fn_cost: float, threshold: Optional[float] = None):
    model.eval()
    ys: List[int] = []
    doms: List[int] = []
    probs: List[np.ndarray] = []
    losses: List[float] = []
    for x, y, d in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _, _ = model(x, grl_coeff=0.0)
        loss = F.cross_entropy(logits, y)
        losses.append(float(loss.detach().cpu()) * y.numel())
        prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
        probs.append(prob)
        ys.extend(y.detach().cpu().numpy().tolist())
        doms.extend(d.numpy().tolist())
    y = np.asarray(ys, dtype=np.int64)
    prob = np.concatenate(probs, axis=0)
    score = prob[:, positive_idx]
    pred_argmax = prob.argmax(axis=1)
    y_bin = (y == positive_idx).astype(np.int64)
    op = select_operating_point(y_bin, score, fp_cost, fn_cost)
    source_worst = select_source_worst_operating_point(y_bin, score, np.asarray(doms, dtype=np.int64), fp_cost, fn_cost)
    th = float(threshold) if threshold is not None else op["min_expected_cost"]["threshold"]
    th_row = threshold_metrics(y_bin, score, th, fp_cost, fn_cost)
    out = {
        "loss": float(sum(losses) / max(len(y), 1)),
        "accuracy": float(accuracy_score(y, pred_argmax)) if len(y) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred_argmax)) if len(np.unique(y)) > 1 else 0.0,
        "average_precision": float(average_precision_score(y_bin, score)) if len(np.unique(y_bin)) > 1 else 0.0,
        "auc": float(roc_auc_score(y_bin, score)) if len(np.unique(y_bin)) > 1 else 0.0,
        "best_f1": op["best_f1"],
        "min_expected_cost": op["min_expected_cost"],
        "source_worst_min_cost": source_worst,
        "threshold_metrics": th_row,
        "n": int(len(y)),
        "positive_support": int(y_bin.sum()),
        "negative_support": int((1 - y_bin).sum()),
        "domain_counts": dict(Counter(doms)),
        "confusion_matrix_argmax": confusion_matrix(y, pred_argmax).tolist() if len(y) else [],
    }
    return out


def selection_score(metrics: Dict, name: str) -> float:
    if name == "balanced_accuracy":
        return float(metrics["balanced_accuracy"])
    if name == "best_f1":
        return float(metrics["best_f1"]["f1"])
    if name == "min_expected_cost":
        return -float(metrics["min_expected_cost"]["expected_cost_per_sample"])
    if name == "source_worst_min_cost":
        row = metrics["source_worst_min_cost"]
        return -float(row.get("worst_domain_cost", row["expected_cost_per_sample"]))
    raise ValueError(name)


def summarize_dataset(ds: DomainImageFolder) -> Dict:
    rows = []
    for path, target in ds.samples:
        rows.append((ds.classes[target], infer_domain(path)))
    return {
        "n": len(ds),
        "classes": ds.classes,
        "class_counts": dict(Counter(cls for cls, _ in rows)),
        "domain_counts": dict(Counter(dom for _, dom in rows)),
        "class_domain_counts": {f"{cls}|{dom}": cnt for (cls, dom), cnt in Counter(rows).items()},
        "domain_to_idx": ds.domain_to_idx,
    }


def jsonable_args(args: argparse.Namespace) -> Dict:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def parse_extra_test(spec: str) -> Tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), Path(path)
    path = Path(spec)
    return path.name, path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_ds, val_ds, test_ds, loaders = make_loaders(args)
    if args.positive_class not in train_ds.class_to_idx:
        raise ValueError(f"positive class {args.positive_class!r} not in {train_ds.classes}")
    positive_idx = int(train_ds.class_to_idx[args.positive_class])
    num_classes = len(train_ds.classes)
    num_domains = len(train_ds.domain_to_idx)
    model = DomainAdaptedRiskNet(
        args.model,
        num_classes=num_classes,
        num_domains=num_domains,
        pretrained=args.pretrained,
        domain_hidden=args.domain_hidden,
    ).to(device)
    counts = class_counts(train_ds)
    class_weight = None
    if args.loss == "weighted_ce":
        inv = counts.sum() / np.maximum(counts, 1.0)
        inv = inv / inv.mean()
        class_weight = torch.as_tensor(inv, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    history = []
    best = {"score": -1e18, "epoch": -1, "path": str(args.out / "best_model.pt")}
    bad_epochs = 0
    started = time.time()
    run_meta = {
        "args": jsonable_args(args),
        "classes": train_ds.classes,
        "positive_class": args.positive_class,
        "positive_idx": positive_idx,
        "domains": train_ds.domain_to_idx,
        "datasets": {
            "train": summarize_dataset(train_ds),
            "val": summarize_dataset(val_ds),
            "test": summarize_dataset(test_ds),
        },
        "model_description": (
            "GAFR-RegNet = RegNet semantic trunk + physics-guided feature recalibration + "
            "trace-sequence branch + frequency-statistics branch + gated residual fusion. "
            "The training wrapper adds gradient-reversal domain alignment and class-conditional CORAL."
        ),
    }
    (args.out / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    for epoch in range(args.epochs):
        model.train()
        total = defaultdict(float)
        n_seen = 0
        batches = len(loaders["train"])
        for batch_idx, (x, y, d) in enumerate(loaders["train"]):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            d = d.to(device, non_blocking=True)
            grl_coeff = grl_schedule(epoch, batch_idx, batches, args.epochs)
            logits, domain_pack, features = model(x, grl_coeff=grl_coeff)
            if isinstance(domain_pack, tuple):
                domain_logits, branch_domain_logits = domain_pack
            else:
                domain_logits, branch_domain_logits = domain_pack, []
            risk_loss = F.cross_entropy(logits, y, weight=class_weight)
            domain_loss = F.cross_entropy(domain_logits, d)
            if branch_domain_logits and args.branch_domain_adv_weight > 0:
                branch_domain_loss = torch.stack([F.cross_entropy(branch_logits, d) for branch_logits in branch_domain_logits]).mean()
            else:
                branch_domain_loss = features.new_tensor(0.0)
            if args.ccoral_weight > 0:
                ccoral_loss = class_conditional_coral(features, y, d)
            else:
                ccoral_loss = features.new_tensor(0.0)
            loss = (
                risk_loss
                + args.domain_adv_weight * domain_loss
                + args.branch_domain_adv_weight * branch_domain_loss
                + args.ccoral_weight * ccoral_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            bs = y.numel()
            n_seen += bs
            total["loss"] += float(loss.detach().cpu()) * bs
            total["risk_loss"] += float(risk_loss.detach().cpu()) * bs
            total["domain_loss"] += float(domain_loss.detach().cpu()) * bs
            total["branch_domain_loss"] += float(branch_domain_loss.detach().cpu()) * bs
            total["ccoral_loss"] += float(ccoral_loss.detach().cpu()) * bs
            total["grl_coeff"] += float(grl_coeff) * bs
        scheduler.step()
        val_metrics = evaluate(model, loaders["val"], device, positive_idx, args.fp_cost, args.fn_cost)
        score = selection_score(val_metrics, args.selection_metric)
        row = {
            "epoch": epoch + 1,
            "train_loss": total["loss"] / max(n_seen, 1),
            "train_risk_loss": total["risk_loss"] / max(n_seen, 1),
            "train_domain_loss": total["domain_loss"] / max(n_seen, 1),
            "train_branch_domain_loss": total["branch_domain_loss"] / max(n_seen, 1),
            "train_ccoral_loss": total["ccoral_loss"] / max(n_seen, 1),
            "train_grl_coeff_mean": total["grl_coeff"] / max(n_seen, 1),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_average_precision": val_metrics["average_precision"],
            "val_auc": val_metrics["auc"],
            "val_best_f1": val_metrics["best_f1"]["f1"],
            "val_best_f1_threshold": val_metrics["best_f1"]["threshold"],
            "val_min_expected_cost": val_metrics["min_expected_cost"]["expected_cost_per_sample"],
            "val_min_expected_cost_threshold": val_metrics["min_expected_cost"]["threshold"],
            "selection_score": score,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if score > best["score"]:
            best.update({"score": score, "epoch": epoch + 1})
            torch.save({"model": model.state_dict(), "args": vars(args), "classes": train_ds.classes, "domains": train_ds.domain_to_idx}, best["path"])
            bad_epochs = 0
        else:
            bad_epochs += 1
        if args.patience > 0 and bad_epochs >= args.patience:
            break

    if os.path.exists(best["path"]):
        state = torch.load(best["path"], map_location=device)
        model.load_state_dict(state["model"])
    val_final = evaluate(model, loaders["val"], device, positive_idx, args.fp_cost, args.fn_cost)
    if args.selection_metric == "min_expected_cost":
        threshold_key = "min_expected_cost"
    elif args.selection_metric == "source_worst_min_cost":
        threshold_key = "source_worst_min_cost"
    else:
        threshold_key = "best_f1"
    selected_threshold = float(val_final[threshold_key]["threshold"])
    test_final = evaluate(model, loaders["test"], device, positive_idx, args.fp_cost, args.fn_cost, threshold=selected_threshold)
    extra_tests = {}
    for spec in args.extra_test_data:
        name, root = parse_extra_test(spec)
        extra_ds = DomainImageFolder(
            root / "test",
            transform=test_ds.inner.transform,
            domain_to_idx=train_ds.domain_to_idx,
            max_samples=args.max_test_samples,
        )
        extra_loader = DataLoader(
            extra_ds,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        extra_tests[name] = {
            "dataset": summarize_dataset(extra_ds),
            "metrics": evaluate(model, extra_loader, device, positive_idx, args.fp_cost, args.fn_cost, threshold=selected_threshold),
        }
    summary = {
        "model": args.model,
        "method": "GAFR-RegNet-MSDA",
        "best_epoch": best["epoch"],
        "best_selection_score": best["score"],
        "selected_threshold_from_val": selected_threshold,
        "threshold_key": threshold_key,
        "val": val_final,
        "test": test_final,
        "extra_tests": extra_tests,
        "seconds": time.time() - started,
    }
    (args.out / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "detailed_val.json").write_text(json.dumps(val_final, indent=2), encoding="utf-8")
    (args.out / "detailed_test.json").write_text(json.dumps(test_final, indent=2), encoding="utf-8")
    with open(args.out / "history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()) if history else [])
        if history:
            writer.writeheader()
            writer.writerows(history)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
