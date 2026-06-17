"""Model definitions for GPR image-level risk recognition."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models


class PhysicsGuidedFeatureRecalibration(nn.Module):
    """Residual channel and dual-axis spatial recalibration for B-scan features."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        hidden = max(channels // reduction, 32)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        pad = spatial_kernel // 2
        self.spatial_iso = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=pad, bias=False)
        self.spatial_time = nn.Conv2d(2, 1, kernel_size=(spatial_kernel, 1), padding=(pad, 0), bias=False)
        self.spatial_trace = nn.Conv2d(2, 1, kernel_size=(1, spatial_kernel), padding=(0, pad), bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        channel_gate = torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(mx))

        spatial_stats = torch.cat(
            [torch.mean(x, dim=1, keepdim=True), torch.amax(x, dim=1, keepdim=True)],
            dim=1,
        )
        spatial_gate = torch.sigmoid(
            self.spatial_iso(spatial_stats)
            + self.spatial_time(spatial_stats)
            + self.spatial_trace(spatial_stats)
        )
        recalibrated = x * channel_gate * spatial_gate
        return x + torch.tanh(self.alpha) * (recalibrated - x)


class GAFRRegNet(nn.Module):
    """Gated Axial-Frequency Risk head on a RegNet trunk.

    The model keeps a strong image-level semantic trunk and adds two
    B-scan-specific evidence branches:

    * a lateral trace-sequence branch, modelling how responses evolve along
      the scanning direction;
    * a frequency-statistics branch, summarising depth and trace spectral bands.

    A learnable gate injects each branch as residual logits, so auxiliary
    evidence can help when reliable without overwhelming the base classifier.
    """

    def __init__(
        self,
        base: nn.Module,
        num_classes: int,
        use_trace: bool = True,
        use_frequency: bool = True,
        use_input_gate: bool = True,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.use_trace = bool(use_trace)
        self.use_frequency = bool(use_frequency)
        self.input_gate = nn.Parameter(torch.zeros(1, in_channels, 1, 1)) if use_input_gate else None
        self.stem = base.stem
        self.trunk_output = base.trunk_output

        final_channels = int(base.fc.in_features)
        self.recalibration = PhysicsGuidedFeatureRecalibration(final_channels)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool_head = nn.AdaptiveMaxPool2d(1)
        self.base_dim = final_channels * 2
        self.base_head = nn.Sequential(nn.Dropout(p=0.15), nn.Linear(self.base_dim, num_classes))

        trace_dim = 0
        if self.use_trace:
            trace_hidden = max(64, final_channels // 4)
            self.trace_gru = nn.GRU(final_channels, trace_hidden, batch_first=True, bidirectional=True)
            trace_dim = trace_hidden * 4
            self.trace_norm = nn.LayerNorm(trace_dim)
            self.trace_head = nn.Linear(trace_dim, num_classes)

        freq_dim = 0
        if self.use_frequency:
            freq_dim = final_channels * 6
            self.freq_norm = nn.LayerNorm(freq_dim)
            self.freq_head = nn.Linear(freq_dim, num_classes)

        gate_in = self.base_dim + trace_dim + freq_dim
        gate_count = int(self.use_trace) + int(self.use_frequency)
        self.evidence_gate = nn.Linear(gate_in, gate_count) if gate_count else None
        if self.evidence_gate is not None:
            nn.init.zeros_(self.evidence_gate.weight)
            nn.init.constant_(self.evidence_gate.bias, -2.0)

        self.branch_dims = [self.base_dim]
        if self.use_trace:
            self.branch_dims.append(trace_dim)
        if self.use_frequency:
            self.branch_dims.append(freq_dim)
        self.feature_dim = int(sum(self.branch_dims))

    @staticmethod
    def _band_stats(x: torch.Tensor, dim: int) -> List[torch.Tensor]:
        spectrum = torch.fft.rfft(x.float(), dim=dim).abs()
        n = spectrum.shape[dim]
        cuts = [0, max(1, n // 3), max(2, 2 * n // 3), n]
        bands = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if hi <= lo:
                hi = min(lo + 1, n)
            band = spectrum.narrow(dim, lo, hi - lo)
            bands.append(band.mean(dim=dim))
        return bands

    def forward_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.input_gate is not None:
            x = x * (1.0 + 0.10 * torch.tanh(self.input_gate))
        z = self.stem(x)
        z = self.trunk_output(z)
        z = self.recalibration(z)
        avg = torch.flatten(self.avgpool(z), 1)
        mx = torch.flatten(self.maxpool_head(z), 1)
        base = torch.cat([avg, mx], dim=1)

        trace = None
        if self.use_trace:
            trace_seq = z.mean(dim=2).transpose(1, 2)
            trace_seq, _ = self.trace_gru(trace_seq)
            trace = self.trace_norm(torch.cat([trace_seq.mean(dim=1), trace_seq.amax(dim=1)], dim=1))

        freq = None
        if self.use_frequency:
            depth_signal = z.mean(dim=3)
            trace_signal = z.mean(dim=2)
            freq = torch.cat(
                self._band_stats(depth_signal, dim=2) + self._band_stats(trace_signal, dim=2),
                dim=1,
            )
            freq = self.freq_norm(freq)
        return base, trace, freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base, trace, freq = self.forward_features(x)
        return self.logits_from_features((base, trace, freq))

    def logits_from_features(
        self,
        features: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
    ) -> torch.Tensor:
        base, trace, freq = features
        logits = self.base_head(base)
        extras = []
        gate_inputs = [base]
        if trace is not None:
            extras.append(self.trace_head(trace))
            gate_inputs.append(trace)
        if freq is not None:
            extras.append(self.freq_head(freq))
            gate_inputs.append(freq)
        if extras:
            gates = torch.sigmoid(self.evidence_gate(torch.cat(gate_inputs, dim=1)))
            for idx, extra in enumerate(extras):
                logits = logits + gates[:, idx : idx + 1] * extra
        return logits


def _weights(weight_enum_name: str, pretrained: bool):
    if not pretrained:
        return None
    enum = getattr(models, weight_enum_name, None)
    if enum is None:
        return None
    return getattr(enum, "DEFAULT", None) or getattr(enum, "IMAGENET1K_V1", None)


def _replace_head(model: nn.Module, name: str, num_classes: int) -> nn.Module:
    if hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name.startswith("densenet"):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name.startswith("mobilenet") or name.startswith("efficientnet") or name.startswith("convnext"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name.startswith("swin"):
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif name.startswith("vit"):
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif name.startswith("maxvit"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported classifier head for model {name!r}")
    return model


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build GAFR-RegNet or a public backbone baseline."""

    name = name.lower()
    if name in {"gafr_regnet", "gafr_regnet_y_8gf", "gafr_regnet_y_8gf_no_trace", "gafr_regnet_y_8gf_no_frequency"}:
        base = models.regnet_y_8gf(weights=_weights("RegNet_Y_8GF_Weights", pretrained))
        return GAFRRegNet(
            base,
            num_classes=num_classes,
            use_trace=not name.endswith("_no_trace"),
            use_frequency=not name.endswith("_no_frequency"),
        )

    factories = {
        "regnet_y_8gf": ("regnet_y_8gf", "RegNet_Y_8GF_Weights"),
        "resnet34": ("resnet34", "ResNet34_Weights"),
        "resnet50": ("resnet50", "ResNet50_Weights"),
        "densenet121": ("densenet121", "DenseNet121_Weights"),
        "mobilenet_v3_small": ("mobilenet_v3_small", "MobileNet_V3_Small_Weights"),
        "efficientnet_b0": ("efficientnet_b0", "EfficientNet_B0_Weights"),
        "efficientnet_v2_s": ("efficientnet_v2_s", "EfficientNet_V2_S_Weights"),
        "convnext_tiny": ("convnext_tiny", "ConvNeXt_Tiny_Weights"),
        "convnext_small": ("convnext_small", "ConvNeXt_Small_Weights"),
        "swin_t": ("swin_t", "Swin_T_Weights"),
        "vit_b_16": ("vit_b_16", "ViT_B_16_Weights"),
        "maxvit_t": ("maxvit_t", "MaxVit_T_Weights"),
    }
    if name not in factories:
        raise ValueError(f"Unsupported model {name!r}.")
    factory_name, weight_name = factories[name]
    factory = getattr(models, factory_name)
    model = factory(weights=_weights(weight_name, pretrained))
    return _replace_head(model, name, num_classes)
