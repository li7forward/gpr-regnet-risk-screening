"""Image transforms and GPR-oriented augmentation."""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
from PIL import Image
from torchvision import transforms


def imagenet_norm() -> transforms.Normalize:
    return transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


class GPRPhysicsAugment:
    """Lightweight B-scan augmentation for domain shift stress.

    The transform perturbs lateral trace position, time/depth alignment, gain,
    low-frequency clutter and noise. It is intentionally simple and transparent:
    it does not create new labels, only simulates common acquisition changes.
    """

    def __init__(
        self,
        p: float = 0.6,
        max_time_shift: float = 0.04,
        max_trace_shift: float = 0.03,
        noise_std: float = 0.025,
        gain_range: Tuple[float, float] = (0.85, 1.15),
        clutter_strength: float = 0.18,
        phase_jitter: float = 0.35,
    ) -> None:
        self.p = float(p)
        self.max_time_shift = float(max_time_shift)
        self.max_trace_shift = float(max_trace_shift)
        self.noise_std = float(noise_std)
        self.gain_range = tuple(float(x) for x in gain_range)
        self.clutter_strength = float(clutter_strength)
        self.phase_jitter = float(phase_jitter)

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return image
        arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
        h, w, _ = arr.shape

        max_dy = int(round(self.max_time_shift * h))
        if max_dy > 0:
            arr = np.roll(arr, random.randint(-max_dy, max_dy), axis=0)

        max_dx = int(round(self.max_trace_shift * w))
        if max_dx > 0:
            arr = np.roll(arr, random.randint(-max_dx, max_dx), axis=1)

        arr *= random.uniform(*self.gain_range)

        if self.clutter_strength > 0:
            mean_trace = arr.mean(axis=1, keepdims=True)
            lateral = np.linspace(0.0, 2.0 * np.pi, w, dtype=np.float32)[None, :, None]
            phase = random.uniform(0.0, 2.0 * np.pi) * max(self.phase_jitter, 1e-6)
            modulation = 0.5 + 0.5 * np.sin(lateral + phase)
            arr += self.clutter_strength * (mean_trace - mean_trace.mean()) * modulation

        if self.noise_std > 0:
            arr += np.random.normal(0.0, self.noise_std, size=arr.shape).astype(np.float32)

        arr = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((arr * 255.0).astype(np.uint8))


def build_transforms(
    imgsz: int,
    input_mode: str = "rgb",
    gpr_physics_aug: bool = False,
    gpr_aug_p: float = 0.6,
    gpr_max_time_shift: float = 0.04,
    gpr_max_trace_shift: float = 0.03,
    gpr_noise_std: float = 0.025,
    gpr_gain_range: Tuple[float, float] = (0.85, 1.15),
    gpr_clutter_strength: float = 0.18,
    gpr_dielectric_jitter: float = 0.10,
    gpr_phase_jitter: float = 0.35,
):
    """Return train/eval transforms for ImageFolder-style B-scan datasets."""

    if input_mode != "rgb":
        raise ValueError("This open-source release expects RGB-rendered B-scan images.")

    train_steps = [transforms.Resize((imgsz, imgsz))]
    if gpr_physics_aug:
        train_steps.append(
            GPRPhysicsAugment(
                p=gpr_aug_p,
                max_time_shift=gpr_max_time_shift * (1.0 + gpr_dielectric_jitter),
                max_trace_shift=gpr_max_trace_shift,
                noise_std=gpr_noise_std,
                gain_range=gpr_gain_range,
                clutter_strength=gpr_clutter_strength,
                phase_jitter=gpr_phase_jitter,
            )
        )
    train_steps.extend([transforms.ToTensor(), imagenet_norm()])
    eval_steps = [transforms.Resize((imgsz, imgsz)), transforms.ToTensor(), imagenet_norm()]
    return transforms.Compose(train_steps), transforms.Compose(eval_steps)
