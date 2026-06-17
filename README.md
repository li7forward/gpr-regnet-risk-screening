# GAFR-RegNet for GPR Image-Level Risk Recognition

Official code for GAFR-RegNet, a GPR B-scan image-level internal-defect risk recognition model.

Given one B-scan image, the model predicts whether defect risk is present. The repository contains the model, training
pipeline, dataset-view builders and a lightweight inference script.

Author: li7forward

## Method

`GAFR-RegNet` uses a RegNetY-8GF semantic trunk with B-scan-specific risk evidence:

- physics-guided channel, time/depth and trace-axis feature recalibration;
- a lateral trace-sequence branch;
- a depth/trace frequency-statistics branch;
- gated residual fusion of semantic, trace and frequency logits.

The training wrapper supports multi-source domain generalization with gradient-reversal domain alignment and
class-conditional CORAL alignment.

## Repository Layout

```text
src/gpr_risk/
  models.py              # GAFR-RegNet and public backbone builders
  transforms.py          # B-scan transforms and GPR-style augmentation
  train.py               # training, validation and domain-alignment losses
scripts/
  create_lodo_views.py   # leave-one-domain-out ImageFolder views
  create_allsource_view.py
  train_risk_model.py
  predict_images.py
  run_lodo_baselines.sh
configs/
  gafr_regnet_lodo.yaml
docs/
  model.md
  reproducibility.md
```

## Installation

```bash
conda create -n gpr-risk python=3.10 -y
conda activate gpr-risk
pip install -r requirements.txt
pip install -e .
```

Install the PyTorch build that matches your CUDA version if the default wheel is not suitable.

## Dataset Format

The training scripts use ImageFolder-style splits:

```text
dataset_root/
  train/
    damage/
    no_damage/
  val/
    damage/
    no_damage/
  test/
    damage/
    no_damage/
```

For multi-source training, filenames should start with a source prefix:

```text
tigpr__train__sample.png
utility__val__sample.png
urdd__test__sample.png
```

The prefix is used to infer the domain id for domain-alignment losses and per-domain validation.

## Training

Create leave-one-domain-out views:

```bash
python scripts/create_lodo_views.py \
  --tigpr-view datasets/tigpr_binary_damage \
  --utility-view datasets/utility_void_binary \
  --urdd-view datasets/urdd_multiclass \
  --out-root datasets/gpr_multisource_risk_lodo
```

Train GAFR-RegNet:

```bash
python scripts/train_risk_model.py \
  --data datasets/gpr_multisource_risk_lodo/risk_lodo_to_urdd_seed2037 \
  --out runs/gafr_regnet_lodo_urdd \
  --model gafr_regnet_y_8gf \
  --pretrained \
  --gpr-physics-aug \
  --selection-metric source_worst_min_cost
```

Train a public backbone baseline under the same protocol:

```bash
python scripts/train_risk_model.py \
  --data datasets/gpr_multisource_risk_lodo/risk_lodo_to_urdd_seed2037 \
  --out runs/convnext_small_lodo_urdd \
  --model convnext_small \
  --pretrained \
  --selection-metric source_worst_min_cost
```

## Inference

```bash
python scripts/predict_images.py \
  --input path/to/bscan_or_folder \
  --run-dir runs/gafr_regnet_lodo_urdd \
  --out predictions.csv
```

## Open-Source Scope

This repository only includes code that is suitable for public release. It does not include datasets, checkpoints,
manuscript-specific post-processing utilities, private validation scripts, server paths or credentials.

## License

MIT License. See `LICENSE`.
