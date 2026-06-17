# GAFR-RegNet for GPR Image-Level Risk Recognition

This repository contains the code for image-level ground-penetrating radar (GPR) internal-defect risk recognition.
Given one B-scan image, the model predicts whether defect risk is present. The current release focuses on cross-domain
generalization across public GPR-style datasets and real concrete specimens.

Author: li7forward

## Main Idea

`GAFR-RegNet` uses a RegNetY-8GF semantic trunk and adds three B-scan-aware components:

- physics-guided feature recalibration for channel, time/depth and trace-axis responses;
- a lateral trace-sequence branch for scan-direction response evolution;
- a frequency-statistics branch for depth and trace spectral evidence;
- gated residual fusion so trace/frequency evidence can help without overwhelming the base classifier.

The training wrapper adds multi-source domain adaptation:

- gradient-reversal domain alignment;
- class-conditional CORAL alignment;
- validation-selected operating points for risk screening.

Public backbones such as ConvNeXt, MaxViT, Swin, ViT, DenseNet, ResNet and EfficientNet are included as controlled
baselines, not as the proposed method.

## Repository Layout

```text
src/gpr_risk/
  models.py              # GAFR-RegNet and public backbone builders
  transforms.py          # B-scan transforms and GPR physics-style augmentation
  train.py               # training, validation, threshold selection and MSDA losses
scripts/
  create_lodo_views.py   # leave-one-domain-out dataset views
  create_allsource_view.py
  train_risk_model.py
  summarize_runs.py
  benchmark_complexity.py
  predict_real_specimens.py
  fewshot_real_calibration.py
  visualize_gradcam.py
  build_result_figures.py
configs/
  gafr_regnet_lodo.yaml
docs/
  experiments.md
  model.md
examples/
  real_specimens/README.md
```

Datasets, trained weights, server paths and manuscript files are intentionally not included.

## Installation

```bash
conda create -n gpr-risk python=3.10 -y
conda activate gpr-risk
pip install -r requirements.txt
pip install -e .
```

Install the PyTorch build that matches your CUDA version if the default wheel is not suitable.

## Dataset Format

Training uses ImageFolder-style splits:

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

For leave-one-domain-out training, filenames should start with a source prefix such as:

```text
tigpr__train__sample.png
utility__val__sample.png
urdd__test__sample.png
```

The prefix is used only for domain-adaptation losses and per-domain validation.

## Quick Start

Create LODO views:

```bash
python scripts/create_lodo_views.py \
  --tigpr-view datasets/tigpr_binary_damage \
  --utility-view datasets/utility_void_binary \
  --urdd-view datasets/urdd_multiclass \
  --out-root datasets/gpr_multisource_risk_lodo
```

Train GAFR-RegNet on one view:

```bash
python scripts/train_risk_model.py \
  --data datasets/gpr_multisource_risk_lodo/risk_lodo_to_urdd_seed2037 \
  --out runs/gafr_regnet_lodo_urdd \
  --model gafr_regnet_y_8gf \
  --pretrained \
  --gpr-physics-aug \
  --selection-metric source_worst_min_cost
```

Train a public strong baseline under the same protocol:

```bash
python scripts/train_risk_model.py \
  --data datasets/gpr_multisource_risk_lodo/risk_lodo_to_urdd_seed2037 \
  --out runs/convnext_small_lodo_urdd \
  --model convnext_small \
  --pretrained \
  --selection-metric source_worst_min_cost
```

Summarize runs:

```bash
python scripts/summarize_runs.py --runs runs/gafr_regnet_lodo_urdd runs/convnext_small_lodo_urdd
```

Generate Grad-CAM overlays:

```bash
python scripts/visualize_gradcam.py \
  --run-dir runs/gafr_regnet_lodo_urdd \
  --images examples/real_specimens/foam_line1.png \
  --out-dir outputs/gradcam
```

## Real Specimen Validation

The real-specimen scripts assume four specimen types with two scan lines each:

- foam surrogate: low-density void surrogate;
- plain concrete: negative control;
- steel pipe: metallic pipe / strong reflector;
- plastic pipe: non-metallic pipe or PVC-like inclusion.

Run fixed-threshold screening:

```bash
python scripts/predict_real_specimens.py \
  --input-root examples/real_specimens \
  --run-dir runs/gafr_regnet_allsource \
  --out-dir outputs/source_data
```

Run line-heldout few-shot calibration:

```bash
python scripts/fewshot_real_calibration.py \
  --input-root examples/real_specimens \
  --run-dir runs/gafr_regnet_allsource \
  --out-dir outputs/source_data
```

## Open-Source Scope

This release contains only code and lightweight documentation. It does not include:

- private datasets or downloaded public datasets;
- model weights or experiment logs;
- server credentials, absolute server paths or manuscript packages.

## License

MIT License. See `LICENSE`.
