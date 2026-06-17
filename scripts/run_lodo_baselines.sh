#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-datasets/gpr_multisource_risk_lodo}"
RUN_ROOT="${RUN_ROOT:-runs/lodo}"
DEVICE="${DEVICE:-cuda:0}"

models=(
  "gafr_regnet_y_8gf"
  "regnet_y_8gf"
  "convnext_small"
  "maxvit_t"
  "swin_t"
  "efficientnet_v2_s"
  "vit_b_16"
  "densenet121"
  "resnet34"
)

heldouts=("tigpr" "utility" "urdd")

for heldout in "${heldouts[@]}"; do
  data="${DATA_ROOT}/risk_lodo_to_${heldout}_seed2037"
  for model in "${models[@]}"; do
    out="${RUN_ROOT}/${model}_to_${heldout}"
    python scripts/train_risk_model.py \
      --data "${data}" \
      --out "${out}" \
      --model "${model}" \
      --pretrained \
      --gpr-physics-aug \
      --selection-metric source_worst_min_cost \
      --device "${DEVICE}"
  done
done
