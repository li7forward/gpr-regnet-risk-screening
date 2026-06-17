# Experiments

Recommended main evidence:

- LODO cross-domain table: each public source is held out as test domain, with train/val using the remaining sources.
- Strong public backbone table: GAFR-RegNet versus RegNetY-8GF, ConvNeXt, MaxViT, Swin, ViT, DenseNet, ResNet and EfficientNet under the same data protocol.
- Ablation table: full GAFR-RegNet, no trace branch, no frequency branch, no GPR physics augmentation, and no MSDA losses.
- Complexity table: parameters, GMACs and inference speed.
- Grad-CAM figure: compare GAFR-RegNet against strong public baselines on correctly detected and difficult samples.
- Real specimen validation: foam, steel pipe, plastic pipe and plain concrete, with scan-line level results, specimen-majority results and reverse-consistency checks.

Do not directly compare private-paper reported numbers with this repository's numbers as strict baselines unless data,
splits and labels are identical. Non-public papers should be discussed as related work or reference results.
