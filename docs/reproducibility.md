# Reproducibility

This repository provides code for model construction, data preparation, training and basic inference.

The repository intentionally does not include:

- dataset files or downloaded public datasets;
- trained checkpoints;
- manuscript-specific post-processing utilities;
- private validation scripts;
- machine-specific paths or credentials.

To reproduce a public-data experiment, prepare ImageFolder-style datasets, create the desired leave-one-domain-out view,
train the proposed model and train each public baseline under the same split and hyperparameter protocol.
