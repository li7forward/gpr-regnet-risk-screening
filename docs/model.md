# Model

The proposed model is `GAFR-RegNet`.

The semantic branch uses RegNetY-8GF because controlled public-backbone screening showed that it is a strong
image-level B-scan classifier while remaining easier to train than larger transformer-only alternatives. The proposed
contribution is not the backbone alone; the B-scan-specific part is the gated axial-frequency risk head.

## Structural Components

- Input amplitude gate: a small learnable per-channel scale for B-scan rendering differences.
- Physics-guided feature recalibration: residual channel attention plus depth-axis and trace-axis spatial gates.
- Trace branch: averages feature maps along depth, treats lateral traces as a sequence, and uses a bidirectional GRU.
- Frequency branch: computes band statistics from depth and trace signals after the RegNet trunk.
- Gated residual fusion: base logits are corrected by trace and frequency logits through a learned confidence gate.

## Cross-Domain Generalization

The training wrapper adds two constraints designed for multi-source GPR domain shift:

- gradient-reversal domain classification encourages domain-invariant high-level features;
- class-conditional CORAL aligns same-label feature distributions across source domains.

This is intended for acquisition changes such as antenna response, pipe radius, soil/concrete background, clutter and
noise, where mixed-source accuracy alone is not enough to prove robustness.
