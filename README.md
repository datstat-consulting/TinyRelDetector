# TinyRelDetector
Relational coding as cognitive architecture was proposed by Green and Hummel (2004). Lipman et al (2024) confirm it by analyzing neural scans of epileptic patients. We use metric learning with a Wasserstein metric augmented by YOLOv9's mechanisms.

Implemented in `TinyGrad`.

# Data instructions
A `data.yaml` file:
```
path: /absolute/or/relative/root
train: images/train
val: images/val
test: images/test
names: ["person","car","cat", ...]   # or nc: <int> and names list
```
Directory tree:
```
root/
  images/
    train/ ... .jpg|.png
    val/   ...
    test/  ...
  labels/
    train/ ... .txt
    val/   ...
    test/  ...
```
Each label .txt line must have `class_id x_center y_center width height` (all normalized to [0,1]). The script loads those, builds per-image tensors of GT boxes `[Mi,4]` and GT labels `[Mi]`, then matches them to `num_roles` predictions with (un)balanced Optimal Transport (a dustbin handles FPs/FNs and empty images).

# References
- Green, Collin, and John E. Hummel. "Relational perception and cognition: Implications for cognitive architecture and the perceptual-cognitive interface." Psychology of Learning and Motivation 44 (2004): 201-226.
- - Retrieved from http://labs.psychology.illinois.edu/~jehummel/pubs/Green&Hummel04_PLM.pdf
- Lipman, Ofer, et al. "Invariant inter-subject relational structures in the human visual cortex." arXiv preprint arXiv:2407.08714 (2024).
- - Retrieved from https://arxiv.org/abs/2407.08714
