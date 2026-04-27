# TinyRelDetector

TinyRelDetector is a tinygrad implementation of a relational visual learner with a Wasserstein detection head and a YOLO-format data loader.

The model combines:

- patch embeddings for image tokens
- slot extraction by attention
- role-filler binding for relational scene representations
- representational-distance matching between image embeddings and target embeddings
- Sinkhorn optimal transport for detection matching
- a dustbin mechanism for unmatched predictions, false positives, false negatives, and empty images

Relational coding as cognitive architecture was proposed by Green and Hummel (2004). Lipman et al. (2024) provide recent neural evidence for invariant relational structure in human visual cortex. TinyRelDetector experiments with that idea in a compact detection setting.

## Layout

```text
TinyRelDetector/
  README.md
  pyproject.toml
  main.py
  TinyRelDetector/
    __init__.py
    tinyrel_detector.py
```

`TinyRelDetector/tinyrel_detector.py` contains the model, losses, YOLO loader, synthetic fallback data, and training functions.

`main.py` contains the CLI entry point:

```python
from TinyRelDetector.tinyrel_detector import ParseArgs, TrainDemo


if __name__ == "__main__":
    args = ParseArgs()
    TrainDemo(
        steps=args.steps,
        batch=args.batch,
        imgHw=tuple(args.img),
        dModel=args.d_model,
        numRoles=args.roles,
        dataYaml=args.data,
        split=args.split,
    )
```

## Installation

Install locally from the repository root:

```bash
pip install -e .
```

The install/distribution name is:

```text
tiny-rel-detector
```

The Python import package is:

```python
import TinyRelDetector
```

Core dependencies:

```bash
pip install tinygrad numpy
```

For YOLO-format datasets, also install:

```bash
pip install pillow pyyaml
```

## Basic imports

```python
from TinyRelDetector import (
    DetectionHead,
    LossWeights,
    RelationalBackbone,
    TargetProjection,
    TrainDemo,
    ParseArgs,
    SyntheticBatch,
    YoloIterBatches,
    OtDetectionLoss,
    RelationalLoss,
)
```

## Quick smoke test

Run a small synthetic training job:

```bash
python main.py --steps 2 --batch 1 --img 64 64 --roles 3 --d_model 32
```

For a faster minimal sanity check:

```bash
python main.py --steps 2 --batch 1 --img 8 8 --roles 1 --d_model 4
```

Image height and width must be divisible by the patch size, which is currently `8`.

## Synthetic training

With no `--data` argument, TinyRelDetector uses a synthetic fallback dataset with simple colored blobs and one detection target per image.

```bash
python main.py --steps 40 --batch 8 --img 64 64 --roles 6 --d_model 64
```

The training log reports:

```text
[010] total=... det=... RSA≈...
```

`RSA` is the correlation between pairwise distances in learned image embeddings and projected target embeddings.

## YOLO-format data

Use a `data.yaml` file:

```yaml
path: /absolute/or/relative/root
train: images/train
val: images/val
test: images/test
names: ["person", "car", "cat"]
```

You can also include `nc`:

```yaml
path: /absolute/or/relative/root
train: images/train
val: images/val
test: images/test
nc: 3
names: ["person", "car", "cat"]
```

Expected directory tree:

```text
root/
  images/
    train/
      image_001.jpg
      image_002.jpg
    val/
      image_003.jpg
    test/
      image_004.jpg
  labels/
    train/
      image_001.txt
      image_002.txt
    val/
      image_003.txt
    test/
      image_004.txt
```

Each label file must use standard YOLO normalized box format:

```text
class_id x_center y_center width height
```

Example:

```text
0 0.512 0.438 0.120 0.200
2 0.250 0.700 0.080 0.140
```

The loader builds per-image tensors:

```text
gt_boxes:  [Mi, 4]
gt_labels: [Mi]
```

where `Mi` is the number of objects in image `i`.

## YOLO-format training

```bash
python main.py \
  --data data.yaml \
  --split train \
  --steps 100 \
  --batch 4 \
  --img 640 640 \
  --roles 20 \
  --d_model 128
```

For a smaller first run:

```bash
python main.py \
  --data data.yaml \
  --split train \
  --steps 10 \
  --batch 1 \
  --img 128 128 \
  --roles 5 \
  --d_model 32
```

## Model components

### Relational backbone

The backbone maps an image to a relational embedding:

```text
image
  -> patch tokens
  -> attention slots
  -> role-filler bindings
  -> scene embedding
```

The role-filler binding stage produces role-specific features used by the detection head.

### Detection head

The detection head predicts one proposal per role:

```text
boxes:      [batch, num_roles, 4]
logits:     [batch, num_roles, num_classes]
objectness: [batch, num_roles]
```

Boxes are represented as normalized YOLO-style coordinates:

```text
cx, cy, w, h
```

### Optimal transport detection loss

Predictions are matched to ground-truth boxes with a Sinkhorn optimal transport objective. The cost combines:

- class cross-entropy
- box L1 distance
- IoU penalty

A dustbin row and column handle unmatched predictions and unmatched ground-truth objects.

### Relational loss

The representation loss combines:

- representational dissimilarity matching
- Wasserstein alignment between learned embeddings and target embeddings
- role orthogonality regularization
- embedding L2 regularization

## CLI

```bash
python main.py --help
```

Arguments:

```text
--data       path to YOLO data.yaml
--split      train, val, or test
--img        image height and width
--roles      number of roles / max proposals
--steps      number of training steps
--batch      batch size
--d_model    hidden dimension
```

## References

- Green, C., & Hummel, J. E. (2004). Relational perception and cognition: Implications for cognitive architecture and the perceptual-cognitive interface. *Psychology of Learning and Motivation*, 44, 201–226. http://labs.psychology.illinois.edu/~jehummel/pubs/Green&Hummel04_PLM.pdf
- Lipman, O., et al. (2024). Invariant inter-subject relational structures in the human visual cortex. *arXiv:2407.08714*. https://arxiv.org/abs/2407.08714

## License

MIT