"""Training, checkpointing and image inference for the native relational model."""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from tinygrad import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save

from .tinyrel_detector import (
    CxcywhToXyxy, DetectionHead, EnableGrad, LossWeights, OtDetectionLoss,
    RelationalBackbone, RelationalLoss, SyntheticBatch, TargetProjection, YoloIterBatches,
)


def _nms(boxes, scores, labels, threshold):
    order = scores.argsort()[::-1]
    kept = []
    sizes = np.maximum(boxes[:, 2:] - boxes[:, :2], 0)
    areas = sizes[:, 0] * sizes[:, 1]
    while len(order):
        current, rest = order[0], order[1:]
        kept.append(current)
        top_left = np.maximum(boxes[current, :2], boxes[rest, :2])
        bottom_right = np.minimum(boxes[current, 2:], boxes[rest, 2:])
        sizes = np.maximum(bottom_right - top_left, 0)
        intersection = sizes[:, 0] * sizes[:, 1]
        union = areas[current] + areas[rest] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[(labels[rest] != labels[current]) | (iou <= threshold)]
    return np.asarray(kept, dtype=int)


class RelationalDetector:
    """Train a tinygrad relational backbone and predict pixel-coordinate boxes."""

    def __init__(self, num_classes, *, image_size=(64, 64), model_width=32, roles=6):
        if (len(image_size) != 2 or any(s < 8 or s % 8 for s in image_size)
                or model_width < 4 or model_width % 4 or roles < 1 or num_classes < 1):
            raise ValueError("Require positive class and role counts, image size divisible by 8, "
                             "and width divisible by 4.")
        self.num_classes = num_classes
        self.image_size = tuple(image_size)
        self.model_width = model_width
        self.roles = roles
        self.backbone = RelationalBackbone(
            dModel=model_width, patch=8, numSlots=2 * roles, numRoles=roles, nHead=4,
        )
        self.head = DetectionHead(dModel=model_width, numRoles=roles, numClasses=num_classes)

    def fit(self, data=None, *, steps=40, batch=4, learning_rate=0.002, split="train"):
        """Fit on a YOLO-format YAML dataset or the synthetic demo generator."""
        if steps < 1 or batch < 1 or learning_rate <= 0:
            raise ValueError("Training settings must be positive.")
        height, width = self.image_size
        dataset = None if data is None else YoloIterBatches(str(data), split, height, width, batch)

        def next_batch():
            if dataset is not None:
                return next(dataset)
            return SyntheticBatch(batch, height, width, self.num_classes)

        sample = next_batch()
        projection = TargetProjection(dIn=sample[1].shape[1], dOut=self.model_width)
        parameters = self.backbone.Parameters() + projection.Parameters() + self.head.Parameters()
        EnableGrad(parameters)
        optimizer = Adam(parameters, lr=learning_rate)
        weights = LossWeights()
        losses = []
        with Tensor.train():
            for step in range(steps):
                images, targets, gt_boxes, gt_labels, classes = sample if step == 0 else next_batch()
                if classes != self.num_classes:
                    raise ValueError("Dataset class count must equal num_classes.")
                embedding, roles, bound = self.backbone(images)
                boxes, logits, objectness = self.head(bound)
                detection = sum(OtDetectionLoss(
                    boxes[i], logits[i], gt_boxes[i], gt_labels[i], objLogit=objectness[i],
                    eps=0.07, uot=True,
                ) for i in range(len(gt_boxes))) / len(gt_boxes)
                loss = RelationalLoss(embedding, projection(targets), roles, weights) + detection
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().numpy()))
                if not np.isfinite(losses[-1]):
                    raise FloatingPointError("Training loss became nonfinite.")
        return losses

    def predict(self, source, *, confidence=0.25, iou=0.5):
        """Predict class IDs, confidence scores and xyxy boxes for a single image."""
        if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
            raise ValueError("confidence and iou must be in [0, 1].")
        if isinstance(source, Image.Image):
            image = source.convert("RGB")
        else:
            with Image.open(source) as handle:
                image = handle.convert("RGB")
        original_width, original_height = image.size
        height, width = self.image_size
        pixels = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR),
                            dtype=np.float32) / 255
        inputs = Tensor(pixels.transpose(2, 0, 1)[None])
        with Tensor.train(False):
            _, _, bound = self.backbone(inputs)
            boxes, logits, objectness = self.head(bound)
            probabilities = logits[0].softmax(axis=-1).numpy()
            labels = probabilities.argmax(axis=1)
            scores = probabilities.max(axis=1) * objectness[0].sigmoid().numpy().reshape(-1)
            boxes = CxcywhToXyxy(boxes[0]).numpy().clip(0, 1)
        boxes *= np.array([original_width, original_height] * 2, dtype=boxes.dtype)
        valid = (scores >= confidence) & (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
        boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
        keep = _nms(boxes, scores, labels, iou)
        return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}

    def save(self, path):
        """Save native weights and JSON architecture metadata in one safetensors file."""
        metadata = {
            "num_classes": self.num_classes, "image_size": self.image_size,
            "model_width": self.model_width, "roles": self.roles,
        }
        weights = get_state_dict({"backbone": self.backbone, "head": self.head})
        safe_save(weights, str(Path(path)), metadata={"architecture": json.dumps(metadata)})

    @classmethod
    def load(cls, path):
        """Load a native safetensors checkpoint."""
        with Path(path).open("rb") as checkpoint:
            size = int.from_bytes(checkpoint.read(8), "little")
            if size < 2 or size > 16_000_000:
                raise ValueError("Invalid checkpoint header length.")
            header = json.loads(checkpoint.read(size))
        metadata = json.loads(header["__metadata__"]["architecture"])
        model = cls(**metadata)
        weights = safe_load(str(path))
        model_state = {"backbone": model.backbone, "head": model.head}
        expected = get_state_dict(model_state)
        if weights.keys() != expected.keys():
            raise ValueError("Checkpoint weights do not match the architecture.")
        load_state_dict(model_state, weights, strict=True, verbose=False)
        return model
