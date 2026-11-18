"""Multi-resolution tinygrad detector with relational context and tiled inference."""

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image
from safetensors.numpy import load as load_tensors, save as save_tensors
from tinygrad import Tensor, TinyJit, nn
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_state_dict, load_state_dict

from .relational import _nms
from .tinyrel_detector import MultiHeadAttention, RoleFillerBinder, SlotExtractor

STRIDE = 4


class _Conv:
    def __init__(self, inputs, outputs, stride=1, kernel_size=3):
        self.conv = nn.Conv2d(inputs, outputs, kernel_size, stride=stride,
                              padding=kernel_size // 2, bias=False)
        self.norm = nn.GroupNorm(4, outputs)

    def __call__(self, image):
        return self.norm(self.conv(image)).silu()


class _Residual:
    def __init__(self, channels):
        self.first = _Conv(channels, channels, kernel_size=1)
        self.second = _Conv(channels, channels)

    def __call__(self, image):
        return image + self.second(self.first(image))


class _Pyramid:
    def __init__(self, width, roles):
        self.stem = _Conv(3, width, stride=2)
        self.fine = _Conv(width, width, stride=2)
        self.fine_block = _Residual(width)
        self.middle = _Conv(width, 2 * width, stride=2)
        self.middle_block = _Residual(2 * width)
        self.coarse = _Conv(2 * width, 3 * width, stride=2)
        self.coarse_block = _Residual(3 * width)
        self.slots = SlotExtractor(3 * width, 2 * roles, nHead=4)
        self.roles = RoleFillerBinder(3 * width, roles)
        self.readout = MultiHeadAttention(3 * width, 4)
        self.coarse_projection = _Conv(3 * width, 2 * width, kernel_size=1)
        self.middle_projection = _Conv(2 * width, width, kernel_size=1)
        self.merge = _Residual(width)
        self.score = nn.Conv2d(width, 1, 1)
        self.score.bias = Tensor.full((1,), -4.0)
        self.box = nn.Conv2d(width, 4, 1)

    @staticmethod
    def _upsample(tensor, reference):
        return tensor.interpolate(reference.shape[-2:], mode="nearest")

    def __call__(self, image):
        fine = self.fine_block(self.fine(self.stem(image)))
        middle = self.middle_block(self.middle(fine))
        coarse = self.coarse_block(self.coarse(middle))
        batch, channels, height, width = coarse.shape
        tokens = coarse.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        _, _, roles = self.roles(self.slots(tokens))
        context = self.readout(tokens, roles, roles).reshape(batch, height, width, channels)
        coarse = self.coarse_projection(coarse + context.permute(0, 3, 1, 2))
        middle = self.middle_projection(middle + self._upsample(coarse, middle))
        features = self.merge(fine + self._upsample(middle, fine))
        return self.score(features), self.box(features)


def encode_targets(boxes, ignored, size):
    """Encode pixel XYXY boxes as a heatmap, regression targets and ignore mask."""
    cells = size // STRIDE
    target = np.zeros((6, cells, cells), dtype=np.float32)
    target[5] = 1
    for left, top, right, bottom in ignored:
        x0, y0 = np.clip(np.floor([left / STRIDE, top / STRIDE]).astype(int), 0, cells)
        x1, y1 = np.clip(np.ceil([right / STRIDE, bottom / STRIDE]).astype(int), 0, cells)
        target[5, y0:y1, x0:x1] = 0
    # Smaller objects win collisions in a feature cell.
    ordered = sorted(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    for left, top, right, bottom in ordered:
        x, y = (left + right) / (2 * STRIDE), (top + bottom) / (2 * STRIDE)
        ix, iy = int(np.floor(x)), int(np.floor(y))
        width, height = (right - left) / STRIDE, (bottom - top) / STRIDE
        if not (0 <= ix < cells and 0 <= iy < cells) or min(width, height) <= 0:
            continue
        if target[0, iy, ix] == 1:
            continue
        radius = min(4, int(min(width, height) / 3))
        xx = np.arange(max(0, ix - radius), min(cells, ix + radius + 1))
        yy = np.arange(max(0, iy - radius), min(cells, iy + radius + 1))
        sigma = max(0.4, (2 * radius + 1) / 6)
        kernel = np.exp(-((xx[None, :] - ix)**2 + (yy[:, None] - iy)**2) / (2 * sigma**2))
        region = np.ix_(yy, xx)
        target[0][region] = np.maximum(target[0][region], kernel)
        target[1:5, iy, ix] = [x - ix, y - iy, np.log(width), np.log(height)]
        target[5, iy, ix] = 1
    return target


def _objective(scores, boxes, target):
    probability = scores.sigmoid()
    heatmap = target[:, :1]
    positive = (heatmap == 1).cast(scores.dtype)
    valid = target[:, 5:]
    count = positive.sum().maximum(1)
    classification = (
        (-scores).softplus() * (1 - probability).square() * positive
        + scores.softplus() * probability.square() * (1 - heatmap).pow(4) * valid * (1 - positive)
    ).sum() / count
    estimated = boxes[:, :2].sigmoid().cat(boxes[:, 2:], dim=1)
    difference = (estimated - target[:, 1:5]).abs()
    regression = ((difference < 1).where(0.5 * difference.square(), difference - 0.5)
                  * positive).sum() / count
    return classification + 2 * regression


def _tile_starts(length, crop, overlap):
    if length <= crop:
        return [0]
    step = max(1, int(crop * (1 - overlap)))
    return sorted(set(range(0, length - crop + 1, step)) | {length - crop})


def crop_image(image, x, y, crop, size):
    """Pad beyond the image edge and rescale a square crop into a CHW tensor."""
    tile = image.crop((x, y, min(x + crop, image.width), min(y + crop, image.height)))
    if tile.size != (crop, crop):
        padded = Image.new("RGB", (crop, crop), (114, 114, 114))
        padded.paste(tile)
        tile = padded
    if crop != size:
        tile = tile.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(tile, dtype=np.float32).transpose(2, 0, 1) / 255


class PyramidRelationalDetector:
    """Anchor-free person detector with stride-4 features and role-based context."""

    def __init__(self, *, image_size=256, crop_size=192, width=16, roles=4):
        if image_size < 32 or image_size % 16 or crop_size < 16 or width < 4 or width % 4 or roles < 1:
            raise ValueError("Use image_size divisible by 16, crop_size >= 16, and width divisible by 4.")
        self.image_size, self.crop_size, self.width, self.roles = image_size, crop_size, width, roles
        self.network = _Pyramid(width, roles)
        self._inference_batch = None

    def trainer(self, learning_rate):
        """Build one optimizer and return an update callable for fixed-size batches."""
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        parameters = list(get_state_dict(self.network).values())
        for parameter in parameters:
            parameter.requires_grad_(True)
        optimizer = AdamW(parameters, lr=learning_rate, weight_decay=0.0001)

        @TinyJit
        def step(images, targets):
            with Tensor.train():
                optimizer.zero_grad()
                scores, boxes = self.network(images)
                loss = _objective(scores, boxes, targets)
                loss.backward()
                optimizer.step()
                return loss.detach().realize()

        def update(images, targets):
            return float(step(Tensor(np.ascontiguousarray(images)),
                              Tensor(np.ascontiguousarray(targets))).item())

        return update

    def _inference_tensors(self, batch):
        with Tensor.train(False):
            scores, offsets = self.network(batch)
            probability = scores.sigmoid()
            peaks = probability.max_pool2d((3, 3), stride=1, padding=1)
            return peaks.realize(probability, offsets), probability, offsets

    def predict(self, source, *, confidence=0.001, iou=0.5, overlap=0.25, batch=4,
                max_detections=300):
        """Return pixel boxes after overlapping tiled inference and cross-tile NMS."""
        if not 0 <= confidence <= 1 or not 0 <= overlap < 1 or not 0 <= iou <= 1:
            raise ValueError("Require valid confidence, overlap and IoU thresholds.")
        if batch < 1 or max_detections < 1:
            raise ValueError("batch and max_detections must be positive.")
        if isinstance(source, Image.Image):
            image = source.convert("RGB")
        else:
            with Image.open(source) as handle:
                image = handle.convert("RGB")
        crop, size = self.crop_size, self.image_size
        origins = [(x, y) for y in _tile_starts(image.height, crop, overlap)
                   for x in _tile_starts(image.width, crop, overlap)]
        all_boxes, all_scores = [], []
        scale = STRIDE * crop / size
        for start in range(0, len(origins), batch):
            tiles = origins[start:start + batch]
            pixels = np.stack([crop_image(image, x, y, crop, size) for x, y in tiles])
            if len(tiles) < batch:
                pixels = np.concatenate((pixels, np.repeat(pixels[-1:], batch - len(tiles), axis=0)))
            if self._inference_batch != batch:
                self._forward = TinyJit(self._inference_tensors)
                self._inference_batch = batch
            peaks, confidence_maps, offsets = (tensor.numpy() for tensor in self._forward(
                Tensor(np.ascontiguousarray(pixels))
            ))
            for index, (x, y) in enumerate(tiles):
                scores = confidence_maps[index, 0]
                ys, xs = np.nonzero((scores >= confidence) & (scores == peaks[index, 0]))
                if len(xs) > max_detections:
                    best = np.argpartition(scores[ys, xs], -max_detections)[-max_detections:]
                    ys, xs = ys[best], xs[best]
                if len(xs) == 0:
                    continue
                raw = offsets[index][:, ys, xs]
                cx = x + scale * (xs + 1 / (1 + np.exp(-np.clip(raw[0], -30, 30))))
                cy = y + scale * (ys + 1 / (1 + np.exp(-np.clip(raw[1], -30, 30))))
                width, height = scale * np.exp(np.clip(raw[2:], -4, 5))
                valid = (cx < image.width) & (cy < image.height)
                boxes = np.column_stack((cx - width / 2, cy - height / 2,
                                         cx + width / 2, cy + height / 2))
                all_boxes.append(boxes[valid].clip(0, [image.width, image.height] * 2))
                all_scores.append(scores[ys, xs][valid])
        boxes = np.concatenate(all_boxes) if all_boxes else np.empty((0, 4))
        scores = np.concatenate(all_scores) if all_scores else np.empty(0)
        if len(scores) > 4 * max_detections:
            count = 4 * max_detections
            select = np.argpartition(scores, -count)[-count:]
            boxes, scores = boxes[select], scores[select]
        labels = np.zeros(len(scores), dtype=int)
        keep = _nms(boxes, scores, labels, iou)[:max_detections]
        return {"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]}

    def save(self, path):
        """Store weights and architecture metadata without pickled code."""
        architecture = dict(image_size=self.image_size, crop_size=self.crop_size,
                            width=self.width, roles=self.roles)
        path = Path(path)
        arrays = {name: tensor.numpy() for name, tensor in get_state_dict(self.network).items()}
        payload = save_tensors(arrays, metadata={"pyramid_architecture": json.dumps(architecture)})
        pending = None
        try:
            with NamedTemporaryFile(dir=path.parent, prefix=".pyramid-", delete=False) as handle:
                pending = Path(handle.name)
                handle.write(payload)
            pending.replace(path)
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)

    @classmethod
    def load(cls, path, *, crop_size=None):
        payload = Path(path).read_bytes()
        length = int.from_bytes(payload[:8], "little")
        if not 2 <= length <= min(len(payload) - 8, 16_000_000):
            raise ValueError("Invalid checkpoint header length.")
        header = json.loads(payload[8:8 + length])
        architecture = json.loads(header["__metadata__"]["pyramid_architecture"])
        if crop_size is not None:
            architecture["crop_size"] = crop_size
        model = cls(**architecture)
        weights = {name: Tensor(array) for name, array in load_tensors(payload).items()}
        if weights.keys() != get_state_dict(model.network).keys():
            raise ValueError("Checkpoint weights do not match the architecture.")
        load_state_dict(model.network, weights, strict=True, verbose=False)
        return model
