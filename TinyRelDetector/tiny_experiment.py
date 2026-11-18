"""Train and compare the native pyramid on the held-out TinyPerson export."""

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from io import StringIO
import json
from shutil import copyfile
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image
from tinygrad import Device, Tensor

from .benchmarks import benchmark_coco, score_coco
from .pyramid import PyramidRelationalDetector, crop_image, encode_targets
from .tiny_export import _SOURCE, _candidates, _prepare_export
from .tinyrel_detector import ReadYoloTxt


@dataclass(frozen=True)
class _Sample:
    source: str
    file: Path
    width: int
    height: int
    boxes: np.ndarray
    ignored: np.ndarray


def _video(source):
    return source.rsplit("_i", 1)[0]


@lru_cache(maxsize=96)
def _image(path):
    with Image.open(path) as handle:
        return handle.convert("RGB")


def _export_samples(split, files, label_class):
    samples = []
    for source, file in files.items():
        with Image.open(file) as handle:
            width, height = handle.size
        label = split / "labels" / file.with_suffix(".txt").name
        if not label.is_file():
            raise FileNotFoundError(label)
        classes, normalized = ReadYoloTxt(str(label), allow_segments=True)
        center, half_size = normalized[:, :2], normalized[:, 2:] / 2
        corners = np.column_stack((center - half_size, center + half_size))
        boxes = (corners * [width, height, width, height]).clip(0, [width, height]*2)
        samples.append(_Sample(source, file, width, height,
                               boxes[classes == label_class], boxes[classes != label_class]))
    return samples


def _split_sources(root, test, seed, validation_images, label_class):
    held_out_videos = {_video(_SOURCE.match(img["file_name"])[1].lower()) for img in test["images"]}

    def sources(split):
        selected = {}
        for source, file in _candidates(root / split):
            if _video(source) not in held_out_videos:
                selected.setdefault(source, file)
        return selected

    train, validation = sources("train"), sources("valid")
    counts = {}
    for source in validation:
        video = _video(source)
        counts[video] = counts.get(video, 0) + 1
    eligible = sorted((video for video, count in counts.items() if count >= 3))
    if len(eligible) < 2:
        raise ValueError("Need at least two eligible validation videos outside the test set.")
    rng = np.random.default_rng(seed)
    validation_videos = set(rng.choice(eligible, min(4, len(eligible)), replace=False))
    train = {source: path for source, path in train.items() if _video(source) not in validation_videos}
    validation = {source: path for source, path in validation.items()
                  if _video(source) in validation_videos}
    if len(validation) > validation_images:
        subset = set(rng.choice(list(validation), validation_images, replace=False))
        validation = {key: path for key, path in validation.items() if key in subset}
    return (_export_samples(root / "train", train, label_class),
            _export_samples(root / "valid", validation, label_class), held_out_videos)


def _coco_samples(annotation_path, image_dir, limit):
    from pycocotools.coco import COCO

    dataset = COCO(str(annotation_path))
    people = dataset.getCatIds(catNms=["person"])
    if len(people) != 1:
        raise ValueError("COCO annotations must contain the person category.")
    samples = []
    entries = sorted(dataset.dataset["images"], key=lambda row: row["id"])
    for entry in entries[:limit]:
        path = image_dir / entry["file_name"]
        if not path.is_file():
            continue
        rows = dataset.loadAnns(dataset.getAnnIds(imgIds=[entry["id"]], catIds=people))
        positive, ignored = [], []
        for annotation in rows:
            x, y, w, h = annotation["bbox"]
            if min(w, h) <= 0:
                continue
            boxes = ignored if annotation.get("iscrowd") else positive
            boxes.append((x, y, x + w, y + h))
        if positive:
            samples.append(_Sample(str(entry["id"]), path, entry["width"], entry["height"],
                                   np.asarray(positive).reshape(-1, 4), np.asarray(ignored).reshape(-1, 4)))
    return samples


def _area(boxes):
    return np.prod(np.maximum(0, boxes[:, 2:] - boxes[:, :2]), axis=1)


def _random_tile(sample, model, rng):
    image = _image(sample.file)
    size = model.image_size
    crop = max(32, round(model.crop_size * rng.uniform(0.72, 1.38)))
    if len(sample.boxes) and rng.random() < 0.78:
        chosen = sample.boxes[rng.integers(len(sample.boxes))]
        center = (chosen[:2] + chosen[2:]) / 2
        x, y = np.floor(center - crop * rng.uniform(0.22, 0.78, size=2)).astype(int)
        x, y = int(np.clip(x, 0, max(0, sample.width - crop))), int(np.clip(y, 0, max(0, sample.height - crop)))
    else:
        x = int(rng.integers(max(0, sample.width - crop) + 1))
        y = int(rng.integers(max(0, sample.height - crop) + 1))
    scale, origin = size / crop, [x, y, x, y]
    positive = (sample.boxes - origin) * scale
    ignored = ((sample.ignored - origin) * scale).clip(0, size)
    clipped = positive.clip(0, size)
    visible = _area(clipped)
    center_in = ((positive[:, :2] + positive[:, 2:]) >= 0).all(axis=1) & (
        (positive[:, :2] + positive[:, 2:]) < 2 * size
    ).all(axis=1)
    keep = center_in & (visible >= 0.5 * _area(positive)) & (visible > 0)
    ignored = np.concatenate((ignored[_area(ignored) > 0], clipped[(visible > 0) & ~keep]))
    boxes = clipped[keep]
    pixels = crop_image(image, x, y, crop, size)
    if rng.random() < 0.5:
        pixels = pixels[:, :, ::-1]
        boxes[:, [0, 2]] = size - boxes[:, [2, 0]]
        ignored[:, [0, 2]] = size - ignored[:, [2, 0]]
    pixels = (pixels * np.float32(rng.uniform(0.78, 1.19))
              + np.float32(rng.uniform(-0.07, 0.07))).clip(0, 1)
    return pixels, encode_targets(boxes, ignored, size)


def _phase_step(samples, model, optimizer, steps, batch, rng):
    if not samples:
        raise ValueError("No training images available for this phase.")
    remaining = steps * batch
    epoch_plan = []
    while remaining:
        indices = rng.permutation(len(samples))[:remaining]
        epoch_plan.extend(indices)
        remaining -= len(indices)
    losses, begin = [], perf_counter()
    for step in range(steps):
        examples = [_random_tile(samples[index], model, rng)
                    for index in epoch_plan[step*batch:(step + 1)*batch]]
        images, targets = zip(*examples, strict=True)
        loss = optimizer(np.stack(images), np.stack(targets))
        if not np.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step + 1}.")
        losses.append(loss)
        if step < 3 or (step + 1) % 20 == 0 or step + 1 == steps:
            print(f"  step {step + 1}/{steps}, loss={np.mean(losses[-20:]):.4f}, "
                  f"elapsed={perf_counter() - begin:.1f}s", flush=True)
    return float(np.mean(losses))


def _validation_data(samples):
    images, annotations = [], []
    for index, sample in enumerate(samples, 1):
        images.append(dict(id=index, file_name=sample.file.name,
                           width=sample.width, height=sample.height))
        for crowd, boxes in enumerate((sample.boxes, sample.ignored)):
            for x1, y1, x2, y2 in boxes:
                width, height = float(x2 - x1), float(y2 - y1)
                if min(width, height) <= 0:
                    continue
                annotations.append(dict(id=len(annotations) + 1, image_id=index,
                                        category_id=1, iscrowd=crowd, area=width*height,
                                        bbox=[float(x1), float(y1), width, height]))
    return dict(images=images, annotations=annotations, categories=[dict(id=1, name="person")])


def _native_score(model, dataset, image_dir, batch, *, verbose):
    from pycocotools.coco import COCO

    def evaluate():
        ground_truth = COCO()
        ground_truth.dataset = dataset
        ground_truth.createIndex()
        rows = []
        start = perf_counter()
        for row in dataset["images"]:
            path = image_dir / row["file_name"]
            detections = model.predict(_image(path), batch=batch, confidence=0.001, iou=0.5)
            for (x1, y1, x2, y2), score in zip(detections["boxes"], detections["scores"], strict=True):
                rows.append(dict(image_id=row["id"], category_id=1, score=float(score),
                                 bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)]))
        elapsed = perf_counter() - start
        result = score_coco(ground_truth, rows, [row["id"] for row in dataset["images"]])
        return dict(images=len(dataset["images"]), predictions=len(rows), AP=float(result.stats[0]),
                    AP50=float(result.stats[1]), AP75=float(result.stats[2]),
                    AP_small=float(result.stats[3]), images_per_second=len(dataset["images"]) / elapsed)

    if verbose:
        return evaluate()
    with redirect_stdout(StringIO()):
        return evaluate()



def compare_checkpoint(checkpoint, yolov8n, annotations, image_dir, batch=4):
    """Score the trained native checkpoint and stock YOLOv8n on the same manifest."""
    test = json.loads(Path(annotations).read_text(encoding="utf-8"))
    native = _native_score(PyramidRelationalDetector.load(checkpoint), test,
                           Path(image_dir), batch, verbose=True)
    baseline = benchmark_coco(yolov8n, annotations, image_dir,
                              class_ids=[0], batch=8, imgsz=640, device="cpu")
    if baseline["images"] != native["images"]:
        raise RuntimeError("The native and YOLO evaluations disagree on the test set size.")
    return native, baseline


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _model_digest(path):
    with path.open("rb") as handle:
        return sha256(handle.read()).hexdigest()


def run_experiment(args):
    config = vars(args).copy()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    Tensor.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = getattr(Device[Device.DEFAULT], "device_name", Device.DEFAULT)
    if args.initial_checkpoint is not None:
        model = PyramidRelationalDetector.load(args.initial_checkpoint, crop_size=args.crop_size)
    else:
        model = PyramidRelationalDetector(image_size=args.image_size,
                                          crop_size=args.crop_size or 144, width=args.width)
    warmup_count = 0
    if args.initial_checkpoint is None:
        warmup = _coco_samples(args.coco_annotations, args.coco_images, args.warmup_image_limit)
        warmup_count = len(warmup)
        print(f"COCO person warm-up · {warmup_count} images · {args.warmup_steps} updates", flush=True)
        _phase_step(warmup, model, model.trainer(0.0015), args.warmup_steps, args.batch, rng)
        model.save(output / "coco-warmup.safetensors")
    if args.warmup_only:
        return

    root = args.dataset.resolve()
    test, test_summary = _prepare_export(root, args.label_class)
    if args.expected_test_images is not None and len(test["images"]) != args.expected_test_images:
        raise ValueError(f"Expected {args.expected_test_images} held-out images, found {len(test['images'])}.")
    test_annotation_path = output / "holdout-annotations.json"
    _write_json(test_annotation_path, test)
    train, validation, held_out = _split_sources(root, test, args.seed, args.validation_images, args.label_class)
    if not train or not any(len(sample.boxes) for sample in validation):
        raise ValueError("Provide training images and a validation split with positive boxes.")
    train_videos = {_video(sample.source) for sample in train}
    validation_videos = {_video(sample.source) for sample in validation}
    if train_videos & held_out or validation_videos & held_out or train_videos & validation_videos:
        raise RuntimeError("Source videos cross experiment splits.")
    validation_gt = _validation_data(validation)
    _write_json(output / "split.json", {
        "training_sources": [sample.source for sample in train],
        "validation_sources": [sample.source for sample in validation],
        "test_filenames": [row["file_name"] for row in test["images"]],
    })
    print(f"{device} · {len(train)} train images ({len(train_videos)} videos), "
          f"{len(validation)} validation images ({len(validation_videos)} videos), "
          f"{len(test['images'])} test images ({len(held_out)} videos)", flush=True)

    optimizer = model.trainer(args.learning_rate)
    history, best_ap = [], -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"Fine-tuning epoch {epoch}/{args.epochs}", flush=True)
        average = _phase_step(train, model, optimizer, args.steps_per_epoch, args.batch, rng)
        record = {"epoch": epoch, "mean_training_loss": average}
        if epoch % args.validate_every == 0 or epoch == args.epochs:
            metrics = _native_score(model, validation_gt, root / "valid/images",
                                    args.batch, verbose=False)
            print(f"  validation AP={metrics['AP']:.4f} AP50={metrics['AP50']:.4f}", flush=True)
            record["validation"] = metrics
            if metrics["AP"] > best_ap:
                best_ap, best_epoch = metrics["AP"], epoch
                candidate = output / f"native-epoch-{epoch:03d}.safetensors"
                model.save(candidate)
        history.append(record)
        _write_json(output / "history.json", history)

    checkpoint = output / "tiny-person-native.safetensors"
    copyfile(candidate, checkpoint)
    print(f"Testing validation-selected checkpoint from epoch {best_epoch}", flush=True)
    native, baseline = compare_checkpoint(checkpoint, args.yolo, test_annotation_path,
                                          root / "test/images", args.batch)
    result = {
        "configuration": config, "device": device, "best_epoch": best_epoch,
        "native_architecture": dict(image_size=model.image_size, crop_size=model.crop_size,
                                    width=model.width, roles=model.roles),
        "training_images": len(train), "training_videos": len(train_videos),
        "validation_images": len(validation), "validation_videos": len(validation_videos),
        "test_videos": len(held_out), "test_summary": test_summary,
        "warmup_images": warmup_count, "checkpoint_sha256": _model_digest(checkpoint),
        "yolov8n_weights_sha256": _model_digest(args.yolo),
        "native": native, "yolov8n": baseline,
    }
    _write_json(output / "comparison.json", result)
    print(json.dumps({"native": native, "yolov8n": baseline}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--yolo", type=Path)
    parser.add_argument("--coco-annotations", type=Path)
    parser.add_argument("--coco-images", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--warmup-only", action="store_true", help="Save a COCO initializer and exit.")
    parser.add_argument("--expected-test-images", type=int)
    parser.add_argument("--label-class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--crop-size", type=int, help="Override crop size in source-image pixels.")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=80)
    parser.add_argument("--warmup-image-limit", type=int)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--validate-every", type=int, default=2)
    parser.add_argument("--validation-images", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if args.warmup_only and args.initial_checkpoint is not None:
        parser.error("warmup-only starts from a new model")
    if not args.warmup_only and (args.dataset is None or args.yolo is None):
        parser.error("fine-tuning requires --dataset and --yolo")
    if args.initial_checkpoint is None and (args.coco_annotations is None or args.coco_images is None):
        parser.error("provide a native initializer or both COCO warm-up paths")
    for name in ("batch", "warmup_steps", "epochs", "steps_per_epoch",
                 "validate_every", "validation_images"):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.warmup_image_limit is not None and args.warmup_image_limit < 1:
        parser.error("warmup-image-limit must be positive")
    run_experiment(args)


if __name__ == "__main__":
    main()
