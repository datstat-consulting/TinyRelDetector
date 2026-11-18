"""Evaluate the TinyPerson-derived portion of a YOLO dataset export."""

from collections import Counter
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from PIL import Image

from .benchmarks import benchmark_coco
from .tinyrel_detector import ReadYoloTxt


_SOURCE = re.compile(r"^(bb_V\d+_I\d+)(?:_[^.]+)?\.rf\.", re.IGNORECASE)


def _candidates(split):
    for path in sorted((split / "images").iterdir()):
        match = _SOURCE.match(path.name)
        if match and path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            yield match[1].lower(), path


def _prepare_export(root, label_class):
    root = Path(root)
    other_splits = ("train", "valid", "val")
    excluded = {
        source for name in other_splits if (root / name / "images").is_dir()
        for source, _ in _candidates(root / name)
    }
    candidates = list(_candidates(root / "test"))
    sources = {}
    for source, path in candidates:
        if source not in excluded:
            sources.setdefault(source, path)
    if not sources:
        raise ValueError("The test export has no matching source frames absent from train/validation.")

    images, annotations = [], []
    label_counts = Counter()
    for image_id, path in enumerate(sources.values(), 1):
        with Image.open(path) as image:
            width, height = image.size
        images.append(dict(id=image_id, file_name=path.name, width=width, height=height))
        label = root / "test/labels" / path.with_suffix(".txt").name
        if not label.is_file():
            raise FileNotFoundError(label)
        classes, boxes = ReadYoloTxt(str(label), allow_segments=True)
        label_counts.update(map(int, classes))
        for class_id, (cx, cy, w, h) in zip(classes, boxes, strict=True):
            left = (float(cx) - float(w) / 2) * width
            top = (float(cy) - float(h) / 2) * height
            box = [left, top, float(w) * width, float(h) * height]
            annotations.append(dict(
                id=len(annotations) + 1, image_id=image_id, category_id=1,
                bbox=box, area=box[2] * box[3], iscrowd=int(class_id != label_class),
            ))
    if not label_counts[label_class]:
        raise ValueError(f"No boxes of label class {label_class} in the selected test images.")
    dataset = dict(
        images=images, annotations=annotations, categories=[dict(id=1, name="person")],
    )
    summary = dict(candidate_files=len(candidates), test_sources=len({src for src, _ in candidates}),
                   selected_sources=len(images), ground_truth_boxes=label_counts[label_class],
                   label_class_counts=dict(sorted(label_counts.items())))
    return dataset, summary


def benchmark_tiny_export(model, dataset, *, label_class, model_class=0, **options):
    """Score the local export using COCO boxes on a deduplicated test subset."""
    data, summary = _prepare_export(dataset, label_class)
    with TemporaryDirectory() as temp:
        path = Path(temp) / "annotations.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = benchmark_coco(
            model, path, Path(dataset) / "test/images", class_ids=[model_class], **options
        )
    return {"dataset": "TinyPerson-derived YOLO export", **summary, **result}
