"""Dataset evaluation with Ultralytics, COCOeval and TrackEval."""

from configparser import ConfigParser
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from .yolo import VisionModel


DATASET_SUITES = {
    "coco": ("detect", "coco.yaml"),
    "coco-seg": ("segment", "coco.yaml"),
    "coco-pose": ("pose", "coco-pose.yaml"),
    "imagenet": ("classify", "imagenet"),
    "dota": ("obb", "DOTAv1.yaml"),
}


def benchmark_dataset(model, suite, *, data=None, **options):
    """Validate with the corresponding Ultralytics task and benchmark dataset."""
    task, default_data = DATASET_SUITES[suite]
    vision = VisionModel(model, task=task)
    metrics = vision.val(default_data if data is None else data, **options)
    return {name: float(value) for name, value in metrics.results_dict.items()}


def _detection_rows(result, image_id, categories, task, mask_api):
    if result.boxes is None or not len(result.boxes):
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.int().cpu().numpy()
    if task == "segment":
        masks = result.masks.data.cpu().numpy().astype(np.uint8)
    elif task == "pose":
        keypoints = result.keypoints.data.cpu().numpy()

    predictions = []
    for index, ((x1, y1, x2, y2), score, label) in enumerate(zip(boxes, scores, classes)):
        row = {"image_id": image_id, "category_id": categories[label], "score": float(score)}
        if task == "segment":
            encoded = mask_api.encode(np.asfortranarray(masks[index]))
            encoded["counts"] = encoded["counts"].decode("ascii")
            row["segmentation"] = encoded
        elif task == "pose":
            row["keypoints"] = keypoints[index].ravel().tolist()
        else:
            row["bbox"] = list(map(float, (x1, y1, x2 - x1, y2 - y1)))
        predictions.append(row)
    return predictions


def benchmark_coco(model, annotations, images, *, task="detect", limit=None, batch=8,
                   imgsz=640, device=None, confidence=0.001, class_ids=None):
    """Use the official COCO API on an annotated COCO validation split."""
    from pycocotools import mask as mask_api
    from pycocotools.coco import COCO
    modes = {"detect": "bbox", "segment": "segm", "pose": "keypoints"}
    if task not in modes or batch < 1 or (limit is not None and limit < 1):
        raise ValueError("Use a COCO task and positive batch and limit values.")
    dataset = COCO(str(annotations))
    image_ids = sorted(dataset.getImgIds())[:limit]
    if not image_ids:
        raise ValueError("The annotation file contains no image IDs.")
    paths = [Path(images) / item["file_name"] for item in dataset.loadImgs(image_ids)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    category_ids = sorted(dataset.getCatIds())
    if class_ids is None:
        categories = dict(enumerate(category_ids))
    else:
        if len(set(class_ids)) != len(category_ids) or len(class_ids) != len(category_ids):
            raise ValueError("Provide one distinct model class per annotation category, in order.")
        categories = dict(zip(class_ids, category_ids))
    model = VisionModel(model, task=task)
    options = dict(imgsz=imgsz, conf=confidence, device=device, verbose=False, max_det=300)
    if class_ids is not None:
        options["classes"] = list(class_ids)
    if task == "segment":
        options["retina_masks"] = True
    rows = []
    start = perf_counter()
    for index in range(0, len(paths), batch):
        frame_batch = paths[index:index + batch]
        results = model.predict([str(path) for path in frame_batch], **options)
        for image_id, result in zip(image_ids[index:index + batch], results, strict=True):
            rows.extend(_detection_rows(result, image_id, categories, task, mask_api))
    elapsed = perf_counter() - start
    evaluation = score_coco(dataset, rows, image_ids, modes[task])
    metrics = {"images": len(image_ids), "predictions": len(rows), "task": task,
               "AP": float(evaluation.stats[0]), "AP50": float(evaluation.stats[1]),
               "AP75": float(evaluation.stats[2]), "images_per_second": len(image_ids) / elapsed}
    if task != "pose":
        metrics["AP_small"] = float(evaluation.stats[3])
    return metrics



def score_coco(dataset, rows, image_ids, metric="bbox"):
    """Apply COCO's standard matching, area bins and maximum-detection rules."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if rows:
        # Some valid COCO files omit the optional info block.
        dataset.dataset.setdefault("info", {})
        predictions = dataset.loadRes(rows)
    else:
        predictions = COCO()
        predictions.dataset = {
            "images": dataset.dataset["images"],
            "categories": dataset.dataset["categories"],
            "annotations": [],
        }
        predictions.createIndex()
    evaluation = COCOeval(dataset, predictions, metric)
    evaluation.params.imgIds = image_ids
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    return evaluation


def _sequence_info(sequence, limit):
    sequence = Path(sequence)
    metadata = ConfigParser()
    if not metadata.read(sequence / "seqinfo.ini", encoding="utf-8"):
        raise FileNotFoundError(sequence / "seqinfo.ini")
    info = metadata["Sequence"]
    length = info.getint("seqLength")
    if length < 1 or (limit is not None and limit < 1):
        raise ValueError("Provide a positive frame limit and a nonempty sequence.")
    frames = length if limit is None else min(limit, length)
    images = [sequence / info["imDir"] / f"{i:06d}{info['imExt']}" for i in range(1, frames + 1)]
    for path in images:
        if not path.is_file():
            raise FileNotFoundError(path)
    ground_truth = sequence / "gt" / "gt.txt"
    if not ground_truth.is_file():
        raise FileNotFoundError(ground_truth)
    return info["name"], images, ground_truth


def _track_rows(frame, result):
    if result.boxes is None or result.boxes.id is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    identities = result.boxes.id.int().cpu().numpy()
    confidence = result.boxes.conf.cpu().numpy()
    rows = []
    for identifier, (x, y, right, bottom), score in zip(identities, boxes, confidence):
        rows.append((frame, int(identifier), float(x + 1), float(y + 1),
                     float(right - x), float(bottom - y), float(score), -1, -1, -1))
    return rows


def _evaluate_mot(trackeval, name, frames, ground_truth, output):
    # Use TrackEval's MOT17 preprocessing to ignore distractors and invalid targets.
    with TemporaryDirectory() as temp:
        gt = Path(temp) / name / "gt"
        gt.mkdir(parents=True)
        with ground_truth.open(encoding="utf-8", newline="") as source, (
            gt / "gt.txt"
        ).open("w", encoding="utf-8", newline="") as filtered:
            writer = csv.writer(filtered)
            writer.writerows(row for row in csv.reader(source) if row and 1 <= int(row[0]) <= frames)
        dataset = trackeval.datasets.MotChallenge2DBox({
            "GT_FOLDER": temp,
            "TRACKERS_FOLDER": str(output),
            "TRACKERS_TO_EVAL": ["tracker"],
            "BENCHMARK": "MOT17",
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": {name: frames},
            "PRINT_CONFIG": False,
        })
        raw = dataset.get_raw_seq_data("tracker", name)
        data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
        hota = trackeval.metrics.HOTA().eval_sequence(data)
        clear = trackeval.metrics.CLEAR({"PRINT_CONFIG": False}).eval_sequence(data)
        identity = trackeval.metrics.Identity({"PRINT_CONFIG": False}).eval_sequence(data)
    return {
        "HOTA": float(np.mean(hota["HOTA"])),
        "MOTA": float(clear["MOTA"]),
        "IDF1": float(identity["IDF1"]),
        "identity_switches": int(clear["IDSW"]),
    }


def benchmark_mot(model, sequence, output, *, tracker="bytetrack.yaml", limit=None,
                  class_id=0, imgsz=640, confidence=0.1, device=None):
    """Track one annotated MOT17 image sequence and score standard MOT metrics."""
    import trackeval

    name, images, ground_truth = _sequence_info(sequence, limit)
    if Path(name).name != name or name in {"", "..", "."}:
        raise ValueError("Invalid sequence name.")
    detector = VisionModel(model, task="detect")
    rows = []
    start = perf_counter()
    for frame, path in enumerate(images, start=1):
        result, = detector.track(path, tracker=tracker, persist=True, classes=[class_id],
                                 imgsz=imgsz, conf=confidence, device=device, verbose=False)
        rows.extend(_track_rows(frame, result))
    elapsed = perf_counter() - start
    output = Path(output)
    predictions = output / "tracker" / "data" / f"{name}.txt"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    with predictions.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    metrics = _evaluate_mot(trackeval, name, len(images), ground_truth, output)
    return {
        "sequence": name, "frames": len(images), "tracker": str(tracker),
        "tracks": len({row[1] for row in rows}), "boxes": len(rows),
        "frames_per_second": len(images) / elapsed, **metrics,
    }


def write_metrics(metrics, output=None):
    """Print JSON metrics and optionally save a copy."""
    payload = json.dumps(metrics, indent=2, sort_keys=True)
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload + "\n", encoding="utf-8")
    print(payload)
