"""Command-line entry points for detection and vision tasks."""

import argparse

from .benchmarks import DATASET_SUITES, benchmark_coco, benchmark_dataset, benchmark_mot, write_metrics
from .tiny_export import benchmark_tiny_export
from .yolo import VisionModel


def _common(command, task=False):
    command.add_argument("--model", required=True, help="Model config or trusted checkpoint.")
    command.add_argument("--device")
    command.add_argument("--imgsz", type=int)
    if task:
        command.add_argument("--task", choices=("detect", "segment", "classify", "pose", "obb"))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tinyrel", description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("train", "val", "predict", "track", "export"):
        command = modes.add_parser(mode)
        _common(command, task=True)
        if mode != "export":
            command.add_argument("--batch", type=int)
            command.add_argument("--project")
            command.add_argument("--name")
        if mode in {"train", "val"}:
            command.add_argument("--data", required=mode == "train")
            command.add_argument("--workers", type=int)
        if mode == "train":
            command.add_argument("--epochs", type=int, default=100)
        if mode in {"predict", "track"}:
            command.add_argument("--source", required=True)
            command.add_argument("--conf", type=float)
            command.add_argument("--save", action="store_true")
        if mode == "track":
            command.add_argument("--tracker", default="bytetrack.yaml",
                                 help="Tracker YAML, for example bytetrack.yaml or botsort.yaml.")
        if mode == "export":
            command.add_argument("--format", default="onnx")
            command.add_argument("--dynamic", action="store_true")

    suite = modes.add_parser("benchmark", help="Ultralytics benchmark datasets.")
    _common(suite)
    suite.add_argument("--suite", choices=DATASET_SUITES, required=True)
    suite.add_argument("--data", help="Local YAML or dataset root overriding the preset.")
    suite.add_argument("--batch", type=int)
    suite.add_argument("--workers", type=int)
    suite.add_argument("--metrics", help="Save metrics as JSON.")

    coco = modes.add_parser("benchmark-coco", help="Official COCO evaluation.")
    _common(coco)
    coco.add_argument("--task", choices=("detect", "segment", "pose"), default="detect")
    coco.add_argument("--annotations", required=True)
    coco.add_argument("--images", required=True)
    coco.add_argument("--limit", type=int)
    coco.add_argument("--batch", type=int)
    coco.add_argument("--metrics", help="Save metrics as JSON.")

    mot = modes.add_parser("benchmark-mot", help="MOT17 pedestrian tracking evaluation.")
    _common(mot)
    mot.add_argument("--sequence", required=True, help="MOT17 sequence with frames, GT and seqinfo.ini.")
    mot.add_argument("--output", required=True, help="Directory for MOTChallenge prediction rows.")
    mot.add_argument("--tracker", default="bytetrack.yaml")
    mot.add_argument("--limit", type=int)
    mot.add_argument("--class-id", type=int)
    mot.add_argument("--confidence", type=float)
    mot.add_argument("--metrics", help="Save metrics as JSON.")

    tiny = modes.add_parser("benchmark-tiny-export", help="Score TinyPerson-derived YOLO test images.")
    _common(tiny)
    tiny.add_argument("--dataset", required=True, help="YOLO export root with test/images and test/labels.")
    tiny.add_argument("--label-class", type=int, required=True, help="Person class ID in the export.")
    tiny.add_argument("--model-class", type=int, default=0, help="Person class ID in the checkpoint.")
    tiny.add_argument("--batch", type=int)
    tiny.add_argument("--metrics", help="Save metrics as JSON.")

    args = {key: value for key, value in vars(parser.parse_args(argv)).items() if value is not None}
    mode = args.pop("mode")
    benchmark = {"benchmark": benchmark_dataset, "benchmark-coco": benchmark_coco,
                 "benchmark-mot": benchmark_mot,
                 "benchmark-tiny-export": benchmark_tiny_export}.get(mode)
    if benchmark is not None:
        metrics_path = args.pop("metrics", None)
        write_metrics(benchmark(**args), metrics_path)
        return

    model = VisionModel(args.pop("model"), task=args.pop("task", None))
    result = getattr(model, mode)(**args)
    if mode == "export":
        print(result)
