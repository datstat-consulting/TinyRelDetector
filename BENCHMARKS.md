# Vision benchmarks

These runs used the Ultralytics YOLOv8n family on Windows CPU with Ultralytics
8.1.45, PyTorch 2.5.1 and an AMD Ryzen 5 6600H. They report results for the
specified validation samples. The relational model has a separate checkpoint
format and these scores apply to the Ultralytics checkpoints.

## COCO 2017 validation

The sample uses the first 128 image IDs in sorted order from `val2017`.
COCOeval scored the original instance and keypoint annotations. The detector,
segmenter and pose estimator used 640-pixel inference, batches of 8 and
a confidence floor of 0.001. Metrics are percentages. Pose uses object
keypoint similarity for its thresholds. [COCO](https://cocodataset.org/#download) ·
[Ultralytics COCO configuration](https://docs.ultralytics.com/datasets/detect/coco/)

| Task | Checkpoint | AP | AP50 | AP75 | Small-object AP |
| --- | --- | ---: | ---: | ---: | ---: |
| Box detection | yolov8n.pt | 43.04 | 59.32 | 47.82 | 29.49 |
| Instance segmentation | yolov8n-seg.pt | 35.67 | 55.35 | 36.28 | 16.13 |
| Pose estimation | yolov8n-pose.pt | 41.30 | 72.88 | 40.21 | — |

```bash
pip install '.[vision,benchmark]'
tinyrel benchmark-coco --task detect --model yolov8n.pt --annotations coco/annotations/instances_val2017.json --images coco/val2017 --limit 128 --device cpu
tinyrel benchmark-coco --task segment --model yolov8n-seg.pt --annotations coco/annotations/instances_val2017.json --images coco/val2017 --limit 128 --device cpu
tinyrel benchmark-coco --task pose --model yolov8n-pose.pt --annotations coco/annotations/person_keypoints_val2017.json --images coco/val2017 --limit 128 --device cpu
```

Omit `--limit` with a complete validation directory to score all 5,000
images. The `tinyrel benchmark` command also supports the Ultralytics
`coco`, `coco-seg`, `coco-pose`, `imagenet` and `dota` presets. Override the
preset with `--data` to point at a local dataset.

## MOT17 pedestrian tracking

Both trackers used the same YOLOv8n checkpoint, its COCO person class, 640-pixel
inference and a 0.1 confidence floor. The comparison used frames 1–120 of the
`MOT17-05-FRCNN` training sequence. TrackEval 1.3.0 applied MOT17 preprocessing
and evaluated the pedestrian class. [MOT17](https://motchallenge.net/data/MOT17/) ·
[TrackEval](https://github.com/JonathonLuiten/TrackEval)

| Tracker | HOTA | IDF1 | MOTA | Identity switches |
| --- | ---: | ---: | ---: | ---: |
| ByteTrack | 41.07% | 49.86% | 31.56% | 6 |
| BoT-SORT | 41.64% | 51.71% | 27.99% | 1 |

```bash
tinyrel benchmark-mot --model yolov8n.pt --sequence MOT17/train/MOT17-05-FRCNN --tracker bytetrack.yaml --limit 120 --device cpu --output runs/bytetrack --metrics runs/bytetrack/metrics.json
tinyrel benchmark-mot --model yolov8n.pt --sequence MOT17/train/MOT17-05-FRCNN --tracker botsort.yaml --limit 120 --device cpu --output runs/botsort --metrics runs/botsort/metrics.json
```

The command accepts a sequence directory with `seqinfo.ini`, numbered
frames and `gt/gt.txt`. It writes ten-column MOTChallenge detections and
can save machine-readable metrics. Train-split clips support local evaluation.
Leaderboard results use the benchmark's full evaluation protocol.

## TinyPerson-derived YOLO export

The local benchmark matched 144 test files to TinyPerson-style source names.
These represented 136 distinct source frames. The evaluator excluded source
frames also present in the training or validation folders and selected one
exported image per remaining source. It scored 359 class-0 boxes across 38
images. It converted polygons to enclosing rectangles and treated the 85
class-1 annotations as COCO crowd regions. It used the same YOLOv8n checkpoint,
640-pixel inference and confidence floor as the COCO run.

| Images | Box AP | AP50 | AP75 | Small-object AP |
| ---: | ---: | ---: | ---: | ---: |
| 38 | 3.02% | 7.18% | 1.70% | 3.07% |

~~~bash
tinyrel benchmark-tiny-export --model yolov8n.pt --dataset Datasets --label-class 0 --device cpu --metrics runs/tiny-export.json
~~~

The folder should contain the original YOLO export's split directories, each
with images and labels. The benchmark uses the export's annotations and
COCOeval. The official TinyPerson test protocol uses the original ignore,
uncertain and dense annotations. [TinyPerson annotation documentation](https://github.com/ucas-vg/PointTinyBenchmark/blob/master/dataset/TinyPerson_annotation.md)

## Fine-tuned native pyramid

The native tinygrad pyramid combines stride-4 spatial features with role-based
context. The COCO initializer used 100 updates on the 73 person-positive images
among the first 128 sorted COCO `val2017` image IDs. Fine-tuning used 2,400
updates on 255 TinyPerson-derived training images. A 21-image validation split
selected the checkpoint by box AP. The 16 training videos, 4 validation videos,
and 17 test videos were disjoint.

Both models use the same frozen manifest of 38 held-out images, with the same
359 class-0 boxes and 85 crowd regions. The native model detects on overlapping
112-pixel crops resized to 192 pixels. Stock YOLOv8n uses full-image 640-pixel
inference. Both paths use a confidence floor of 0.001 and COCOeval for box
accuracy.

Epoch 24 was selected at 0.70% validation AP.

| Model | Box AP | AP50 | AP75 | Small-object AP |
| --- | ---: | ---: | ---: | ---: |
| Fine-tuned native pyramid | 0.33% | 1.73% | 0.03% | 0.33% |
| Stock pretrained YOLOv8n | 3.02% | 7.18% | 1.70% | 3.07% |

~~~bash
pip install '.[benchmark]'
python -m TinyRelDetector.tiny_experiment --warmup-only --coco-annotations coco/annotations/instances_val2017.json --coco-images coco/val2017 --warmup-image-limit 128 --warmup-steps 100 --width 12 --image-size 192 --crop-size 144 --batch 4 --seed 19 --output runs/person-init
python -m TinyRelDetector.tiny_experiment --dataset Datasets --initial-checkpoint runs/person-init/coco-warmup.safetensors --yolo yolov8n.pt --expected-test-images 38 --epochs 24 --steps-per-epoch 100 --validate-every 4 --batch 4 --crop-size 112 --seed 19 --output runs/person-comparison
~~~

The comparison directory contains the selected safetensors checkpoint,
validation history, split manifest, and JSON scores.
