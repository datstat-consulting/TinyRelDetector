import numpy as np
from PIL import Image

from TinyRelDetector import RelationalDetector
from TinyRelDetector.cli import main


def test_native_fit_save_load_and_inference(tmp_path):
    model = RelationalDetector(3, image_size=(16, 16), model_width=4, roles=2)
    losses = model.fit(steps=1, batch=2)
    assert np.isfinite(losses).all()
    source = Image.new("RGB", (17, 12), "red")
    original = model.predict(source, confidence=0)
    path = tmp_path / "native.safetensors"
    model.save(path)
    restored = RelationalDetector.load(path).predict(source, confidence=0)
    for field in ("boxes", "scores", "labels"):
        np.testing.assert_allclose(original[field], restored[field])
    assert restored["boxes"].shape[1] == 4
    assert (restored["boxes"] <= [17, 12, 17, 12]).all()


def test_cli_forwards_predict_arguments(monkeypatch):
    calls = {}

    class Backend:
        def __init__(self, model, task):
            calls.update(model=model, task=task)

        def predict(self, **options):
            calls.update(options)

    monkeypatch.setattr("TinyRelDetector.cli.VisionModel", Backend)
    main(["predict", "--model", "weights.pt", "--source", "image.png",
          "--task", "detect", "--conf", "0.4", "--save"])
    assert calls == dict(model="weights.pt", task="detect", source="image.png", conf=0.4, save=True)



def test_native_yolo_loader_resolves_nested_labels_and_cycles(tmp_path):
    from TinyRelDetector import YoloIterBatches

    images = tmp_path / "images/train/sub"
    labels = tmp_path / "labels/train/sub"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (16, 16), "blue").save(images / "square.png")
    (labels / "square.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    config = tmp_path / "dataset.yaml"
    config.write_text("train: images/train\nnames: {0: square}\n")
    batches = YoloIterBatches(str(config), "train", 16, 16, 3)
    for _ in range(2):
        batch, _, boxes, classes, count = next(batches)
        assert batch.shape == (3, 3, 16, 16) and count == 1
        assert len(classes) == 3
        np.testing.assert_allclose(boxes[0].numpy(), [[0.5, 0.5, 0.4, 0.4]])



def test_track_and_benchmark_cli_select_backend(monkeypatch, tmp_path):
    import json

    from TinyRelDetector import benchmarks

    calls = []

    class Backend:
        def __init__(self, model, *, task):
            calls.append((model, task))

        def track(self, **options):
            calls.append(options)

        def val(self, data, **options):
            from types import SimpleNamespace
            calls.append((data, options))
            return SimpleNamespace(results_dict={"metrics/mAP50-95(M)": 0.4})

    monkeypatch.setattr("TinyRelDetector.cli.VisionModel", Backend)
    monkeypatch.setattr(benchmarks, "VisionModel", Backend)
    main(["track", "--model", "weights.pt", "--source", "video.mp4",
          "--tracker", "botsort.yaml"])
    assert calls[-1]["tracker"] == "botsort.yaml"
    metrics_path = tmp_path / "metrics.json"
    main(["benchmark", "--suite", "coco-seg", "--model", "weights.pt", "--device", "cpu",
          "--metrics", str(metrics_path)])
    assert calls[-2] == ("weights.pt", "segment")
    assert calls[-1] == ("coco.yaml", {"device": "cpu"})
    assert json.loads(metrics_path.read_text()) == {"metrics/mAP50-95(M)": 0.4}


def test_mot_sequence_limit_only_requires_the_selected_frames(tmp_path):
    from TinyRelDetector.benchmarks import _sequence_info

    (tmp_path / "seqinfo.ini").write_text(
        "[Sequence]\nname=sequence\nseqLength=20\nimDir=img1\nimExt=.jpg\n"
    )
    (tmp_path / "gt").mkdir()
    (tmp_path / "gt/gt.txt").touch()
    (tmp_path / "img1").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "img1/000001.jpg")
    name, images, ground_truth = _sequence_info(tmp_path, 1)
    assert name == "sequence" and [path.name for path in images] == ["000001.jpg"]
    assert ground_truth.name == "gt.txt"



def test_tiny_export_excludes_shared_sources_and_deduplicates_test(tmp_path):
    from TinyRelDetector.tiny_export import _prepare_export

    for split in ("train", "test"):
        for part in ("images", "labels"):
            (tmp_path / split / part).mkdir(parents=True)
    names = [
        ("train", "bb_V0000_I0000100_jpg.rf.a.jpg"),
        ("test", "bb_V0000_I0000100_jpg.rf.b.jpg"),
        ("test", "bb_V0000_I0000200_jpg.rf.c.jpg"),
        ("test", "bb_V0000_I0000200_jpg.rf.d.jpg"),
        ("test", "unrelated.jpg"),
    ]
    for split, name in names:
        Image.new("RGB", (20, 10)).save(tmp_path / split / "images" / name)
        label = tmp_path / split / "labels" / (name[:-4] + ".txt")
        label.write_text("2 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n"
                         "4 0.2 0.2 0.1 0.1\n")
    data, summary = _prepare_export(tmp_path, label_class=2)
    assert summary["candidate_files"] == 3 and summary["selected_sources"] == 1
    assert data["images"][0]["file_name"].endswith("rf.c.jpg")
    assert data["annotations"][0]["bbox"] == [5, 2.5, 10, 5]
    assert summary["ground_truth_boxes"] == 1
    assert [row["iscrowd"] for row in data["annotations"]] == [0, 1]
