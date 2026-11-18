"""Ultralytics vision workflows exposed from TinyRelDetector."""

from pathlib import Path


class VisionModel:
    """Use Ultralytics models for standard computer-vision workflows."""

    def __init__(self, model, *, task=None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Install the vision extra with pip install 'tiny-rel-detector[vision]'.") from exc
        self.model = YOLO(str(model), task=task)

    def train(self, data, **kwargs):
        return self.model.train(data=str(data), **kwargs)

    def val(self, data=None, **kwargs):
        if data is not None:
            kwargs["data"] = str(data)
        return self.model.val(**kwargs)

    def predict(self, source, **kwargs):
        if isinstance(source, Path):
            source = str(source)
        return self.model.predict(source=source, **kwargs)

    def track(self, source, **kwargs):
        if isinstance(source, Path):
            source = str(source)
        return self.model.track(source=source, **kwargs)

    def export(self, *, format="onnx", **kwargs):
        return self.model.export(format=format, **kwargs)
