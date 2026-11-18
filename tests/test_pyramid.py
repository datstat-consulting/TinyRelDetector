import numpy as np
from PIL import Image
from tinygrad import Tensor

from TinyRelDetector import PyramidRelationalDetector
from TinyRelDetector.pyramid import encode_targets


def test_dense_center_encoding_preserves_positives_inside_ignored_regions():
    target = encode_targets([[4, 4, 8, 12], [2, 2, 11, 14]], [[0, 0, 16, 16]], 32)
    assert target[0, 2, 1] == 1
    assert target[5, 2, 1] == 1
    assert target[5, 0, 0] == 0
    assert target[5, 6, 6] == 1
    np.testing.assert_allclose(target[1:5, 2, 1], [0.5, 0, 0, np.log(2)], atol=1e-7)


def test_pyramid_checkpoint_and_tiled_pixel_output(tmp_path):
    Tensor.manual_seed(5)
    detector = PyramidRelationalDetector(image_size=32, crop_size=24, width=4, roles=1)
    image = Image.fromarray(np.random.default_rng(5).integers(0, 256, (29, 31, 3), dtype=np.uint8))
    first = detector.predict(image, confidence=0, batch=4, max_detections=8)
    checkpoint = tmp_path / "pyramid.safetensors"
    detector.save(checkpoint)
    detector.save(checkpoint)
    restored = PyramidRelationalDetector.load(checkpoint).predict(
        image, confidence=0, batch=4, max_detections=8
    )
    assert 0 < len(first["boxes"]) <= 8
    assert np.all((first["boxes"] >= 0) & (first["boxes"] <= [31, 29, 31, 29]))
    for name in first:
        np.testing.assert_allclose(first[name], restored[name], atol=1e-6)
