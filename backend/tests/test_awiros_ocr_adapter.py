from __future__ import annotations

import numpy as np

from services.awirosOcr import AwirosOCR, _preprocess


def test_awiros_preprocess_matches_release_shape_and_range():
    crop = np.full((32, 128, 3), 127, dtype=np.uint8)
    tensor = _preprocess(crop)
    assert tensor.shape == (1, 3, 48, 320)
    assert tensor.dtype == np.float32
    assert float(tensor.min()) >= -1.0
    assert float(tensor.max()) <= 1.0


def test_awiros_health_does_not_initialize_or_download(tmp_path):
    engine = AwirosOCR(
        weights_path=tmp_path / "model.safetensors",
        dictionary_path=tmp_path / "en_dict.txt",
        paddleocr_dir=tmp_path / "PaddleOCR",
        device="cpu",
    )
    health = engine.health()
    assert health["loaded"] is False
    assert health["weights_present"] is False
    assert health["paddleocr_present"] is False
