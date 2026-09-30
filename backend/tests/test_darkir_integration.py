import hashlib
from pathlib import Path

import numpy as np
import pytest

from services.darkIR import DarkIRRestorer, _build_darkir_classes


def test_darkir_architecture_parameter_count():
    torch = pytest.importorskip("torch")
    model = _build_darkir_classes(torch)(
        img_channel=3,
        width=32,
        middle_blk_num_enc=2,
        middle_blk_num_dec=2,
        enc_blk_nums=[1, 2, 3],
        dec_blk_nums=[3, 1, 1],
        dilations=[1, 4, 9],
        extra_depth_wise=True,
    )
    assert sum(p.numel() for p in model.parameters()) == 3_321_638


def test_darkir_missing_checkpoint_is_safe():
    restorer = DarkIRRestorer(
        Path("does-not-exist-DarkIR_384.pt"),
        device="cpu",
        strict=False,
    )
    result = restorer.enhance(np.zeros((64, 64, 3), dtype=np.uint8))
    assert result["model_used"] is False
    assert result["frame"].shape == (64, 64, 3)
