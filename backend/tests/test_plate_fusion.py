from services.plateFusion import fuse_plate_candidates, looks_like_indian_plate


def test_ollama_alone_cannot_confirm():
    result = fuse_plate_candidates([{"text": "GJ01AB1234", "confidence": 0.99, "source": "ollama"}])
    assert result["decision"] == "POSSIBLE"
    assert result["confidence"] <= 0.49


def test_primary_and_temporal_agreement_can_confirm():
    result = fuse_plate_candidates([
        {"text": "GJ01AB1234", "confidence": 0.91, "source": "ppocr"},
        {"text": "GJ01AB1234", "confidence": 0.88, "source": "temporal"},
        {"text": "GJ01AB1234", "confidence": 0.70, "source": "ollama"},
    ])
    assert result["plate"] == "GJ01AB1234"
    assert result["decision"] == "CONFIRMED"


def test_indian_bh_format():
    assert looks_like_indian_plate("22BH1234AA")
