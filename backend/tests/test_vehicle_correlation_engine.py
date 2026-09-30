from services.vehicleCorrelationEngine import evidence_fusion


def test_full_evidence_fusion():
    result = evidence_fusion(
        anpr_score=0.98,
        reid_score=0.91,
        temporal_score=0.95,
        location_score=0.92,
        direction_score=0.90,
        attribute_score=1.0,
        track_quality=0.95,
        anpr_available=True,
        reid_available=True,
        metadata_available=True,
    )
    assert result["score"] > 0.90
    assert result["available_branches"] == ["anpr", "reid", "metadata"]


def test_missing_reid_is_not_positive_evidence():
    result = evidence_fusion(
        anpr_score=0.98,
        reid_score=0.0,
        temporal_score=0.95,
        location_score=0.92,
        direction_score=0.90,
        attribute_score=1.0,
        track_quality=0.95,
        anpr_available=True,
        reid_available=False,
        metadata_available=True,
    )
    assert "reid" not in result["available_branches"]
    assert result["reid_score"] == 0.0
    assert result["score"] > 0.0


def test_no_evidence_scores_zero():
    result = evidence_fusion(
        anpr_score=0,
        reid_score=0,
        temporal_score=0,
        location_score=0,
        direction_score=0,
        attribute_score=0,
        track_quality=0,
        anpr_available=False,
        reid_available=False,
        metadata_available=False,
    )
    assert result["score"] == 0.0
