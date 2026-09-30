from services.analyticsReporting import public_rows, render_csv, render_pdf, report_summary


def sample_rows():
    return [{
        "record_type": "ANPR",
        "event_id": "PLATE-1",
        "vehicle_id": "GV-1",
        "plate": "TN01AB1234",
        "timestamp": "2026-09-04T12:00:00Z",
        "camera_id": "CAM-1",
        "camera_name": "Main Gate",
        "location": "Chennai",
        "latitude": 13.0827,
        "longitude": 80.2707,
        "confidence": 0.94,
        "watchlist_result": "MATCH",
        "watchlist_category": "STOLEN",
        "rule": "ANPR_WATCHLIST",
        "severity": "CRITICAL",
        "timestamp_quality": "SOURCE_PTS",
        "evidence_path": "/snapshot/1",
        "evidence_sha256": "a" * 64,
        "_snapshot_bytes": None,
    }]


def test_csv_and_pdf_exports_have_evaluator_fields():
    rows = sample_rows()
    csv_data = render_csv(rows, "REPORT-1")
    assert b"camera_name" in csv_data
    assert b"timestamp_quality" in csv_data
    assert b"TN01AB1234" in csv_data
    pdf_data = render_pdf(rows, "REPORT-1", {})
    assert pdf_data.startswith(b"%PDF")
    assert len(pdf_data) > 1500
    assert "_snapshot_bytes" not in public_rows(rows)[0]
    assert report_summary(rows)["watchlist_match_count"] == 1

