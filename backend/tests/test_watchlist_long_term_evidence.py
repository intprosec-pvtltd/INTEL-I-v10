from services.watchlist_service import watchlist_journey_lookback_seconds
from telegramAlert import build_caption


def test_watchlist_journey_default_is_one_year(monkeypatch):
    monkeypatch.delenv("WATCHLIST_JOURNEY_WINDOW_SECONDS", raising=False)
    monkeypatch.delenv("WATCHLIST_JOURNEY_LOOKBACK_DAYS", raising=False)

    assert watchlist_journey_lookback_seconds() == 365 * 86400


def test_watchlist_journey_allows_month_and_year_windows(monkeypatch):
    monkeypatch.delenv("WATCHLIST_JOURNEY_WINDOW_SECONDS", raising=False)
    monkeypatch.setenv("WATCHLIST_JOURNEY_LOOKBACK_DAYS", "730")

    assert watchlist_journey_lookback_seconds() == 730 * 86400


def test_day_setting_overrides_legacy_short_window(monkeypatch):
    monkeypatch.setenv("WATCHLIST_JOURNEY_WINDOW_SECONDS", "3600")
    monkeypatch.setenv("WATCHLIST_JOURNEY_LOOKBACK_DAYS", "365")

    assert watchlist_journey_lookback_seconds() == 365 * 86400


def test_legacy_seconds_are_not_capped_at_24_hours(monkeypatch):
    expected = 400 * 86400
    monkeypatch.setenv("WATCHLIST_JOURNEY_WINDOW_SECONDS", str(expected))

    assert watchlist_journey_lookback_seconds() == expected


def test_person_watchlist_telegram_caption_contains_name_and_location():
    caption = build_caption(
        {
            "alert_id": 81,
            "severity": "HIGH",
            "rule": "PERSON_WATCHLIST_WANTED",
            "cam_id": "CAM04",
            "source_type": "rtsp",
            "zone": "Naroda Junction",
            "track_id": "42",
            "match_confidence": 0.946,
            "created_at": "2026-09-15T07:12:00Z",
            "watchlist_person_name": "Rajesh Kumar",
            "watchlist_category": "WANTED",
            "message": "Watchlist person Rajesh Kumar detected in Naroda Junction",
        }
    )

    assert "Watchlist Person: Rajesh Kumar" in caption
    assert "Watchlist: WANTED" in caption
    assert "Camera: CAM04" in caption
    assert "Naroda Junction" in caption
