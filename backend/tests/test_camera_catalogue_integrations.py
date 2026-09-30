from cryptography.fernet import Fernet
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from connectors.sentinel import SentinelCatalogueConnector
from connectors.vms import GenericVMSCatalogueConnector
from db.database import Base
from db.model import Camera, CameraIntegration, CameraIntegrationSyncRun, User
import db.watchlist_model  # noqa: F401 - registers referenced tables
import db.intelligence_model  # noqa: F401 - registers referenced tables
import db.advanced_intelligence_model  # noqa: F401 - registers referenced tables
from services.cameraIntegration import _safe_provider_metadata, encrypt_integration_config, sync_integration


def test_sentinel_catalogue_normalizes_metadata_without_exposing_urls():
    payload = {
        "revision": "gov-42",
        "cameras": [{
            "cameraId": "ignored",
            "camera_id": "GOV-CAM-04",
            "camera_name": "Highway Camera 04",
            "live_status": "live",
            "codec": "h264",
            "location": {"name": "NH48 Toll", "lat": 23.02, "lng": 72.51},
            "streams": [
                {"url": "https://cctv.example/live/4/index.m3u8", "codec": "h264", "width": 1920, "height": 1080},
                {"url": "rtsp://cctv.example:8554/stream/4", "codec": "h265", "fps": 25},
            ],
        }],
    }

    def handler(request):
        assert request.url.path == "/api/ingest"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(200, json=payload, headers={"etag": "v42"})

    result = SentinelCatalogueConnector(
        {"base_url": "https://sentinel.example", "auth_type": "bearer", "token": "secret"},
        transport=httpx.MockTransport(handler),
    ).fetch_catalogue()
    assert result.source_revision == "gov-42"
    assert result.etag == "v42"
    assert result.cameras[0].preferred_stream().source_type == "rtsp"
    safe = result.safe_dict()
    assert safe["cameras"][0]["location_name"] == "NH48 Toll"
    assert "url" not in safe["cameras"][0]["streams"][0]
    assert "rtsp://" not in str(safe)


def test_generic_vms_mapping_supports_heterogeneous_schema():
    payload = {"data": {"devices": [{
        "uuid": "NVR-7-CH-2",
        "label": "Warehouse channel 2",
        "availability": True,
        "geo": {"y": 12.9716, "x": 77.5946},
        "play": "rtsp://nvr.example/channel/2",
    }]}}

    connector = GenericVMSCatalogueConnector({
        "base_url": "https://vms.example",
        "catalogue_path": "/v2/devices",
        "mapping": {
            "items": "data.devices",
            "id": "uuid",
            "name": "label",
            "status": "availability",
            "latitude": "geo.y",
            "longitude": "geo.x",
            "streams": "missing",
            "stream_url": "play",
        },
    }, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    result = connector.fetch_catalogue()
    # A mapping-driven VMS can still use standard fallback stream fields.
    # Map this vendor's play field explicitly as the primary URL.
    assert len(result.cameras) == 1
    assert result.cameras[0].external_id == "NVR-7-CH-2"
    assert result.cameras[0].live_status == "ONLINE"
    assert result.cameras[0].latitude == 12.9716


def test_provider_metadata_redacts_secret_keys_and_embedded_urls():
    safe = _safe_provider_metadata({
        "description": "Public intersection",
        "authorization": "Bearer top-secret",
        "playback": "rtsp://user:pass@camera.example/live",
        "nested": {"cookie": "session=secret", "status": "healthy"},
    })
    assert safe["description"] == "Public intersection"
    assert safe["playback"] == "[REDACTED_URL]"
    assert safe["nested"] == {"status": "healthy"}
    assert "authorization" not in safe


def test_sync_is_idempotent_and_updates_existing_external_camera(monkeypatch):
    monkeypatch.setenv("CAMERA_SOURCE_ENCRYPTION_KEY", Fernet.generate_key().decode())
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    state = {"name": "Sentinel A"}

    def handler(request):
        return httpx.Response(200, json={"cameras": [{
            "camera_id": "A-1",
            "camera_name": state["name"],
            "status": "online",
            "rtsp_url": "rtsp://sentinel.example/stream/a1",
            "codec": "h264",
            "latitude": 13.1,
            "longitude": 80.2,
        }]})

    encrypted, fingerprint = encrypt_integration_config({"base_url": "https://sentinel.example"})
    with Session() as db:
        user = User(full_name="Admin", email="admin@example.com", password="hash", role="super_admin", is_active=True)
        db.add(user)
        db.flush()
        integration = CameraIntegration(
            user_id=user.id,
            name="Government Sentinel",
            provider_type="sentinel",
            config_encrypted=encrypted,
            config_fingerprint=fingerprint,
            enabled=True,
            auto_sync=True,
            sync_interval_seconds=300,
        )
        db.add(integration)
        db.commit()
        db.refresh(integration)

        first = sync_integration(db, integration, transport=httpx.MockTransport(handler))
        assert first["created_count"] == 1
        camera = db.query(Camera).one()
        first_id = camera.cam_id
        assert camera.external_camera_id == "A-1"
        assert camera.source == camera.cam_id
        assert "rtsp://" not in camera.source

        state["name"] = "Sentinel A Updated"
        second = sync_integration(db, integration, transport=httpx.MockTransport(handler))
        assert second["created_count"] == 0
        assert db.query(Camera).count() == 1
        camera = db.query(Camera).one()
        assert camera.cam_id == first_id
        assert camera.camera_name == "Sentinel A Updated"
        assert db.query(CameraIntegrationSyncRun).count() == 2
