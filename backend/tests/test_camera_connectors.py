import unittest

from connectors.manager import validate_connector_config
from db.crud import _normalize_source_type, _validate_camera_source


class CameraConnectorValidationTests(unittest.TestCase):
    def test_supported_source_types(self):
        for source_type in (
            "rtsp", "http", "https", "hls", "webcam",
            "onvif", "vendor_api", "vendor_sdk",
        ):
            self.assertEqual(_normalize_source_type(source_type), source_type)

    def test_onvif_config_requires_host_and_user(self):
        with self.assertRaises(ValueError):
            validate_connector_config("onvif", {"host": "10.0.0.2"})
        config = validate_connector_config(
            "onvif",
            {"host": "10.0.0.2", "username": "admin", "password": "secret"},
        )
        self.assertEqual(config["host"], "10.0.0.2")

    def test_vendor_api_config(self):
        config = validate_connector_config(
            "vendor_api",
            {
                "base_url": "https://vendor.example",
                "stream_path": "/camera/1",
                "stream_json_path": "data.stream_url",
            },
        )
        self.assertEqual(config["stream_path"], "/camera/1")

    def test_vendor_sdk_requires_allowlisted_adapter_identity(self):
        with self.assertRaises(ValueError):
            validate_connector_config("vendor_sdk", {"module": "x"})
        config = validate_connector_config(
            "vendor_sdk",
            {
                "module": "connectors.vendor_sdk_plugins.hikvision",
                "class_name": "HikvisionAdapter",
            },
        )
        self.assertEqual(config["class_name"], "HikvisionAdapter")

    def test_source_validation(self):
        self.assertEqual(
            _validate_camera_source(
                "onvif://10.0.0.2:80",
                "onvif",
            ),
            "onvif://10.0.0.2:80",
        )
        self.assertEqual(
            _validate_camera_source(
                "sdk://vendor/device",
                "vendor_sdk",
            ),
            "sdk://vendor/device",
        )
        with self.assertRaises(ValueError):
            _validate_camera_source(
                "https://example.com/page",
                "hls",
            )


if __name__ == "__main__":
    unittest.main()
