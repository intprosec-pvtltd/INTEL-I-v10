"""Standard-library unit tests for external watchlist boundary validation."""
import os
import unittest
from integrations.government_watchlist.mapper import normalize_record
from integrations.government_watchlist.providers.rest import validate_remote_url
from integrations.government_watchlist.security import decrypt_credential, encrypt_credential

class ExternalWatchlistBoundaryTests(unittest.TestCase):
    def test_nested_configurable_mapping_normalizes_vehicle(self):
        raw = {"vehicle": {"id": "gov-42", "type": "vehicle", "registration": "GJ 01 AB 1234", "state": "stolen"}, "case": {"fir": "CR-9"}}
        mapped = normalize_record(raw, {"external_id": "vehicle.id", "entity_type": "vehicle.type", "category": "vehicle.state", "plate": "vehicle.registration", "case_reference": "case.fir"})
        self.assertEqual(mapped["external_id"], "gov-42")
        self.assertEqual(mapped["entity_type"], "VEHICLE")
        self.assertEqual(mapped["plate"], "GJ01AB1234")
        self.assertEqual(mapped["case_reference"], "CR-9")

    def test_invalid_entity_and_required_data_are_rejected(self):
        with self.assertRaises(ValueError): normalize_record({"id": "x", "entity_type": "OTHER"}, {})
        with self.assertRaises(ValueError): normalize_record({"id": "x", "entity_type": "PERSON", "category": "WANTED"}, {})

    def test_private_and_non_https_urls_are_rejected(self):
        for url in ("http://public.example/api", "https://127.0.0.1/api", "https://localhost/api", "https://user:pass@example.com/api"):
            with self.assertRaises(ValueError): validate_remote_url(url)

    def test_credentials_are_encrypted_and_round_trip(self):
        from cryptography.fernet import Fernet
        old = os.environ.get("EXTERNAL_WATCHLIST_ENCRYPTION_KEY")
        os.environ["EXTERNAL_WATCHLIST_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
        try:
            cipher = encrypt_credential("server-token")
            self.assertNotEqual(cipher, b"server-token")
            self.assertNotIn(b"server-token", cipher)
            self.assertEqual(decrypt_credential(cipher), "server-token")
        finally:
            if old is None: os.environ.pop("EXTERNAL_WATCHLIST_ENCRYPTION_KEY", None)
            else: os.environ["EXTERNAL_WATCHLIST_ENCRYPTION_KEY"] = old


    def test_rest_provider_uses_server_side_auth_and_tls_verification(self):
        import json
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        from integrations.government_watchlist.providers.rest import RestJsonProvider
        captured = {}
        class Response:
            status_code = 200
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def raise_for_status(self): pass
            def iter_bytes(self): yield json.dumps({"data": {"items": [{"id": "a"}]}}).encode()
        class Client:
            def __init__(self, **kwargs): captured["client"] = kwargs
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def stream(self, method, url, headers, params=None):
                captured["headers"] = headers
                return Response()
        fake_httpx = SimpleNamespace(Client=Client, Timeout=lambda *args, **kwargs: (args, kwargs), RequestError=OSError)
        source = SimpleNamespace(base_url="https://unit.example", auth_type="BEARER", api_key_header=None,
                                 tls_verify=True, pagination={}, records_path="data.items")
        with patch.dict(sys.modules, {"httpx": fake_httpx}), patch("integrations.government_watchlist.providers.rest.validate_remote_url", side_effect=lambda x: x):
            rows = RestJsonProvider().fetch_records(source, "secure-token")
        self.assertEqual(rows, [{"id": "a"}])
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secure-token")
        self.assertTrue(captured["client"]["verify"])

    def test_none_credentials_remain_none(self):
        self.assertIsNone(encrypt_credential(None))
        self.assertIsNone(decrypt_credential(None))

if __name__ == "__main__":
    unittest.main()
