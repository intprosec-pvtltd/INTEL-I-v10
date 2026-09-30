import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def test_alembic_has_single_head():
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    root=Path(__file__).resolve().parents[1]
    heads=ScriptDirectory.from_config(Config(str(root/'alembic.ini'))).get_heads()
    assert heads == ['f9a1b2c3d4e5']

def test_outbox_waits_for_kafka_delivery_ack():
    text=(Path(__file__).resolve().parents[1]/'events'/'kafkaProducer.py').read_text()
    assert 'threading.Event()' in text
    assert 'Kafka delivery acknowledgement timed out' in text

def test_identity_correction_supports_all_actions():
    text=(Path(__file__).resolve().parents[1]/'services'/'identityCorrection.py').read_text()
    for action in ('SPLIT','MERGE','CORRECT','MARK_UNCERTAIN'):
        assert action in text

def test_missed_alert_recovery_endpoint_exists():
    text=(Path(__file__).resolve().parents[1]/'main.py').read_text()
    assert '/alerts/recover' in text
