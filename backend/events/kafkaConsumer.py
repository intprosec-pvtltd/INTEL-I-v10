import os
import json
import asyncio
from dotenv import load_dotenv
from confluent_kafka import Consumer

from db.database import getDB
from db.crud import save_alert, save_snapshot_to_db
from config import RULE_COOLDOWN_SECONDS
from events.kafkaProducer import init_kafka, publish_event
from monitoring.metrics import db_alert_save_total

load_dotenv()

RAW_TOPIC = os.getenv("KAFKA_RAW_ALERT_TOPIC", "cctv.alerts.raw")
SAVED_TOPIC = os.getenv("KAFKA_SAVED_ALERT_TOPIC", "cctv.alerts.saved")


def consumer_config():
    config = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        "group.id": os.getenv("KAFKA_CONSUMER_GROUP", "intel-i-alert-db-writer"),
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
        "security.protocol": os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT"),
    }

    if config["security.protocol"] in ["SASL_SSL", "SASL_PLAINTEXT"]:
        config["sasl.mechanism"] = os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512")
        config["sasl.username"] = os.getenv("KAFKA_SASL_USERNAME")
        config["sasl.password"] = os.getenv("KAFKA_SASL_PASSWORD")

    ca = os.getenv("KAFKA_SSL_CA_LOCATION")
    if ca:
        config["ssl.ca.location"] = ca

    return config


def save_event_to_db(event: dict):
    data = event.get("data", {})

    db = next(getDB())

    try:
        rule = data.get("rule")
        level = data.get("level") or data.get("alert_type") or "LOW"
        cam_id = data.get("cam_id")
        track_id = data.get("track_id")
        source_type = data.get("source_type")
        zone = data.get("zone")

        if not rule or not cam_id:
            return None

        alert_obj, created = save_alert(
            db=db,
            alert_type=level,
            alert_rule=rule,
            track_id=track_id,
            cam_id=cam_id,
            source_type=source_type,
            zone=zone,
            level=level,
            cooldown_seconds=RULE_COOLDOWN_SECONDS.get(rule, 180),
        )

        if not created or alert_obj is None:
            db_alert_save_total.labels(status="duplicate").inc()
            return None

        snapshot_data = data.get("snapshot_data")
        if snapshot_data:
            save_snapshot_to_db(db, alert_obj.id, snapshot_data)

        saved_event = {
            "event_type": "CCTV_ALERT_SAVED",
            "raw_event_id": event.get("event_id"),
            "alert_id": alert_obj.id,
            "data": {
                **data,
                "id": alert_obj.id,
            },
        }

        publish_event(
            topic=SAVED_TOPIC,
            key=cam_id,
            event=saved_event,
        )

        db_alert_save_total.labels(status="success").inc()
        return saved_event

    except Exception as e:
        db.rollback()
        db_alert_save_total.labels(status="failed").inc()
        print("DB consumer save error:", e)
        raise
    finally:
        db.close()


def consume_alerts():
    init_kafka()

    consumer = Consumer(consumer_config())
    consumer.subscribe([RAW_TOPIC])

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue

            if msg.error():
                print("Kafka consumer error:", msg.error())
                continue

            event = json.loads(msg.value().decode("utf-8"))

            try:
                save_event_to_db(event)
                consumer.commit(msg)
            except Exception:
                pass

    finally:
        consumer.close()


if __name__ == "__main__":
    consume_alerts()