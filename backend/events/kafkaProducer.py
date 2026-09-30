import os
import json
import logging
import threading
import time
from confluent_kafka import Producer, KafkaException

logger = logging.getLogger(__name__)

KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_CLIENT_ID = os.getenv("KAFKA_CLIENT_ID", "intel-i-backend")
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")

producer: Producer | None = None


def init_kafka() -> bool:
    global producer

    if not KAFKA_ENABLED:
        logger.warning("Kafka disabled by KAFKA_ENABLED=false")
        producer = None
        return False

    try:
        config = {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "client.id": KAFKA_CLIENT_ID,
            "security.protocol": KAFKA_SECURITY_PROTOCOL,
            "acks": "all",
            "retries": 5,
            "retry.backoff.ms": 500,
            "enable.idempotence": True,
            "compression.type": "snappy",
            "linger.ms": 10,
            "socket.keepalive.enable": True,
        }

        sasl_username = os.getenv("KAFKA_SASL_USERNAME")
        sasl_password = os.getenv("KAFKA_SASL_PASSWORD")
        sasl_mechanism = os.getenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512")
        ssl_ca_location = os.getenv("KAFKA_SSL_CA_LOCATION")

        if KAFKA_SECURITY_PROTOCOL in {"SASL_SSL", "SASL_PLAINTEXT"}:
            if not sasl_username or not sasl_password:
                raise RuntimeError("Kafka SASL username/password missing")

            config.update(
                {
                    "sasl.username": sasl_username,
                    "sasl.password": sasl_password,
                    "sasl.mechanism": sasl_mechanism,
                }
            )

        if KAFKA_SECURITY_PROTOCOL in {"SSL", "SASL_SSL"} and ssl_ca_location:
            config["ssl.ca.location"] = ssl_ca_location

        producer = Producer(config)

        producer.list_topics(timeout=10)

        logger.info("Kafka connected")
        return True

    except Exception:
        producer = None
        logger.exception("Kafka connection failed")
        return False


def kafka_status() -> bool:
    return producer is not None


def delivery_report(err, msg):
    if err is not None:
        logger.error("Kafka delivery failed: %s", err)
    else:
        logger.debug(
            "Kafka delivered topic=%s partition=%s offset=%s",
            msg.topic(),
            msg.partition(),
            msg.offset(),
        )


def produce_event(topic: str, key: str | None, value: dict, timeout: float = 10.0):
    if producer is None:
        raise RuntimeError("Kafka producer is not initialized")
    if not topic:
        raise ValueError("Kafka topic is required")

    payload = json.dumps(value, default=str).encode("utf-8")
    kafka_key = key.encode("utf-8") if key else None
    done = threading.Event()
    result = {"error": None}

    def callback(err, msg):
        result["error"] = err
        if err is None:
            logger.debug("Kafka delivered topic=%s partition=%s offset=%s", msg.topic(), msg.partition(), msg.offset())
        else:
            logger.error("Kafka delivery failed: %s", err)
        done.set()

    try:
        producer.produce(topic=topic, key=kafka_key, value=payload, callback=callback)
        deadline = time.monotonic() + max(0.1, float(timeout))
        while not done.is_set() and time.monotonic() < deadline:
            producer.poll(min(0.25, max(0.0, deadline - time.monotonic())))
        if not done.is_set():
            raise TimeoutError("Kafka delivery acknowledgement timed out")
        if result["error"] is not None:
            raise KafkaException(result["error"])
    except BufferError as exc:
        producer.poll(1)
        raise RuntimeError("Kafka producer queue is full") from exc
    except KafkaException:
        logger.exception("Kafka produce failed")
        raise


def flush_kafka(timeout: int = 10):
    if producer is not None:
        producer.flush(timeout)


def close_kafka():
    global producer

    if producer is not None:
        try:
            producer.flush(10)
        finally:
            producer = None
