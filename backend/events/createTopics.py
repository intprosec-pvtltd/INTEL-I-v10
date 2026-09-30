import os
from dotenv import load_dotenv
from confluent_kafka.admin import AdminClient, NewTopic

load_dotenv()


def admin_config():
    config = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
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


def create_topics():
    admin = AdminClient(admin_config())

    topics = [
        NewTopic(
            os.getenv("KAFKA_RAW_ALERT_TOPIC", "cctv.alerts.raw"),
            num_partitions=3,
            replication_factor=int(os.getenv("KAFKA_REPLICATION_FACTOR", "1")),
            config={
                "cleanup.policy": "delete",
                "retention.ms": os.getenv("KAFKA_RAW_RETENTION_MS", "604800000"),
            },
        ),
        NewTopic(
            os.getenv("KAFKA_SAVED_ALERT_TOPIC", "cctv.alerts.saved"),
            num_partitions=3,
            replication_factor=int(os.getenv("KAFKA_REPLICATION_FACTOR", "1")),
            config={
                "cleanup.policy": "delete",
                "retention.ms": os.getenv("KAFKA_SAVED_RETENTION_MS", "604800000"),
            },
        ),
    ]

    futures = admin.create_topics(topics)

    for topic, future in futures.items():
        try:
            future.result()
            print("Created topic:", topic)
        except Exception as e:
            if "already exists" in str(e).lower():
                print("Topic already exists:", topic)
            else:
                print("Topic create failed:", topic, e)


if __name__ == "__main__":
    create_topics()