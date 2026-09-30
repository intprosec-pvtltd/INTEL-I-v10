#!/usr/bin/env bash
set -euo pipefail
# Run on the labeled EC2 node. Does not format, partition, mount or delete disks.
for name in postgres redis kafka minio; do
  sudo mkdir -p "/var/lib/intel-i/$name"
done
# Apache Kafka official image uses appuser UID/GID 1000.
sudo chown 1000:1000 /var/lib/intel-i/kafka
sudo chmod 0750 /var/lib/intel-i/kafka
# Postgres/Redis entrypoints initialize ownership of their data directories.
df -h /var/lib/intel-i
findmnt -T /var/lib/intel-i
