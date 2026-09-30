import os
import secrets

from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import Request, HTTPException
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST


# ============================================================
# ENVIRONMENT / PROMETHEUS CONFIGURATION
# ============================================================

ENV = os.getenv("ENV", "dev").strip().lower()
IS_PROD = ENV == "prod"

PROMETHEUS_ENABLED = (
    os.getenv(
        "PROMETHEUS_ENABLED",
        "false" if IS_PROD else "true",
    )
    .strip()
    .lower()
    == "true"
)

PROMETHEUS_METRICS_PATH = os.getenv(
    "PROMETHEUS_METRICS_PATH",
    "/metrics",
).strip()

if not PROMETHEUS_METRICS_PATH.startswith("/"):
    PROMETHEUS_METRICS_PATH = f"/{PROMETHEUS_METRICS_PATH}"

PROMETHEUS_TOKEN = os.getenv("PROMETHEUS_TOKEN")


if IS_PROD and PROMETHEUS_ENABLED and not PROMETHEUS_TOKEN:
    raise RuntimeError(
        "PROMETHEUS_TOKEN is required when PROMETHEUS_ENABLED=true in production"
    )


# ============================================================
# GENERAL CCTV METRICS
# ============================================================

alerts_detected_total = Counter(
    "cctv_alerts_detected_total",
    "Alerts detected before Kafka",
    ["level", "rule", "source_type"],
)


db_alert_save_total = Counter(
    "cctv_db_alert_save_total",
    "Alert DB save results",
    ["status"],
)


kafka_publish_total = Counter(
    "cctv_kafka_publish_total",
    "Kafka publish results",
    ["status"],
)


frames_processed_total = Counter(
    "cctv_frames_processed_total",
    "Frames processed",
    ["source_type"],
)


outbox_pending = Gauge(
    "cctv_outbox_pending",
    "Pending local outbox events",
)


redis_health = Gauge(
    "cctv_redis_health",
    "Redis health 1 up 0 down",
)


kafka_health = Gauge(
    "cctv_kafka_health",
    "Kafka health 1 up 0 down",
)


active_ws_clients = Gauge(
    "cctv_active_websocket_clients",
    "Active websocket clients",
)


active_camera_workers = Gauge(
    "cctv_active_camera_workers",
    "Active camera workers",
)

# Distributed worker/lease observability.  Labels intentionally avoid user IDs
# or source URLs.
distributed_worker_owned_cameras = Gauge(
    "intel_i_distributed_worker_owned_cameras",
    "Cameras currently owned by this distributed analytics worker",
    ["worker_id"],
)

camera_lease_events_total = Counter(
    "intel_i_camera_lease_events_total",
    "Distributed camera lease lifecycle events",
    ["event"],
)

camera_failover_total = Counter(
    "intel_i_camera_failover_total",
    "Cameras recovered by a worker after ownership became available",
)


camera_read_failures_total = Counter(
    "cctv_camera_read_failures_total",
    "Camera frame read failures before reconnect",
    ["source_type"],
)


camera_reconnect_total = Counter(
    "cctv_camera_reconnect_total",
    "Camera reconnect attempts",
    ["source_type"],
)


camera_decoder_failures_total = Counter(
    "cctv_camera_decoder_failures_total",
    "Camera decoder open/initialization failures",
    ["source_type"],
)


alert_processing_seconds = Histogram(
    "cctv_alert_processing_seconds",
    "Alert handling latency",
)


# ============================================================
# VEHICLE AI METRICS
# ============================================================

vehicle_inference_seconds = Histogram(
    "cctv_vehicle_inference_seconds",
    "Vehicle detection and inference latency in seconds",
    buckets=(
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.1,
        0.15,
        0.25,
        0.5,
        1.0,
        2.0,
    ),
)


vehicle_detections_total = Counter(
    "cctv_vehicle_detections_total",
    "Total vehicle detections",
    ["vehicle_class"],
)


vehicle_tracking_active = Gauge(
    "cctv_vehicle_tracking_active",
    "Currently active vehicle tracks",
)


vehicle_correlation_seconds = Histogram(
    "cctv_vehicle_correlation_seconds",
    "Vehicle correlation processing latency in seconds",
    buckets=(
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
    ),
)


vehicle_correlation_score = Histogram(
    "cctv_vehicle_correlation_score",
    "Vehicle correlation confidence score",
    buckets=(
        0.0,
        0.25,
        0.5,
        0.6,
        0.7,
        0.8,
        0.85,
        0.9,
        0.95,
        0.98,
        0.99,
        1.0,
    ),
)


# ============================================================
# VEHICLE RE-ID METRICS
# ============================================================

vehicle_reid_inference_seconds = Histogram(
    "cctv_vehicle_reid_inference_seconds",
    "Vehicle Re-ID inference latency in seconds",
    buckets=(
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
    ),
)


# Existing pipeline compatibility name.
# This points to the SAME Prometheus Histogram and does not
# register a duplicate metric.
reid_inference_seconds = vehicle_reid_inference_seconds


# ============================================================
# VEHICLE ANPR / LPR METRICS
# ============================================================

vehicle_anpr_inference_seconds = Histogram(
    "cctv_vehicle_anpr_inference_seconds",
    "Vehicle ANPR/LPR inference latency in seconds",
    buckets=(
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
    ),
)


# Existing INTEL-I pipeline compatibility name.
# This points to the SAME Prometheus Histogram.
anpr_inference_seconds = vehicle_anpr_inference_seconds


# ============================================================
# VEHICLE PIPELINE QUALITY METRICS
# ============================================================

vehicle_track_created_total = Counter(
    "cctv_vehicle_track_created_total",
    "Vehicle tracks created",
)


vehicle_track_lost_total = Counter(
    "cctv_vehicle_track_lost_total",
    "Vehicle tracks lost",
)


vehicle_observations_total = Counter(
    "cctv_vehicle_observations_total",
    "Persisted vehicle observations",
)


vehicle_correlation_total = Counter(
    "cctv_vehicle_correlation_total",
    "Vehicle correlation decisions",
    ["decision"],
)


vehicle_correlation_rejected_total = Counter(
    "cctv_vehicle_correlation_rejected_total",
    "Vehicle correlation candidates rejected",
    ["reason"],
)


vehicle_anpr_reads_total = Counter(
    "cctv_vehicle_anpr_reads_total",
    "ANPR/LPR reads generated",
    ["status"],
)


vehicle_reid_comparisons_total = Counter(
    "cctv_vehicle_reid_comparisons_total",
    "Vehicle Re-ID comparisons",
)


vehicle_gis_projection_total = Counter(
    "cctv_vehicle_gis_projection_total",
    "Vehicle GIS projection attempts",
    ["status"],
)


vehicle_live_position_total = Counter(
    "cctv_vehicle_live_position_total",
    "Live vehicle GIS position events emitted",
    ["position_type"],
)


# ============================================================
# CAMERA PERFORMANCE METRICS
# ============================================================

camera_fps = Gauge(
    "cctv_camera_fps",
    "Current camera processing FPS",
    ["camera_id"],
)


camera_frame_latency_seconds = Histogram(
    "cctv_camera_frame_latency_seconds",
    "Camera frame processing latency in seconds",
    ["camera_id"],
    buckets=(
        0.005,
        0.01,
        0.02,
        0.03,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
    ),
)


camera_dropped_frames_total = Counter(
    "cctv_camera_dropped_frames_total",
    "Dropped camera frames",
    ["camera_id"],
)


camera_queue_depth = Gauge(
    "cctv_camera_queue_depth",
    "Current camera processing queue depth",
    ["camera_id"],
)


# ============================================================
# GPU METRICS
# ============================================================

gpu_utilization_percent = Gauge(
    "cctv_gpu_utilization_percent",
    "GPU utilization percentage",
    ["gpu_id"],
)


gpu_memory_percent = Gauge(
    "cctv_gpu_memory_percent",
    "GPU memory utilization percentage",
    ["gpu_id"],
)


gpu_memory_used_bytes = Gauge(
    "cctv_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["gpu_id"],
)


gpu_temperature_celsius = Gauge(
    "cctv_gpu_temperature_celsius",
    "GPU temperature in Celsius",
    ["gpu_id"],
)


# ============================================================
# GIS / LIVE VEHICLE TRACKING METRICS
# ============================================================

gis_calibration_total = Gauge(
    "cctv_gis_calibrated_cameras",
    "Number of cameras with valid GIS calibration",
)


gis_projection_seconds = Histogram(
    "cctv_gis_projection_seconds",
    "Pixel to geographic coordinate projection latency",
    buckets=(
        0.0001,
        0.0005,
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
    ),
)


live_vehicle_positions = Gauge(
    "cctv_live_vehicle_positions",
    "Number of currently active live vehicle positions",
)


# ============================================================
# EVIDENCE / INCIDENT METRICS
# ============================================================

incident_snapshots_total = Counter(
    "cctv_incident_snapshots_total",
    "Incident snapshots created",
    ["status"],
)


evidence_snapshots_total = Counter(
    "cctv_evidence_snapshots_total",
    "Evidence snapshots stored",
    ["type"],
)


evidence_integrity_failures_total = Counter(
    "cctv_evidence_integrity_failures_total",
    "Evidence integrity verification failures",
)


# ============================================================
# PROMETHEUS SECURITY
# ============================================================

def _protect_metrics_request(request: Request):
    """
    Protect the Prometheus endpoint in production.

    Development:
        /metrics is available without authentication.

    Production:
        Authorization: Bearer <PROMETHEUS_TOKEN>
        is required.

    A 404 is deliberately returned for unauthorized access so
    the existence of the metrics endpoint is not disclosed.
    """

    if not IS_PROD:
        return

    auth_header = request.headers.get("Authorization", "")

    expected_header = f"Bearer {PROMETHEUS_TOKEN}"

    if not secrets.compare_digest(
        auth_header,
        expected_header,
    ):
        raise HTTPException(
            status_code=404,
            detail="Not found",
        )


# ============================================================
# PROMETHEUS SETUP
# ============================================================

def setup_prometheus(app):
    """
    Configure Prometheus instrumentation for FastAPI.

    The function is safe to call during application startup.
    When PROMETHEUS_ENABLED=false, no metrics endpoint is
    registered.
    """

    if not PROMETHEUS_ENABLED:
        return

    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        excluded_handlers=[
            PROMETHEUS_METRICS_PATH,
        ],
    )

    instrumentator.instrument(app)

    @app.get(
        PROMETHEUS_METRICS_PATH,
        include_in_schema=False,
    )
    def metrics(request: Request):
        _protect_metrics_request(request)

        return Response(
            generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
# ============================================================
# PHASE 8 — AI SCHEDULER / RESOURCE PROTECTION
# ============================================================
ai_queue_depth = Gauge(
    "intel_i_ai_queue_depth",
    "Current AI scheduler queue depth",
)
ai_workers_active = Gauge(
    "intel_i_ai_workers_active",
    "Currently active AI scheduler workers",
)
ai_jobs_submitted_total = Counter(
    "intel_i_ai_jobs_submitted_total",
    "AI jobs admitted to the scheduler",
)
ai_jobs_completed_total = Counter(
    "intel_i_ai_jobs_completed_total",
    "AI jobs completed successfully",
)
ai_jobs_failed_total = Counter(
    "intel_i_ai_jobs_failed_total",
    "AI jobs failed",
)
ai_jobs_dropped_total = Counter(
    "intel_i_ai_jobs_dropped_total",
    "AI jobs dropped by backpressure or protection",
    ["reason"],
)
ai_gpu_utilization_percent = Gauge(
    "intel_i_ai_gpu_utilization_percent",
    "Cached maximum GPU utilization across visible GPUs",
)
ai_gpu_memory_percent = Gauge(
    "intel_i_ai_gpu_memory_percent",
    "Cached maximum GPU memory utilization across visible GPUs",
)

# ============================================================
# PHASE 9 — VEHICLE OBSERVATION / PIPELINE METRICS
# ============================================================
vehicle_observations_accepted_total = Counter(
    "intel_i_vehicle_observations_accepted_total",
    "Vehicle observations accepted by the Phase 9 validation gate",
)
vehicle_observations_rejected_total = Counter(
    "intel_i_vehicle_observations_rejected_total",
    "Vehicle observations rejected by the Phase 9 validation gate",
    ["reason"],
)
vehicle_reid_attempts_total = Counter(
    "intel_i_vehicle_reid_attempts_total",
    "Vehicle Re-ID inference attempts",
)
vehicle_anpr_attempts_total = Counter(
    "intel_i_vehicle_anpr_attempts_total",
    "Vehicle ANPR inference attempts",
)
vehicle_pipeline_errors_total = Counter(
    "intel_i_vehicle_pipeline_errors_total",
    "Vehicle pipeline stage failures",
    ["stage"],
)
