import os

def _bounded_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default

    return max(minimum, min(value, maximum))


def _bounded_float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)

    if raw is None or not raw.strip():
        return default

    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default

    return max(minimum, min(value, maximum))


MODEL_PERSON = os.getenv("MODEL_PERSON", "models/yolov8m.pt")
MODEL_POSE = os.getenv("MODEL_POSE", "models/yolov8m-pose.pt")
MODEL_CRIME = os.getenv("MODEL_CRIME", "models/Suspicious_Activities_nano.pt")
MODEL_PPE = os.getenv("MODEL_PPE", "models/best.pt")

DEVICE = os.getenv("AI_DEVICE", "0")


SNAPSHOT_ONCE_PER_TRACK_RULE = True

FRAME_WIDTH = 960
FRAME_HEIGHT = 540
DEFAULT_FPS = 15
PROCESS_EVERY_N_FRAMES = 2

PERSON_CONF = 0.60
POSE_CONF = 0.45
CRIME_CONF = 0.18
PPE_CONF = 0.35
MIN_PERSON_BOX_AREA = 2200
MAX_BOX_ASPECT_RATIO = 4.5

MIN_POSE_VISIBILITY = 0.32
MIN_VISIBLE_KEYPOINTS = 5

MIN_TRACK_AGE_SECONDS = 1.8
MIN_TRACK_HITS = 2

TRACK_HISTORY_SECONDS = 20
STATE_CLEANUP_SECONDS = 20

MIN_BRIGHTNESS = 5
MAX_BRIGHTNESS = 240
MIN_BLUR_SCORE = 2
MAX_DARK_RATIO = 0.6

RUNNING_MIN_SECONDS = 3.0
CLOSE_INTERACTION_MIN_SECONDS = 120.0
LOITERING_MIN_SECONDS = 300.0
SUSPICIOUS_ESCALATION_MIN_SECONDS = 2.5
FIGHTING_MIN_SECONDS = 4.0
CRIME_CONFIRM_SECONDS = 2.5
WEAPON_CONFIRM_SECONDS = 0.8
FALL_CONFIRM_SECONDS = 3.0
PERIMETER_CONFIRM_SECONDS = 3.0

FACE_COVER_CONFIRM_SECONDS = 3.0
HELMET_CONFIRM_SECONDS = 2.0

FOLLOW_MIN_SECONDS = 10.0
FOLLOW_DISTANCE_NORM = 0.18
FOLLOW_DIRECTION_COSINE = 0.65
FOLLOW_SPEED_DIFF_MAX = 0.04

CHAIN_SNATCH_MIN_SECONDS = 1.2
CHAIN_SNATCH_DISTANCE_NORM = 0.10
CHAIN_SNATCH_WRIST_SPEED_NORM = 0.14
CHAIN_SNATCH_AGGRESSIVE_SCORE = 0.20

ABDUCTION_CONFIRM_SECONDS = 2.0
ABDUCTION_DISTANCE_NORM = 0.11
ABDUCTION_WRIST_SPEED_NORM = 0.12
ABDUCTION_AGGRESSIVE_SCORE = 0.22
ABDUCTION_MIN_SPEED_NORM = 0.015

REPEATED_ENTRY_EXIT_MIN_COUNT = 4

RUNNING_SPEED_NORM = 0.10
FACE_MASK_MIN_SPEED_NORM = 0.00
HELMET_MIN_SPEED_NORM = 0.00

LOITERING_RADIUS_NORM = 0.020
CLOSE_DISTANCE_NORM = 0.09
FIGHT_DISTANCE_NORM = 0.12

AGGRESSIVE_WRIST_SPEED_NORM = 0.22
SUSPICIOUS_WRIST_SPEED_NORM = 0.16

ALERT_LEVEL_CRITICAL = "CRITICAL"
ALERT_LEVEL_WARNING = "WARNING"
ALERT_LEVEL_MEDIUM = "MEDIUM"
ALERT_LEVEL_LOW = "LOW"

SNAPSHOT_LEVELS = {
    ALERT_LEVEL_CRITICAL,
    ALERT_LEVEL_WARNING,
}

NOTIFICATION_LEVELS = {
    ALERT_LEVEL_CRITICAL,
    ALERT_LEVEL_WARNING,
}

LEVEL_PRIORITY = {
    ALERT_LEVEL_LOW: 1,
    ALERT_LEVEL_MEDIUM: 2,
    ALERT_LEVEL_WARNING: 3,
    ALERT_LEVEL_CRITICAL: 4,
}

CRITICAL_CRIME_CLASSES = {
    "assaulting",
    "fighting",
    "kidnapping",
    "man_with_gun",
    "man_with_knife",
    "terrorist_with_time_bomb",
    "theaf_robbery",
}

IGNORED_CRIME_CLASSES = {
    "people",
    "person",
    "police",
    "prisoner",
}

ABDUCTION_CLASSES = {"kidnapping"}

WEAPON_CLASSES = {
    "man_with_gun",
    "man_with_knife",
}

GUN_CLASSES = {"man_with_gun"}
KNIFE_CLASSES = {"man_with_knife"}
THEFT_CLASSES = {"theaf_robbery"}
FIGHT_CLASSES = {"fighting"}
ASSAULT_CLASSES = {"assaulting"}

SUSPICIOUS_CLASSES = {
    "terrorist_with_time_bomb",
}

PPE_MASK_CLASSES = {"mask"}
PPE_HELMET_CLASSES = {"hardhat", "helmet"}

FACE_COVER_CLASSES = {
    "mask",
    "hardhat",
    "helmet",
}

BAG_CLASSES = {
    "bag",
    "backpack",
    "handbag",
}

MIN_ALERT_CONFIDENCE = {
    "Sudden Running": 0.50,
    "Close Interaction": 0.50,
    "Loitering": 0.50,
    "Repeated Entry Exit": 0.50,
    "Suspicious Behaviour": 0.50,
    "Suspicious Escalation": 0.50,

    "Following Behaviour": 0.50,
    "Chain Snatching Risk": 0.60,
    "Possible Abduction Risk": 0.70,

    "Perimeter Breach": 0.50,
    "Face Covered Walking": 0.60,
    "Helmet Walking": 0.45,
    "Abandoned Object": 0.50,
    "Fighting": 0.70,
    "Gun Handling": 0.50,
    "Knife Handling": 0.50,
    "Theft / Robbery": 0.70,
    "Assault Risk": 0.80,
    "Fall Emergency": 0.80,
    "Crime Activity": 0.50,

    "Crowd Count": 0.40,
    "Person Count": 0.40,
    "Vehicle Count": 0.40,
}

RULE_LEVEL = {
    "Suspicious Escalation": ALERT_LEVEL_CRITICAL,
    "Chain Snatching Risk": ALERT_LEVEL_CRITICAL,
    "Possible Abduction Risk": ALERT_LEVEL_CRITICAL,
    "Perimeter Breach": ALERT_LEVEL_CRITICAL,
    "Fighting": ALERT_LEVEL_CRITICAL,
    "Gun Handling": ALERT_LEVEL_CRITICAL,
    "Knife Handling": ALERT_LEVEL_CRITICAL,
    "Theft / Robbery": ALERT_LEVEL_CRITICAL,
    "Assault Risk": ALERT_LEVEL_CRITICAL,
    "Fall Emergency": ALERT_LEVEL_CRITICAL,
    "Crime Activity": ALERT_LEVEL_CRITICAL,

    "Sudden Running": ALERT_LEVEL_WARNING,
    "Repeated Entry Exit": ALERT_LEVEL_WARNING,
    "Suspicious Behaviour": ALERT_LEVEL_WARNING,
    "Following Behaviour": ALERT_LEVEL_WARNING,
    "Face Covered Walking": ALERT_LEVEL_WARNING,
    "Helmet Walking": ALERT_LEVEL_WARNING,
    "Abandoned Object": ALERT_LEVEL_WARNING,

    "Close Interaction": ALERT_LEVEL_MEDIUM,
    "Loitering": ALERT_LEVEL_MEDIUM,

    "Crowd Count": ALERT_LEVEL_LOW,
    "Person Count": ALERT_LEVEL_LOW,
    "Vehicle Count": ALERT_LEVEL_LOW,
}

RULE_COOLDOWN_SECONDS = {
    "Sudden Running": 120,
    "Close Interaction": 180,
    "Loitering": 300,
    "Repeated Entry Exit": 240,
    "Suspicious Behaviour": 120,
    "Suspicious Escalation": 120,
    "Following Behaviour": 240,
    "Chain Snatching Risk": 400,
    "Possible Abduction Risk": 300,
    "Perimeter Breach": 240,
    "Face Covered Walking": 300,
    "Helmet Walking": 180,
    "Abandoned Object": 300,
    "Fighting": 120,
    "Gun Handling": 180,
    "Knife Handling": 180,
    "Theft / Robbery": 180,
    "Assault Risk": 180,
    "Fall Emergency": 240,
    "Crime Activity": 180,

    "Crowd Count": 300,
    "Person Count": 300,
    "Vehicle Count": 300,
}

SNAPSHOT_COOLDOWN = 90
SNAPSHOT_RETENTION_DAYS = 3

ZONE_CONFIG_PATH = "zones.json"
DEFAULT_ZONE_MODE = "allow_all"


# ---------------------------------------------------------------------------
# Stream ingestion / deterministic analytics timing
# ---------------------------------------------------------------------------
OUTPUT_STREAM_FPS = 15.0
RTSP_TRANSPORT = "tcp"
RTSP_READ_TIMEOUT_SECONDS = 5.0
RTSP_RECONNECT_INITIAL_SECONDS = 1.0
RTSP_RECONNECT_MAX_SECONDS = 15.0
STREAM_GAP_THRESHOLD_SECONDS = 3.0
SCENE_DISCONTINUITY_THRESHOLD_SECONDS = 3.0
MAX_STREAM_WIDTH = 4096
MAX_STREAM_HEIGHT = 4096
MEDIA_DECODER = os.getenv("MEDIA_DECODER", "pyav").strip().lower()
STRICT_PTS_REQUIRED = os.getenv("STRICT_PTS_REQUIRED", "true").strip().lower() in {
    "1", "true", "yes", "on",
}
PTS_GAP_THRESHOLD_SECONDS = _bounded_float_env(
    "PTS_GAP_THRESHOLD_SECONDS", 3.0, 0.25, 120.0
)
PTS_BACKWARD_TOLERANCE_SECONDS = _bounded_float_env(
    "PTS_BACKWARD_TOLERANCE_SECONDS", 0.10, 0.0, 10.0
)
SCENE_HARD_CUT_THRESHOLD = _bounded_float_env(
    "SCENE_HARD_CUT_THRESHOLD", 0.42, 0.10, 1.0
)


# ---------------------------------------------------------------------------
# Phase 8 — AI Scheduler / backpressure / GPU protection
# ---------------------------------------------------------------------------
AI_WORKERS = _bounded_int_env("AI_WORKERS", 2, 1, 32)
AI_GLOBAL_QUEUE_LIMIT = _bounded_int_env("AI_GLOBAL_QUEUE_LIMIT", 32, 1, 512)
AI_CAMERA_QUEUE_LIMIT = _bounded_int_env("AI_CAMERA_QUEUE_LIMIT", 2, 1, 16)
AI_JOB_TIMEOUT_SECONDS = _bounded_float_env("AI_JOB_TIMEOUT_SECONDS", 8.0, 0.5, 120.0)
AI_MAX_JOB_AGE_SECONDS = _bounded_float_env("AI_MAX_JOB_AGE_SECONDS", 2.0, 0.1, 30.0)

AI_GPU_PROTECTION_ENABLED = os.getenv(
    "AI_GPU_PROTECTION_ENABLED",
    "true",
).strip().lower() in {"1", "true", "yes", "on"}

AI_GPU_MAX_UTILIZATION_PERCENT = _bounded_float_env(
    "AI_GPU_MAX_UTILIZATION_PERCENT",
    95.0,
    50.0,
    100.0,
)

AI_GPU_MAX_MEMORY_PERCENT = _bounded_float_env(
    "AI_GPU_MAX_MEMORY_PERCENT",
    92.0,
    50.0,
    100.0,
)


# ---------------------------------------------------------------------------
# Phase 7 — TensorRT artifact/runtime policy
# ---------------------------------------------------------------------------
TENSORRT_ENABLED = os.getenv(
    "TENSORRT_ENABLED",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}

TENSORRT_REQUIRED = os.getenv(
    "TENSORRT_REQUIRED",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}

TENSORRT_MANIFEST = os.getenv(
    "TENSORRT_MANIFEST",
    "models/tensorrt/manifest.json",
)

TENSORRT_VEHICLE_ROLE = os.getenv(
    "TENSORRT_VEHICLE_ROLE",
    "vehicle_detector",
)

TENSORRT_PLATE_ROLE = os.getenv(
    "TENSORRT_PLATE_ROLE",
    "plate_detector",
)

AI_GPU_PROTECTION_REQUIRE_TELEMETRY = os.getenv(
    "AI_GPU_PROTECTION_REQUIRE_TELEMETRY",
    "false",
).strip().lower() in {"1", "true", "yes", "on"}
