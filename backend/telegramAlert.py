import os
import time
import logging
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from telegram import Bot
from telegram.error import TelegramError, InvalidToken

from db.database import SessionLocal
from db.crud import get_snapshot


# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger("telegram-alert")


# ============================================================================
# ENVIRONMENT
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_PATH,
    override=True,
)


# ============================================================================
# TELEGRAM CONFIGURATION
# ============================================================================

BOT_TOKEN = os.getenv(
    "BOTTOKEN",
    "",
).strip()

CHAT_ID = os.getenv(
    "CHATID",
    "",
).strip()


TELEGRAM_COOLDOWN_SECONDS = int(
    os.getenv(
        "TELEGRAM_COOLDOWN_SECONDS",
        "300",
    )
)

TELEGRAM_MEMORY_CLEAN_SECONDS = int(
    os.getenv(
        "TELEGRAM_MEMORY_CLEAN_SECONDS",
        "900",
    )
)


# ============================================================================
# BOT
# ============================================================================

bot = (
    Bot(token=BOT_TOKEN)
    if BOT_TOKEN
    else None
)


telegramAlertMemory = {}


def _masked_token() -> str:
    """
    Return a safe representation of the bot token.
    """

    if not BOT_TOKEN:
        return "missing"

    if len(BOT_TOKEN) <= 10:
        return "***"

    return (
        BOT_TOKEN[:6]
        + "..."
        + BOT_TOKEN[-4:]
    )


# ============================================================================
# CONFIG CHECK
# ============================================================================

def telegram_config_ok() -> bool:
    """
    Verify that Telegram configuration exists.
    """

    if not BOT_TOKEN:
        logger.error(
            "TELEGRAM_CONFIG_ERROR | BOTTOKEN missing | env=%s",
            ENV_PATH,
        )

        print(
            "[TELEGRAM] BOTTOKEN missing"
        )

        return False

    if not CHAT_ID:
        logger.error(
            "TELEGRAM_CONFIG_ERROR | CHATID missing | env=%s",
            ENV_PATH,
        )

        print(
            "[TELEGRAM] CHATID missing"
        )

        return False

    if not bot:
        logger.error(
            "TELEGRAM_CONFIG_ERROR | bot not initialized"
        )

        print(
            "[TELEGRAM] Bot not initialized"
        )

        return False

    return True


# ============================================================================
# TELEGRAM AUTHENTICATION TEST
# ============================================================================

async def test_telegram_auth() -> bool:
    """
    Verify that the configured Telegram bot token is valid.
    """

    if not telegram_config_ok():
        return False

    try:

        me = await bot.get_me()

        logger.info(
            "TELEGRAM_AUTH_OK | bot=%s | id=%s",
            me.username,
            me.id,
        )

        print(
            f"[TELEGRAM] Auth OK | "
            f"bot=@{me.username} | "
            f"id={me.id}"
        )

        return True

    except InvalidToken as exc:

        logger.exception(
            "TELEGRAM_AUTH_INVALID_TOKEN | token=%s | error=%s",
            _masked_token(),
            exc,
        )

        print(
            "[TELEGRAM] Invalid bot token"
        )

        return False

    except TelegramError as exc:

        logger.exception(
            "TELEGRAM_AUTH_FAILED | error=%s",
            exc,
        )

        print(
            f"[TELEGRAM] Authentication failed: {exc}"
        )

        return False

    except Exception as exc:

        logger.exception(
            "TELEGRAM_AUTH_UNEXPECTED_ERROR | error=%s",
            exc,
        )

        print(
            f"[TELEGRAM] Unexpected auth error: {exc}"
        )

        return False


# ============================================================================
# MEMORY CLEANUP
# ============================================================================

def clean_telegram_memory(
    timeout: int = TELEGRAM_MEMORY_CLEAN_SECONDS,
):
    """
    Remove expired Telegram cooldown records.
    """

    now = time.time()

    for key in list(
        telegramAlertMemory.keys()
    ):

        last_time = telegramAlertMemory.get(
            key
        )

        if not last_time:
            continue

        if (
            now - last_time
            > timeout
        ):
            del telegramAlertMemory[key]


# ============================================================================
# ALERT KEY
# ============================================================================

def get_telegram_alert_key(
    alert_data: dict,
) -> str:
    """
    Build the deduplication key.
    """

    cam_id = str(
        alert_data.get(
            "cam_id"
        )
        or alert_data.get(
            "camera_id"
        )
        or "unknown"
    )

    rule = str(
        alert_data.get(
            "alert_rule"
        )
        or alert_data.get(
            "rule"
        )
        or alert_data.get(
            "alert_type"
        )
        or "unknown"
    )

    track_id = str(
        alert_data.get(
            "track_id"
        )
        or "global"
    )

    alert_id = str(
        alert_data.get(
            "alert_id"
        )
        or alert_data.get(
            "id"
        )
        or ""
    )
    if alert_id:
        return (
            f"{cam_id}:"
            f"{rule}:"
            f"{track_id}:"
            f"{alert_id}"
        )

    return (
        f"{cam_id}:"
        f"{rule}:"
        f"{track_id}"
    )


# ============================================================================
# COOLDOWN CHECK
# ============================================================================

def telegram_allowed(alert_data: dict) -> bool:
    clean_telegram_memory()
    now = time.time()
    key = get_telegram_alert_key(
        alert_data
    )

    last_time = (
        telegramAlertMemory.get(
            key
        )
    )

    if last_time:

        elapsed = (
            now - last_time
        )

        if (
            elapsed
            < TELEGRAM_COOLDOWN_SECONDS
        ):

            logger.info(
                "TELEGRAM_SKIPPED_COOLDOWN | "
                "key=%s | elapsed=%.1fs | cooldown=%ss",
                key,
                elapsed,
                TELEGRAM_COOLDOWN_SECONDS,
            )

            return False

    return True


def mark_telegram_sent(
    alert_data: dict,
):
    key = get_telegram_alert_key(
        alert_data
    )

    telegramAlertMemory[key] = (
        time.time()
    )

    logger.debug(
        "TELEGRAM_COOLDOWN_RECORDED | key=%s",
        key,
    )


# ============================================================================
# SEVERITY NORMALIZATION
# ============================================================================

def normalize_level(
    alert_data: dict,
) -> str:

    raw = str(
        alert_data.get(
            "severity"
        )
        or alert_data.get(
            "level"
        )
        or alert_data.get(
            "alert_level"
        )
        or alert_data.get(
            "priority"
        )
        or ""
    ).upper().strip()

    if raw in {
        "CRITICAL",
        "HIGH",
    }:
        return "HIGH"

    if raw in {
        "WARNING",
        "MEDIUM",
    }:
        return "MEDIUM"

    if raw == "LOW":
        return "LOW"

    if raw in {
        "INFO",
        "INFORMATION",
    }:
        return "INFO"

    return "INFO"


# ============================================================================
# TELEGRAM SEVERITY POLICY
# ============================================================================

def telegram_severity_allowed(
    alert_data: dict,
) -> bool:

    level = normalize_level(
        alert_data
    )

    if level == "INFO":

        logger.info(
            "TELEGRAM_SKIPPED_INFO | alert_id=%s",
            alert_data.get(
                "alert_id"
            )
            or alert_data.get(
                "id"
            ),
        )

        return False

    return True


# ============================================================================
# CAPTION
# ============================================================================

def build_caption(
    alert_data: dict,
) -> str:

    alert_id = (
        alert_data.get(
            "alert_id"
        )
        or alert_data.get(
            "id"
        )
        or "N/A"
    )

    level = normalize_level(
        alert_data
    )

    rule = (
        alert_data.get(
            "alert_rule"
        )
        or alert_data.get(
            "rule"
        )
        or alert_data.get(
            "alert_type"
        )
        or "Unknown"
    )

    camera_id = (
        alert_data.get(
            "cam_id"
        )
        or alert_data.get(
            "camera_id"
        )
        or "N/A"
    )

    source_type = (
        alert_data.get(
            "source_type"
        )
        or alert_data.get(
            "source"
        )
        or "N/A"
    )

    zone = (
        alert_data.get(
            "zone"
        )
        or "N/A"
    )

    track_id = (
        alert_data.get(
            "track_id"
        )
        or "N/A"
    )

    confidence = (
        alert_data.get(
            "confidence"
        )
        or alert_data.get(
            "match_confidence"
        )
        or "N/A"
    )

    timestamp = (
        alert_data.get(
            "timestamp"
        )
        or alert_data.get(
            "created_at"
        )
        or "N/A"
    )

    plate = (
        alert_data.get(
            "plate"
        )
        or alert_data.get(
            "plate_number"
        )
        or alert_data.get(
            "license_plate"
        )
    )

    watchlist_category = (
        alert_data.get(
            "watchlist_category"
        )
        or alert_data.get(
            "person_category"
        )
    )

    watchlist_person_name = (
        alert_data.get("watchlist_person_name")
        or alert_data.get("person_name")
        or alert_data.get("full_name")
    )

    location_name = (
        alert_data.get("location_name")
        or alert_data.get("camera_location")
    )

    match_type = (
        alert_data.get(
            "watchlist_match_type"
        )
        or alert_data.get(
            "match_type"
        )
    )

    message = (
        alert_data.get(
            "message"
        )
        or ""
    )

    lines = [
        "🚨 INTEL-I ALERT",
        "━━━━━━━━━━━━━━━━━━",
        f"🆔 Alert ID: {alert_id}",
        f"📌 Severity: {level}",
        f"📋 Rule: {rule}",
        f"📹 Camera: {camera_id}",
        f"🎥 Source: {source_type}",
        f"📍 Zone: {zone}",
        f"🔎 Track ID: {track_id}",
        f"🎯 Confidence: {confidence}",
        f"🕐 Time: {timestamp}",
    ]

    if watchlist_person_name:
        lines.append(
            f"👤 Watchlist Person: {watchlist_person_name}"
        )

    if location_name and str(location_name) != str(zone):
        lines.append(
            f"🗺 Location: {location_name}"
        )

    if plate:
        lines.append(
            f"🔢 Plate: {plate}"
        )

    if watchlist_category:
        lines.append(
            f"🚨 Watchlist: {watchlist_category}"
        )

    if match_type:
        lines.append(
            f"🎯 Match: {match_type}"
        )

    lines.extend(
        [
            "━━━━━━━━━━━━━━━━━━",
            message,
        ]
    )

    return "\n".join(
        lines
    ).strip()


# ============================================================================
# SNAPSHOT ID
# ============================================================================

def get_snapshot_id(
    alert_data: dict,
):
    """
    Resolve snapshot ID from payload.
    """

    snapshot_id = alert_data.get(
        "snapshot_id"
    )

    if snapshot_id:

        try:
            return int(
                snapshot_id
            )

        except Exception:

            logger.warning(
                "TELEGRAM_INVALID_SNAPSHOT_ID | value=%s",
                snapshot_id,
            )

    snapshot_path = alert_data.get(
        "snapshot_path"
    )

    if snapshot_path:

        try:

            path_value = (
                str(snapshot_path)
                .rstrip("/")
            )

            return int(
                path_value
                .split("/")
                [-1]
            )

        except Exception:

            logger.warning(
                "TELEGRAM_INVALID_SNAPSHOT_PATH | path=%s",
                snapshot_path,
            )

    return None


# ============================================================================
# SEND TELEGRAM ALERT
# ============================================================================

async def sendTelegramAlert(
    alert_data: dict,
) -> bool:
    """
    Send an INTEL-I alert to Telegram.

    Returns:

        True  = successfully delivered
        False = not delivered
    """

    alert_id = (
        alert_data.get(
            "alert_id"
        )
        or alert_data.get(
            "id"
        )
        or "unknown"
    )

    logger.info(
        "TELEGRAM_SEND_START | "
        "alert_id=%s | "
        "level=%s | "
        "cam_id=%s | "
        "rule=%s",
        alert_id,
        normalize_level(
            alert_data
        ),
        alert_data.get(
            "cam_id"
        ),
        alert_data.get(
            "alert_rule"
        )
        or alert_data.get(
            "rule"
        ),
    )

    # ========================================================================
    # CONFIG
    # ========================================================================

    if not telegram_config_ok():
        logger.error(
            "TELEGRAM_SEND_ABORTED | configuration invalid | alert_id=%s",
            alert_id,
        )

        return False

    # ========================================================================
    # SEVERITY
    # ========================================================================

    if not telegram_severity_allowed(
        alert_data
    ):
        return False

    # ========================================================================
    # COOLDOWN
    # ========================================================================

    if not telegram_allowed(
        alert_data
    ):
        return False

    # ========================================================================
    # CAPTION
    # ========================================================================

    caption = build_caption(
        alert_data
    )

    # ========================================================================
    # SNAPSHOT
    # ========================================================================

    snapshot_id = get_snapshot_id(
        alert_data
    )

    # ========================================================================
    # SEND
    # ========================================================================

    try:

        # --------------------------------------------------------------------
        # WITH SNAPSHOT
        # --------------------------------------------------------------------

        if snapshot_id:

            logger.info(
                "TELEGRAM_SNAPSHOT_LOOKUP | "
                "alert_id=%s | "
                "snapshot_id=%s",
                alert_id,
                snapshot_id,
            )

            db = SessionLocal()

            try:

                snapshot = get_snapshot(
                    db,
                    snapshot_id,
                )

                if (
                    snapshot
                    and snapshot.image_data
                ):

                    image_file = BytesIO(
                        snapshot.image_data
                    )

                    image_file.name = (
                        f"alert_{snapshot_id}.jpg"
                    )

                    logger.info(
                        "TELEGRAM_SEND_PHOTO | "
                        "alert_id=%s | "
                        "snapshot_id=%s | "
                        "chat_id=%s",
                        alert_id,
                        snapshot_id,
                        CHAT_ID,
                    )

                    await bot.send_photo(
                        chat_id=CHAT_ID,
                        photo=image_file,
                        caption=caption[:1024],
                    )

                    # IMPORTANT:
                    # Cooldown is recorded ONLY after successful send.
                    mark_telegram_sent(
                        alert_data
                    )

                    logger.info(
                        "TELEGRAM_PHOTO_SENT | "
                        "alert_id=%s | "
                        "snapshot_id=%s",
                        alert_id,
                        snapshot_id,
                    )

                    print(
                        f"[TELEGRAM] PHOTO SENT | "
                        f"alert_id={alert_id} | "
                        f"snapshot_id={snapshot_id}"
                    )

                    return True

                logger.warning(
                    "TELEGRAM_SNAPSHOT_NOT_FOUND | "
                    "alert_id=%s | "
                    "snapshot_id=%s",
                    alert_id,
                    snapshot_id,
                )

            finally:

                db.close()

        # --------------------------------------------------------------------
        # TEXT FALLBACK
        # --------------------------------------------------------------------

        logger.info(
            "TELEGRAM_SEND_TEXT | "
            "alert_id=%s | "
            "chat_id=%s",
            alert_id,
            CHAT_ID,
        )

        await bot.send_message(
            chat_id=CHAT_ID,
            text=caption[:4096],
        )

        # IMPORTANT:
        # Cooldown is recorded ONLY after successful send.
        mark_telegram_sent(
            alert_data
        )

        logger.info(
            "TELEGRAM_TEXT_SENT | alert_id=%s",
            alert_id,
        )

        print(
            f"[TELEGRAM] TEXT SENT | "
            f"alert_id={alert_id}"
        )

        return True

    except InvalidToken as exc:

        logger.exception(
            "TELEGRAM_INVALID_TOKEN | "
            "alert_id=%s | "
            "token=%s | "
            "error=%s",
            alert_id,
            _masked_token(),
            exc,
        )

        print(
            "[TELEGRAM] Invalid bot token"
        )

        return False

    except TelegramError as exc:

        logger.exception(
            "TELEGRAM_API_ERROR | "
            "alert_id=%s | "
            "error=%s",
            alert_id,
            exc,
        )

        print(
            f"[TELEGRAM] API error: {exc}"
        )

        return False

    except Exception as exc:

        logger.exception(
            "TELEGRAM_UNEXPECTED_ERROR | "
            "alert_id=%s | "
            "error=%s",
            alert_id,
            exc,
        )

        print(
            f"[TELEGRAM] Unexpected error: {exc}"
        )

        return False
