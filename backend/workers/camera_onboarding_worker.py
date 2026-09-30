from __future__ import annotations
import asyncio
import logging
import os
import signal
from typing import Final

import db.model  # noqa: F401

# Register models stored in separate modules.
import db.advanced_intelligence_model  # noqa: F401
import db.auth_security_model  # noqa: F401
import db.intelligence_model  # noqa: F401
import db.watchlist_model  # noqa: F401

from db.database import SessionLocal
from db.model import CameraOnboardingJob
from services.cameraOnboarding import run_job


logging.basicConfig(
    level=os.getenv(
        "LOG_LEVEL",
        "INFO",
    ).upper(),
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "intel_i.camera_onboarding_worker"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_POLL_SECONDS: Final[float] = 2.0
MIN_POLL_SECONDS: Final[float] = 0.5
MAX_POLL_SECONDS: Final[float] = 30.0

DEFAULT_ERROR_BACKOFF_SECONDS: Final[float] = 5.0
MAX_ERROR_BACKOFF_SECONDS: Final[float] = 60.0


def _environment_flag(
    name: str,
    default: bool,
) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return raw_value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _bounded_float(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    raw_value = os.getenv(
        name,
        str(default),
    )

    try:
        value = float(raw_value)

    except (TypeError, ValueError):
        logger.warning(
            "Invalid numeric environment value; using default | "
            "name=%s default=%s",
            name,
            default,
        )

        value = default

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


# ============================================================
# SHUTDOWN CONTROL
# ============================================================

stop_event = asyncio.Event()


def _request_shutdown() -> None:
    """Request graceful worker shutdown."""

    if not stop_event.is_set():
        logger.info(
            "Camera onboarding worker shutdown requested"
        )

        stop_event.set()


def _register_signal_handlers() -> None:
    """Register graceful shutdown handlers.

    Windows may not support loop.add_signal_handler().
    In that case, signal.signal() is used as a fallback.
    """

    loop = asyncio.get_running_loop()

    for signal_name in (
        "SIGINT",
        "SIGTERM",
    ):
        signal_value = getattr(
            signal,
            signal_name,
            None,
        )

        if signal_value is None:
            continue

        try:
            loop.add_signal_handler(
                signal_value,
                _request_shutdown,
            )

        except (
            NotImplementedError,
            RuntimeError,
        ):
            try:
                signal.signal(
                    signal_value,
                    lambda *_args: _request_shutdown(),
                )

            except (
                ValueError,
                OSError,
            ):
                logger.warning(
                    "Unable to register shutdown signal | "
                    "signal=%s",
                    signal_name,
                )


# ============================================================
# JOB CLAIMING
# ============================================================

def _claim_next_job() -> str | None:
    """Atomically claim the oldest queued onboarding job.

    PostgreSQL's SKIP LOCKED allows multiple onboarding workers
    to claim different jobs without processing the same job
    simultaneously.
    """

    db = SessionLocal()

    try:
        job = (
            db.query(
                CameraOnboardingJob
            )
            .filter(
                CameraOnboardingJob.status
                == "QUEUED"
            )
            .order_by(
                CameraOnboardingJob.created_at.asc()
            )
            .with_for_update(
                skip_locked=True
            )
            .first()
        )

        if job is None:
            db.rollback()
            return None

        job.status = "CLAIMED"

        job_id = str(job.id)

        db.commit()

        logger.info(
            "Camera onboarding job claimed | "
            "job_id=%s total=%s",
            job_id,
            job.total,
        )

        return job_id

    except Exception:
        db.rollback()

        logger.exception(
            "Unable to claim camera onboarding job"
        )

        raise

    finally:
        db.close()


# ============================================================
# WAITING
# ============================================================

async def _wait_for_next_poll(
    seconds: float,
) -> None:
    """Wait for either the poll interval or shutdown signal."""

    try:
        await asyncio.wait_for(
            stop_event.wait(),
            timeout=seconds,
        )

    except asyncio.TimeoutError:
        # A timeout is the expected polling behavior.
        pass


# ============================================================
# WORKER
# ============================================================

async def main() -> None:
    """Run the durable onboarding job-consumer loop."""

    if not _environment_flag(
        "CAMERA_ONBOARDING_ENABLED",
        True,
    ):
        logger.warning(
            "Camera onboarding worker is disabled | "
            "CAMERA_ONBOARDING_ENABLED=false"
        )

        return

    poll_seconds = _bounded_float(
        name=(
            "CAMERA_ONBOARDING_WORKER_POLL_SECONDS"
        ),
        default=DEFAULT_POLL_SECONDS,
        minimum=MIN_POLL_SECONDS,
        maximum=MAX_POLL_SECONDS,
    )

    _register_signal_handlers()

    logger.info(
        "Camera onboarding worker started | "
        "poll_seconds=%s",
        poll_seconds,
    )

    consecutive_claim_errors = 0

    while not stop_event.is_set():
        try:
            job_id = await asyncio.to_thread(
                _claim_next_job
            )

            consecutive_claim_errors = 0

        except Exception:
            consecutive_claim_errors += 1

            error_backoff = min(
                MAX_ERROR_BACKOFF_SECONDS,
                DEFAULT_ERROR_BACKOFF_SECONDS
                * (
                    2
                    ** min(
                        consecutive_claim_errors - 1,
                        4,
                    )
                ),
            )

            logger.warning(
                "Onboarding worker will retry job claim | "
                "retry_seconds=%s consecutive_errors=%s",
                error_backoff,
                consecutive_claim_errors,
            )

            await _wait_for_next_poll(
                error_backoff
            )

            continue

        if job_id is None:
            await _wait_for_next_poll(
                poll_seconds
            )

            continue

        try:
            logger.info(
                "Camera onboarding job processing started | "
                "job_id=%s",
                job_id,
            )

            await run_job(
                job_id
            )

            logger.info(
                "Camera onboarding job processing finished | "
                "job_id=%s",
                job_id,
            )

        except asyncio.CancelledError:
            logger.warning(
                "Camera onboarding job cancelled because "
                "the worker is shutting down | job_id=%s",
                job_id,
            )

            raise

        except Exception:
            # run_job() owns per-job and per-camera status persistence.
            # The worker remains alive so one failed job cannot stop
            # later onboarding jobs.
            logger.exception(
                "Camera onboarding job processing failed | "
                "job_id=%s",
                job_id,
            )

    logger.info(
        "Camera onboarding worker stopped"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        asyncio.run(
            main()
        )

    except KeyboardInterrupt:
        # Windows can raise KeyboardInterrupt directly before
        # the asyncio signal handler completes.
        logger.info(
            "Camera onboarding worker interrupted"
        )