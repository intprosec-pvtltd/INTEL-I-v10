"""Add canonical confidence score to alerts.

Revision ID: e7a3c9d1f5b2
Revises: c8f1d2e4a6b7
Create Date: 2026-09-10

This migration:

1. Adds alerts.confidence_score as DOUBLE PRECISION.
2. Backfills existing vehicle-watchlist confidence values.
3. Backfills confidence from linked incident evaluations.
4. Supports both incident_evidence and incident_evidences table names.
5. Constrains confidence values to the inclusive range 0.0..1.0.
6. Preserves NULL when no reliable confidence value exists.

Confidence convention:

    0.0 = 0%
    1.0 = 100%

The canonical database representation is a ratio, not a percentage.

Examples:

    0.8564 -> 85.64%
    0.9989 -> 99.89%

Frontend/API presentation may convert the ratio to a percentage.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op


# ---------------------------------------------------------------------------
# Alembic identifiers
# ---------------------------------------------------------------------------

revision: str = "e7a3c9d1f5b2"
down_revision: Union[str, Sequence[str], None] = "c8f1d2e4a6b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Database object names
# ---------------------------------------------------------------------------

ALERTS_TABLE = "alerts"
CONFIDENCE_COLUMN = "confidence_score"

CONFIDENCE_CONSTRAINT = "ck_alert_confidence_score_range"


def upgrade() -> None:
    """Add canonical alert confidence and backfill historical records."""

    # ------------------------------------------------------------------
    # 1. Add canonical confidence column.
    #
    # DOUBLE PRECISION is used because confidence values originate from
    # AI / ML inference and correlation pipelines.
    #
    # NULL is intentional:
    # - some historical alerts have no meaningful confidence score;
    # - some rule-based alerts may not originate from an ML decision;
    # - fabricating 0.0 would incorrectly imply zero confidence.
    #
    # Canonical storage convention:
    #
    #     0.0 <= confidence_score <= 1.0
    #
    # The frontend can multiply by 100 for presentation.
    # ------------------------------------------------------------------

    op.execute(
        """
        ALTER TABLE public.alerts
        ADD COLUMN IF NOT EXISTS confidence_score DOUBLE PRECISION;
        """
    )

    # ------------------------------------------------------------------
    # 2. Document the canonical representation.
    # ------------------------------------------------------------------

    op.execute(
        """
        COMMENT ON COLUMN public.alerts.confidence_score IS
        'Canonical alert confidence ratio constrained to the inclusive '
        'range 0.0..1.0';
        """
    )

    # ------------------------------------------------------------------
    # 3. Backfill from existing watchlist confidence.
    #
    # INTEL-I already stores watchlist_match_confidence for some alerts,
    # especially vehicle/watchlist correlation results.
    #
    # GREATEST/LEAST protects the canonical confidence field against
    # historical floating-point values outside the intended 0..1 range.
    #
    # Existing confidence_score values are never overwritten.
    # ------------------------------------------------------------------

    op.execute(
        """
        UPDATE public.alerts
        SET confidence_score = GREATEST(
            0.0,
            LEAST(
                1.0,
                watchlist_match_confidence
            )
        )
        WHERE confidence_score IS NULL
          AND watchlist_match_confidence IS NOT NULL;
        """
    )

    # ------------------------------------------------------------------
    # 4. Backfill using linked incident evaluation confidence.
    #
    # Some INTEL-I installations may have:
    #
    #     incident_evidence
    #
    # while others may contain:
    #
    #     incident_evidences
    #
    # The migration detects either table safely.
    #
    # Where multiple evidence rows refer to the same alert, the latest
    # evidence row by evidence.id is selected.
    #
    # Existing canonical values are deliberately preserved.
    # ------------------------------------------------------------------

    op.execute(
        """
        DO $migration$
        DECLARE
            evidence_table TEXT;
        BEGIN

            IF to_regclass('public.incident_evidence') IS NOT NULL THEN

                evidence_table := 'incident_evidence';

            ELSIF to_regclass(
                'public.incident_evidences'
            ) IS NOT NULL THEN

                evidence_table := 'incident_evidences';

            END IF;


            IF evidence_table IS NOT NULL THEN

                EXECUTE format(
                    $sql$

                    UPDATE public.alerts AS alert_row

                    SET confidence_score =
                        linked.confidence_score

                    FROM (
                        SELECT DISTINCT ON (
                            evidence.alert_id
                        )

                            evidence.alert_id,

                            GREATEST(
                                0.0,
                                LEAST(
                                    1.0,
                                    incident.evaluation_confidence
                                )
                            ) AS confidence_score

                        FROM public.%I AS evidence

                        JOIN public.incidents AS incident
                          ON incident.id =
                             evidence.incident_id

                        WHERE evidence.alert_id IS NOT NULL
                          AND incident.evaluation_confidence
                              IS NOT NULL

                        ORDER BY
                            evidence.alert_id,
                            evidence.id DESC

                    ) AS linked

                    WHERE alert_row.id =
                          linked.alert_id

                      AND alert_row.confidence_score
                          IS NULL

                    $sql$,

                    evidence_table
                );

            END IF;

        END
        $migration$;
        """
    )

    # ------------------------------------------------------------------
    # 5. Add database-level confidence invariant.
    #
    # NOT VALID first is preferable for production migrations because
    # PostgreSQL can install the constraint separately from validation.
    #
    # We normalize imported/backfilled values before validating.
    # ------------------------------------------------------------------

    op.execute(
        """
        DO $migration$
        BEGIN

            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname =
                    'ck_alert_confidence_score_range'
                  AND conrelid =
                    'public.alerts'::regclass
            )
            THEN

                ALTER TABLE public.alerts

                ADD CONSTRAINT
                    ck_alert_confidence_score_range

                CHECK (
                    confidence_score IS NULL
                    OR (
                        confidence_score >= 0.0
                        AND confidence_score <= 1.0
                    )
                )

                NOT VALID;

            END IF;

        END
        $migration$;
        """
    )

    # ------------------------------------------------------------------
    # 6. Validate the constraint.
    #
    # Once this succeeds, PostgreSQL guarantees all existing and future
    # non-NULL confidence values are inside 0.0..1.0.
    # ------------------------------------------------------------------

    op.execute(
        """
        ALTER TABLE public.alerts
        VALIDATE CONSTRAINT
            ck_alert_confidence_score_range;
        """
    )


def downgrade() -> None:
    """Remove the canonical alert confidence column safely."""

    # ------------------------------------------------------------------
    # 1. Remove constraint first because it depends on the column.
    # ------------------------------------------------------------------

    op.execute(
        """
        ALTER TABLE public.alerts
        DROP CONSTRAINT IF EXISTS
            ck_alert_confidence_score_range;
        """
    )

    # ------------------------------------------------------------------
    # 2. Remove canonical confidence column.
    #
    # Existing watchlist_match_confidence and incident confidence values
    # remain untouched.
    # ------------------------------------------------------------------

    op.execute(
        """
        ALTER TABLE public.alerts
        DROP COLUMN IF EXISTS confidence_score;
        """
    )