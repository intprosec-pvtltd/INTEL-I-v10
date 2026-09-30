"""Add production PostGIS location support to cameras.

Revision ID: c8f1d2e4a6b7
Revises: b2e7c4f9a1d6
Create Date: 2026-09-05

This migration:

1. Ensures the PostGIS extension is available.
2. Adds cameras.location as geometry(Point, 4326).
3. Backfills geometry from existing longitude/latitude values.
4. Creates a GiST spatial index.
5. Keeps legacy latitude/longitude writes synchronized to location.
6. Preserves latitude and longitude for backward compatibility.

Coordinate convention:
    X = longitude
    Y = latitude
    SRID = 4326 (WGS84)

The PostGIS extension is intentionally NOT removed during downgrade because
other database objects or applications may depend on it.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op


# ---------------------------------------------------------------------------
# Alembic identifiers
# ---------------------------------------------------------------------------

revision: str = "c8f1d2e4a6b7"
down_revision: Union[str, Sequence[str], None] = "b2e7c4f9a1d6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


CAMERAS_TABLE = "cameras"
LOCATION_COLUMN = "location"

LOCATION_INDEX = "idx_cameras_location_gist"

SYNC_FUNCTION = "sync_camera_location_from_coordinates"
SYNC_TRIGGER = "trg_cameras_sync_location"


def upgrade() -> None:
    """Add and populate the canonical PostGIS camera location."""

    # ------------------------------------------------------------------
    # 1. Ensure PostGIS is installed.
    #
    # The production INTEL-I database is expected to already have
    # PostGIS installed. IF NOT EXISTS keeps this operation idempotent
    # for environments where it is already enabled.
    # ------------------------------------------------------------------

    op.execute(
        """
        CREATE EXTENSION IF NOT EXISTS postgis;
        """
    )

    # ------------------------------------------------------------------
    # 2. Add canonical WGS84 geometry column.
    #
    # Nullable is intentional:
    # - cameras may legitimately have no known location;
    # - legacy records may contain incomplete coordinates;
    # - invalid coordinates must never be converted into geometry.
    #
    # latitude/longitude remain in the table for compatibility with
    # existing INTEL-I services and APIs.
    # ------------------------------------------------------------------

    op.execute(
        """
        ALTER TABLE public.cameras
        ADD COLUMN location geometry(Point, 4326);
        """
    )

    # ------------------------------------------------------------------
    # 3. Backfill existing valid coordinates.
    #
    # IMPORTANT:
    # PostGIS point order is:
    #
    #     ST_MakePoint(longitude, latitude)
    #
    # NOT latitude, longitude.
    #
    # Range comparisons also reject +/-Infinity and PostgreSQL NaN for
    # the purposes of this conversion because they cannot satisfy all
    # of the bounded comparisons below.
    #
    # Existing latitude/longitude values are not modified.
    # ------------------------------------------------------------------

    op.execute(
        """
        UPDATE public.cameras
        SET location = ST_SetSRID(
            ST_MakePoint(longitude, latitude),
            4326
        )
        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL
          AND latitude >= -90.0
          AND latitude <= 90.0
          AND longitude >= -180.0
          AND longitude <= 180.0;
        """
    )

    # ------------------------------------------------------------------
    # 4. Create production spatial index.
    #
    # Geometry GiST index supports bounding/spatial operations and
    # allows PostGIS to efficiently reduce candidate sets.
    # ------------------------------------------------------------------

    op.execute(
        """
        CREATE INDEX idx_cameras_location_gist
        ON public.cameras
        USING GIST (location);
        """
    )

    # ------------------------------------------------------------------
    # 5. Synchronization function.
    #
    # Existing INTEL-I code currently writes latitude/longitude.
    # This trigger guarantees those writes also maintain the canonical
    # PostGIS geometry without requiring every existing code path to be
    # changed immediately.
    #
    # Rules:
    # - valid lat/lon -> POINT(longitude latitude), SRID 4326
    # - incomplete coordinates -> NULL geometry
    # - invalid coordinates -> NULL geometry
    #
    # We intentionally do NOT rewrite latitude/longitude here.
    # ------------------------------------------------------------------

    op.execute(
        """
        CREATE OR REPLACE FUNCTION public.sync_camera_location_from_coordinates()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
            IF NEW.latitude IS NOT NULL
               AND NEW.longitude IS NOT NULL
               AND NEW.latitude >= -90.0
               AND NEW.latitude <= 90.0
               AND NEW.longitude >= -180.0
               AND NEW.longitude <= 180.0
            THEN
                NEW.location :=
                    ST_SetSRID(
                        ST_MakePoint(
                            NEW.longitude,
                            NEW.latitude
                        ),
                        4326
                    );
            ELSE
                NEW.location := NULL;
            END IF;

            RETURN NEW;
        END;
        $$;
        """
    )

    # ------------------------------------------------------------------
    # 6. Synchronization trigger.
    #
    # INSERT:
    #   Every newly-created camera gets canonical geometry automatically.
    #
    # UPDATE OF latitude, longitude:
    #   Any legacy code updating coordinates automatically updates
    #   location.
    #
    # A direct location-only update does not invoke this trigger. This
    # avoids recursive/bidirectional synchronization and keeps legacy
    # latitude/longitude as the compatibility write interface.
    # ------------------------------------------------------------------

    op.execute(
        """
        CREATE TRIGGER trg_cameras_sync_location
        BEFORE INSERT OR UPDATE OF latitude, longitude
        ON public.cameras
        FOR EACH ROW
        EXECUTE FUNCTION public.sync_camera_location_from_coordinates();
        """
    )


def downgrade() -> None:
    """Remove INTEL-I camera PostGIS location integration safely."""

    # Trigger must be removed before its function.

    op.execute(
        """
        DROP TRIGGER IF EXISTS trg_cameras_sync_location
        ON public.cameras;
        """
    )

    op.execute(
        """
        DROP FUNCTION IF EXISTS
        public.sync_camera_location_from_coordinates();
        """
    )

    # Remove spatial index before removing the geometry column.

    op.execute(
        """
        DROP INDEX IF EXISTS public.idx_cameras_location_gist;
        """
    )

    op.execute(
        """
        ALTER TABLE public.cameras
        DROP COLUMN IF EXISTS location;
        """
    )

    